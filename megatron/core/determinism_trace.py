# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Opt-in structured tracing for training determinism investigations.

The recorder is inactive by default. When an iteration trace is active, every
rank writes its own JSON Lines file so diagnostics never add collectives or
cross-rank ordering constraints to the training step being observed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass
from enum import Enum
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import torch

TRACE_SCHEMA_VERSION = 1

_RUNTIME_PACKAGES = ("flash-attn", "mamba-ssm", "transformer-engine", "triton")


class EventKind(str, Enum):
    """Semantic categories emitted by a determinism trace."""

    RUNTIME = "runtime"
    PHASE = "phase"
    RECOMPUTE = "recompute"
    COLLECTIVE = "collective"
    OPTIMIZER = "optimizer"
    TENSOR = "tensor"
    OP = "op"


@dataclass(frozen=True)
class TensorFingerprint:
    """Stable metadata and an optional exact byte hash for one tensor."""

    shape: list[int]
    dtype: str
    device: str
    layout: str
    numel: int
    requires_grad: bool
    sha256: str | None = None


@dataclass(frozen=True)
class TraceEvent:
    """Versioned JSONL event written by :class:`DeterminismTrace`."""

    schema_version: int
    sequence: int
    rank: int
    iteration: int
    kind: str
    name: str
    payload: dict[str, Any]


@dataclass
class RecomputeTraceHandle:
    """Identity shared by a checkpoint's original forward and recomputation."""

    trace: DeterminismTrace
    checkpoint_id: int
    callable_name: str
    input_fingerprints: list[dict[str, Any]]
    forward_fingerprints: list[dict[str, Any]] | None = None


@dataclass
class CollectiveTraceHandle:
    """State shared by the begin and end events for one collective."""

    trace: DeterminismTrace
    name: str
    operation: str
    metadata: dict[str, Any]
    completed: bool = False


class _CollectiveTraceWork:
    """Delegate a distributed work handle and record its result after ``wait``."""

    def __init__(self, work: Any, results: list[tuple[CollectiveTraceHandle, Any]]) -> None:
        self._work = work
        self._results = results
        self._recorded = False

    def wait(self, *args, **kwargs):
        """Wait for the underlying work and record completed outputs once."""
        result = self._work.wait(*args, **kwargs)
        if result is not False and not self._recorded:
            for handle, outputs in self._results:
                record_collective_result(handle, outputs)
            self._recorded = True
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._work, name)


_ACTIVE_TRACE: ContextVar[DeterminismTrace | None] = ContextVar(
    "megatron_determinism_trace", default=None
)
_PROCESS_ACTIVE_TRACE: DeterminismTrace | None = None
_COLLECTIVE_PHASE: ContextVar[str] = ContextVar(
    "megatron_determinism_collective_phase", default="forward"
)


def _distributed_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", 0))


def _callable_name(function: Any) -> str:
    """Return a stable module-qualified name for a traced callable."""
    module = getattr(function, "__module__", None)
    name = getattr(function, "__qualname__", type(function).__qualname__)
    return f"{module}.{name}" if module else name


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach()
    if value.layout != torch.strided:
        value = value.to_dense()
    # A size-one expanded tensor can report contiguous while retaining a zero
    # stride, which prevents byte reinterpretation. Materialize canonical dense
    # storage before hashing every strided layout.
    value = value.cpu().clone(memory_format=torch.contiguous_format)
    return value.reshape(-1).view(torch.uint8).numpy().tobytes()


def fingerprint_tensor(tensor: torch.Tensor, *, include_hash: bool) -> TensorFingerprint:
    """Describe a tensor and optionally hash all of its bytes exactly."""
    digest = hashlib.sha256(_tensor_bytes(tensor)).hexdigest() if include_hash else None
    return TensorFingerprint(
        shape=list(tensor.shape),
        dtype=str(tensor.dtype),
        device=str(tensor.device),
        layout=str(tensor.layout),
        numel=tensor.numel(),
        requires_grad=tensor.requires_grad,
        sha256=digest,
    )


# ---------------------------------------------------------------------------
# Device-side tensor signature.
#
# A 128-bit digest of a tensor's raw bytes computed on the tensor's own device,
# ported from Megatron-Bridge's determinism debug tool (``signature.py`` on the
# ``zhiyul/determinism-debug-tool`` branch). The bytes are reinterpreted as
# ``int64`` lanes and reduced with two integer reductions that mix value and
# position. Integer addition is exact and wraps mod 2**64, so the digest is
# identical regardless of reduction order, chunk size, GPU, process, or
# physical topology — and only two 64-bit scalars ever cross the PCIe bus,
# which is why this is cheap enough to fingerprint every ATen op output while
# the SHA256 path above (a full device-to-host copy per tensor) is not. It is
# bit-exact over all bytes: it distinguishes ``-0.0`` from ``+0.0`` and
# differing NaN payloads.
# ---------------------------------------------------------------------------

_SIG_C1 = 6364136223846793005  # 0x5851F42D4C957F2D (LCG multiplier)
_SIG_C2 = 1442695040888963407  # 0x14057B7EF767814F (LCG increment)
_SIG_MASK64 = 0xFFFFFFFFFFFFFFFF
# At most 16M int64 lanes (~128 MB) per reduction chunk. Chunk sums accumulate
# exactly, so the digest is independent of the chunk size.
_SIG_CHUNK_LANES = 1 << 24
# Sentinel for empty tensors. A real all-zero tensor digests to a nonzero value
# through the position term, so this cannot collide with actual data.
_SIG_EMPTY_DIGEST = "0" * 32


def _device_digest(x: torch.Tensor) -> str:
    """128-bit hex digest of ``x``'s raw bytes, reduced on ``x``'s own device.

    ``x`` must be contiguous, real, and non-empty.
    """
    # reshape(-1) first so 0-dim scalars (e.g. a loss) work — ``view`` cannot
    # reinterpret a 0-dim tensor.
    u8 = x.reshape(-1).view(torch.uint8)
    pad = (-u8.numel()) % 8
    if pad:
        u8 = torch.cat([u8, u8.new_zeros(pad)])
    lanes = u8.view(torch.int64)

    h1 = 0
    h2 = 0
    n = lanes.numel()
    for start in range(0, n, _SIG_CHUNK_LANES):
        seg = lanes[start : start + _SIG_CHUNK_LANES]
        idx = torch.arange(start, start + seg.numel(), device=seg.device, dtype=torch.int64)
        # Mix each lane with its absolute position, then take a linear and a
        # nonlinear (squared) reduction so multi-lane cancellations in the
        # linear sum are still caught. ``idx + 1`` is never zero, so lane 0 of
        # an all-zero tensor still receives a nonzero position term.
        mixed = seg * _SIG_C1 + (idx + 1) * _SIG_C2
        both = torch.stack((mixed.sum(), (mixed * mixed).sum()))
        s1, s2 = both.tolist()  # one D2H transfer of two scalars per chunk
        h1 = (h1 + s1) & _SIG_MASK64
        h2 = (h2 + s2) & _SIG_MASK64
    return f"{h1:016x}{h2:016x}"


def tensor_signature(tensor: Any) -> dict[str, Any] | None:
    """Cross-process-stable fingerprint of a tensor as a JSON-ready dict.

    Returns ``None`` for non-tensors. The digest covers every raw byte of the
    tensor (complex tensors are viewed as interleaved real/imaginary floats).
    """
    if not isinstance(tensor, torch.Tensor):
        return None
    shape = list(tensor.shape)
    dtype = str(tensor.dtype)
    numel = tensor.numel()
    if numel == 0:
        return {"shape": shape, "dtype": dtype, "numel": 0, "digest": _SIG_EMPTY_DIGEST}
    x = tensor.detach()
    if x.is_complex():
        x = torch.view_as_real(x)
    if x.layout != torch.strided:
        x = x.to_dense()
    x = x.contiguous()
    return {"shape": shape, "dtype": dtype, "numel": numel, "digest": _device_digest(x)}


def _tensor_tree(value: Any, *, include_hash: bool, path: str = "") -> list[dict[str, Any]]:
    fingerprints = []
    if isinstance(value, torch.Tensor):
        fingerprints.append(
            {
                "path": path or "<root>",
                **asdict(fingerprint_tensor(value, include_hash=include_hash)),
            }
        )
    elif isinstance(value, Mapping):
        for key in sorted(value, key=str):
            child = f"{path}/{key}" if path else str(key)
            fingerprints.extend(_tensor_tree(value[key], include_hash=include_hash, path=child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child = f"{path}/{index}" if path else str(index)
            fingerprints.extend(_tensor_tree(item, include_hash=include_hash, path=child))
    return fingerprints


def _same_tensor_values(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> bool:
    """Compare fingerprinted values while ignoring expected autograd metadata changes."""
    value_fields = ("path", "shape", "dtype", "device", "layout", "numel", "sha256")
    return [tuple(item[field] for field in value_fields) for item in left] == [
        tuple(item[field] for field in value_fields) for item in right
    ]


def _runtime_payload() -> dict[str, Any]:
    package_versions = {}
    for package_name in _RUNTIME_PACKAGES:
        try:
            package_versions[package_name] = importlib_metadata.version(package_name)
        except importlib_metadata.PackageNotFoundError:
            package_versions[package_name] = None

    payload: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "package_versions": package_versions,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUBLAS_WORKSPACE_CONFIG",
                "CUDA_DEVICE_MAX_CONNECTIONS",
                "MAMBA_DETERMINISTIC",
                "NCCL_ALGO",
                "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
                "NVTE_FLASH_ATTN",
                "NVTE_FUSED_ATTN",
                "NVTE_UNFUSED_ATTN",
                "TRITON_CACHE_AUTOTUNING",
            )
        },
    }
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        allocator_getter = getattr(torch.cuda.memory, "get_allocator_backend", None)
        payload["device_name"] = torch.cuda.get_device_name(device)
        payload["device_capability"] = list(torch.cuda.get_device_capability(device))
        payload["cudnn_version"] = torch.backends.cudnn.version()
        payload["allocator_backend"] = allocator_getter() if allocator_getter else None
    return payload


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, Enum):
        return value.value
    return value


class DeterminismTrace:
    """Rank-local writer for one selected training iteration."""

    def __init__(
        self,
        trace_dir: str | Path,
        iteration: int,
        *,
        hash_tensors: bool = False,
        rank: int | None = None,
    ) -> None:
        self.iteration = iteration
        self.rank = _distributed_rank() if rank is None else rank
        self.hash_tensors = hash_tensors
        self.path = Path(trace_dir) / f"iter_{iteration:07d}" / f"rank_{self.rank:05d}.jsonl"
        self._sequence = 0
        self._checkpoint_sequence = 0
        self._file = None
        self._token: Token | None = None
        self._previous_process_trace: DeterminismTrace | None = None
        self._pending_collectives = 0
        self._lock = threading.RLock()

    def __enter__(self) -> DeterminismTrace:
        global _PROCESS_ACTIVE_TRACE
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")
        self._token = _ACTIVE_TRACE.set(self)
        try:
            self.record(EventKind.RUNTIME, "runtime", _runtime_payload())
            self.record(EventKind.PHASE, "iteration.begin")
        except Exception:
            _ACTIVE_TRACE.reset(self._token)
            self._token = None
            self._file.close()
            self._file = None
            raise
        self._previous_process_trace = _PROCESS_ACTIVE_TRACE
        _PROCESS_ACTIVE_TRACE = self
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        global _PROCESS_ACTIVE_TRACE
        with self._lock:
            try:
                if exc_type is not None:
                    self.record(
                        EventKind.PHASE, "iteration.error", {"exception_type": exc_type.__name__}
                    )
                self.record(
                    EventKind.PHASE,
                    "iteration.end",
                    {"pending_collectives": self._pending_collectives},
                )
            finally:
                if self._token is not None:
                    _ACTIVE_TRACE.reset(self._token)
                    self._token = None
                if _PROCESS_ACTIVE_TRACE is self:
                    _PROCESS_ACTIVE_TRACE = self._previous_process_trace
                self._previous_process_trace = None
                if self._file is not None:
                    self._file.close()
                    self._file = None

    def record(self, kind: EventKind, name: str, payload: Mapping[str, Any] | None = None) -> None:
        """Write one event and flush it so a failed job leaves a usable prefix."""
        if not self.record_if_open(kind, name, payload):
            raise RuntimeError("DeterminismTrace must be entered before recording events")

    def record_if_open(
        self, kind: EventKind, name: str, payload: Mapping[str, Any] | None = None
    ) -> bool:
        """Write one event if the trace is open, returning whether it was recorded."""
        with self._lock:
            if self._file is None:
                return False
            event = TraceEvent(
                schema_version=TRACE_SCHEMA_VERSION,
                sequence=self._sequence,
                rank=self.rank,
                iteration=self.iteration,
                kind=kind.value,
                name=name,
                payload=_json_safe(dict(payload or {})),
            )
            self._sequence += 1
            self._file.write(json.dumps(asdict(event), sort_keys=True, allow_nan=False) + "\n")
            self._file.flush()
            return True

    def next_checkpoint_id(self) -> int:
        """Return a stable per-iteration checkpoint occurrence number."""
        with self._lock:
            checkpoint_id = self._checkpoint_sequence
            self._checkpoint_sequence += 1
            return checkpoint_id

    @property
    def is_open(self) -> bool:
        """Return whether events can still be appended to this trace."""
        with self._lock:
            return self._file is not None

    @property
    def pending_collectives(self) -> int:
        """Return the number of collective begin events without an in-window end."""
        with self._lock:
            return self._pending_collectives


@contextmanager
def trace_iteration(
    trace_dir: str | Path | None, iteration: int, *, hash_tensors: bool = False
) -> Iterator[DeterminismTrace | None]:
    """Activate a rank-local trace, or yield ``None`` when tracing is disabled."""
    if trace_dir is None:
        yield None
        return
    with DeterminismTrace(trace_dir, iteration, hash_tensors=hash_tensors) as trace:
        yield trace


def active_trace() -> DeterminismTrace | None:
    """Return the trace active in the current execution context, if any."""
    return _ACTIVE_TRACE.get() or _PROCESS_ACTIVE_TRACE


def collective_trace_phase() -> str:
    """Return the semantic execution phase for newly launched collectives."""
    return _COLLECTIVE_PHASE.get()


@contextmanager
def use_determinism_trace(
    trace: DeterminismTrace | None, *, collective_phase: str | None = None
) -> Iterator[None]:
    """Temporarily propagate an existing trace into an autograd execution context."""
    if trace is None:
        yield
        return
    trace_token = _ACTIVE_TRACE.set(trace)
    phase_token = _COLLECTIVE_PHASE.set(collective_phase) if collective_phase is not None else None
    try:
        yield
    finally:
        if phase_token is not None:
            _COLLECTIVE_PHASE.reset(phase_token)
        _ACTIVE_TRACE.reset(trace_token)


def trace_tensor_hashes_enabled() -> bool:
    """Return whether the active trace requests exact tensor byte hashes."""
    trace = active_trace()
    return trace is not None and trace.hash_tensors


def record_event(kind: EventKind, name: str, payload: Mapping[str, Any] | None = None) -> None:
    """Record an event when tracing is active; otherwise do nothing."""
    trace = active_trace()
    if trace is not None:
        trace.record_if_open(kind, name, payload)


def record_tensor(name: str, tensor: torch.Tensor, **payload: Any) -> None:
    """Record tensor metadata and, when enabled, its exact byte hash."""
    trace = active_trace()
    if trace is None:
        return
    with trace._lock:
        if trace.is_open:
            trace.record(
                EventKind.TENSOR,
                name,
                {**payload, **asdict(fingerprint_tensor(tensor, include_hash=trace.hash_tensors))},
            )


def record_optimizer_state(name: str, optimizer: Any) -> None:
    """Record exact local optimizer parameters and state using stable ordinals."""
    trace = active_trace()
    if trace is None or not trace.hash_tensors:
        return

    optimizer_instances = getattr(optimizer, "chained_optimizers", None) or [optimizer]
    for optimizer_index, optimizer_instance in enumerate(optimizer_instances):
        if getattr(optimizer_instance, "is_stub_optimizer", False):
            continue
        param_groups = getattr(optimizer_instance, "param_groups", None)
        optimizer_state = getattr(optimizer_instance, "state", None)
        if param_groups is None or optimizer_state is None:
            continue
        for group_index, param_group in enumerate(param_groups):
            for param_index, param in enumerate(param_group["params"]):
                if not isinstance(param, torch.Tensor):
                    continue
                prefix = f"{name}/optimizer{optimizer_index}/group{group_index}/param{param_index}"
                record_tensor(f"{prefix}/main_param", param)
                scalar_state = {}
                for state_name, state_value in sorted(
                    optimizer_state.get(param, {}).items(), key=lambda item: str(item[0])
                ):
                    if isinstance(state_value, torch.Tensor):
                        record_tensor(f"{prefix}/{state_name}", state_value)
                    elif state_value is None or isinstance(state_value, (bool, int, float, str)):
                        scalar_state[str(state_name)] = state_value
                if scalar_state:
                    record_event(EventKind.OPTIMIZER, f"{prefix}/scalars", scalar_state)


def begin_collective_trace(
    name: str,
    operation: str,
    inputs: Any,
    *,
    group: torch.distributed.ProcessGroup | None = None,
    metadata: Mapping[str, Any] | None = None,
    trace: DeterminismTrace | None = None,
) -> CollectiveTraceHandle | None:
    """Record collective inputs without introducing communication."""
    trace = active_trace() if trace is None else trace
    if trace is None:
        return None
    event_metadata = dict(metadata or {})
    if group is not None:
        event_metadata.update(
            {
                "group_size": group.size(),
                "group_rank": group.rank(),
                "backend": str(torch.distributed.get_backend(group)),
            }
        )
    with trace._lock:
        if not trace.is_open:
            return None
        trace.record(
            EventKind.COLLECTIVE,
            f"{name}.begin",
            {
                "operation": operation,
                **event_metadata,
                "inputs": _tensor_tree(inputs, include_hash=trace.hash_tensors),
            },
        )
        trace._pending_collectives += 1
    return CollectiveTraceHandle(
        trace=trace, name=name, operation=operation, metadata=event_metadata
    )


def record_collective_result(handle: CollectiveTraceHandle | None, outputs: Any) -> None:
    """Record collective outputs after the caller has made them ready."""
    if handle is None:
        return
    trace = handle.trace
    with trace._lock:
        if handle.completed:
            return
        if trace.is_open:
            trace.record(
                EventKind.COLLECTIVE,
                f"{handle.name}.end",
                {
                    "operation": handle.operation,
                    **handle.metadata,
                    "outputs": _tensor_tree(outputs, include_hash=trace.hash_tensors),
                },
            )
            assert trace._pending_collectives > 0
            trace._pending_collectives -= 1
        handle.completed = True


def trace_collective_work(work: Any, handle: CollectiveTraceHandle | None, outputs: Any) -> Any:
    """Return a work handle that records outputs after its existing wait call."""
    if handle is None:
        return work
    return _CollectiveTraceWork(work, [(handle, outputs)])


def trace_collective_work_group(
    work: Any, results: list[tuple[CollectiveTraceHandle | None, Any]]
) -> Any:
    """Record several collective outputs completed by one grouped work handle."""
    active_results = [(handle, outputs) for handle, outputs in results if handle is not None]
    if not active_results:
        return work
    return _CollectiveTraceWork(work, active_results)


def begin_recompute_trace(function: Any, inputs: Any) -> RecomputeTraceHandle | None:
    """Allocate the identity shared by a checkpoint forward and recomputation."""
    trace = active_trace()
    if trace is None:
        return None
    with trace._lock:
        if not trace.is_open:
            return None
        checkpoint_id = trace.next_checkpoint_id()
        return RecomputeTraceHandle(
            trace=trace,
            checkpoint_id=checkpoint_id,
            callable_name=_callable_name(function),
            input_fingerprints=_tensor_tree(inputs, include_hash=trace.hash_tensors),
        )


def record_recompute_phase(handle: RecomputeTraceHandle | None, phase: str, outputs: Any) -> None:
    """Record checkpoint outputs and compare recomputation to the original forward."""
    if handle is None:
        return
    trace = handle.trace
    with trace._lock:
        if not trace.is_open:
            return
        fingerprints = _tensor_tree(outputs, include_hash=trace.hash_tensors)
        payload: dict[str, Any] = {
            "checkpoint_id": handle.checkpoint_id,
            "callable": handle.callable_name,
            "inputs": handle.input_fingerprints,
            "outputs": fingerprints,
        }
        if phase == "forward":
            handle.forward_fingerprints = fingerprints
        elif trace.hash_tensors and handle.forward_fingerprints is not None:
            payload["matches_forward"] = _same_tensor_values(
                fingerprints, handle.forward_fingerprints
            )
        trace.record(EventKind.RECOMPUTE, f"checkpoint.{phase}", payload)
