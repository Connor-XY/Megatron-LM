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
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch

TRACE_SCHEMA_VERSION = 1


class EventKind(str, Enum):
    """Semantic categories emitted by a determinism trace."""

    RUNTIME = "runtime"
    PHASE = "phase"
    RECOMPUTE = "recompute"
    COLLECTIVE = "collective"
    OPTIMIZER = "optimizer"
    TENSOR = "tensor"


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


class _CollectiveTraceWork:
    """Delegate a distributed work handle and record its result after ``wait``."""

    def __init__(self, work: Any, results: list[tuple[CollectiveTraceHandle, Any]]) -> None:
        self._work = work
        self._results = results
        self._recorded = False

    def wait(self, *args, **kwargs):
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
_COLLECTIVE_PHASE: ContextVar[str] = ContextVar(
    "megatron_determinism_collective_phase", default="forward"
)


def _distributed_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", 0))


def _callable_name(function: Any) -> str:
    module = getattr(function, "__module__", None)
    name = getattr(function, "__qualname__", type(function).__qualname__)
    return f"{module}.{name}" if module else name


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach()
    if value.layout != torch.strided:
        value = value.to_dense()
    value = value.cpu().contiguous()
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
    payload: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUBLAS_WORKSPACE_CONFIG",
                "CUDA_DEVICE_MAX_CONNECTIONS",
                "NCCL_ALGO",
                "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
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

    def __enter__(self) -> DeterminismTrace:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")
        self._token = _ACTIVE_TRACE.set(self)
        self.record(EventKind.RUNTIME, "runtime", _runtime_payload())
        self.record(EventKind.PHASE, "iteration.begin")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is not None:
            self.record(EventKind.PHASE, "iteration.error", {"exception_type": exc_type.__name__})
        self.record(EventKind.PHASE, "iteration.end")
        if self._token is not None:
            _ACTIVE_TRACE.reset(self._token)
            self._token = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def record(self, kind: EventKind, name: str, payload: Mapping[str, Any] | None = None) -> None:
        """Write one event and flush it so a failed job leaves a usable prefix."""
        if self._file is None:
            raise RuntimeError("DeterminismTrace must be entered before recording events")
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

    def next_checkpoint_id(self) -> int:
        """Return a stable per-iteration checkpoint occurrence number."""
        checkpoint_id = self._checkpoint_sequence
        self._checkpoint_sequence += 1
        return checkpoint_id


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
    return _ACTIVE_TRACE.get()


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
        trace.record(kind, name, payload)


def record_tensor(name: str, tensor: torch.Tensor, **payload: Any) -> None:
    """Record tensor metadata and, when enabled, its exact byte hash."""
    trace = active_trace()
    if trace is None:
        return
    trace.record(
        EventKind.TENSOR,
        name,
        {**payload, **asdict(fingerprint_tensor(tensor, include_hash=trace.hash_tensors))},
    )


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
    trace.record(
        EventKind.COLLECTIVE,
        f"{name}.begin",
        {
            "operation": operation,
            **event_metadata,
            "inputs": _tensor_tree(inputs, include_hash=trace.hash_tensors),
        },
    )
    return CollectiveTraceHandle(
        trace=trace, name=name, operation=operation, metadata=event_metadata
    )


def record_collective_result(handle: CollectiveTraceHandle | None, outputs: Any) -> None:
    """Record collective outputs after the caller has made them ready."""
    if handle is None:
        return
    handle.trace.record(
        EventKind.COLLECTIVE,
        f"{handle.name}.end",
        {
            "operation": handle.operation,
            **handle.metadata,
            "outputs": _tensor_tree(outputs, include_hash=handle.trace.hash_tensors),
        },
    )


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
    return RecomputeTraceHandle(
        trace=trace,
        checkpoint_id=trace.next_checkpoint_id(),
        callable_name=_callable_name(function),
        input_fingerprints=_tensor_tree(inputs, include_hash=trace.hash_tensors),
    )


def record_recompute_phase(handle: RecomputeTraceHandle | None, phase: str, outputs: Any) -> None:
    """Record checkpoint outputs and compare recomputation to the original forward."""
    if handle is None:
        return
    trace = handle.trace
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
        payload["matches_forward"] = _same_tensor_values(fingerprints, handle.forward_fingerprints)
    trace.record(EventKind.RECOMPUTE, f"checkpoint.{phase}", payload)
