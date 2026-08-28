# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Low-perturbation module flight recorder for determinism investigations.

The recorder is intentionally private and opt-in. On the hot path it performs
no host reads, file I/O, explicit CUDA synchronization, or communication. Each
tensor is reduced by one prewarmed device kernel into a unique row of a single
preallocated digest arena. The arena is copied to the host once from an
``atexit`` handler and written in chronological hook order. That final dump
waits for all device streams, after training has finished, before reading the
arena.

This is a fingerprint, not a byte-for-byte tensor dump. Two independent,
position-sensitive 64-bit integer reductions cover every logical tensor byte.
Integer addition wraps modulo 2**64, so CTA arrival order cannot change the
digest. The probability of an accidental collision is small enough for
localization, but a suspected birth should still be confirmed with an exact
snapshot of the narrowed tensor.

The hook API and primary environment variables are compatible with the older
``_nt3_modprobe`` implementation::

    NT3_MOD_PROBE=1
    NT3_MOD_PROBE_RANKS=0,1,64,65
    NT3_MOD_PROBE_DEPTH=9
    NT3_MOD_PROBE_CALLS=400
    NT3_MOD_PROBE_DIR=/path/to/run-a

Additional dose controls::

    NT3_MOD_PROBE_INCLUDE=decoder.layers.0  # also filters manual boundary keys
    NT3_MOD_PROBE_LEAF_ONLY=1
    NT3_MOD_PROBE_SLOTS=2000000
    NT3_MOD_PROBE_HOOKS=full        # full, forward, or none (manual boundaries)
    NT3_MOD_PROBE_BACKEND=triton    # triton (default) or explicit torch fallback

``NT3_MOD_PROBE_DIR`` is expanded at dump time, so a literal
``$SLURM_JOB_ID`` can safely separate two jobs submitted by the same launcher.

The one-kernel hash still adds stream work; no content observer can be both
faithful and literally zero-perturbation. Use the validity gate: an instrumented
dirty control must still diverge before treating a clean trace as evidence.
"""

from __future__ import annotations

import atexit
import itertools
import os
import zlib
from dataclasses import dataclass
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised in environments without Triton
    triton = None
    tl = None


_ENABLED = os.environ.get("NT3_MOD_PROBE") == "1"
_RANK_SPEC = os.environ.get("NT3_MOD_PROBE_RANKS", "0")
_ALL_RANKS = _RANK_SPEC.strip().lower() == "all"
_RANKS = set() if _ALL_RANKS else {int(rank) for rank in _RANK_SPEC.split(",") if rank.strip()}
_DEPTH = int(os.environ.get("NT3_MOD_PROBE_DEPTH", "4") or "4")
_MAX_CALLS = int(os.environ.get("NT3_MOD_PROBE_CALLS", "400") or "400")
_OUTPUT_DIR = os.environ.get("NT3_MOD_PROBE_DIR", "/tmp/nt3modprobe")
_INCLUDE = os.environ.get("NT3_MOD_PROBE_INCLUDE", "")
_LEAF_ONLY = os.environ.get("NT3_MOD_PROBE_LEAF_ONLY", "0") == "1"
_MAX_SLOTS = int(os.environ.get("NT3_MOD_PROBE_SLOTS", "2000000") or "2000000")
_HOOK_MODE = os.environ.get("NT3_MOD_PROBE_HOOKS", "full").strip().lower()
_REQUESTED_BACKEND = os.environ.get("NT3_MOD_PROBE_BACKEND", "triton").strip().lower()

_HASH_WORDS = 2
_BLOCK_SIZE = 1024


if triton is not None:

    @triton.jit
    def _triton_hash_kernel(data_ptr, nbytes, digest_ptr, salt, BLOCK_SIZE: tl.constexpr):
        """Hash raw bytes with associative integer reductions.

        Constants fit signed int64 so the same formula is available in the
        PyTorch fallback. Arithmetic intentionally wraps modulo 2**64.
        """
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < nbytes
        values = tl.load(data_ptr + offsets, mask=mask, other=0).to(tl.int64)
        positions = offsets.to(tl.int64) + 1 + salt * 1315423911

        first = (values ^ (positions * 6364136223846793005)) * 1442695040888963407
        second = values + positions * 3202034522624059733
        second = second * second * 3935559000370003845

        first_sum = tl.sum(first, axis=0)
        second_sum = tl.sum(second, axis=0)
        tl.atomic_add(digest_ptr, first_sum)
        tl.atomic_add(digest_ptr + 1, second_sum)


@dataclass(frozen=True)
class _FieldRecord:
    name: str
    slot: int | None
    metadata: str
    status: str


@dataclass(frozen=True)
class _EventRecord:
    sequence: int
    key: str
    call: int
    fields: tuple[_FieldRecord, ...]


class _TritonHashBackend:
    name = "triton"

    def __init__(self) -> None:
        if triton is None:
            raise RuntimeError("Triton is unavailable")

    def hash_into(self, raw_bytes: torch.Tensor, output: torch.Tensor, salt: int) -> None:
        if raw_bytes.numel() == 0:
            return
        grid = (triton.cdiv(raw_bytes.numel(), _BLOCK_SIZE),)
        _triton_hash_kernel[grid](
            raw_bytes, raw_bytes.numel(), output, salt, BLOCK_SIZE=_BLOCK_SIZE, num_warps=4
        )

    def warmup(self, output: torch.Tensor) -> None:
        # Compilation and temporary allocation happen before hooks are installed.
        # All operations use the current stream, so training cannot overtake them.
        dummy = torch.arange(64, dtype=torch.uint8, device=output.device)
        self.hash_into(dummy, output, 0)
        output.zero_()


class _TorchHashBackend:
    """Sync-free compatibility fallback; higher dose than the Triton backend."""

    name = "torch"

    def hash_into(self, raw_bytes: torch.Tensor, output: torch.Tensor, salt: int) -> None:
        if raw_bytes.numel() == 0:
            return
        values = raw_bytes.to(torch.int64)
        positions = torch.arange(1, values.numel() + 1, dtype=torch.int64, device=values.device)
        positions = positions + salt * 1315423911
        first = (values ^ (positions * 6364136223846793005)) * 1442695040888963407
        second = values + positions * 3202034522624059733
        second = second * second * 3935559000370003845
        output[0].add_(first.sum())
        output[1].add_(second.sum())

    def warmup(self, output: torch.Tensor) -> None:
        output.zero_()


_digest_buffer: torch.Tensor | None = None
_backend: _TritonHashBackend | _TorchHashBackend | None = None
_backend_error: str | None = None
_slots_used = 0
_dropped_slots = 0
_events: list[_EventRecord] = []
_counts: dict[str, int] = {}
_sequence = itertools.count()
_hook_handles: list[Any] = []
_dumped = False


def _rank() -> int:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) or "0")


def _active() -> bool:
    return _ENABLED and (_ALL_RANKS or _rank() in _RANKS)


def _flat_tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        tensors = []
        for item in value:
            tensors.extend(_flat_tensors(item))
        return tensors
    if isinstance(value, dict):
        tensors = []
        for key in sorted(value, key=str):
            tensors.extend(_flat_tensors(value[key]))
        return tensors
    return []


_QUANTIZED_BUFFER_ATTRS = (
    "_data",
    "_rowwise_data",
    "_columnwise_data",
    "_scale_inv",
    "_rowwise_scale_inv",
    "_columnwise_scale_inv",
)


def _tensor_parts(tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
    if type(tensor) is torch.Tensor:  # pylint: disable=unidiomatic-typecheck
        return [("data", tensor)]
    parts = []
    for name in _QUANTIZED_BUFFER_ATTRS:
        value = getattr(tensor, name, None)
        if type(value) is torch.Tensor:  # pylint: disable=unidiomatic-typecheck
            parts.append((name, value))
    return parts or [("data", tensor)]


def _raw_bytes(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach()
    if value.layout != torch.strided:
        value = value.to_dense()
    try:
        return value.contiguous().reshape(-1).view(torch.uint8)
    except RuntimeError:
        return value.clone(memory_format=torch.contiguous_format).reshape(-1).view(torch.uint8)


def _metadata(tensor: torch.Tensor) -> str:
    shape = "x".join(str(size) for size in tensor.shape) or "scalar"
    stride = (
        "x".join(str(value) for value in tensor.stride())
        if tensor.layout == torch.strided
        else "na"
    )
    return f"{tensor.dtype}:{shape}:{stride}:{tensor.numel()}"


def _reserve_slot() -> int | None:
    global _dropped_slots, _slots_used
    if _slots_used >= _MAX_SLOTS:
        _dropped_slots += 1
        return None
    slot = _slots_used
    _slots_used += 1
    return slot


def _snapshot_field(name: str, tensor: torch.Tensor) -> _FieldRecord:
    assert _backend is not None
    assert _digest_buffer is not None
    slot = _reserve_slot()
    metadata = _metadata(tensor)
    if slot is None:
        return _FieldRecord(name, None, metadata, "arena_full")

    try:
        for part_index, (part_name, part) in enumerate(_tensor_parts(tensor)):
            salt_text = f"{metadata}|{part_index}|{part_name}".encode("utf-8")
            salt = zlib.crc32(salt_text) & 0x7FFFFFFF
            _backend.hash_into(_raw_bytes(part), _digest_buffer[slot], salt)
    except Exception as error:  # noqa: BLE001 - diagnostics must never stop training
        return _FieldRecord(name, slot, metadata, f"hash_error:{type(error).__name__}")
    return _FieldRecord(name, slot, metadata, "ok")


def enabled_for(key: str) -> bool:
    """Return whether this rank and manual boundary key should record."""
    return _active() and _digest_buffer is not None and (not _INCLUDE or _INCLUDE in key)


def record_tensors(key: str, pairs: list[tuple[str, torch.Tensor | None]]) -> None:
    """Record a chronological boundary without any host read or file write.

    This public helper can also be called at FSDP boundaries that module hooks
    cannot observe, for example immediately before and after reduce-scatter.
    """
    if not enabled_for(key):
        return
    call = _counts.get(key, 0)
    _counts[key] = call + 1
    if call >= _MAX_CALLS:
        return
    fields = tuple(
        _snapshot_field(field, tensor)
        for field, tensor in pairs
        if isinstance(tensor, torch.Tensor)
    )
    if fields:
        _events.append(_EventRecord(next(_sequence), key, call, fields))


def _format_digest(words: list[int]) -> str:
    return "".join(f"{word & 0xFFFFFFFFFFFFFFFF:016x}" for word in words)


def dump() -> None:
    """Copy the digest arena once and write chronological rank-local records."""
    global _dumped
    if _dumped or _digest_buffer is None:
        return
    _dumped = True
    try:
        # Hashes may have been launched from autograd or communication streams.
        # Synchronize only here, after training, so the D2H copy observes all
        # rows without putting a synchronization point on the training path.
        torch.cuda.synchronize(_digest_buffer.device)
        # This is the only device-to-host transfer in the recorder.
        host_rows = _digest_buffer[:_slots_used].cpu().tolist()
        output_dir = os.path.expandvars(_OUTPUT_DIR)
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"rank_{_rank():05d}.txt")
        with open(path, "w", encoding="utf-8") as output_file:
            output_file.write(
                "# nt3-modprobe-v2 "
                f"backend={_backend.name if _backend else 'none'} "
                f"backend_error={_backend_error or 'none'} "
                f"events={len(_events)} slots={_slots_used} dropped_slots={_dropped_slots}\n"
            )
            for event in _events:
                tokens = []
                for field in event.fields:
                    digest = (
                        _format_digest(host_rows[field.slot])
                        if field.slot is not None
                        else "unavailable"
                    )
                    tokens.extend(
                        (
                            f"{field.name}={digest}",
                            f"{field.name}.meta={field.metadata}",
                            f"{field.name}.status={field.status}",
                        )
                    )
                output_file.write(
                    f"rank={_rank()} seq={event.sequence} mod={event.key} "
                    f"call={event.call} {' '.join(tokens)}\n"
                )
    except Exception:  # noqa: BLE001 - process teardown must not be made fatal by diagnostics
        pass


def _initialize_backend(warmup_output: torch.Tensor) -> None:
    global _backend, _backend_error
    assert _digest_buffer is not None
    if _REQUESTED_BACKEND not in ("triton", "torch"):
        raise ValueError("NT3_MOD_PROBE_BACKEND must be 'triton' or 'torch'")
    if _REQUESTED_BACKEND == "triton":
        try:
            _backend = _TritonHashBackend()
            _backend.warmup(warmup_output)
            return
        except Exception as error:  # noqa: BLE001 - report initialization failure clearly
            _backend_error = f"{type(error).__name__}:{error}"
            raise RuntimeError(
                "the low-dose Triton hash backend failed to initialize; "
                "set NT3_MOD_PROBE_BACKEND=torch only if its higher observer dose is acceptable"
            ) from error
    _backend = _TorchHashBackend()
    _backend.warmup(warmup_output)


def install(model: Any) -> int:
    """Install module hooks and preallocate all persistent device storage."""
    global _digest_buffer
    if not _active():
        return 0
    if _MAX_CALLS <= 0 or _MAX_SLOTS <= 0:
        raise ValueError("NT3_MOD_PROBE_CALLS and NT3_MOD_PROBE_SLOTS must be positive")
    if _HOOK_MODE not in ("full", "forward", "none"):
        raise ValueError("NT3_MOD_PROBE_HOOKS must be 'full', 'forward', or 'none'")
    if _digest_buffer is not None:
        return len(_hook_handles)

    _digest_buffer = torch.zeros(
        (_MAX_SLOTS, _HASH_WORDS), dtype=torch.int64, device=torch.cuda.current_device()
    )
    # Keep JIT warmup writes out of the arena: its first row may be consumed by
    # a hook running from an autograd worker thread with a different current
    # stream.
    warmup_output = torch.zeros((_HASH_WORDS,), dtype=torch.int64, device=_digest_buffer.device)
    _initialize_backend(warmup_output)

    chunks = (
        [] if _HOOK_MODE == "none" else (model if isinstance(model, (list, tuple)) else [model])
    )
    modules = []
    for chunk in chunks:
        if not isinstance(chunk, torch.nn.Module):
            continue
        for name, module in chunk.named_modules():
            if not name or name.count(".") > _DEPTH:
                continue
            if _INCLUDE and _INCLUDE not in name:
                continue
            if _LEAF_ONLY and any(True for _ in module.children()):
                continue
            modules.append((name, module))

    for name, module in modules:

        def forward_hook(_module, inputs, outputs, module_name=name):
            record_tensors(
                f"{module_name}|fwd",
                [(f"in{index}", tensor) for index, tensor in enumerate(_flat_tensors(inputs))]
                + [(f"out{index}", tensor) for index, tensor in enumerate(_flat_tensors(outputs))],
            )

        def backward_hook(_module, grad_inputs, grad_outputs, module_name=name):
            record_tensors(
                f"{module_name}|bwd",
                [
                    (f"gout{index}", tensor)
                    for index, tensor in enumerate(_flat_tensors(grad_outputs))
                ]
                + [
                    (f"gin{index}", tensor)
                    for index, tensor in enumerate(_flat_tensors(grad_inputs))
                ],
            )

        _hook_handles.append(module.register_forward_hook(forward_hook))
        if _HOOK_MODE == "full":
            try:
                _hook_handles.append(module.register_full_backward_hook(backward_hook))
            except Exception:  # noqa: BLE001 - a few extension modules reject full backward hooks
                pass

    atexit.register(dump)
    return len(modules)
