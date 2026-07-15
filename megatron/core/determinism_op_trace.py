# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Blanket ATen-op fingerprinting for determinism debugging.

This is the op-level capture layer that complements the semantic tracer in
:mod:`megatron.core.determinism_trace`. A :class:`~torch.utils._python_dispatch.TorchDispatchMode`
fingerprints the output of **every** ATen operation with the cheap device-side
signature (:func:`megatron.core.determinism_trace.tensor_signature`) and records
it into the active rank-local trace. The semantic tracer names divergences at
instrumented boundaries (collectives, recompute, optimizer, P2P); this layer
names the exact compute op *between* those boundaries — non-determinism that
originates in a compute kernel (an atomic ``scatter_add``, an attention
backward) is invisible to boundary instrumentation until the bad value reaches
the next boundary, whereas the op layer names it directly.

Root-cause property: within one rank's stream, per-op-name occurrence counters
give every record a run-independent identity, so the offline comparator
(``tools/determinism/compare_traces.py``) aligns two runs record-by-record and
the first record whose output digest differs is the divergence site —
everything before it matched, so that op (or the collective feeding it)
introduced the difference.

Design notes, matching the Megatron-Bridge determinism debug tool this ports
(``zhiyul/determinism-debug-tool``):

* ``empty``-family ops are skipped — they return uninitialized memory whose
  contents differ run-to-run by definition (or are fill-controlled by torch
  deterministic mode), so fingerprinting them reports a spurious "root cause".
* ``c10d`` collective ops are skipped — collectives are the semantic tracer's
  job, and it records them at their existing completion boundaries; the
  dispatch mode would fingerprint an in-flight buffer for async collectives.
* Backward ops are captured too: PyTorch propagates the dispatch-mode TLS into
  autograd engine threads, and the per-name counters are guarded by a lock so
  identities stay deterministic (Megatron schedules execute forward/backward
  segments sequentially per rank).
* **Incompatible with CUDA-graph capture** — the mode runs Python per ATen op,
  which cannot run inside ``torch.cuda.graph`` capture. Do not enable together
  with CUDA graphs.
* Extension kernels that bypass the torch dispatcher (Transformer Engine GEMMs
  and fused attention, Triton/FLA kernels, DeepEP) are not captured directly; a
  divergence born inside one surfaces at the first ATen op that consumes its
  output. Validated example: non-deterministic cuDNN fused-attention backward
  reports its first divergence at the RoPE-backward ``aten.slice`` that consumes
  the attention gradients, with every earlier op matching.
* This is a debugging aid for bounded iteration windows, not a training
  feature: one device reduction per op output adds real overhead (far less
  than byte-exact hashing, but not free).
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import Any, Iterator

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from megatron.core.determinism_trace import (
    EventKind,
    active_trace,
    collective_trace_phase,
    tensor_signature,
)

# Guards the per-trace occurrence counters (backward ops arrive from autograd
# engine threads) and suppresses recursive capture while the tracer itself
# runs tensor ops.
_LOCK = threading.Lock()
_SUSPENDED = threading.local()

# Substring skip rules (see module docstring).
_SKIP_SUBSTRINGS = ("empty", "c10d")

# Optional per-output size cap. Fingerprinting a tensor allocates a device
# temporary the size of the output, so on a very large model the per-op cost
# during the live-activation forward can OOM a rank. Set DET_TRACE_OP_MAXNUMEL
# to skip outputs above that element count (0 = no cap). A skipped output still
# records its shape/dtype (no digest), so a divergence surfaces at the first
# under-cap tensor downstream.
_MAX_NUMEL = int(os.environ.get("DET_TRACE_OP_MAXNUMEL", "0") or "0")


def _op_identity(trace: Any, name: str) -> int:
    """Deterministic per-(trace, op-name) occurrence index."""
    counters = getattr(trace, "_op_trace_counters", None)
    if counters is None:
        counters = {}
        setattr(trace, "_op_trace_counters", counters)
    index = counters.get(name, 0)
    counters[name] = index + 1
    return index


def _signature_outputs(out: Any) -> list[dict[str, Any]]:
    """Signatures of all non-empty tensor outputs (every dtype: integer/bool
    divergences such as routing indices or argmax results are real
    non-determinism and must not be dropped)."""
    signatures: list[dict[str, Any]] = []

    def add(value: Any) -> None:
        if isinstance(value, torch.Tensor) and value.numel() > 0:
            if _MAX_NUMEL and value.numel() > _MAX_NUMEL:
                signatures.append(
                    {"shape": list(value.shape), "dtype": str(value.dtype),
                     "numel": value.numel(), "digest": "skipped_oversize"}
                )
                return
            signature = tensor_signature(value)
            if signature is not None:
                signatures.append(signature)

    if isinstance(out, torch.Tensor):
        add(out)
    elif isinstance(out, (list, tuple)):
        for item in out:
            add(item)
    return signatures


class OpTraceMode(TorchDispatchMode):
    """Fingerprint every ATen op output into the active determinism trace."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        if getattr(_SUSPENDED, "value", False):
            return out
        trace = active_trace()
        if trace is None or not trace.is_open:
            return out
        name = str(func)
        if any(token in name for token in _SKIP_SUBSTRINGS):
            return out
        _SUSPENDED.value = True
        try:
            signatures = _signature_outputs(out)
            if signatures:
                with _LOCK:
                    index = _op_identity(trace, name)
                phase = collective_trace_phase()
                trace.record_if_open(
                    EventKind.OP,
                    name,
                    {
                        # Unique, run-independent identity so the offline
                        # comparator aligns records semantically instead of by
                        # arrival order (reuses the identity field the
                        # comparator already consumes).
                        "checkpoint_id": f"aten:{name}:{index}",
                        "phase": phase,
                        "outputs": signatures,
                    },
                )
        finally:
            _SUSPENDED.value = False
        return out


@contextmanager
def op_trace_mode(enabled: bool) -> Iterator[None]:
    """Enable blanket ATen-op fingerprinting for the duration of the context.

    A no-op when ``enabled`` is falsy so callers can wrap a training step
    unconditionally. The dispatch mode only records while a determinism trace
    is active, so enabling it outside a traced iteration adds dispatch
    overhead but writes nothing.
    """
    if not enabled:
        yield
        return
    with OpTraceMode():
        yield
