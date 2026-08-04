# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""All-rank, all-module activation digest trace.

Locates which module first produces a differing activation when some ranks of a
single job disagree. With tensor-parallel size 1, every rank sees the same
tokens, so any non-expert activation is expected to be bit-identical across
ranks. A module whose per-rank digests partition into a large group and a small
one is the producer, and one such run is self-contained evidence: the comparison
is across ranks inside a single process group, not across runs, so it does not
depend on reproducing an intermittent failure twice.

Expert-parallel modules are excluded from that expectation. After dispatch each
rank holds a different expert shard, so all-ranks-differ is normal there and is
a distinguishable signature from a minority split.

Usage, from the training or calibration entry point after the model is built:

    from tools.determinism.cudnn_attention_repro.all_rank_digest_trace import (
        install_all_rank_digest_trace,
    )

    handles = install_all_rank_digest_trace(model)
    ...
    for h in handles:
        h.remove()

Configured by environment variable, and inert unless MCORE_DIGEST_TRACE is set:

    MCORE_DIGEST_TRACE            output path template, must contain {rank}
    MCORE_DIGEST_TRACE_MAX_LAYER  highest decoder layer index to trace (default 7)
    MCORE_DIGEST_TRACE_MAX_CALLS  calls recorded per module (default 32)

Keep the layer and call ranges small. The trace is a lower bound on divergence:
a split occurring outside the traced window is invisible here, so absence of a
recorded split is not proof of a clean run.
"""

from __future__ import annotations

import hashlib
import json
import os
import re

import torch

LAYER_PATTERN = re.compile(r"^decoder\.layers\.(\d+)(\.|$)")


def _first_tensor(value):
    """Return the first tensor found in a nested structure, or None."""
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _tensor_signature(tensor):
    """Digest a tensor plus the layout metadata needed to interpret a mismatch.

    The digest is taken over the raw bytes of a contiguous copy, so it is exact
    rather than tolerance-based: a one-ULP difference in a single element of 64
    million changes it. Shape, stride and offset travel with it because two
    ranks holding equal values in different layouts is a different finding from
    two ranks holding different values.
    """
    if tensor is None:
        return None
    detached = tensor.detach()
    contiguous = detached.contiguous()
    as_bytes = contiguous.view(torch.uint8) if contiguous.dtype != torch.uint8 else contiguous
    digest = hashlib.md5(as_bytes.cpu().numpy().tobytes()).hexdigest()
    return {
        "digest": digest,
        "dtype": str(detached.dtype),
        "shape": list(detached.shape),
        "stride": list(detached.stride()),
        "storage_offset": detached.storage_offset(),
        "numel": detached.numel(),
        "is_contiguous": detached.is_contiguous(),
    }


def install_all_rank_digest_trace(model):
    """Register digest-recording forward hooks. Returns the handles to remove."""
    template = os.environ.get("MCORE_DIGEST_TRACE")
    if not template:
        return []

    max_layer = int(os.environ.get("MCORE_DIGEST_TRACE_MAX_LAYER", "7"))
    max_calls = int(os.environ.get("MCORE_DIGEST_TRACE_MAX_CALLS", "32"))
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    path = template.format(rank=rank)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Line buffered, so a trace survives a job that is killed mid-run.
    trace_file = open(path, "w", encoding="utf-8", buffering=1)

    def wanted(name):
        if not name:
            return False
        match = LAYER_PATTERN.match(name)
        if match:
            return int(match.group(1)) <= max_layer
        return name.startswith("embedding")

    counters = {}
    order = {"value": 0}

    def make_hook(name, module_type):
        def hook(module, args, kwargs, output):
            call = counters.get(name, 0)
            if call >= max_calls:
                return
            counters[name] = call + 1
            sequence = order["value"]
            order["value"] = sequence + 1
            trace_file.write(
                json.dumps(
                    {
                        "rank": rank,
                        "pid": os.getpid(),
                        "module": name,
                        "module_type": module_type,
                        "call": call,
                        "sequence": sequence,
                        "input": _tensor_signature(_first_tensor(args)),
                        "output": _tensor_signature(_first_tensor(output)),
                    },
                    sort_keys=True,
                )
                + "\n"
            )

        return hook

    handles = []
    for name, module in model.named_modules():
        if not wanted(name):
            continue
        handles.append(
            module.register_forward_hook(make_hook(name, type(module).__name__), with_kwargs=True)
        )

    print(
        "MCORE_DIGEST_TRACE_INSTALLED rank=%d modules=%d max_layer=%d max_calls=%d path=%s"
        % (rank, len(handles), max_layer, max_calls, path),
        flush=True,
    )
    return handles
