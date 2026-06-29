<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software and related documentation.
-->

# Deterministic Training

> **Audience:** people launching Megatron training. This guide explains how to
> enable reproducible training and what to expect. It intentionally excludes
> internal test names, cluster labels, job identifiers, and engineering
> provenance.

Deterministic mode makes two equivalent training runs produce the same model
outputs, gradients, and recorded training metrics. It is useful when
investigating a regression, reproducing a training issue, or validating a
change to the training stack.

## Enable deterministic mode

Add `--deterministic-mode` to the normal training command:

```bash
python pretrain_gpt.py \
  --deterministic-mode \
  <other training options>
```

Use the same command, input data order, random seed, container or software
versions, hardware class, and parallel layout for both runs. Deterministic mode
cannot make two intentionally different configurations identical.

## What Megatron changes

Megatron enables deterministic PyTorch behavior and selects conservative
collective and kernel settings. It also avoids or rejects optimizations whose
numerical order is not currently certified.

| Area | What to expect |
| --- | --- |
| Gradient reduction | Reproducible accumulation order where floating-point addition would otherwise depend on runtime topology. |
| Attention and kernel libraries | Only supported deterministic kernel choices are allowed. Megatron may choose a slower compatible implementation. |
| Mixture-of-experts routing | Uses the certified dispatch path instead of unsupported fused dispatchers. |
| Hybrid sequence models | Uses deterministic workspaces and fixed kernel-selection settings. |

You normally do not need to set environment variables manually. Megatron
configures the required runtime values early in startup. If your launcher must
set them itself, use the same values for both runs and set them before importing
GPU kernel libraries.

## Features that are currently incompatible

Deterministic mode stops rather than silently weakening its guarantees when a
known unsupported feature is enabled. In particular, do not combine it with:

- fused cross-entropy loss;
- tensor-parallel communication overlap;
- uncertified fused mixture-of-experts dispatch;
- multiple distributed-optimizer instances;
- collective averaging inside the distributed optimizer; or
- Megatron-FSDP.

The error message identifies the incompatible option. Remove that option or use
a supported configuration before retrying.

## Verify a run

For an initial check, run the same command twice and compare the complete
per-iteration metrics, not only the final loss. For a production investigation,
capture a short deterministic trace or selected tensor dumps from both runs and
compare them with the supplied tools.

A matching log is a useful signal, but it is not a full numerical certificate:
different tensors can round to the same printed metric. The developer guide
explains the stricter trace and dump workflows.

## Performance expectations

Deterministic mode can be slower because it replaces timing-sensitive or
order-dependent operations with reproducible implementations. Measure a
representative workload before enabling it broadly. The implementation avoids
extra work where the result is provably unchanged, but the remaining cost
depends on the model, sequence length, hardware, and enabled features.

## Need more detail?

- [Developer determinism reference](../developer/determinism/README.md) —
  architecture, operation catalog, profiling, and debugging tools.
- [Determinism status](../developer/determinism/status.md) — supported
  behavior, limitations, and the standard for validation claims.
