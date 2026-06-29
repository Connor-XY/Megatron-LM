---
orphan: true
---

# Determinism validation evidence

> **Audience:** developers and reviewers who need to understand what the
> determinism work establishes. This is a conclusion-level summary, not a run
> ledger: it intentionally omits cluster names, scheduler identifiers, artifact
> locations, source revisions, and immutable artifact identifiers. Those raw
> records remain only
> in an ignored local file.

## What was evaluated

Validation combines code-path review, focused operation tests, exact comparison
of independent training launches, and paired performance profiles. The target
workloads are DeepSeek-V3-style MoE/MLA configurations and Nemotron-3-Ultra-style
hybrid MoE/Mamba configurations.

The scope is deliberately narrower than “every Megatron model is deterministic.”
Each claim is tied to a selected branch, model shape, and parallel layout. The
[operation catalog](./op-catalog.md) records those branches; the
[training path](./training-path.md) shows when each one executes.

## Findings

- Deterministic mode selects explicit safe paths for routing, supported
  reductions, attention eligibility, and Mamba configuration before model
  construction.
- DeepSeek-V3-style and Nemotron-3-Ultra-style proxy runs have exact-output and
  exact-gradient coverage across the supported model and parallelism matrix.
- MoE all-to-all dispatch and combine are rank-indexed permutations, not
  floating-point reductions. External fused dispatchers are rejected in
  deterministic mode because their ordering cannot yet be certified.
- One-rank reduce-scatter is an identity operation and supported two-rank
  reduce-scatter has exactly one addition per output. Both can use native
  shortcuts without weakening the numerical contract; larger groups use a
  fixed logical FP32 accumulation order.
- The deterministic MoE unpermute and routing implementations preserve the
  conservative deterministic fallback’s results while avoiding unnecessary
  allocation and indexed-reduction overhead on supported shapes.
- Transformer Engine backend selection is observable from real forwards.
  Certification requires a selected-backend event, rather than assuming that a
  configured preference took effect.
- Structured traces and tensor-dump comparison can locate the first observable
  difference across forward, recompute, optimizer, and completed collective
  boundaries without introducing a new training collective.

## Performance conclusions

Determinism has no single overhead number. It depends on model shape, parallel
layout, kernel availability, and overlap. The implementation therefore removes
cost only where the arithmetic proves it safe:

- use identity and native two-rank reduction shortcuts;
- use guarded fixed-order unpermute and routing fast paths;
- pin Mamba/Triton configuration before kernel import; and
- let Transformer Engine choose an eligible deterministic backend instead of
  forcing a global attention backend.

The remaining material costs are generally attention backward,
vocab-parallel loss/gather backward, and some grouped-GEMM or larger
floating-reduction paths. A proposed optimization needs both an exactness
comparison and an alternating-order end-to-end profile; kernel-only timing is
not enough to establish a training-step win.

## What this supports

The evidence supports deterministic training for the documented configuration
set when the model configuration, inputs, seed, software, hardware class, and
parallel layout are fixed. It also supports a repeatable debugging workflow:
use a trace or dump comparison to find the first mismatch, then inspect the
corresponding operation-catalog row.

## What it does not claim

- Tensor-parallel floating reductions in some mappings, linears, and
  vocab-parallel cross entropy remain topology-sensitive outside their
  supported paths.
- Fused cross entropy, tensor-parallel communication overlap, multiple
  distributed-optimizer instances, collective averaging, Megatron-FSDP, and
  packed-sequence SSM lack the required cross-allocation contract.
- A production-scale Bridge qualification is a maintainer validation gate, not
  a general user-facing support promise.

## Evidence standard for future changes

For any new deterministic branch, require:

1. A code-path explanation of the selected deterministic algorithm.
2. Focused tests for its correctness and fallback behavior.
3. Exact comparison of independent launches at the intended topology.
4. A paired performance measurement when the branch changes runtime cost.
5. An updated operation-catalog row describing scope and remaining limitations.

This keeps the public documentation useful to engineers while leaving
environment-specific provenance out of the upstream repository.
