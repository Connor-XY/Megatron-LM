---
orphan: true
---

# Determinism validation evidence

> **Audience:** developers and reviewers. It records findings, conclusions, and
> explanations, and omits environment-specific run provenance such as scheduler
> details, artifacts, and source identity.

## Scope and method

The validation work combines four kinds of evidence:

1. A code-path review identifies the deterministic branch and the ordinary
   branch it replaces.
2. Focused tests exercise correctness, fallbacks, dtype/shape limits, and
   CUDA-graph behavior.
3. Independent model runs compare outputs, gradients, trace events, or
   serialized training metrics exactly at the intended parallel layout.
4. Paired profiles compare deterministic and ordinary execution in alternating
   order before a performance change is accepted.

The target workloads are DeepSeek-V3-style MoE/MLA configurations and
Nemotron-3-Ultra-style hybrid MoE/Mamba configurations. This is a scoped
training claim, not a claim that every Megatron model or optional feature is
deterministic.

## Baseline coverage

- The module-level suite uses `BitExactRunner` to restore RNG state and compare
  bit-identical outputs and gradients across GPT, Hybrid, and Transformer
  components under tensor, expert, pipeline, virtual-pipeline, and
  data-parallel layouts. Scheduling stressors are included to expose latent
  collective-ordering races.
- The performance gate runs a small GPT recipe in deterministic and ordinary
  modes under Nsight Systems and reports a per-range leaderboard. It rejects a
  regression beyond the documented step-time threshold.
- DeepSeek-V3-style and Nemotron-3-Ultra-style proxies exercise group-limited
  routing, expert bias, grouped GEMM, all-to-all dispatch, hybrid Mamba plus
  attention plus MoE, and supported parallel-layout combinations.
- The full-width all-to-all matrix covers DeepSeek top-k-8 and Nemotron top-k-6
  configurations, including tensor/expert parallelism, pipeline variants, and
  the supported FSDP repeatability cells. The latter are not a
  cross-allocation Megatron-FSDP certificate.

## Operator and routing findings

- PyTorch LayerNorm/RMSNorm fallbacks and DSA sparse-mask paths have focused
  coverage of their forward, backward, and recompute passes. The DSA writes
  are deterministic where their indices are unique.
- Deterministic top-k keeps `sorted=True` in both the original forward pass
  and the checkpoint recompute. This avoids probability drift, but its
  radix-sort work remains a performance hotspot.
- Standard MoE all-to-all is a rank-indexed permutation and produces matching
  dispatch and combine payloads when the inputs match. External fused
  dispatchers are rejected because the order in which their tokens arrive and
  combine cannot yet be certified.
- The implementation that produces the routing map and probabilities uses
  collision-free writes, and its backward pass gathers the selected entries.
  Tests cover the ordinary PyTorch fallback, Triton acceleration, unsupported
  inputs, and CUDA-graph replay.
- The deterministic unpermute fast path preserves the conservative
  `index_add_` contribution order while lowering allocation and latency on its
  guarded shapes. The fallback still handles unsupported routes, padding,
  capacity/drop configurations, dtypes, and layouts.
- Router expert-bias counts use integer accumulation and compare values in
  integer space, so near-tie decisions do not change even after the counts
  exceed the range where floating point represents integers exactly.

## Trace and divergence-diagnosis findings

The structured tracer records semantic events without introducing a training
collective. It covers:

- forward, backward, optimizer, and activation-recompute boundaries;
- MoE dispatch/combine, pipeline send/receive, data-parallel reduction, and
  distributed-optimizer gather boundaries;
- synchronous tensor-parallel mappings and tensor-parallel linear operations at
  their existing completion points;
- deterministic reduction implementation and group size;
- local optimizer state when explicitly enabled; and
- the attention backend selected by an actual Transformer Engine forward.

The comparator aligns events by semantic identity rather than pipeline arrival
order and reports the first causal mismatch. The certifier checks trace
coverage, recompute correspondence, completed collective balance, runtime
configuration, and selected attention-backend visibility. It rejects incomplete
or mixed trace trees rather than producing a partial success result. Its
streaming implementation keeps memory proportional to one file's active
collective names rather than the total event count. Large-scale rank sampling,
an all-rank summary tier, device-side 128-bit digests, buffered writes, and
event caps are localization controls. Summary comparison identifies the first
divergent iteration and affected ranks before heavier semantic or op-level
capture is enabled. Only all-rank, untruncated, full-byte comparisons should be
described as cryptographic trace certificates.

## Reduction and optimizer findings

- One-rank reduce-scatter is an identity operation: the selected path does not
  launch a collective or allocate an unnecessary result buffer.
- A supported two-rank reduce-scatter output has one addition only. The native
  path therefore preserves the deterministic arithmetic contract and is
  substantially faster than the generic ordered helper.
- Larger floating reductions use a fixed logical-rank FP32 accumulation order.
  The hierarchical form avoids the avoidable latency of a flat ordered
  implementation while retaining a topology-independent result.
- Final tensor- and pipeline-parallel gradient synchronization, gradient norm,
  reported losses, and supported MoE metrics use the corresponding ordered
  reduction path. Traces expose which implementation executed.
- Distributed-optimizer parameter all-gather is rank-indexed copying and needs
  no numerical reduction. Native floating data-parallel reductions outside the
  supported distributed-optimizer path remain a gap.

## Mamba and attention findings

- Deterministic Mamba execution pins the relevant environment before the
  kernels are imported and fixes its workspace and configuration choice. This
  prevents the autotuner from drifting on a cold cache, and it does so without
  having to disable the fused path.
- Transformer Engine selection is validated from the backend that an actual
  forward uses. DeepSeek-style MLA and Nemotron-style attention can choose
  different valid backends under the same policy.
- A global forced attention backend is not safe: availability depends on
  dropout, layout, dtype, mask, architecture, and installed library behavior.
  Backend selection must remain observable in the trace.
- Attention backward remains one of the material deterministic costs and is an
  upstream kernel-optimization target rather than a reason to weaken the
  correctness contract.

## Model-level and Bridge findings

Independent model comparisons establish exact semantic traces for the supported
DeepSeek-V3-style and Nemotron-3-Ultra-style configurations. They include
activation recompute, MoE routing, all-to-all dispatch/combine, ordered
data-parallel reductions, and supported optimizer behavior.

The Bridge path has separate validation for configuration resolution,
process-group hierarchy propagation, strict comparison of serialized training
metrics, and longer sequential deterministic runs. The result establishes that
the supported configuration can retain an identical trajectory; it does not yet
turn every production-scale layout into a general support promise.

## Performance conclusions

The evidence supports these engineering decisions:

| Decision | Why |
| --- | --- |
| Keep sorted deterministic top-k | Post-selection alternatives either regress end-to-end performance or change tie gradients. |
| Keep guarded unpermute and routing fast paths | They preserve the conservative deterministic result while removing measurable overhead on supported shapes. |
| Use identity and native two-rank reduction shortcuts | Their arithmetic is already uniquely ordered and avoids the generic helper’s cost. |
| Use hierarchical ordered reduction for larger groups | It preserves the required accumulation order and is materially better than a flat ordered implementation. |
| Pin Mamba/Triton configuration early | It prevents configuration-dependent drift without a demonstrated proxy slowdown. |
| Do not force one attention backend | Eligibility and performance depend on the actual input and installed kernel library. |

The remaining full-step gap comes from attention backward, the vocab-parallel
loss and gather backward, grouped GEMM, and topology and overlap effects.
End-to-end ABBA measurements improve some workloads but vary across
allocations, so the aggressive performance target is not yet a general claim.

## Conditions and remaining limitations

The following remain intentionally unsupported or conditional in deterministic
mode:

- fused cross entropy;
- tensor-parallel communication overlap;
- multiple distributed-optimizer instances and collective averaging;
- Megatron-FSDP floating reduction paths;
- packed-sequence SSM;
- topology-independent floating reductions in several tensor-parallel paths;
  and
- any external fused MoE dispatcher without an independent, exact
  cross-allocation certificate.

## Conclusion

The work establishes deterministic branches and validation tools for the
documented DeepSeek-V3-style and Nemotron-3-Ultra-style training surfaces. It
also establishes where determinism still depends on a configuration precondition
or is deliberately rejected. New changes must extend the code-path explanation,
focused tests, independent exact comparison, and paired performance evidence
before their scope is expanded.
