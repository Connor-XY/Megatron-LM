---
orphan: true
---

# Determinism in Megatron-LM — Current Status

> **Audience:** Megatron-LM / Megatron-Core developers working on deterministic
> training. For setup and operational constraints, see the
> [Deterministic Training user guide](../../user-guide/deterministic-training.md).
>
> **Companion docs in this directory**
> - [`validation-evidence.md`](./validation-evidence.md) — detailed findings,
>   conclusions, and explanations from the validation work.
> - [`training-path.md`](./training-path.md) — the forward+backward branch map:
>   every point in the training step where the code (or an underlying kernel)
>   chooses a deterministic vs non-deterministic path.
> - [`op-catalog.md`](./op-catalog.md) — the per-op catalog table (det? / det path /
>   non-det path / how selected / perf Δ / gap).

---

## 1. What "deterministic" means here

**Bitwise determinism:** given a fixed input and seed, the model yields an
*identical* loss/grad curve across every run, with everything else held fixed —
input data + order + seed, model recipe + config, parallelism layout
(TP/PP/DP/EP/CP/VPP), container image, library versions (MCore, CUDA, cuDNN,
NCCL, Transformer Engine, PyTorch, driver), NCCL env/runtime settings, and
hardware type + cluster topology.

The definition is about **numerical results, not the computation graph.** What is
explicitly *allowed* to vary between runs:

- The computation graph may differ slightly if final numerics are bit-identical.
- Execution ordering may change, **as long as reduction orderings are the same**
  (floating-point addition is non-associative — this is the root cause of nearly
  all non-determinism; see §2).
- Kernel scheduling / implementation may vary if outputs/grads are unchanged.
- Deterministic mode may select different (slower / more memory-hungry) kernels or
  fallback paths than non-deterministic mode — this is the design we use.

## 2. Why it is hard

The primary enemy is **floating-point non-associativity**: `(a+b)+c ≠ a+(b+c)` in
bf16/fp16/fp32. Any kernel whose result depends on the *order* of a parallel
reduction can produce different round-off between runs. The usual culprits are
atomic adds, multi-block reductions, async collectives whose completion order
varies, and allocator-driven kernel autotuning. Determinism therefore means
pinning reduction order everywhere it matters.

## 3. Targets and motivation

Customers are moving determinism from a debug/CI nicety to a **production / hero-run
requirement**. They want to replay loss spikes deterministically, resume from a
checkpoint along the original trajectory, and have a proof point that the GPU
stack produces correct math, which mitigates SDC/SDE concerns. The blocker to
enabling it by default is the performance penalty.

| Milestone | Determinism overhead (step time) |
| --- | --- |
| Current baseline (mcore) | **~15%** overall; Nemotron-3-Ultra **+17.4%**, DSV3 **+8.5%** |
| Adoption target | **< 10%** (would let us default `deterministic_mode=True`) |
| Aggressive target | **≈ 5%** (per Meituan Longcat: deterministic FAG + ScatterAdd + grouped GEMM + fused GemmAdd) |

Profiling attributes most of the gap to a handful of operations — see
[`op-catalog.md`](./op-catalog.md) “Hotspots”. The recurring offenders are
`aten::fill_`, `aten::empty`, `aten::index_put_`, `scatter_add`, and
`cub::DeviceRadixSort` in the MoE top-k path.

Paired profiles establish three useful conclusions:

- The fixed logical hierarchy removes the isolated data-parallel reduction
  penalty. In the measured proxy it runs close to native, whereas a flat ordered
  path is materially slower.
- Collision-free routing and fixed-order unpermute remove avoidable
  deterministic overhead. The routing work improves isolated latency strongly
  and produces modest end-to-end gains, but it does not eliminate the full
  training-step gap by itself.
- Full deterministic overhead remains workload- and allocation-sensitive.
  Attention backward, vocab-parallel loss/gather backward, and grouped-GEMM
  paths can dominate after the routing and reduction improvements land.

The current performance result is therefore a direction, not a global claim.
Each optimization needs an exactness comparison and an alternating-order
end-to-end measurement before it is accepted.

## 4. Control plane — how determinism is turned on

### 4.1 Upstream state

PR #5041 **merged into `NVIDIA/main` on 2026-07-08**, so the determinism test
scaffold (`BitExactRunner`, the det-vs-nondet nsys leaderboard, and the
`determinism.py` argument helpers) is upstream. This branch was stacked on that
PR's pre-merge head and adds the implementation, tests, diagnostics, and
evidence described below; those additional commits have **no separate upstream
PR yet** and are not an upstream guarantee. A rebase onto post-merge `main` is
pending.

The config flag is `model_parallel_config.py` `deterministic_mode: bool = False`,
threaded into `TransformerConfig`; library code reads that flag or
`torch.are_deterministic_algorithms_enabled()`.

### 4.2 This branch

`--deterministic-mode` delegates to reusable setup helpers:

- **`megatron.determinism_env.set_determinism_env_vars()`** is intentionally
  lightweight enough to run before importing torch or Mamba kernels. It defaults
  `NCCL_ALGO=Ring`, `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`, and
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`; it also forces
  `MAMBA_DETERMINISTIC=1` and `TRITON_CACHE_AUTOTUNING=0`. The latter is
  load-bearing: a cold autotune cache can choose different reduction tilings.
  `pretrain_hybrid.py` runs this bootstrap before importing `HybridModel`.
- **`megatron.training.determinism.apply_determinism_to_args(args)`** rejects CE
  fusion, replaces uncertified DeepEP/HybridEP dispatch with standard all-to-all,
  validates `NCCL_ALGO` (excluding `Tree`), forces `tp_comm_overlap=False`, and
  finally enables torch deterministic algorithms.
  For the distributed optimizer it also requires one optimizer instance,
  rejects collective AVG, and forces ordered fp32 reduce-scatter.
- **Small floating-point statistics** (gradient norm, reported loss, and MoE
  metrics) use rank-ordered all-gather plus a fixed local sum, so their result
  does not depend on NCCL's reduction order. Measured on GB300 (NCCL 2.29.2, 16
  ranks, `NCCL_ALGO=Ring`, NVLS disabled): two disjoint allocations of the same
  shape built the same ring and returned bit-identical all-reduce and
  reduce-scatter results for fp32 and bf16. The same probe changed every hash
  when the rank-to-input assignment was permuted, so it does detect a change of
  reduction order. Allocations of a differing shape, larger scales, and other
  topologies are untested.

Flash attention is permitted when Transformer Engine honors
`NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`; the bit-exact suite covers supported
configurations.

## 5. Enforced limitations in deterministic mode

These features are currently **incompatible** with deterministic mode:

| Feature | Where enforced | Reason |
| --- | --- | --- |
| `cross_entropy_loss_fusion` | `megatron/training/determinism.py` assert (`apply_determinism_to_args`, reached from `arguments.py` `validate_args`) | Fused CE kernel is non-deterministic |
| Flex MoE dispatcher (DeepEP/HybridEP) | `determinism.py` switches to standard all-to-all; `TransformerConfig` rejects any surviving flex config | External fused dispatch/combine kernels do not yet have a fixed-order cross-allocation certificate |
| `tp_comm_overlap` (async TP) | `determinism.py` (forces off) | Async NCCL collective ordering varies |
| Multiple distributed-optimizer instances | `determinism.py` assert | The cross-instance floating-point reduction is not ordered |
| `ddp_average_in_collective` | `determinism.py` assert | AVG must follow the rank-ordered SUM, not occur inside NCCL |
| Megatron-FSDP | `determinism.py` assert | Its native all-reduce/reduce-scatter path does not yet provide fixed-rank accumulation |
| Packed sequence (`thd`) in gated-delta-net | `ssm/gated_delta_net.py:314` assert | No deterministic packed-seq SSM path |

> **Open question (tracked in the roadmap docs):** it is not fully established
> *whether* CE-fusion and tp_comm_overlap can be made deterministic. Resolving
> this is part of the performance roadmap — lifting a limitation is often cheaper than
> a slow fallback.

## 6. Surface area — the det/non-det branches that exist

The complete per-operation classification is in
[`op-catalog.md`](./op-catalog.md). The load-bearing branches are:

- **MoE unpermute** (`moe_utils.py`,
  `moe/ops/deterministic_index_select.py`): for supported dropless, unpadded
  all-to-all shapes, deterministic mode uses a fixed-order Triton segment sum
  with a collision-free gather backward. Unsupported shapes retain a
  deterministic `index_add_`, while normal mode uses `scatter_add_`. The guarded
  path is CUDA-graph safe and preserves
  the fallback’s contribution order.
- **MoE routing map and probabilities** (`moe_utils.py`,
  `moe/ops/deterministic_routing.py`): deterministic mode uses collision-free
  row-wise writes and a selected-entry gather backward. Unsupported inputs
  retain an in-place `scatter_` fallback, while normal mode keeps the
  out-of-place `scatter`.
- **Flex MoE dispatch** (`transformer/moe/token_dispatcher.py`): external
  fused dispatchers combine permutation, communication, and reduction order
  outside MCore’s control. Deterministic configuration therefore selects the
  standard all-to-all dispatcher and fails closed if a flex configuration
  survives.
- **Top-k and expert bias** (`moe_utils.py`, `router.py`): deterministic mode
  keeps `sorted=True` across activation recompute, and expert-bias counts use
  exact integer accumulation and comparison.
- **Embedding and vocabulary-parallel loss** (`tensor_parallel/layers.py`,
  `tensor_parallel/cross_entropy.py`): direct indexing avoids the atomics in
  embedding backward, and the local cross-entropy backward uses collision-free
  selected-class updates. The remaining tensor-parallel floating collectives
  retain their documented limitation.
- **Mamba / SSM** (`ssm/ops/determinism.py`,
  `ssm/mamba_mixer.py`, `ssm/gated_delta_net.py`): deterministic mode fixes
  configuration before kernel import, uses deterministic workspaces, and keeps
  the fast fused path where supported. Packed sequence remains unsupported.
- **Data-parallel and small-statistic reductions**
  (`distributed/param_and_grad_buffer.py`,
  `distributed/deterministic_collectives.py`): deterministic mode uses
  identity and native two-rank shortcuts where the arithmetic is already
  uniquely ordered, and otherwise falls back to a fixed logical-rank FP32
  reduction.
- **Final tensor/pipeline-parallel gradient synchronization**
  (`distributed/finalize_model_grads.py`): supported one- and two-rank groups
  take their semantic shortcuts, while larger groups use bounded chunks, a fixed
  local sum, and rank-indexed communication. The trace records the
  implementation selected for each reduction.
- **Megatron-FSDP reduction** remains unsupported at the required
  cross-allocation scope because its native reduction pipeline does not consume
  the ordered-FP32 branch.
- **Transformer Engine attention**
  (`extensions/transformer_engine.py`): deterministic mode requires
  `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`, and the trace records the backend and
  sub-backend selected by a real forward. Eligibility is input-specific, so no
  backend is forced globally.
- **Inference/RL scheduling** (`dynamic_engine.py`,
  `data_parallel_inference_coordinator.py`, `rl/rl_utils.py`): scheduling
  uses a stable key rather than completion order where it affects observable
  output ordering.

## 7. Validation status

The detailed validation findings, conclusions, and explanations are collected in
[validation-evidence.md](./validation-evidence.md). They omit environment-specific
provenance.

## 8. Known gaps (feeding the roadmap)

1. **Production-scale Bridge validation is not yet a checked-in recurring
   gate.** The model-level qualification establishes that two deterministic
   launches can match all recorded nonvolatile metrics under the supported
   configuration. It should be repeated regularly at the intended scale,
   including a same-mode control without profiling instrumentation. The local
   provenance record retains the detailed run material.

2. **The production performance target is not consistently closed.** The
   hierarchy and routing changes remove verified sources of overhead, but
   full-step results still vary with topology, overlap, and kernel selection.
   Attention backward, Transformer Engine normalization/compiled paths,
   vocab-parallel cross entropy, expert-score gather, and grouped GEMM remain
   the main optimization targets. A successful kernel microbenchmark does not
   by itself establish a training-step win.

3. **First-divergence tooling has deliberate coverage limits.**
   `compare_dumps.py` localizes saved activations, parameters, and gradients.
   The structured trace covers phase ordering, recompute correspondence,
   optimizer boundaries, completed collective boundaries, and selected
   attention backends. It does not yet independently capture Transformer
   Engine FP8/FP4 quantizer state, tensor-parallel userbuffer payloads, or
   selection state for arbitrary auto-dispatching libraries.

4. **Some floating reductions still lack topology-independent paths.**
   Tensor-parallel mappings and linears, vocab-parallel cross entropy, and the
   non-distributed-optimizer data-parallel all-reduce remain conditional or
   unsupported at the required scope. Megatron-FSDP, fused cross entropy,
   tensor-parallel communication overlap, multiple distributed optimizers,
   collective averaging, and packed-sequence SSM remain intentionally blocked
   in deterministic mode.

## 9. References

- Determinism roadmap & meeting notes (internal Google Docs).
- Customer technical reports: Meituan Longcat (deterministic FAG / ScatterAdd /
  grouped GEMM / fused GemmAdd), MSFT, DeepSeek-V4.
- Thinking Machines, "Defeating nondeterminism in LLM inference."
- PyTorch reproducibility & `torch.use_deterministic_algorithms` docs.
