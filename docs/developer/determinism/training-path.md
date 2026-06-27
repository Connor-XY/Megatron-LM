# Determinism Branch Map — A Walk Through One Training Step

This is a forward→backward→optimizer walk through a Megatron training step,
calling out **every place where determinism enters or is decided**. Each entry
links to the file:line and to the row in [`op-catalog.md`](./op-catalog.md). Read
[`status.md`](./status.md) first for the control plane and definitions.

Legend for the "Determinism" column:
- 🟢 deterministic (by formula, or pinned reduction order)
- 🔵 has an explicit deterministic branch (selected by `deterministic_mode` /
  `torch.are_deterministic_algorithms_enabled()`)
- 🟡 deterministic only under a precondition (env var / fixed NCCL algo / unique
  indices) — **verify** at scale
- 🔴 non-deterministic with no in-repo deterministic path (a gap)

---

## Stage 0 — Data & RNG (before the model)

| Step | Where | Determinism | Notes |
| --- | --- | --- | --- |
| Sample order / shuffle | `megatron/training/datasets/data_samplers.py`, `megatron/core/datasets/` | 🟡 | Reproducible iff seed + data path + DP layout are fixed. Part of the "fixed input" contract, not a kernel concern. |
| Global RNG seeding | `megatron/training/initialize.py`, `tensor_parallel/random.py` (`CudaRNGStatesTracker`) | 🟢 | Per-rank RNG is tracked and restorable — this is exactly what `BitExactRunner` snapshots/restores between its two runs. |
| Dropout masks | attention/MLP dropout layers | 🟡 | Reproducible under fixed RNG state; set dropout=0 for hero runs to remove a source entirely. |

## Stage 1 — Forward

### 1a. Embedding

| Step | Where | Determinism | Notes |
| --- | --- | --- | --- |
| Vocab-parallel embedding lookup | `tensor_parallel/layers.py:299-303` | 🔵 | det: direct index `self.weight[masked_input]`; non-det: `F.embedding` (whose **backward** scatters grads with atomics → non-deterministic with duplicate tokens). Branch on `self.deterministic_mode`. |
| Rotary / position embeddings (RoPE) | `models/common/embeddings/rotary_pos_embedding.py` | 🟢 | Pure trig formula, no scatter/atomics. |

### 1b. Attention

| Step | Where | Determinism | Notes |
| --- | --- | --- | --- |
| QKV / output projection (TP linear) | `tensor_parallel/layers.py` (Column/RowParallelLinear), TE linear in `extensions/transformer_engine.py` | 🟡 | GEMM forward is deterministic given fixed cuBLAS workspace (`CUBLAS_WORKSPACE_CONFIG=:4096:8`). TP all-reduce/all-gather reproducible under fixed `NCCL_ALGO`. |
| Core attention (fwd) | TE `DotProductAttention` (`extensions/transformer_engine.py:1697`) | 🔵🟡 | When `deterministic_mode` on, asserts `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` so TE picks deterministic kernels. The **backward** is the real concern (see Stage 3). |
| Softmax / scale / mask | `transformer/dot_product_attention.py` | 🟢 | Elementwise. |

### 1c. Normalization

| Step | Where | Determinism | Notes |
| --- | --- | --- | --- |
| LayerNorm / RMSNorm (fwd) | `transformer/torch_norm.py`, TE norm in `extensions/transformer_engine.py` | 🟢 | Forward reduction is deterministic. Backward weight-grad reduction is the concern (Stage 3). |

### 1d. MLP / MoE (the determinism hot zone)

| Step | Where | Determinism | Notes |
| --- | --- | --- | --- |
| Router gating + softmax/sigmoid | `transformer/moe/moe_utils.py:789-818` | 🟢 | Computed in fp32; elementwise. |
| Top-k expert selection | `moe_utils.py` (`torch.topk`) | 🔵 | Deterministic mode forces `sorted=True` even in the no-grad activation-checkpoint forward; normal mode keeps the faster no-grad `sorted=False` path. This avoids forward/recompute probability drift for sigmoid top-k normalization. The sorted path is the source of the `cub::DeviceRadixSort` det-vs-nondet delta (**hotspot**). |
| Expert-bias selected-score gather | `moe_utils.py` (`torch.gather`) | 🔵 | PyTorch switches gather backward from scatter-add to its deterministic indexed-write path. A collision-free Triton candidate was exact and faster in isolation/profile but regressed two balanced production ABBAs by 3.2–3.3%, so it is not in the source tree. |
| Flextron soft-mask routing | `elastification/flextron_elasticity_hooks.py` | 🟡 | Flextron replaces the main router with weighted sigmoid accumulation, always-sorted top-k, selected-score gather, and unique-index scatter. Its `deterministic_mode` parameter is deprecated and unused; global deterministic algorithms still select PyTorch's deterministic gather backward. The path is cataloged but not yet model-certified. |
| Group-limited (node-limited) top-k | `moe_utils.py:617-634` (`group_mask.scatter_`) | 🟡 | `scatter_` writes 1s at unique group indices → deterministic in forward; no explicit det branch (**verify**). |
| Routing map / probs construction | `moe_utils.py`, `moe/ops/deterministic_routing.py` | 🔵 | det: a fused row-wise Triton kernel initializes probabilities and the boolean map, writes unique top-k entries, and gathers selected gradients in backward. It has no atomics or reductions; unsupported inputs retain in-place `scatter_`. Non-det uses out-of-place `scatter`. The fused path is exact to the scatter and former `index_put_(accumulate=False)` implementations and passes CUDA-graph replay. Also `compute_routing_scores_for_aux_loss` and capacity masks use collision-free `scatter` with **no** det branch. |
| Capacity-factor drop | `moe_utils.py:940-951` | 🟡 | `scatter` of capacity mask; unique indices. |
| Token permute (dispatch sort) | `moe_utils.py:299-431` (`argsort(stable=True)` + `index_select`, or fused TE permute) | 🟢 | Stable sort + gather is deterministic. |
| EP all-to-all dispatch | `transformer/moe/token_dispatcher.py`, `tensor_parallel/mappings.py` | 🟢 | All-to-all is a rank-indexed permutation, not a floating-point reduction. EP32 traces certify exact dispatch/combine payloads when the inputs match. |
| Grouped GEMM (expert FFN) | `extensions/transformer_engine.py` `TEGroupedLinear` | 🟡 | Forward deterministic; backward weight-grad accumulation order is the concern + a perf target (Longcat "optimized grouped GEMM"). |
| Token unpermute (combine) | `moe_utils.py:513-531` | 🔵 | det: `index_add_` (CUDA-graph safe); non-det: `scatter_add_`. The post-change rank profile separates routing-probability scatter, vocab cross-entropy, and gather backward as distinct indexed-write costs; do not attribute the aggregate `index_put_` range to this combine branch. |
| Router replay (optional) | `transformer/moe/router_replay.py` | 🟢 | Records top-k indices once and replays them — forces identical routing across runs (a determinism *tool*, not on the default path). |

### 1e. SSM / Mamba (hybrid models: nemotron-3-ultra)

| Step | Where | Determinism | Notes |
| --- | --- | --- | --- |
| Triton kernel autotune | `megatron/determinism_env.py`, `ssm/ops/determinism.py` and `mamba_ssm.utils.determinism` | 🔵 | det: `MAMBA_DETERMINISTIC=1` plus `TRITON_CACHE_AUTOTUNING=0` before kernel imports selects one fixed config; normal mode may timing-autotune. A cold cached-autotune run was the Nemotron backward divergence root cause. |
| Tiled reduction workspace | `ssm/ops/determinism.py:106-123` | 🔵 | det: allocate `zeros(..., tile_dim)` and `.sum(-1)` (ordered reduction); non-det: `empty(...)`. |
| Gated-delta-rule kernel | `ssm/gated_delta_net.py:213-216` | 🔵 | det: torch `chunk_gated_delta_rule`; non-det: FLA fused kernel. |
| Causal conv1d | `ssm/gated_delta_net.py:430-446` | 🔵 | det: `F.conv1d` (+ transposes); non-det: FLA `causal_conv1d`. |
| Packed sequence (`thd`) | `ssm/gated_delta_net.py:314` | 🔴 | `assert not deterministic_mode` — **no deterministic packed-seq path** (gap). |

## Stage 2 — Loss

| Step | Where | Determinism | Notes |
| --- | --- | --- | --- |
| Vocab-parallel cross-entropy | `tensor_parallel/cross_entropy.py`, `tensor_parallel/deterministic_cross_entropy.py` | 🟡 | The local deterministic backward fuses the one-selected-class subtract, optional label-smoothing subtract, and output-gradient scaling in one collision-free Triton pass. It reads the training loss's common non-contiguous `[sequence, batch]` gradient directly in logical row order instead of allocating a contiguous copy (no atomics/reductions; PyTorch fallback; CUDA-graph coverage). The 3 all-reduces (MAX, SUM, SUM) still use native TP collectives. `NCCL_ALGO=Ring` pins the algorithm but not necessarily the physical rank order across allocations; TP>1 remains a cross-allocation verification gap. |
| Fused CE | `cross_entropy_loss_fusion` | 🔴→forbidden | Non-deterministic; **asserted off** in deterministic mode (`arguments.py:1502`). |
| MoE aux loss | `moe_utils.py:842-890` | 🟡 | `scatter` for routing map; aux-loss scalar reduction. |

## Stage 3 — Backward

| Step | Where | Determinism | Notes |
| --- | --- | --- | --- |
| Autograd traversal | PyTorch engine | 🟢 | Order is deterministic for a fixed graph. |
| FlashAttention backward | TE, gated by `NVTE_ALLOW_NONDETERMINISTIC_ALGO` (`extensions/transformer_engine.py:1697`) | 🔵🟡 | **The classic non-determinism source** — atomic dQ/dK/dV accumulation. Deterministic TE path exists but is slower (Longcat "deterministic FAG" = independent accumulation buffers + global deterministic sum) — a top perf target. The 0.1% tolerance comments in `param_and_grad_buffer.py:326/334/345` exist for the *non-deterministic* default mode. |
| LayerNorm/RMSNorm backward | TE / torch norm | 🟢 | PyTorch LayerNorm/RMSNorm fallback backward is bit-exact at hidden 128/2048 on H100 and GB200; TE norms are covered by the model proxies. |
| Embedding backward | see 1a | 🔵 | det path = direct-index `index_put(accumulate=True)` (deterministic under `use_deterministic_algorithms`), not `F.embedding`'s atomic scatter. |
| MoE unpermute/permute backward | see 1d | 🔵 | Mirror of forward `index_add_`/`scatter_add_` branch. |
| Grouped-GEMM backward | TE `TEGroupedLinear` | 🟡 | wgrad accumulation order; perf target. |

## Stage 4 — Gradient reduction (DP / FSDP)

| Step | Where | Determinism | Notes |
| --- | --- | --- | --- |
| Grad bucket all-reduce / reduce-scatter | `distributed/param_and_grad_buffer.py` | 🔵 | Deterministic distributed-optimizer mode forces the ordered fp32 path below. Native floating-point all-reduce/reduce-scatter remains topology-sensitive across allocations. The trace hashes every bucket before launch and after its existing completion boundary. |
| fp32-accumulation reduce-scatter | `distributed/reduce_scatter_with_fp32_accumulation.py` (forced by `apply_determinism_to_args` for the distributed optimizer) | 🟢 | Rank-indexed all-to-all then **ordered `torch.sum(..., dtype=fp32)`**. The optional fixed two-level hierarchy reduces contiguous logical groups before exchanging fp32 partials. Production EP32 DSV3 and Nemotron traces match 6,080/6,080 and 9,664/9,664 semantic events across independent 32-GPU allocations. |
| Distributed-optimizer param all-gather | `distributed/param_and_grad_buffer.py` | 🟢 | All-gather copies rank-indexed parameter shards and performs no floating-point reduction. The trace hashes each bucket before launch and after the existing completion boundary. |
| Async param gather / grad reduce overlap | `distributed/param_and_grad_buffer.py`, `tensor_parallel/layers.py:544/565/577` (`async_op=True`) | 🟡 | Async completion order can vary; determinism relies on existing `wait()` or stream barriers before use. The trace adds no wait and reports operations that outlive its window through `iteration.end.pending_collectives`. `tp_comm_overlap` is force-disabled in det mode. |

## Stage 5 — Optimizer

| Step | Where | Determinism | Notes |
| --- | --- | --- | --- |
| Distributed optimizer param sharding/order | `optimizer/distrib_optimizer.py:1094` | 🟢 | Param→shard mapping "preserves deterministic ordering across ranks". |
| Adam / Muon update | `optimizer/` | 🟢 | Elementwise; deterministic given deterministic grads. |
| Grad clipping (global norm) | `optimizer/clip_grads.py`, `distributed/deterministic_collectives.py` | 🔵 | det: rank-ordered all-gather plus fixed local fp32 sum; normal: native all-reduce. This closes the scalar divergence that otherwise changes every optimizer update after an exact gradient reduction. |
| Reported losses / MoE metrics | `training.py`, `training/utils/common_utils.py`, `transformer/moe/moe_logging.py` | 🔵 | det: rank-ordered sum/average for small statistics; normal: native SUM/AVG collectives. This keeps logs and scheduler-facing aggregates cross-allocation exact. |

## Cross-cutting (affect every stage)

| Concern | Control | Determinism | Notes |
| --- | --- | --- | --- |
| cuBLAS GEMM workspace | `CUBLAS_WORKSPACE_CONFIG=:4096:8` | 🟡 | Required for deterministic GEMM algo selection. |
| NCCL collective algorithm | `NCCL_ALGO=Ring` | 🟡 | Pins the algorithm, not the physical ring chosen for an allocation. It is insufficient by itself for floating-point reduction reproducibility; ordered reduction branches above are the guarantee. `Tree` remains excluded. |
| TE non-deterministic algos | `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` | 🟡 | Forces TE deterministic attention/norm kernels. |
| CUDA caching allocator | (none) | 🟡 | Allocation pattern can influence kernel autotuning/selection; flagged in the roadmap as worth investigating. |
| PP / VPP microbatch interleave | schedules and `p2p_communication.py` in `core/pipeline_parallel/` | 🟢→🟡 | Schedule is deterministic, but interleaving scrambles observed event order. Rank-local P2P trace events are semantically named and complete at existing waits, so comparison does not depend on cross-kind arrival order. |
| Structured runtime trace | `core/determinism_trace.py`, `training.py`, `tensor_parallel/random.py`, `tensor_parallel/mappings.py`, `pipeline_parallel/p2p_communication.py`, `distributed/param_and_grad_buffer.py` | 🟢 | Opt-in rank-local events add no collectives or waits. Semantic comparison can hash recompute outputs, MoE EP all-to-all, pipeline P2P, DP gradient-reduction, distributed-optimizer parameter-gather, and optimizer boundary tensors on selected iterations. |

---

## How to read this against the catalog

Each 🟡/🔴 above is a candidate for either a **test** (Workstream 2 — does it stay
bit-exact at scale?) or an **optimization** (Workstream 3 — is its deterministic
path the perf bottleneck?). The 🔵 rows already have a deterministic branch and are
the *template* for adding new ones (follow the `moe_utils.py:517` pattern:
`if torch.are_deterministic_algorithms_enabled(): <det> else: <fast>`).
