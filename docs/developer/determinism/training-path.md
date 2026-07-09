---
orphan: true
---

# Determinism Branch Map — A Walk Through One Training Step

> **Audience:** developers tracing the implementation. For setup instructions,
> use the [user guide](../../user-guide/deterministic-training.md); for
> conclusion-level validation, see [validation evidence](./validation-evidence.md).
> Abbreviations are defined in the [glossary](./glossary.md).

This document walks through a single Megatron training step — from the forward
pass, through the backward pass, to the optimizer — calling out **every place
where determinism enters or is decided**. Each entry
links to the file:line and to the row in [`op-catalog.md`](./op-catalog.md). Read
[`status.md`](./status.md) first for the control plane and definitions.

This legend explains the "Determinism" column. Statuses are distinguished by their
**glyph shape** (not color), and each has a name used in the prose:

- ✔ **deterministic** — deterministic as implemented (by formula, or pinned
  reduction order)
- ◆ **branch** — an explicit deterministic branch exists (selected by
  `deterministic_mode` / `torch.are_deterministic_algorithms_enabled()`)
- ▲ **conditional** — deterministic only under a stated precondition (env var,
  fixed NCCL algorithm, unique indices) — **verify** at scale
- ✖ **gap** — non-deterministic with no in-repo deterministic path;
  `✖→forbidden` means deterministic mode rejects the feature (fails closed)
  rather than running it

Combined markers narrow a claim: `◆▲` = a deterministic branch exists but is
subject to a precondition; `✔▲` = deterministic on the verified surface,
conditional beyond it; `✔→▲` = deterministic in itself but conditional in what
an observer sees.

---

## Stage 0 — Data & RNG (before the model)

| Step | Where | Determinism | Notes |
| --- | --- | --- | --- |
| Sample order / shuffle | `megatron/training/datasets/data_samplers.py`, `megatron/core/datasets/` | ▲ | Reproducible iff seed + data path + DP layout are fixed. Part of the "fixed input" contract, not a kernel concern. |
| Global RNG seeding | `megatron/training/initialize.py`, `tensor_parallel/random.py` (`CudaRNGStatesTracker`) | ✔ | Per-rank RNG is tracked and restorable — this is exactly what `BitExactRunner` snapshots/restores between its two runs. |
| Dropout masks | attention/MLP dropout layers | ▲ | Reproducible under the tracked RNG state — dropout is not a divergence source by itself. Note that dropout also changes TE attention-backend *eligibility* in deterministic mode (dropout > 0 can exclude the fused backend; see Stage 3), so dropout=0 both removes the RNG dependence and widens deterministic backend choice. |

## Stage 1 — Forward

### 1a. Embedding

| Step | Where | Determinism | Notes |
| --- | --- | --- | --- |
| Vocab-parallel embedding lookup | `tensor_parallel/layers.py:299-303` | ◆ | det: direct index `self.weight[masked_input]`; non-det: `F.embedding` (whose **backward** scatters grads with atomics → non-deterministic with duplicate tokens). Branch on `self.deterministic_mode`. |
| Rotary / position embeddings (RoPE) | `models/common/embeddings/rotary_pos_embedding.py` | ✔ | Pure trig formula, no scatter/atomics. |

### 1b. Attention

| Step | Where | Determinism | Notes |
| --- | --- | --- | --- |
| QKV / output projection (TP linear) | `tensor_parallel/layers.py` (Column/RowParallelLinear), TE linear in `extensions/transformer_engine.py` | ▲ | GEMM kernels are deterministic under the pinned cuBLAS workspace (`CUBLAS_WORKSPACE_CONFIG=:4096:8`). The TP all-gather is a rank-indexed copy (no floating-point reduction). The row-parallel all-reduce **is** a floating-point SUM: `NCCL_ALGO=Ring` pins the algorithm but not the physical ring across allocations, so cross-allocation TP reduction order remains an open catalog gap (see Cross-cutting and the TP-linear catalog rows). |
| Core attention (fwd) | TE `DotProductAttention` (`extensions/transformer_engine.py:1697`) | ◆▲ | When `deterministic_mode` on, asserts `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` so TE picks deterministic kernels. The **backward** is the real concern (see Stage 3). |
| Softmax / scale / mask | `transformer/dot_product_attention.py` | ✔ | Elementwise. |

### 1c. Normalization

| Step | Where | Determinism | Notes |
| --- | --- | --- | --- |
| LayerNorm / RMSNorm (fwd) | `transformer/torch_norm.py`, TE norm in `extensions/transformer_engine.py` | ✔ | Forward reduction is deterministic. Backward weight-grad reduction is the concern (Stage 3). |

### 1d. MLP / MoE (the determinism hot zone)

| Step | Where | Determinism | Notes |
| --- | --- | --- | --- |
| Router gating + softmax/sigmoid | `transformer/moe/moe_utils.py:789-818` | ✔ | Computed in fp32; elementwise. |
| Top-k expert selection | `moe_utils.py` (`torch.topk`) | ◆ | Deterministic mode forces `sorted=True` even in the no-grad activation-checkpoint forward; normal mode keeps the faster no-grad `sorted=False` path. This avoids forward/recompute probability drift for sigmoid top-k normalization. The sorted path is the source of the `cub::DeviceRadixSort` det-vs-nondet delta (**hotspot**). |
| Expert-bias selected-score gather | `moe_utils.py` (`torch.gather`) | ◆ | PyTorch switches gather backward from scatter-add to its deterministic indexed-write path. A collision-free Triton candidate was exact and faster in isolation/profile but regressed two balanced production ABBAs by 3.2–3.3%, so it is not in the source tree. |
| Flextron soft-mask routing | `elastification/flextron_elasticity_hooks.py` | ▲ | Flextron replaces the main router with weighted sigmoid accumulation, always-sorted top-k, selected-score gather, and unique-index scatter. Its `deterministic_mode` parameter is deprecated and unused; global deterministic algorithms still select PyTorch's deterministic gather backward. The path is cataloged but not yet model-certified. |
| Group-limited (node-limited) top-k | `moe_utils.py:617-634` (`group_mask.scatter_`) | ▲ | `scatter_` writes 1s at unique group indices, so the forward is deterministic, with no explicit deterministic branch (**verify**). |
| Routing map / probs construction | `moe_utils.py`, `moe/ops/deterministic_routing.py` | ◆ | det: a fused row-wise Triton kernel initializes probabilities and the boolean map, writes unique top-k entries, and gathers selected gradients in backward. It has no atomics or reductions; unsupported inputs retain in-place `scatter_`. Non-det uses out-of-place `scatter`. The fused path is exact to the scatter and former `index_put_(accumulate=False)` implementations and passes CUDA-graph replay. Also `compute_routing_scores_for_aux_loss` and capacity masks use collision-free `scatter` with **no** det branch. |
| Capacity-factor drop | `moe_utils.py:940-951` | ▲ | `scatter` of capacity mask; unique indices. |
| Token permute (dispatch sort) | `moe_utils.py:299-431` (`argsort(stable=True)` + `index_select`, or fused TE permute) | ✔ | Stable sort + gather is deterministic. |
| EP all-to-all dispatch | `transformer/moe/token_dispatcher.py`, `tensor_parallel/mappings.py` | ✔ | All-to-all is a rank-indexed permutation, not a floating-point reduction. EP32 traces certify exact dispatch/combine payloads when the inputs match. |
| Flex EP dispatch (DeepEP/HybridEP) | `transformer/moe/token_dispatcher.py` | ✖→forbidden | The external fused kernels do not expose a certified fixed token-arrival/combination order. CLI and Bridge deterministic overrides switch to standard all-to-all; direct deterministic configs that retain `flex` fail closed. |
| Grouped GEMM (expert FFN) | `extensions/transformer_engine.py` `TEGroupedLinear` | ▲ | Forward deterministic; backward weight-grad accumulation order is the concern + a perf target (Longcat "optimized grouped GEMM"). |
| Token unpermute (combine) | `moe_utils.py`, `moe/ops/deterministic_index_select.py` | ◆ | Dropless, unpadded deterministic A2A with top-k ≥4 and hidden ≥2,048 uses a fixed-order Triton segment sum and collision-free gather backward. It stably sorts the expert-major row map back into token order and downcasts after every addition to preserve `index_add_` rounding exactly. Unsupported shapes retain CUDA-graph-safe deterministic `index_add_`; normal mode uses `scatter_add_`. |
| Router replay (optional) | `transformer/moe/router_replay.py` | ✔ | Records top-k indices once and replays them — forces identical routing across runs (a determinism *tool*, not on the default path). |

### 1e. SSM / Mamba (hybrid models: nemotron-3-ultra)

| Step | Where | Determinism | Notes |
| --- | --- | --- | --- |
| Triton kernel autotune | `megatron/determinism_env.py`, `ssm/ops/determinism.py` and `mamba_ssm.utils.determinism` | ◆ | det: `MAMBA_DETERMINISTIC=1` plus `TRITON_CACHE_AUTOTUNING=0` before kernel imports selects one fixed config; normal mode may timing-autotune. A cold cached-autotune run was the Nemotron backward divergence root cause. |
| Tiled reduction workspace | `ssm/ops/determinism.py:106-123` | ◆ | det: allocate `zeros(..., tile_dim)` and `.sum(-1)` (ordered reduction); non-det: `empty(...)`. |
| Gated-delta-rule kernel | `ssm/gated_delta_net.py:213-216` | ◆ | det: torch `chunk_gated_delta_rule`; non-det: FLA fused kernel. |
| Causal conv1d | `ssm/gated_delta_net.py:430-446` | ◆ | det: `F.conv1d` (+ transposes); non-det: FLA `causal_conv1d`. |
| Packed sequence (`thd`) | `ssm/gated_delta_net.py:314` | ✖ | `assert not deterministic_mode` — **no deterministic packed-seq path** (gap). |

## Stage 2 — Loss

| Step | Where | Determinism | Notes |
| --- | --- | --- | --- |
| Vocab-parallel cross-entropy | `tensor_parallel/cross_entropy.py`, `tensor_parallel/deterministic_cross_entropy.py` | ▲ | The local deterministic backward fuses the one-selected-class subtract, optional label-smoothing subtract, and output-gradient scaling in one collision-free Triton pass. It reads the training loss's common non-contiguous `[sequence, batch]` gradient directly in logical row order instead of allocating a contiguous copy (no atomics/reductions; PyTorch fallback; CUDA-graph coverage). The 3 all-reduces (MAX, SUM, SUM) still use native TP collectives. `NCCL_ALGO=Ring` pins the algorithm but not necessarily the physical rank order across allocations; TP>1 remains a cross-allocation verification gap. |
| Fused CE | `cross_entropy_loss_fusion` | ✖→forbidden | Non-deterministic; **asserted off** in deterministic mode (`megatron/training/determinism.py`, `apply_determinism_to_args`). |
| MoE aux loss | `moe_utils.py:842-890` | ✔ | The aux-loss routing map is a unique-index `scatter` — verified bit-exact by the DSV3 proxy with `seq_aux_loss` and the EP32 certificates (catalog: "Aux-loss routing map"). The loss value itself is a local ordered reduction over that map. |

## Stage 3 — Backward

| Step | Where | Determinism | Notes |
| --- | --- | --- | --- |
| Autograd traversal | PyTorch engine | ✔ | Order is deterministic for a fixed graph. |
| TE attention backend + backward | TE, gated by `NVTE_ALLOW_NONDETERMINISTIC_ALGO` (`extensions/transformer_engine.py`) | ◆▲ | TE filters Flash/Fused/Unfused backends by deterministic support. The structured trace records the backend and sub-backend selected after each real forward, or an explicit unavailable event if private TE diagnostics changed. TP2×EP16 certification observed DSV3 MLA select unfused under auto and Nemotron select fused `NVTE_F16_arbitrary_seqlen`; both choices repeated exactly in forward and recompute. Dense dropout 0.1 selects deterministic Flash versus native fused because no deterministic fused backend supports that input; dropout 0 selects fused in both modes. Forcing one backend globally is therefore unsafe and shows no measured benefit. With both modes fixed to Flash, forward is neutral but deterministic Flash backward is +37.4%; with auto fused at dropout 0, deterministic fused backward is +24.7%. The remaining kernel optimization belongs in TE (Longcat "deterministic FAG" = independent accumulation buffers + global deterministic sum). |
| LayerNorm/RMSNorm backward | TE / torch norm | ✔ | PyTorch LayerNorm/RMSNorm fallback backward is bit-exact at hidden 128/2048 on H100 and GB200; TE norms are covered by the model proxies. |
| Embedding backward | see 1a | ◆ | det path = direct-index `index_put(accumulate=True)` (deterministic under `use_deterministic_algorithms`), not `F.embedding`'s atomic scatter. |
| MoE unpermute/permute backward | see 1d | ◆ | The fast unpermute's custom backward is a collision-free row gather. Unsupported unpermute shapes retain PyTorch `IndexAddBackward`; the optimized token-permute backward independently uses the same fixed-order segment-sum kernel. |
| Grouped-GEMM backward | TE `TEGroupedLinear` | ▲ | wgrad accumulation order; perf target. |

## Stage 4 — Gradient reduction (DP / FSDP)

| Step | Where | Determinism | Notes |
| --- | --- | --- | --- |
| End-of-backward TP / PP gradient synchronization | `distributed/finalize_model_grads.py`, `distributed/deterministic_collectives.py` | ◆ | Sequence-parallel and Q/K-layernorm grads use TP SUM; explicitly marked parameters use TP AVG; tied embeddings and replicated conditional/Flextron-router parameters use PP SUM. Deterministic mode keeps the native identity path for one rank and the native exact path for two ranks, where each element has only one addition. Groups larger than two use chunked all-to-all, a fixed logical-rank fp32 sum, optional post-sum division for AVG, and all-gather. The 64 MiB communication chunks bound temporary storage independently of gradient size. The trace records inputs/outputs and the selected implementation. Independent target-shape comparisons establish the zero-overhead native two-rank path. |
| Router expert-bias update | `transformer/moe/router.py`, `transformer/moe/moe_utils.py:get_updated_expert_bias` | ✔ | Per-expert assignments accumulate into an int64 buffer and use an exact TP×CP×DP integer SUM. The update compares `total_count - num_experts * expert_count` in integer space before changing the fp32 bias, so near-ties remain exact even beyond fp32's exact-integer range. The opt-in trace records the existing synchronous collective boundary. |
| Grad bucket all-reduce / reduce-scatter | `distributed/param_and_grad_buffer.py` | ◆ | Deterministic distributed-optimizer mode forces the ordered fp32 path below. Native floating-point all-reduce/reduce-scatter remains topology-sensitive across allocations. The trace hashes every bucket before launch and after its existing completion boundary. |
| fp32-accumulation reduce-scatter | `distributed/reduce_scatter_with_fp32_accumulation.py` (forced by `apply_determinism_to_args` for the distributed optimizer) | ✔ | One-rank groups return the identity without a collective or allocation; the async path retains a waitable handle and records a CUDA event only when a distinct output needs a device copy. Two-rank fp16/bf16/fp32 groups use native reduce-scatter: every output has exactly two operands and one exactly rounded addition, so physical topology cannot change an accumulation tree. Larger groups retain rank-indexed all-to-all plus ordered `torch.sum(..., dtype=fp32)` or the fixed two-level logical hierarchy. The identity path is 23.5–42.4× faster than the former helper on target bucket sizes and improves TP1×EP32 DSV3 step time 53.625→52.85 ms (-1.45%) in same-allocation ABBA; native TP2 is 1.56–1.89× faster. TP1×EP32 and TP2×EP16 DSV3/Nemotron certificates exercise both shortcuts. |
| Megatron-FSDP gradient reduction | `distributed/fsdp/src/megatron_fsdp/param_and_grad_buffer.py` | ✖→forbidden | The synchronous helpers and overlapped bucket pipeline call native all-reduce/reduce-scatter directly, including optional native AVG and premultiplied SUM. They do not consume `reduce_scatter_with_fp32_accumulation`, so deterministic CLI mode rejects Megatron-FSDP. Existing FSDP8 model cells repeat twice within one initialized topology and must not be interpreted as cross-allocation certificates. The production Ultra launcher does not enable Megatron-FSDP. |
| Distributed-optimizer param all-gather | `distributed/param_and_grad_buffer.py` | ✔ | All-gather copies rank-indexed parameter shards and performs no floating-point reduction. The trace hashes each bucket before launch and after the existing completion boundary. |
| Async param gather / grad reduce overlap | `distributed/param_and_grad_buffer.py`, `tensor_parallel/layers.py:544/565/577` (`async_op=True`) | ▲ | Async completion order can vary; determinism relies on existing `wait()` or stream barriers before use. The trace adds no wait and reports operations that outlive its window through `iteration.end.pending_collectives`. `tp_comm_overlap` is force-disabled in det mode. |

## Stage 5 — Optimizer

| Step | Where | Determinism | Notes |
| --- | --- | --- | --- |
| Distributed optimizer param sharding/order | `optimizer/distrib_optimizer.py:1094` | ✔ | Param→shard mapping "preserves deterministic ordering across ranks". |
| Adam / Muon update | `optimizer/` | ✔ | Adam is elementwise. Muon's Newton–Schulz orthogonalization is GEMM-based — deterministic given deterministic gradients and the pinned cuBLAS workspace, but not elementwise; its layer-wise distributed-optimizer variant remains outside the target certificate (see the audit disposition index). |
| Grad clipping (global norm) | `optimizer/clip_grads.py`, `distributed/deterministic_collectives.py` | ◆ | det: rank-ordered all-gather plus fixed local fp32 sum; normal: native all-reduce. This closes the scalar divergence that otherwise changes every optimizer update after an exact gradient reduction. |
| Reported losses / MoE metrics | `training.py`, `training/utils/common_utils.py`, `transformer/moe/moe_logging.py` | ◆ | det: rank-ordered sum/average for small statistics; normal: native SUM/AVG collectives. This keeps logs and scheduler-facing aggregates cross-allocation exact. |

## Cross-cutting (affect every stage)

| Concern | Control | Determinism | Notes |
| --- | --- | --- | --- |
| cuBLAS GEMM workspace | `CUBLAS_WORKSPACE_CONFIG=:4096:8` | ▲ | Required for deterministic GEMM algo selection. |
| NCCL collective algorithm | `NCCL_ALGO=Ring` | ▲ | Pins the algorithm, not the physical ring chosen for an allocation. It is insufficient by itself for floating-point reduction reproducibility; ordered reduction branches above are the guarantee. `Tree` remains excluded. |
| TE non-deterministic algos | `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` | ▲ | Forces TE deterministic attention/norm kernels. |
| CUDA caching allocator | (none) | ▲ | Allocation pattern can influence kernel autotuning/selection; flagged in the roadmap as worth investigating. |
| PP / VPP microbatch interleave | schedules and `p2p_communication.py` in `core/pipeline_parallel/` | ✔→▲ | Schedule is deterministic, but interleaving scrambles observed event order. Rank-local P2P trace events are semantically named and complete at existing waits, so comparison does not depend on cross-kind arrival order. |
| Structured runtime trace | `core/determinism_trace.py`, `extensions/transformer_engine.py`, `training.py`, `tensor_parallel/random.py`, `tensor_parallel/mappings.py`, `tensor_parallel/layers.py`, `pipeline_parallel/p2p_communication.py`, `distributed/param_and_grad_buffer.py`, `distributed/finalize_model_grads.py` | ✔ | Opt-in rank-local events add no collectives or waits. Runtime events record kernel-library versions and deterministic/backend-selection environment controls without importing optional libraries. TE attention additionally reports the actual backend/sub-backend selected after a real forward, with layer, phase, mask, QKV layout, and tensor metadata; unsupported private diagnostics emit a distinct unavailable event and fail the weekly certificate. Semantic comparison can hash paired forward/recompute outputs for Megatron checkpoint, Transformer Engine checkpoint, and output-discarding checkpoint paths; synchronous TP mapping collectives; core TP linear synchronous/async collectives at existing waits; MoE EP all-to-all; pipeline P2P; DP gradient-reduction; distributed-optimizer parameter-gather; final TP/PP gradient and token-count synchronization; and optimizer boundary tensors on selected iterations. TP userbuffer payloads and internal TE FP8/FP4 quantizer state remain open. |

---

## How to read this against the catalog

Each ▲/✖ above is a candidate for either a **test** (does it stay bit-exact at
scale?) or an **optimization** (is its deterministic path the perf bottleneck?).
The ◆ rows already have a deterministic branch and are the *template* for adding
new ones: guard with `torch.are_deterministic_algorithms_enabled()` or
`config.deterministic_mode`, keep the fast path in the other arm, and provide a
conservative deterministic fallback for unsupported shapes — the fused routing
and unpermute implementations under `megatron/core/transformer/moe/ops/` are the
current reference shape.
