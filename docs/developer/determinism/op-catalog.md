---
orphan: true
---

# Determinism Op Catalog

> **Audience:** developers and maintainers reviewing implementation paths. This
> is not a quick-start guide. Terms such as DP, TP, PP, and EP are defined in
> the [glossary](./glossary.md). Conclusion-level results are summarized in
> [validation-evidence.md](./validation-evidence.md).

This is the per-operation catalog of determinism in Megatron-Core. For the
narrative walk, see [`training-path.md`](./training-path.md); for the control plane
and targets, see [`status.md`](./status.md).

## How this catalog is maintained (evidence, not guesses)

Every row is classified by one of three evidence sources. **Do not estimate** the
perf numbers. Fill the "Perf Δ" column from the PR #5041 nsys leaderboard
(`tests/performance_tests/shell_test_utils/determinism/print_nsys_leaderboard.py`)
run on a real recipe, and draw the determinism verdict from one of the following:

- **doc** — guaranteed by the PyTorch, TE, NCCL, or cuBLAS documentation (for
  example, `index_add` and `index_put(accumulate)` are documented as deterministic
  under `torch.use_deterministic_algorithms(True)`, while `scatter_add`,
  `F.embedding` backward, and FA backward are documented as non-deterministic).
- **code** — an explicit deterministic branch exists in the source.
- **test** — confirmed by `BitExactRunner`: with `deterministic_mode` toggled, the
  same input must give bit-identical output and gradients. Rows still needing this
  are marked **⚠ verify**.

Before declaring the catalog complete, run
`python tools/determinism/audit_sensitive_ops.py megatron --git-tracked-only`.
The AST report is a review queue for collective reductions, indexed operations,
ordering calls, and determinism controls. Every training-path candidate needs
either a row here or an explicit reason that it is out of scope. The tracked-only
mode prevents local scratch files from changing the branch catalog. The scanner
intentionally does not infer a verdict, and it filters the known non-tensor
collisions that come from asynchronous or control-plane `put`, `gather`, and
module `embedding` calls.

Status legend (matches `training-path.md`): ✔ deterministic · ◆ deterministic
branch · ▲ conditional (verify) · ✖ gap (no deterministic path; `✖→forbidden` =
deterministic mode rejects the feature and fails closed).

<!-- sensitive-op-audit count=285 files=88 fingerprint=d0e3c25f90d86c6dcd3d0f3c948f0839d3b4a37a1c130e3c6121c2a0b2567a41 -->

### Repository-wide audit disposition index

The static audit currently finds **285 sensitive calls in 88 tracked source
files**. The detailed tables below remain the authority for operation-level
verdicts and performance evidence. This index gives every audited file an
explicit disposition, including paths outside the DeepSeek-V3 and Nemotron-3-Ultra
training target. `--verify-catalog` checks the count, the operation fingerprint
that is independent of line numbers, and the presence of every path, so that a new
or changed call cannot silently fall out of the review queue.

Run the gate with:

```bash
uv run python tools/determinism/audit_sensitive_ops.py megatron \
  --git-tracked-only \
  --verify-catalog docs/developer/determinism/op-catalog.md
```

| Audited source file | Disposition | Current conclusion |
| --- | --- | --- |
| `megatron/core/datasets/data_schedule.py` | target deterministic | Hybrid-CP sequence lengths and routes use integer rank-ordered gathers, a stable Python sort, and rank-indexed all-to-all copies; no floating reduction. |
| `megatron/core/determinism_trace.py` | diagnostic | Reads the deterministic-algorithms flag; tracing adds no numerical operation. |
| `megatron/core/dist_checkpointing/exchange_utils.py` | checkpoint/control | Rank-indexed tensor exchange; checkpoint load is outside the per-step numerical certificate. |
| `megatron/core/dist_checkpointing/strategies/async_utils.py` | checkpoint/control | Completion-state reductions do not feed model arithmetic. |
| `megatron/core/dist_checkpointing/strategies/filesystem_async.py` | checkpoint/control | Deterministic size/type ordering for I/O scheduling. |
| `megatron/core/dist_checkpointing/strategies/state_dict_saver.py` | checkpoint/control | Metadata agreement and rank-indexed broadcast; no optimizer value reduction. |
| `megatron/core/distributed/deterministic_collectives.py` | detailed below | Fixed logical-rank floating reductions and rank-indexed collectives implement several deterministic branches. |
| `megatron/core/distributed/distributed_data_parallel.py` | target deterministic | Initial parameter synchronization is a rank-zero broadcast. |
| `megatron/core/distributed/finalize_model_grads.py` | detailed below | Final TP/PP gradients use the ordered deterministic reduction branch. |
| `megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py` | target forbidden | Megatron-FSDP is rejected by deterministic CLI mode; this file contributes rank-indexed parameter broadcast only. |
| `megatron/core/distributed/fsdp/src/megatron_fsdp/mixed_precision.py` | target forbidden | Megatron-FSDP is rejected; its FP8 amax reduction has no target certificate. |
| `megatron/core/distributed/fsdp/src/megatron_fsdp/param_and_grad_buffer.py` | target forbidden | Native floating gradient collectives are the reason deterministic mode rejects Megatron-FSDP. |
| `megatron/core/distributed/fsdp/src/megatron_fsdp/uneven_dtensor.py` | target forbidden | Megatron-FSDP is rejected; uneven-DTensor validation/materialization is not certified. |
| `megatron/core/distributed/param_and_grad_buffer.py` | detailed below | Ordered distributed-optimizer reduce-scatter is covered; the non-distributed-optimizer native DP all-reduce remains a gap. |
| `megatron/core/distributed/reduce_scatter_with_fp32_accumulation.py` | detailed below | Single-rank identity, native exact two-rank reduction, or fixed flat/hierarchical logical-rank fp32 accumulation. |
| `megatron/core/energy_monitor.py` | reporting/control | Integer energy counters are summed for telemetry and never feed training state. |
| `megatron/core/export/trtllm/trtllm_weights_converter/distributed_trtllm_model_weights_converter.py` | outside target | Export-only vocabulary-padding metadata reduction. |
| `megatron/core/fault_injector.py` | diagnostic | Rank-indexed fault-control broadcast. |
| `megatron/core/fp8_utils.py` | conditional | Finite amax uses order-independent MAX, but internal FP8/FP4 quantizer state is not yet covered by a cross-allocation target certificate. |
| `megatron/core/fusions/fused_cross_entropy.py` | target forbidden | Fused cross-entropy remains forbidden in deterministic mode. |
| `megatron/core/inference/batch_dimensions_utils.py` | outside target | Inference-only shape agreement and deterministic batch-dimension ordering. |
| `megatron/core/inference/communication_utils.py` | outside target | Inference-only rank-indexed broadcasts. |
| `megatron/core/inference/contexts/dynamic_context.py` | outside target | Inference allocator agreement, not model training arithmetic. |
| `megatron/core/inference/contexts/kv_block_allocator.py` | outside target | Inference LRU ordering. |
| `megatron/core/inference/contexts/mamba_slot_allocator.py` | outside target | Inference LRU ordering. |
| `megatron/core/inference/engines/dynamic_engine.py` | outside target | Inference completion ordering has its own deterministic branch. |
| `megatron/core/inference/sampling/torch_sampling.py` | outside target | RNG-driven sampling; top-k/top-p ordering is not a training determinism claim. |
| `megatron/core/inference/text_generation_controllers/text_generation_controller.py` | outside target | Inference top-k and collision-free gathers. |
| `megatron/core/inference/text_generation_server/dynamic_text_gen_server/endpoints/common.py` | outside target | Inference command broadcast. |
| `megatron/core/inference/text_generation_server/endpoints/common.py` | outside target | Inference command broadcast. |
| `megatron/core/inference/unified_memory.py` | outside target | Inference allocator metadata all-gather. |
| `megatron/core/models/common/embeddings/rope_utils.py` | target deterministic | Context-parallel positional embeddings use collision-free `index_select`. |
| `megatron/core/models/common/language_module/language_module.py` | target deterministic | Tied embedding initialization reduces one initialized copy with one zero copy across the embedding group. |
| `megatron/core/models/hybrid/hybrid_layer_allocation.py` | target deterministic | Python attribute-name sort fixes hybrid-layer construction order. |
| `megatron/core/models/mimo/comm/colocated_communicator.py` | outside target | MIMO rank-indexed all-gather. |
| `megatron/core/models/mimo/optimizer.py` | outside target gap | MIMO uses native floating grad-norm / success reductions and has no ordered target path. |
| `megatron/core/models/mimo/partition/utils.py` | outside target | MIMO collision-free context-parallel index selection. |
| `megatron/core/models/multimodal/context_parallel.py` | outside target gap | Multimodal native floating reduce-scatter has no ordered path; gathers/all-to-all are rank-indexed. |
| `megatron/core/models/multimodal/llava_model.py` | outside target | LLaVA collision-free token selection. |
| `megatron/core/models/vision/radio.py` | outside target | Vision positional-encoding gather. |
| `megatron/core/num_microbatches_calculator.py` | target deterministic | Step schedule is sorted by explicit iteration key. |
| `megatron/core/optimizer/clip_grads.py` | detailed below | Global norm uses ordered scalar reduction in deterministic mode; zero counts are integer reductions. |
| `megatron/core/optimizer/distrib_optimizer.py` | detailed below | State movement is rank-indexed and parameter ordering is explicit. |
| `megatron/core/optimizer/layer_wise_optimizer.py` | outside target gap | Alternative layer-wise optimizer has no DeepSeek/Ultra certificate. |
| `megatron/core/optimizer/optimizer.py` | target deterministic | The audited collective combines the finite/overflow control flag, not gradients. |
| `megatron/core/optimizer/qk_clip.py` | conditional | Finite QK maxima use order-independent MAX; NaN-containing training is outside the certificate. |
| `megatron/core/pipeline_parallel/bridge_communicator.py` | target deterministic | Pipeline bridge transfers are rank-indexed broadcasts without floating reduction. |
| `megatron/core/pipeline_parallel/hybrid_cp_schedule.py` | target deterministic | Hybrid-CP schedule metadata uses rank-indexed broadcast. |
| `megatron/core/rerun_state_machine.py` | reporting/control | Rerun decisions reduce integer booleans; sample sorting affects diagnostics only. |
| `megatron/core/resharding/nvshmem_copy_service/planning/communication_scheduler.py` | checkpoint/control | Explicit deterministic copy-schedule ordering. |
| `megatron/core/resharding/nvshmem_copy_service/planning/workload_packer.py` | checkpoint/control | Explicit deterministic copy-work ordering. |
| `megatron/core/resharding/planner.py` | checkpoint/control | Explicit destination-offset ordering for resharding. |
| `megatron/core/ssm/gated_delta_net.py` | detailed below | Deterministic SSM fallback plus collision-free index selection. |
| `megatron/core/ssm/mamba_context_parallel.py` | target deterministic | Attention-load balancing uses collision-free `index_select`. |
| `megatron/core/ssm/ops/determinism.py` | detailed below | Reads the deterministic flag and fixes Triton configuration/workspaces. |
| `megatron/core/tensor_parallel/cross_entropy.py` | detailed below gap | Local selected-class update is closed; native TP floating reductions remain topology-sensitive. |
| `megatron/core/tensor_parallel/data.py` | target deterministic | Batch dictionaries and tensors are rank-zero broadcasts. |
| `megatron/core/tensor_parallel/layers.py` | detailed below gap | Embedding backward and TP linear native floating reductions remain cataloged gaps. |
| `megatron/core/tensor_parallel/mappings.py` | detailed below gap | Rank-indexed mappings are stable; native TP SUM/reduce-scatter remains topology-sensitive. |
| `megatron/core/transformer/attention.py` | diagnostic | Real-time checks all-gather rank-indexed values and do not update the model. |
| `megatron/core/transformer/experimental_attention_variant/dsa.py` | detailed below | DSA masks are collision-free; logging reductions do not feed model state. |
| `megatron/core/transformer/moe/moe_utils.py` | detailed below | Routing, top-k, permute, unpermute, and expert-bias families have individual rows. |
| `megatron/core/transformer/moe/ops/deterministic_index_select.py` | detailed below | Fixed-order deterministic permute/unpermute helpers. |
| `megatron/core/transformer/moe/ops/deterministic_routing.py` | detailed below | Collision-free fused routing writes. |
| `megatron/core/transformer/moe/paged_stash.py` | reporting/control | Overflow decisions sum integer flags; no floating model value is reduced. |
| `megatron/core/transformer/moe/router.py` | detailed below | Sinkhorn/top-k routing has target proxy coverage. |
| `megatron/core/transformer/moe/router_replay.py` | conditional | Replay gathers are collision-free, but router replay has no target model certificate. |
| `megatron/core/transformer/moe/token_dispatcher.py` | target forbidden | Deterministic mode rejects flex/DeepEP dispatch and uses certified standard all-to-all. |
| `megatron/core/transformer/moe/token_dispatcher_inference.py` | outside target | Inference-only native reduce-scatter remains outside the training certificate. |
| `megatron/core/transformer/multi_token_prediction.py` | conditional | TP argmax is rank-indexed and deterministic; floating MTP loss reductions affect logging only and may vary in low bits. |
| `megatron/core/utils.py` | target deterministic | Audited calls are rank-indexed diagnostics/batch movement or collision-free context-parallel selection. |
| `megatron/elastification/flextron_elasticity_hooks.py` | outside target conditional | Flextron routing is cataloged separately but lacks a target model certificate. |
| `megatron/elastification/loss_func.py` | outside target gap | Elastification loss reporting uses native floating reduction. |
| `megatron/elastification/router/hybrid_flex_router.py` | outside target conditional | Alternate elasticity router uses sorted top-k and collision-free selection but has no target certificate. |
| `megatron/post_training/loss_func.py` | outside target gap | Post-training loss reporting uses native floating reduction. |
| `megatron/rl/agent/api.py` | outside target | Explicit rollout grouping order. |
| `megatron/rl/agent/weighted_multi_task.py` | outside target | Explicit task-count distribution order. |
| `megatron/rl/rl_utils.py` | outside target | RL rollout ordering has a deterministic branch; RL reductions/gathers are not target-certified. |
| `megatron/rl/sequence_packing_utils.py` | outside target | RL packing metadata all-gathers. |
| `megatron/training/checkpointing.py` | checkpoint/control | Metadata agreement reduction does not combine model values. |
| `megatron/training/datasets/fim_dataset.py` | target deterministic | FIM boundaries are explicitly sorted. |
| `megatron/training/determinism.py` | detailed below | Enables deterministic algorithms after validating/overriding unsupported features. |
| `megatron/training/dist_signal_handler.py` | reporting/control | Signal-state all-gather. |
| `megatron/training/ft_integration.py` | diagnostic | Fault-tolerance control broadcast. |
| `megatron/training/gpu_sniff_test.py` | diagnostic | Communication benchmark, not a training numerical path. |
| `megatron/training/inprocess_restart.py` | reporting/control | Zero-valued collective eagerly initializes NCCL; it does not update training state. |
| `megatron/training/training.py` | detailed below | Setup/control collectives and ordered loss/metric reporting are classified in the training-path rows. |
| `megatron/training/utils/common_utils.py` | reporting/control | Parameter norms and max/boolean statistics are diagnostics; native floating diagnostic reductions may vary in low bits. |

---

## Control plane

| Item | File:line | Effect |
| --- | --- | --- |
| `--deterministic-mode` (this branch) | `megatron/training/arguments.py` | delegates the complete setup to `apply_determinism_to_args()` before model construction |
| `set_determinism_env_vars()` (stacked on PR #5041) | `megatron/determinism_env.py` | default `NCCL_ALGO=Ring`, `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`; force `MAMBA_DETERMINISTIC=1`, `TRITON_CACHE_AUTOTUNING=0` before kernel imports |
| `apply_determinism_to_args()` | `megatron/training/determinism.py` | assert no CE-fusion; validate `NCCL_ALGO` (excl. Tree); force `tp_comm_overlap=False`; for distributed optimizer require one instance/no collective AVG and force ordered fp32 reduce-scatter; finally enable torch deterministic algorithms |
| `deterministic_mode` config flag | `megatron/core/model_parallel_config.py:153` | threaded into `TransformerConfig`; read by library branches below |

---

## Catalog table

> Columns: **Op** · **File:line** · **Primitive** · **Det?** · **Det path** ·
> **Non-det path** · **Selected by** · **Evidence** · **Perf Δ** · **Gap / TODO**

### MoE (the determinism hot zone)

| Op | File:line | Primitive | Det? | Det path | Non-det path | Selected by | Evidence | Perf Δ | Gap / TODO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Token unpermute (combine) | `transformer/moe/moe_utils.py`, `transformer/moe/ops/deterministic_index_select.py` | fixed-order segment sum vs scatter-accumulate | ◆ | dropless/unpadded A2A at top-k ≥4 and hidden ≥2,048: stable inverse map + Triton segment sum with gather backward; fallback: `index_add_` | `scatter_add_` | deterministic algorithms plus guarded uniform top-k shape | code+**test** | final-source isolated ABBA: 1.63–4.48× faster and 42–47% lower peak allocation than deterministic `index_add_` | The kernel preserves `index_add_` contribution order and per-add dtype rounding exactly. Smaller shapes, capacity/drop/padding, unsupported dtype/device, and nonuniform routes retain the existing fallback. Full-step gain remains to be isolated because prior NVTX dispatch deltas overlapped other work. |
| Routing map/probs | `moe_utils.py`, `moe/ops/deterministic_routing.py` | collision-free dense write + gather backward | ◆ | fused Triton probability/map kernel; PyTorch `scatter_` fallback | out-of-place `scatter` ×2 | `are_deterministic_algorithms_enabled()` | code+**test** | isolated forward 4.21–4.63× faster and forward+backward 1.66–1.87× faster than deterministic `scatter_`; production scatter/fused ABBA median -2.0% | Top-k indices are unique within each row, so the fused path has no atomics or reductions. One row program initializes both dense outputs, writes selected entries, and backward gathers only selected gradients. Unsupported devices, dtypes, layouts, or missing Triton retain the prior scatter path. |
| Top-k expert select | `moe_utils.py` | `torch.topk(sorted=...)` | ◆ | `sorted=True`, including no-grad checkpoint forward | `sorted=is_grad_enabled()` | deterministic algorithms | code+**test** | `cub::DeviceRadixSort` 6.36 vs 1.23 ms over 3 steps (+5.13 ms) | Fixes sigmoid top-k probability drift between original forward and grad-enabled recompute. Generic unsorted-select + selected-sort and a fused Triton register-sort prototype are both rejected below: the former is slower, while the latter adds a slower backward and changes gradients for quantized ties. A viable optimization must live inside selection/fusion without another autograd boundary and must preserve tie gradients. |
| Expert-bias score gather | `moe_utils.py` | `torch.gather` + dense backward write | ◆ | PyTorch deterministic gather backward (`index_put_`) | PyTorch scatter-add backward | deterministic algorithms + expert bias | code+profile | candidate isolated forward+backward 1.48–2.23× faster, but gather-only production ABBA regressed 3.3% | A collision-free Triton candidate was exact and reduced the profiled range from `GatherBackward0` +1.75 ms to 0.29 ms, but two balanced production ABBAs regressed 3.2–3.3%; it was removed. Keep as an open fusion target rather than landing an isolated-only win. |
| Flextron soft-mask routing | `elastification/flextron_elasticity_hooks.py` | weighted sigmoid + top-k + gather + collision-free scatter | ▲ | inherits global deterministic gather backward; top-k is always sorted | native gather backward when global deterministic algorithms are off | global deterministic algorithms (`deterministic_mode` argument is deprecated/unused) | code | not profiled | This alternate Flextron router is now cataloged, but has no model-level deterministic certificate. Its unique-index scatters are deterministic; gather backward remains the same open generic path as the main router. |
| Group-limited top-k mask | `moe_utils.py:590-645` | `scatter_(1, group_idx, 1)` | ✔ | unique group indices ⇒ det fwd | — (no branch) | always | code+**test** | small | **Verified** by the DSV3 proxy across EP≤4/TP/FSDP/PP/VPP and by the final EP32 certificates (`518849`/`518850`). |
| Aux-loss routing map | `moe_utils.py:888-901` | `scatter` | ✔ | unique indices ⇒ det fwd | — (no branch) | always | code+**test** | small | **Verified** via DSV3 proxy and final EP32 certificates with `seq_aux_loss` enabled. |
| Router expert-bias token counts | `transformer/moe/router.py`, `transformer/moe/moe_utils.py:get_updated_expert_bias` | local accumulation + TP×CP×DP SUM | ✔ | int64 accumulation, native integer all-reduce, and integer-space mean comparison | same | expert-bias routing | code+**test** | not profiled | Counts and the bias-update sign remain exact beyond fp32's `2**24` integer range and are independent of NCCL reduction order. The trace fingerprints the count vector before and after the existing synchronous collective. |
| Capacity-drop mask | `moe_utils.py:950-962` | `scatter` | ✔ | unique indices | — (no branch) | capacity factor | code+**test** | small | **Verified** bit-exact for `probs` and `position` drop policies, including unpadded and fixed-capacity padded A2A dispatch, across EP≤4/TP/FSDP/PP/VPP. ⚠ still unverified at EP>16. |
| Router map (Sinkhorn) | `transformer/moe/router.py:260` | `scatter` | ✔ | unique top-k indices | — | Sinkhorn routing | code+**test** | small | **Verified** bit-exact with the DSV3 Sinkhorn preset across EP≤4/TP/FSDP/PP/VPP. ⚠ still unverified at EP>16. |
| Pad routing map | `moe_utils.py:648-680` / `fusions/fused_pad_routing_map.py` | `cumsum` + mask write | ✔ | ordered cumsum | — | always | doc | small | cumsum is deterministic. |
| Token permute (dispatch) | `moe_utils.py:300-442`; `token_dispatcher.py:645-659`; `moe/ops/deterministic_index_select.py` | `argsort(stable=True)` + row gather | ◆ | fixed-order Triton backward for dropless top-k ≥4 and hidden ≥2048 | PyTorch `index_select` backward / fused TE | deterministic algorithms + guarded shape | code+**test** | H100: `IndexSelectBackward0` 50.34→14.97 ms (**-70%**) over 3 profiled steps; GB200 isolated kernel: 6.5–55.4% lower latency | Model-level A2A top-k-8/top-k-6 presets are bit-exact. Extend the fast path only with shape-specific evidence; supported fallbacks remain unchanged. |
| EP all-to-all dispatch/combine | `transformer/moe/token_dispatcher.py`, `tensor_parallel/mappings.py` | `all_to_all` | ✔ | rank-indexed permutation | — | always | code+**test** | — | EP32 DSV3 and Nemotron-style structured traces fingerprint dispatch/combine in forward, recompute, and backward. The collective does not perform floating-point reduction. |
| Flex EP dispatch/combine | `transformer/moe/token_dispatcher.py` (`MoEFlexTokenDispatcher`) | external DeepEP/HybridEP fused permute + communication | ✖→forbidden | deterministic launchers select standard `alltoall` before config construction | fused flex dispatcher | deterministic mode switch plus `TransformerConfig` fail-closed validation | code+**test** | not profiled | External token-arrival, permutation, and combine order is not controlled or traced by MCore. Focused MCore and Bridge tests verify the guard and recipe override. Independent-allocation tensor certificates are required before either backend can be permitted in deterministic mode. |
| Grouped GEMM (experts) | `extensions/transformer_engine.py` `TEGroupedLinear` | grouped matmul | ✔▲ | TE deterministic kernels | TE fast kernels | `NVTE_ALLOW_NONDETERMINISTIC_ALGO` | doc+**test** | `_GroupedLinearBackward` +19.2 ms (+20%) over 3 profiled steps | Bit-exact in DSV3 + nemotron proxies (`moe_grouped_gemm=True`). wgrad order remains a perf target ("optimized grouped GEMM", "fused GemmAdd"). |
| Router replay | `transformer/moe/router_replay.py` | record/replay top-k | ✔ | replay recorded indices | — | opt-in | code | — | A determinism *tool*, not default path. |

### Attention

| Op | File:line | Primitive | Det? | Det path | Non-det path | Selected by | Evidence | Perf Δ | Gap / TODO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TE attention backend selection | TE; `extensions/transformer_engine.py`; `TransformerConfig.attention_backend` | FlashAttention / cuDNN FusedAttention / unfused DPA | ◆▲ | TE filters out backends/sub-backends that cannot honor deterministic execution | TE selects an eligible backend by internal priority and heuristics | `attention_backend`, `NVTE_{FLASH,FUSED,UNFUSED}_ATTN`, `NVTE_ALLOW_NONDETERMINISTIC_ALGO` | source+runtime+profile+**test** | dense dropout 0.1: det Flash vs native fused; dropout 0: fused in both modes; forcing fused shows no gain | The structured trace records the selected backend/sub-backend after the real TE forward and fails closed when private diagnostic state is unavailable. TP2×EP16 certification observed DSV3 MLA choose unfused under auto and Nemotron choose fused `NVTE_F16_arbitrary_seqlen`, identically in forward/recompute and both launches. Do not globally force a backend. |
| FlashAttention backward | TE; `extensions/transformer_engine.py:1697-1703` | atomic dQ/dK/dV accumulation | ◆▲ | TE deterministic FA kernels | TE atomic FA | `deterministic_mode` asserts `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` | source+profile+code | fixed-Flash dense profile: +6.034 ms (+37.4%) over the capture | **Deterministic FAG** (Longcat/DSV4): independent accumulation buffers + global deterministic sum remains an external TE/kernel target. |
| Core attention fwd | TE `DotProductAttention` | fused attn | ✔ | — | — | — | doc | — | Forward deterministic. |
| Multi-Latent Attention (MLA) | `transformer/multi_latent_attention.py` (q/kv low-rank proj + YaRN rope) | low-rank GEMMs + rope | ✔ | — | — | — | doc+**test** | — | **Verified** bit-exact by `test_deepseek_model.py` (DSV3 MLA + qk_layernorm + YaRN) across EP/TP/FSDP/PP/VPP. |
| Attention dropout | `transformer/dot_product_attention.py:114-220` | RNG mask | ▲ | fixed RNG state | — | RNG | doc | — | Reproducible under the tracked RNG state — not a divergence source by itself. dropout > 0 additionally restricts deterministic TE backend eligibility (see backend-selection row); dropout=0 removes both effects. |

### Normalization

| Op | File:line | Primitive | Det? | Det path | Non-det path | Selected by | Evidence | Perf Δ | Gap / TODO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LayerNorm/RMSNorm fwd | `transformer/torch_norm.py`, TE norm | reduction | ✔ | — | — | backend | doc+**test** | — | PyTorch fallback verified directly; TE norms are exercised by model proxies. |
| LayerNorm/RMSNorm bwd | TE / `transformer/torch_norm.py` | weight-grad reduction | ✔ | PyTorch fallback / TE deterministic kernels | TE fast kernels | backend + `NVTE_ALLOW_NONDETERMINISTIC_ALGO` | code+**test** | TBD | **Verified** bit-exact for PyTorch LayerNorm and RMSNorm at hidden 128/2048 on H100 and GB200; TE norms remain covered by the model proxies. |

### Embedding & TP linear

| Op | File:line | Primitive | Det? | Det path | Non-det path | Selected by | Evidence | Perf Δ | Gap / TODO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vocab-parallel embedding | `tensor_parallel/layers.py:299-303` | indexing vs `F.embedding` | ◆ | direct `weight[idx]` (det bwd) | `F.embedding` (atomic bwd) | `self.deterministic_mode` | code+profile | current Nemotron `IndexBackward0`: 0.728 ms over 2 steps; custom candidate was only 0.61–0.69× as fast | `F.embedding` has a non-deterministic backward. An exact stable-sort/segmented-reduction candidate was rejected on GB200; retain direct indexing and re-profile on dense/Hopper shapes before revisiting. |
| RoPE / position emb | `models/common/embeddings/rotary_pos_embedding.py` | trig | ✔ | — | — | — | doc | — | — |
| Column/Row parallel linear (GEMM) | `tensor_parallel/layers.py` | cuBLAS GEMM | ▲ | fixed cuBLAS workspace | — | `CUBLAS_WORKSPACE_CONFIG` | doc | — | — |
| Synchronous TP mapping collectives | `tensor_parallel/mappings.py` | all-reduce, first/last-dim all-gather and reduce-scatter | ▲ | native NCCL with overlap disabled | same collectives | `tp_comm_overlap=False` in deterministic mode | code+**test** | hash tracing is debug-only | The opt-in trace fingerprints exact inputs/outputs and labels original forward, recompute, and backward without adding a collective or wait. Permutation collectives are deterministic, but native floating-point SUM order remains topology-sensitive across allocations. |
| Core TP linear synchronous/async collectives | `tensor_parallel/layers.py` | forward all-gather; backward async all-gather/all-reduce/reduce-scatter | ▲ | existing `wait()` completion boundaries; overlap disabled in deterministic mode | communication/wgrad overlap | `sequence_parallel`, `allreduce_dgrad`, `tp_comm_overlap` | code+**test** | hash tracing is debug-only | The trace wraps async work and records output only from the existing wait, without adding or moving a wait; forward, recompute, and backward are labeled explicitly. Native floating-point SUM order remains topology-sensitive. TP userbuffer collectives remain uninstrumented. |
| Pipeline P2P send/receive | `pipeline_parallel/p2p_communication.py` | `isend` / `irecv` / batched P2P | ▲ | existing wait pins readiness before use | overlap / VPP | schedule | code+**test** | hash tracing is debug-only | Structured trace fingerprints sends and completed receives without adding waits; PP2×VPP2×EP2 passed two-run comparison. |

### Loss

| Op | File:line | Primitive | Det? | Det path | Non-det path | Selected by | Evidence | Perf Δ | Gap / TODO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vocab-parallel cross-entropy | `tensor_parallel/cross_entropy.py`, `tensor_parallel/deterministic_cross_entropy.py` | 3× all-reduce + fp32 + selected class update | ▲ | fused collision-free Triton selected subtract + smoothing + output scaling, including direct logical-order loads from 2D-strided loss gradients; native NCCL TP reductions | generic `index_put_` selected update + elementwise kernels; native NCCL | deterministic algorithms | code+**test** | fused local backward 3.34–20.41× faster for contiguous gradients and 14.41–16.52× for 2D-strided gradients in isolation; original production ABBA -2.7%; dense strided ABBA -12.9% | The local backward update is closed without atomics and supports label smoothing, the training loss's common strided layout, fallback, and CUDA graphs. `NCCL_ALGO=Ring` still does not pin the physical TP ring across allocations; TP>1 cross-allocation certification and an ordered SUM path remain open. |
| Fused cross-entropy | `cross_entropy_loss_fusion` | fused kernel | ✖ | — (forbidden) | fused | asserted off | code | — | Open: can it be made deterministic? (perf opportunity). |

### Backward / grad reduction / optimizer

| Op | File:line | Primitive | Det? | Det path | Non-det path | Selected by | Evidence | Perf Δ | Gap / TODO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Final TP/PP gradient synchronization | `distributed/finalize_model_grads.py`, `distributed/deterministic_collectives.py` | all-reduce (SUM / AVG) | ◆ | groups ≤2: native identity/single-addition path; groups >2: chunked all-to-all + fixed logical-rank fp32 sum + all-gather; AVG divides after the sum | native NCCL | deterministic algorithms plus model config (`sequence_parallel`, `qk_layernorm`, shared embeddings, replicated conditional/Flextron parameters) | code+**test** | TP2 target path: no added collective; TP4 GB200 primitive: +0.13 ms at 64 MiB and +0.47 ms at 256 MiB (1.53×/1.65×) | The 64 MiB chunk cap bounds temporary storage to a constant multiple of the chunk size. Independent TP2×PP2×DP4 allocations match 1,856/1,856 semantic events and select `native_two_rank_exact`. TP4 training exercises the ordered implementation. Other TP reductions remain separate gaps. |
| Grad bucket all-reduce / reduce-scatter | `distributed/param_and_grad_buffer.py` | floating-point NCCL collective | ◆ | ordered fp32 reduce-scatter for deterministic distributed optimizer | native NCCL | deterministic args | code+**test** | hash tracing is debug-only | Native floating-point collectives are allocation-topology-sensitive even with Ring. Non-distributed-optimizer all-reduce remains a gap. |
| fp32-accum reduce-scatter | `distributed/reduce_scatter_with_fp32_accumulation.py` | identity, native two-rank RS, or all-to-all + ordered `sum(fp32)` | ✔ | group 1: no-op identity; group 2 fp16/bf16/fp32: native single-addition path; other groups: flat rank order or fixed two-level logical-rank tree | native RS | forced in deterministic distributed optimizer; hierarchy is explicit | code+**test** | group-1 identity is 23.5–42.4× faster than the former helper on 32–160 MiB target buckets; TP2 native is 1.56–1.89× faster; group>2 hierarchy remains 1.324–1.346 ms vs flat 3.110–3.118 ms | TP1×EP32 jobs exercise large expert buckets as `single_rank_identity`; TP2 native matches the ordered path in 288 randomized fp16/bf16/fp32 sync/async cells and across two independent allocations. Larger reductions retain fixed fp32 order. One optimizer instance and no collective AVG remain enforced. |
| Megatron-FSDP grad all-reduce / reduce-scatter | `distributed/fsdp/src/megatron_fsdp/param_and_grad_buffer.py` | native NCCL SUM / AVG / premultiplied SUM | ✖→forbidden | none | synchronous helpers plus overlapped inner/outer-DP bucket pipelines | `use_megatron_fsdp` | source+same-topology **test** | not profiled | Deterministic CLI mode now rejects this backend because it does not consume `reduce_scatter_with_fp32_accumulation`. FSDP8 proxy cells repeat inside one process-group topology only. Add an ordered reduction and trace its existing stream/event completion boundaries before claiming cross-allocation support. The Ultra production launcher uses standard distributed optimizer, not this path. |
| Distributed-optimizer param all-gather | `distributed/param_and_grad_buffer.py` | NCCL all-gather | ✔ | rank-indexed byte copies | — | always | code+**test** | hash tracing is debug-only | No floating-point reduction; structured trace fingerprints sync and overlapped gathers without adding a collective or wait. |
| Distributed optimizer param order | `optimizer/distrib_optimizer.py:1094` | shard mapping | ✔ | "preserving deterministic ordering across ranks" | — | always | code | — | — |
| Grad clip global norm | `optimizer/clip_grads.py`, `distributed/deterministic_collectives.py` | SUM reduction | ◆ | rank-ordered all-gather + local sum | native all-reduce | deterministic algorithms | code+**test** | TBD | Closed a one-ulp scalar divergence that otherwise changed every optimizer parameter. Intended only for small statistics. |
| Reported loss / MoE metrics | `training.py`, `training/utils/common_utils.py`, `transformer/moe/moe_logging.py` | SUM / AVG reductions | ◆ | rank-ordered all-gather + local sum/divide | native all-reduce | deterministic algorithms | code+**test** | TBD | Keeps reported loss and expert-bias/load-balancing metrics exact across allocations. |

### SSM / Mamba (hybrid models)

| Op | File:line | Primitive | Det? | Det path | Non-det path | Selected by | Evidence | Perf Δ | Gap / TODO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Triton autotune | `megatron/determinism_env.py`, `ssm/ops/determinism.py`, external `mamba_ssm.utils.determinism` | autotune config select | ◆ | one fixed config (`TRITON_CACHE_AUTOTUNING=0`) | timing-selected config | early deterministic env | code+**test** | paired Nemotron EP32 proxy: fixed 74.5 ms vs autotuned 75.0 ms (no slowdown) | Cold cached-autotune selected different reduction tilings across launches; env setup must precede kernel-module import. |
| Tiled reduction workspace | `ssm/ops/determinism.py:106-123` | `zeros`+`sum` vs `empty` | ◆ | ordered tile sum | unordered | `use_deterministic_mode()` | code | paired Nemotron EP32 proxy: workspace+autotune 75.0 ms vs normal 75.5 ms (no slowdown) | Extra memory for tiled reduction; no measurable step-time cost in the current proxy. |
| Gated-delta-rule kernel | `ssm/gated_delta_net.py:213-216` | torch vs FLA fused | ◆ | `torch_chunk_gated_delta_rule` | FLA fused | `deterministic_mode` | code | TBD | — |
| Causal conv1d | `ssm/gated_delta_net.py:430-446` | `F.conv1d` vs FLA | ◆ | `F.conv1d` (+transpose) | `causal_conv1d` | `deterministic_mode` | code | TBD | — |
| Mamba-2 combined training | `ssm/mamba_mixer.py`, external `mamba_ssm` | fused causal conv + selective scan | ✔▲ | fused path with deterministic workspaces and fixed Triton config | same path with timing autotune / atomic reductions | early env + torch deterministic algorithms | code+**test** | paired proxy: no measurable deterministic Mamba overhead | Exact paired runs establish that fixed autotune configuration, rather than disabling the fused path, prevents cold-cache autotune drift. |
| Packed sequence (`thd`) | `ssm/gated_delta_net.py:314` | — | ✖ | — | thd | asserted off | code | — | **Gap**: no deterministic packed-seq SSM path. |

### Inference / RL (not training-loop, listed for completeness)

| Op | File:line | Det? | Notes |
| --- | --- | --- | --- |
| DP inference coordinator scheduling | `inference/.../dynamic_engine.py:607`, `data_parallel_inference_coordinator.py:181` | ◆ | sorted rank identities vs completion order |
| RL rollout ordering | `rl/rl_utils.py:678` | ◆ | sort by `problem_id` vs completion order |
| Inference token sampling | `inference/sampling/torch_sampling.py:61-69` | ▲ | `cumsum`/`scatter` in top-p; RNG-driven |
| DSA sparse-attention masks | `transformer/experimental_attention_variant/dsa.py:214/385/950` | ✔ | Unique top-k `scatter_` masks are bit-exact in indexer-loss forward, recomputed manual backward, and unfused sparse-attention backward (`test_dsa_paths.py`). |

---

## Hotspots (performance priorities)

The measurements behind these conclusions are retained locally, together with
their run provenance. This page records only the results that affect an
engineering decision, and it is not a profiling log.

### Findings

| Surface | Finding | Engineering conclusion |
| --- | --- | --- |
| Sorted router top-k | Requiring a stable selection order adds visible sort work. Prototype post-selection sorting preserved values but either slowed training or changed tie gradients. | Keep `sorted=True`; a future improvement must preserve both values and tie gradients inside the selection implementation. |
| MoE token unpermute | The guarded fixed-order segment-sum implementation is substantially faster than the deterministic `index_add_` fallback at supported shapes and lowers peak allocation. | Use the guarded fast path; preserve the fallback for unsupported routes and shapes. |
| Routing map construction | Collision-free fused routing is materially faster in isolation and gives a modest end-to-end improvement. | Keep the fused path and its `scatter_` fallback. |
| One- and two-rank reduce-scatter | A one-rank identity needs no communication; a supported two-rank output has one addition only. Both are exact and much faster than the generic ordered helper. | Select these semantic shortcuts before the general fixed-order path. |
| Larger floating reductions | A fixed logical hierarchy removes topology-dependent accumulation order. Its isolated cost can be near native, but full-step benefit depends on overlap and workload shape. | Retain the hierarchy for correctness; judge performance with paired end-to-end profiles. |
| Mamba / Triton | Pinning the deterministic configuration avoids cold-cache autotune drift without a measured proxy slowdown. | Set deterministic environment controls before Mamba or Triton modules are imported. |
| Attention | Backend eligibility is input- and library-dependent. Forcing a global backend is neither generally available nor reliably faster. | Record the backend selected by an actual forward; leave selection to Transformer Engine under deterministic eligibility rules. |
| Remaining cost | Attention backward, vocab-parallel loss/gather backward, and some grouped-GEMM paths can dominate the remaining deterministic gap. | Profile the actual model and topology before optimizing; do not infer end-to-end gain from one kernel range. |

### Profiling interpretation

The profiler workflow compares a deterministic configuration against an ordinary
one on the same recipe, and then attributes only the ranges whose boundaries make
the comparison meaningful. NVTX range totals may overlap, so they do not add up to
step latency. For a candidate change, use paired runs in alternating order and keep
both legs, because a single placement can hide allocator, cache, or overlap
effects.

### Validation coverage and open gaps

The checked-in suites and independent paired launches establish these findings:

- DeepSeek-V3-style and Nemotron-3-Ultra-style proxy configurations cover MoE
  routing, all-to-all dispatch/combine, activation recompute, optimizer state,
  and supported parallel layouts.
- The trace tooling verifies recompute correspondence, completed collective
  boundaries, selected attention backend, and the deterministic implementation
  selected for a reduction.
- Small-group reduction shortcuts are covered by focused semantic tests and
  model-level comparisons. Larger groups retain ordered FP32 accumulation.
- The deterministic unpermute and routing fast paths are checked against their
  conservative deterministic fallbacks, including CUDA-graph cases.

The following are intentionally not claimed as a general deterministic contract:

- Tensor-parallel floating reductions in mappings, linears, and
  vocab-parallel cross entropy remain topology-sensitive beyond the supported
  paths.
- Fused cross entropy, tensor-parallel communication overlap, multiple
  distributed-optimizer instances, collective averaging, Megatron-FSDP, and
  packed-sequence SSM do not have the required cross-allocation contract.
- A production-scale Bridge qualification is a maintainer gate. It is not a
  user-facing support promise.

For the findings, conclusions, and evidence standard, see
[validation-evidence.md](./validation-evidence.md).
