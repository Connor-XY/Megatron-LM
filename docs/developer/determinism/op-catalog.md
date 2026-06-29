# Determinism Op Catalog

The per-operation catalog of determinism in Megatron-Core. For the narrative walk
see [`training-path.md`](./training-path.md); for the control plane and targets see
[`status.md`](./status.md).

## How this catalog is maintained (evidence, not guesses)

Every row is classified by one of three evidence sources. **Do not estimate** perf
numbers — fill the "Perf Δ" column from the PR #5041 nsys leaderboard
(`tests/performance_tests/shell_test_utils/determinism/print_nsys_leaderboard.py`)
on a real recipe, and the determinism verdict from one of:

- **doc** — guaranteed by PyTorch / TE / NCCL / cuBLAS documentation (e.g.
  `index_add`, `index_put(accumulate)` are documented deterministic under
  `torch.use_deterministic_algorithms(True)`; `scatter_add` / `F.embedding`
  backward / FA backward are documented non-deterministic).
- **code** — an explicit deterministic branch exists in the source.
- **test** — confirmed by `BitExactRunner` (toggle `deterministic_mode`; same input
  must give bit-identical out+grad). Rows still needing this are marked **⚠ verify**.

Before declaring the catalog complete, run
`python tools/determinism/audit_sensitive_ops.py megatron`. The AST report is a
review queue for collective reductions, indexed operations, ordering calls, and
determinism controls; every training-path candidate needs either a row here or
an explicit reason it is out of scope. The scanner intentionally does not infer
a verdict and filters known non-tensor collisions from async/control-plane
`put`, `gather`, and module `embedding` calls.

Status legend (matches `training-path.md`): 🟢 deterministic · 🔵 has det branch ·
🟡 conditional (verify) · 🔴 gap (no det path).

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
| Token unpermute (combine) | `transformer/moe/moe_utils.py:526-541` | scatter-accumulate | 🔵 | `index_add_` (CUDA-graph safe) | `scatter_add_` | `are_deterministic_algorithms_enabled()` | doc+code | `index_add_` +85.9 ms NVTX-range time over 3 profiled steps | The redundant second zero-allocation is removed. The cleanup is bit-exact but step-time neutral; the remaining reduction kernel is still a target. |
| Routing map/probs | `moe_utils.py`, `moe/ops/deterministic_routing.py` | collision-free dense write + gather backward | 🔵 | fused Triton probability/map kernel; PyTorch `scatter_` fallback | out-of-place `scatter` ×2 | `are_deterministic_algorithms_enabled()` | code+**test** | isolated forward 4.21–4.63× faster and forward+backward 1.66–1.87× faster than deterministic `scatter_`; production scatter/fused ABBA median -2.0% | Top-k indices are unique within each row, so the fused path has no atomics or reductions. One row program initializes both dense outputs, writes selected entries, and backward gathers only selected gradients. Unsupported devices, dtypes, layouts, or missing Triton retain the prior scatter path. |
| Top-k expert select | `moe_utils.py` | `torch.topk(sorted=...)` | 🔵 | `sorted=True`, including no-grad checkpoint forward | `sorted=is_grad_enabled()` | deterministic algorithms | code+**test** | `cub::DeviceRadixSort` 6.36 vs 1.23 ms over 3 steps (+5.13 ms) | Fixes sigmoid top-k probability drift between original forward and grad-enabled recompute. Investigate a faster stable-select path. |
| Expert-bias score gather | `moe_utils.py` | `torch.gather` + dense backward write | 🔵 | PyTorch deterministic gather backward (`index_put_`) | PyTorch scatter-add backward | deterministic algorithms + expert bias | code+profile | candidate isolated forward+backward 1.48–2.23× faster, but gather-only production ABBA regressed 3.3% | A collision-free Triton candidate was exact and reduced the profiled range from `GatherBackward0` +1.75 ms to 0.29 ms, but two balanced production ABBAs regressed 3.2–3.3%; it was removed. Keep as an open fusion target rather than landing an isolated-only win. |
| Flextron soft-mask routing | `elastification/flextron_elasticity_hooks.py` | weighted sigmoid + top-k + gather + collision-free scatter | 🟡 | inherits global deterministic gather backward; top-k is always sorted | native gather backward when global deterministic algorithms are off | global deterministic algorithms (`deterministic_mode` argument is deprecated/unused) | code | not profiled | This alternate Flextron router is now cataloged, but has no model-level deterministic certificate. Its unique-index scatters are deterministic; gather backward remains the same open generic path as the main router. |
| Group-limited top-k mask | `moe_utils.py:590-645` | `scatter_(1, group_idx, 1)` | 🟢 | unique group indices ⇒ det fwd | — (no branch) | always | code+**test** | small | **Verified** by the DSV3 proxy across EP≤4/TP/FSDP/PP/VPP and by the final EP32 certificates (`518849`/`518850`). |
| Aux-loss routing map | `moe_utils.py:888-901` | `scatter` | 🟢 | unique indices ⇒ det fwd | — (no branch) | always | code+**test** | small | **Verified** via DSV3 proxy and final EP32 certificates with `seq_aux_loss` enabled. |
| Router expert-bias token counts | `transformer/moe/router.py`, `transformer/moe/moe_utils.py:get_updated_expert_bias` | local accumulation + TP×CP×DP SUM | 🟢 | int64 accumulation, native integer all-reduce, and integer-space mean comparison | same | expert-bias routing | code+**test** | not profiled | Counts and the bias-update sign remain exact beyond fp32's `2**24` integer range and are independent of NCCL reduction order. The trace fingerprints the count vector before and after the existing synchronous collective. |
| Capacity-drop mask | `moe_utils.py:950-962` | `scatter` | 🟢 | unique indices | — (no branch) | capacity factor | code+**test** | small | **Verified** bit-exact for `probs` and `position` drop policies, including unpadded and fixed-capacity padded A2A dispatch, across EP≤4/TP/FSDP/PP/VPP. ⚠ still unverified at EP>16. |
| Router map (Sinkhorn) | `transformer/moe/router.py:260` | `scatter` | 🟢 | unique top-k indices | — | Sinkhorn routing | code+**test** | small | **Verified** bit-exact with the DSV3 Sinkhorn preset across EP≤4/TP/FSDP/PP/VPP. ⚠ still unverified at EP>16. |
| Pad routing map | `moe_utils.py:648-680` / `fusions/fused_pad_routing_map.py` | `cumsum` + mask write | 🟢 | ordered cumsum | — | always | doc | small | cumsum is deterministic. |
| Token permute (dispatch) | `moe_utils.py:300-442`; `token_dispatcher.py:645-659`; `moe/ops/deterministic_index_select.py` | `argsort(stable=True)` + row gather | 🔵 | fixed-order Triton backward for dropless top-k ≥4 and hidden ≥2048 | PyTorch `index_select` backward / fused TE | deterministic algorithms + guarded shape | code+**test** | H100: `IndexSelectBackward0` 50.34→14.97 ms (**-70%**) over 3 profiled steps; GB200 isolated kernel: 6.5–55.4% lower latency | Model-level A2A top-k-8/top-k-6 presets are bit-exact. Extend the fast path only with shape-specific evidence; supported fallbacks remain unchanged. |
| EP all-to-all dispatch/combine | `transformer/moe/token_dispatcher.py`, `tensor_parallel/mappings.py` | `all_to_all` | 🟢 | rank-indexed permutation | — | always | code+**test** | — | EP32 DSV3 and Nemotron-style structured traces fingerprint dispatch/combine in forward, recompute, and backward. The collective does not perform floating-point reduction. |
| Grouped GEMM (experts) | `extensions/transformer_engine.py` `TEGroupedLinear` | grouped matmul | 🟢🟡 | TE deterministic kernels | TE fast kernels | `NVTE_ALLOW_NONDETERMINISTIC_ALGO` | doc+**test** | `_GroupedLinearBackward` +19.2 ms (+20%) over 3 profiled steps | Bit-exact in DSV3 + nemotron proxies (`moe_grouped_gemm=True`). wgrad order remains a perf target ("optimized grouped GEMM", "fused GemmAdd"). |
| Router replay | `transformer/moe/router_replay.py` | record/replay top-k | 🟢 | replay recorded indices | — | opt-in | code | — | A determinism *tool*, not default path. |

### Attention

| Op | File:line | Primitive | Det? | Det path | Non-det path | Selected by | Evidence | Perf Δ | Gap / TODO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TE attention backend selection | TE; `extensions/transformer_engine.py:1697-1703`; `TransformerConfig.attention_backend` | FlashAttention / cuDNN FusedAttention / unfused DPA | 🔵🟡 | TE filters out backends/sub-backends that cannot honor deterministic execution | TE selects an eligible backend by internal priority and heuristics | `attention_backend`, `NVTE_{FLASH,FUSED,UNFUSED}_ATTN`, `NVTE_ALLOW_NONDETERMINISTIC_ALGO` | source+profile+**test** | dense dropout 0.1: det Flash vs native fused; dropout 0: fused in both modes; forcing fused shows no gain | Do not globally force a backend: deterministic fused is unavailable for the profiled dropout-0.1 shape, and forcing fused at dropout 0 does not improve the paired ratio. Nemotron's dropout-0 fused recipe is model-certified. |
| FlashAttention backward | TE; `extensions/transformer_engine.py:1697-1703` | atomic dQ/dK/dV accumulation | 🔵🟡 | TE deterministic FA kernels | TE atomic FA | `deterministic_mode` asserts `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` | source+profile+code | fixed-Flash dense profile: +6.034 ms (+37.4%) over the capture | **Deterministic FAG** (Longcat/DSV4): independent accumulation buffers + global deterministic sum remains an external TE/kernel target. |
| Core attention fwd | TE `DotProductAttention` | fused attn | 🟢 | — | — | — | doc | — | Forward deterministic. |
| Multi-Latent Attention (MLA) | `transformer/multi_latent_attention.py` (q/kv low-rank proj + YaRN rope) | low-rank GEMMs + rope | 🟢 | — | — | — | doc+**test** | — | **Verified** bit-exact by `test_deepseek_model.py` (DSV3 MLA + qk_layernorm + YaRN) across EP/TP/FSDP/PP/VPP. |
| Attention dropout | `transformer/dot_product_attention.py:114-220` | RNG mask | 🟡 | fixed RNG state | — | RNG | doc | — | Set dropout=0 for hero runs to eliminate. |

### Normalization

| Op | File:line | Primitive | Det? | Det path | Non-det path | Selected by | Evidence | Perf Δ | Gap / TODO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LayerNorm/RMSNorm fwd | `transformer/torch_norm.py`, TE norm | reduction | 🟢 | — | — | backend | doc+**test** | — | PyTorch fallback verified directly; TE norms are exercised by model proxies. |
| LayerNorm/RMSNorm bwd | TE / `transformer/torch_norm.py` | weight-grad reduction | 🟢 | PyTorch fallback / TE deterministic kernels | TE fast kernels | backend + `NVTE_ALLOW_NONDETERMINISTIC_ALGO` | code+**test** | TBD | **Verified** bit-exact for PyTorch LayerNorm and RMSNorm at hidden 128/2048 on H100 and GB200; TE norms remain covered by the model proxies. |

### Embedding & TP linear

| Op | File:line | Primitive | Det? | Det path | Non-det path | Selected by | Evidence | Perf Δ | Gap / TODO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vocab-parallel embedding | `tensor_parallel/layers.py:299-303` | indexing vs `F.embedding` | 🔵 | direct `weight[idx]` (det bwd) | `F.embedding` (atomic bwd) | `self.deterministic_mode` | code+profile | current Nemotron `IndexBackward0`: 0.728 ms over 2 steps; custom candidate was only 0.61–0.69× as fast | `F.embedding` has a non-deterministic backward. An exact stable-sort/segmented-reduction candidate was rejected on GB200; retain direct indexing and re-profile on dense/Hopper shapes before revisiting. |
| RoPE / position emb | `models/common/embeddings/rotary_pos_embedding.py` | trig | 🟢 | — | — | — | doc | — | — |
| Column/Row parallel linear (GEMM) | `tensor_parallel/layers.py` | cuBLAS GEMM | 🟡 | fixed cuBLAS workspace | — | `CUBLAS_WORKSPACE_CONFIG` | doc | — | — |
| Synchronous TP mapping collectives | `tensor_parallel/mappings.py` | all-reduce, first/last-dim all-gather and reduce-scatter | 🟡 | native NCCL with overlap disabled | same collectives | `tp_comm_overlap=False` in deterministic mode | code+**test** | hash tracing is debug-only | The opt-in trace fingerprints exact inputs/outputs and labels original forward, recompute, and backward without adding a collective or wait. Permutation collectives are deterministic, but native floating-point SUM order remains topology-sensitive across allocations. |
| Core TP linear synchronous/async collectives | `tensor_parallel/layers.py` | forward all-gather; backward async all-gather/all-reduce/reduce-scatter | 🟡 | existing `wait()` completion boundaries; overlap disabled in deterministic mode | communication/wgrad overlap | `sequence_parallel`, `allreduce_dgrad`, `tp_comm_overlap` | code+**test** | hash tracing is debug-only | The trace wraps async work and records output only from the existing wait, without adding or moving a wait; forward, recompute, and backward are labeled explicitly. Native floating-point SUM order remains topology-sensitive. TP userbuffer collectives remain uninstrumented. |
| Pipeline P2P send/receive | `pipeline_parallel/p2p_communication.py` | `isend` / `irecv` / batched P2P | 🟡 | existing wait pins readiness before use | overlap / VPP | schedule | code+**test** | hash tracing is debug-only | Structured trace fingerprints sends and completed receives without adding waits; PP2×VPP2×EP2 passed two-run comparison. |

### Loss

| Op | File:line | Primitive | Det? | Det path | Non-det path | Selected by | Evidence | Perf Δ | Gap / TODO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vocab-parallel cross-entropy | `tensor_parallel/cross_entropy.py`, `tensor_parallel/deterministic_cross_entropy.py` | 3× all-reduce + fp32 + selected class update | 🟡 | fused collision-free Triton selected subtract + smoothing + output scaling, including direct logical-order loads from 2D-strided loss gradients; native NCCL TP reductions | generic `index_put_` selected update + elementwise kernels; native NCCL | deterministic algorithms | code+**test** | fused local backward 3.34–20.41× faster for contiguous gradients and 14.41–16.52× for 2D-strided gradients in isolation; original production ABBA -2.7%; dense strided ABBA -12.9% | The local backward update is closed without atomics and supports label smoothing, the training loss's common strided layout, fallback, and CUDA graphs. `NCCL_ALGO=Ring` still does not pin the physical TP ring across allocations; TP>1 cross-allocation certification and an ordered SUM path remain open. |
| Fused cross-entropy | `cross_entropy_loss_fusion` | fused kernel | 🔴 | — (forbidden) | fused | asserted off | code | — | Open: can it be made deterministic? (perf opportunity). |

### Backward / grad reduction / optimizer

| Op | File:line | Primitive | Det? | Det path | Non-det path | Selected by | Evidence | Perf Δ | Gap / TODO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Final TP/PP gradient synchronization | `distributed/finalize_model_grads.py` | native NCCL all-reduce (SUM / AVG) | 🟡 | no ordered replacement; exact trace fingerprints inputs and outputs | same native collectives | model config (`sequence_parallel`, `qk_layernorm`, shared embeddings, replicated conditional/Flextron parameters) | code+**test** | hash tracing is debug-only | Native floating-point TP/PP reductions remain allocation-topology-sensitive. Ultra TP2×PP3 exercises this surface, but its same-allocation certificate cannot prove cross-topology equivalence. Add an ordered path only after traces show this is the first divergence; native AVG also needs SUM-then-divide semantics. |
| Grad bucket all-reduce / reduce-scatter | `distributed/param_and_grad_buffer.py` | floating-point NCCL collective | 🔵 | ordered fp32 reduce-scatter for deterministic distributed optimizer | native NCCL | deterministic args | code+**test** | hash tracing is debug-only | Native floating-point collectives are allocation-topology-sensitive even with Ring. Non-distributed-optimizer all-reduce remains a gap. |
| fp32-accum reduce-scatter | `distributed/reduce_scatter_with_fp32_accumulation.py` | all-to-all + ordered `sum(fp32)` | 🟢 | flat rank order or fixed two-level logical-rank tree | native RS | forced in deterministic distributed optimizer; hierarchy is explicit | code+**test** | production hierarchy 1.324–1.346 ms vs flat ordered 3.110–3.118 ms and native 1.364–1.374 ms on 32×GB200 cross-domain | One optimizer instance and no collective AVG are enforced. With logical group size 4, DSV3 jobs `520170`/`520203` and Nemotron jobs `520171`/`520204` match exactly across allocations. Single-rank expert-DP reductions remain flat because no inter-rank reduction exists. |
| Megatron-FSDP grad all-reduce / reduce-scatter | `distributed/fsdp/src/megatron_fsdp/param_and_grad_buffer.py` | native NCCL SUM / AVG / premultiplied SUM | 🔴→forbidden | none | synchronous helpers plus overlapped inner/outer-DP bucket pipelines | `use_megatron_fsdp` | source+same-topology **test** | not profiled | Deterministic CLI mode now rejects this backend because it does not consume `reduce_scatter_with_fp32_accumulation`. FSDP8 proxy cells repeat inside one process-group topology only. Add an ordered reduction and trace its existing stream/event completion boundaries before claiming cross-allocation support. The Ultra production launcher uses standard distributed optimizer, not this path. |
| Distributed-optimizer param all-gather | `distributed/param_and_grad_buffer.py` | NCCL all-gather | 🟢 | rank-indexed byte copies | — | always | code+**test** | hash tracing is debug-only | No floating-point reduction; structured trace fingerprints sync and overlapped gathers without adding a collective or wait. |
| Distributed optimizer param order | `optimizer/distrib_optimizer.py:1094` | shard mapping | 🟢 | "preserving deterministic ordering across ranks" | — | always | code | — | — |
| Grad clip global norm | `optimizer/clip_grads.py`, `distributed/deterministic_collectives.py` | SUM reduction | 🔵 | rank-ordered all-gather + local sum | native all-reduce | deterministic algorithms | code+**test** | TBD | Closed a one-ulp scalar divergence that otherwise changed every optimizer parameter. Intended only for small statistics. |
| Reported loss / MoE metrics | `training.py`, `training/utils/common_utils.py`, `transformer/moe/moe_logging.py` | SUM / AVG reductions | 🔵 | rank-ordered all-gather + local sum/divide | native all-reduce | deterministic algorithms | code+**test** | TBD | Keeps reported loss and expert-bias/load-balancing metrics exact across allocations. |

### SSM / Mamba (hybrid models)

| Op | File:line | Primitive | Det? | Det path | Non-det path | Selected by | Evidence | Perf Δ | Gap / TODO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Triton autotune | `megatron/determinism_env.py`, `ssm/ops/determinism.py`, external `mamba_ssm.utils.determinism` | autotune config select | 🔵 | one fixed config (`TRITON_CACHE_AUTOTUNING=0`) | timing-selected config | early deterministic env | code+**test** | paired Nemotron EP32 proxy: fixed 74.5 ms vs autotuned 75.0 ms (no slowdown) | Cold cached-autotune selected different reduction tilings across launches; env setup must precede kernel-module import. |
| Tiled reduction workspace | `ssm/ops/determinism.py:106-123` | `zeros`+`sum` vs `empty` | 🔵 | ordered tile sum | unordered | `use_deterministic_mode()` | code | paired Nemotron EP32 proxy: workspace+autotune 75.0 ms vs normal 75.5 ms (no slowdown) | Extra memory for tiled reduction; no measurable step-time cost in the current proxy. |
| Gated-delta-rule kernel | `ssm/gated_delta_net.py:213-216` | torch vs FLA fused | 🔵 | `torch_chunk_gated_delta_rule` | FLA fused | `deterministic_mode` | code | TBD | — |
| Causal conv1d | `ssm/gated_delta_net.py:430-446` | `F.conv1d` vs FLA | 🔵 | `F.conv1d` (+transpose) | `causal_conv1d` | `deterministic_mode` | code | TBD | — |
| Mamba-2 combined training | `ssm/mamba_mixer.py`, external `mamba_ssm` | fused causal conv + selective scan | 🟢🟡 | fused path with deterministic workspaces and fixed Triton config | same path with timing autotune / atomic reductions | early env + torch deterministic algorithms | code+**test** | paired EP32 proxy: no measurable deterministic Mamba overhead (74.5–75.0 ms vs normal 75.5 ms) | Fused EP32 jobs `518541`/`518542` matched 9,664/9,664 events within and across allocations without launcher-supplied Mamba/Triton env. Disabling only the fused path did not fix cold-cache autotune drift; pinning config did. |
| Packed sequence (`thd`) | `ssm/gated_delta_net.py:314` | — | 🔴 | — | thd | asserted off | code | — | **Gap**: no deterministic packed-seq SSM path. |

### Inference / RL (not training-loop, listed for completeness)

| Op | File:line | Det? | Notes |
| --- | --- | --- | --- |
| DP inference coordinator scheduling | `inference/.../dynamic_engine.py:607`, `data_parallel_inference_coordinator.py:181` | 🔵 | sorted rank identities vs completion order |
| RL rollout ordering | `rl/rl_utils.py:678` | 🔵 | sort by `problem_id` vs completion order |
| Inference token sampling | `inference/sampling/torch_sampling.py:61-69` | 🟡 | `cumsum`/`scatter` in top-p; RNG-driven |
| DSA sparse-attention masks | `transformer/experimental_attention_variant/dsa.py:214/385/950` | 🟢 | Unique top-k `scatter_` masks are bit-exact in indexer-loss forward, recomputed manual backward, and unfused sparse-attention backward (`test_dsa_paths.py`). |

---

## Hotspots (perf priority — measured)

### Measurement provenance

These results are cluster measurements, not estimates. The original dense/MoE
leaderboards used source commit `a6e02bdc5533` plus the indicated local profiling
script. Cluster-local artifacts are retained under:

- Draco: `/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_mcore/users/yxu1/Megatron-LM-det-ws2`
- AWS-DFW: `/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/yxu1`

| experiment | Slurm job / node | durable console evidence | outcome |
| --- | --- | --- | --- |
| dense GPT TP2, det vs nondet | `10427264` / `batch-block1-2002` | `det_perf_console.log`, SHA256 `59ca433e913c…` | 106.0 / 95.8 ms, 1.11× |
| DSV3-style TP2×EP4 baseline | `10427330` / `batch-block1-3404` | `det_perf_moe_console.log`, SHA256 `8aff4bf4b0b5…` | 228.4 / 182.2 ms, raw ratio 1.2536×; job exit 1 is the perf-gate failure |
| remove redundant unpermute zero | `10428334` / `batch-block4-0103` | `det_perf_moe_quickwin_console.log`, SHA256 `efc42f798d58…` | 235.1 / 181.3 ms, 1.30×; no step win |
| zero cleanup + routing scatter candidate | `10428497` / `batch-block3-0152` | `det_perf_moe_routingfix_console.log`, SHA256 `9c0663c4ad55…` | 232.2 / 186.4 ms, 1.25×; no step win |
| top-k-8/hidden-2048 parent→optimized A/B | `10429428` / `batch-block1-3968` | `det-profile-topk8-results/{baseline,candidate}/console.log`, SHA256 `bf5e8d8cc5fe…` / `794c70496a4f…` | parent 242.8 / 189.1 ms; optimized 241.4 / 192.3 ms |
| top-k-8/hidden-2048 optimized→parent B/A | `10429492` / `batch-block1-2042` | `det-profile-topk8-results-repeat2/{candidate,baseline}/console.log`, SHA256 `e4feffa550d5…` / `80af2d8832d2…` | optimized 237.4 / 195.0 ms; parent 238.3 / 194.2 ms |
| production helper isolated ABBA (GB200) | `516088` / `nvl72d040-T11` | `submit_logs/perf_det_idx_select_516088.out`, SHA256 `0cddc145a70e…` | bit-exact; 6.5–55.4% latency reduction across top-k 4/8 and hidden 4096/7168 |
| fp32 reduce-scatter, both A/B orders (GB200) | `519609` / `519610`, 32 GPUs | `bench-det-rs-519609.out`, SHA256 `fd9bd65d42f2…`; `bench-det-rs-ordered-519610.out`, SHA256 `0292b6e73b67…` | ordered 0.505–0.520 ms vs native 0.777–0.820 ms; ordered/native 0.63–0.65× |
| fp32 reduce-scatter, cross-domain (GB200) | `519905`, 32 GPUs across `nvl72d038`/`nvl72d049` | `bench-det-rs-519905.out`, SHA256 `3cc9304aaad3…` | ordered 3.081 ms vs native 1.372 ms; ordered/native 2.24×; max absolute numerical delta 9.54e-6 |
| fp32 reduce-scatter, cross-domain repeat (GB200) | `519924`, 32 GPUs across `nvl72d049`/`nvl72d070` | `bench-det-rs-519924.out`, SHA256 `d301f26b3768…` | ordered 2.389 ms vs native 1.230 ms; ordered/native 1.94×; max delta 9.54e-6 (3.13e-7 of max magnitude) |
| Nemotron Mamba attribution (GB200) | `519888`, 32 GPUs across `nvl72d039`/`nvl72d069` | `perf-nemotron-decompose-519888.out`, SHA256 `fa1e0917d177…` | normal 75.5 ms; deterministic workspace+autotune 75.0 ms; deterministic workspace+fixed config 74.5 ms |
| Nemotron deterministic attribution (GB200) | `519928`, 32 GPUs across `nvl72d038`/`nvl72d039` | `perf-nemotron-dp-attribution-519928.out`, SHA256 `221507ab2f1a…` | native 72.5 ms; ordered-DP-only 75.1 ms (+3.6%); full deterministic 79.9 ms (+10.2%) |
| hierarchical ordered RS, forward order (GB200) | `519943`, 32 GPUs across `nvl72d021`/`nvl72d069` | `bench-det-rs-hier-519943.out`, SHA256 `f0c4184216b6…` | hierarchical 1.153 ms; current ordered 2.314 ms; native 1.183 ms; same 9.54e-6 max delta to native |
| hierarchical ordered RS, reverse order (GB200) | `519967`, 32 GPUs across `nvl72d036`/`nvl72d070` | `bench-det-rs-hier-rev-519967.out`, SHA256 `c8025fb9aa0f…` | hierarchical 1.144 ms; current ordered 2.329 ms; native 1.191 ms; same 9.54e-6 max delta to native |
| hierarchical RS exact-hash cross-domain (GB200) | `520019`, 32 GPUs across `nvl72d039`/`nvl72d070` | `bench-det-rs-hier-rev-520019.out`, SHA256 `b5ce2fa2422f…` | hierarchical 1.117 ms; current ordered 2.249 ms; native 1.140 ms |
| hierarchical RS exact-hash single-domain (GB200) | `520022`, 32 GPUs in `nvl72d039` | `bench-det-rs-hier-hash-520022.out`, SHA256 `de84be49e350…` | hierarchical 0.605 ms; current ordered 0.452 ms; native 0.801 ms; ordered and hierarchical hashes match `520019` on all ranks, native differs on 32/32 |
| production hierarchical RS, both A/B orders (GB200) | `520160`, 32 GPUs across `nvl72d034`/`nvl72d039` | `bench-det-rs-prod-520160.out`, SHA256 `4ae4f06124c4…` | hierarchy 1.324–1.346 ms; flat ordered 3.110–3.118 ms; native 1.364–1.374 ms. Hierarchy and flat hashes each match independent job `520019` on 32/32 ranks; native differs on 32/32. |
| production DSV3 hierarchy cross-allocation certificate | `520170` / `520203`, 32 GPUs each | `dsv3-hier-cross-allocation-520170-520203.json`, SHA256 `b782753fe098…` | 6,080/6,080 events exact; zero divergences; 64 multi-rank hierarchical DP reductions and zero missing per trace tree |
| production Nemotron hierarchy cross-allocation certificate | `520171` / `520204`, 32 GPUs each | `nemotron-hier-cross-allocation-520171-520204.json`, SHA256 `c93842d47013…` | 9,664/9,664 events exact; zero divergences; 128 multi-rank hierarchical DP reductions and zero missing per trace tree |
| production Nemotron hierarchy attribution (GB200) | `520235`, 32 GPUs across four NVL72 domains | `perf-nemotron-hierarchy-520235.out`, SHA256 `1e83b9582b2d…` | native 55.6 ms; flat-DP-only 57.8 ms (+4.0%); hierarchical-DP-only 55.4 ms (-0.4%); full deterministic with hierarchy 66.4 ms (+19.4%). Steady medians use iterations 6–14. |
| full-deterministic flat/hierarchy A/B (GB200) | `520311`, same 32-GPU placement as `520235` | `perf-nemotron-full-pair-520311.out`, SHA256 `47fa84b65289…` | native 57.2 ms; full deterministic flat 66.5 ms (+16.3%); full deterministic hierarchy 66.7 ms (+16.6%, +0.3% vs flat). The exposed DP win is hidden by overlap in the full stack. |
| rank-scoped production Nemotron profile (GB200) | `520457`, 32 GPUs | `profile-nemotron-det-gap-520457/leaderboard.txt`, SHA256 `cc39eca7ea76…` | deterministic MLP +24.86 ms; `index_put_` +11.16 ms; `fill_` +20.33 ms; grouped-linear backward +0.99 ms over the captured steps. Only global rank 0 ran under nsys. |
| deterministic routing construction microbenchmark (GB200) | `520492`, 1 rank on a 4-GPU allocation | `bench-det-route-map-520492.out`, SHA256 `87b3f623a4ba…` | bf16/fp32 × 16/512/4096 tokens, 512 experts, top-k 22: byte-exact outputs and gradients with one repeated hash per case; `scatter_` 1.76–1.80× faster forward and 1.51–1.53× faster forward+backward. |
| production routing `index_put_`/`scatter_` ABBA (GB200) | `520526`, fixed 32-GPU allocation | `perf-nemotron-route-scatter-520526.out`, SHA256 `eb1f1923976b…` | median-of-leg medians 65.7→64.5 ms (-1.8%); forward pair -4.8%, reverse pair +1.2%; loss, sequence-aux-loss, and grad-norm sequences identical. |
| post-change rank-scoped Nemotron profile (GB200) | `520565`, 32 GPUs | `profile-nemotron-det-gap-520565/leaderboard.txt`, SHA256 `b434ff1538dc…` | `scatter_` +5.79 ms and residual `index_put_` +8.97 ms. SQLite nesting attributes the residual to routing probability scatter, vocab cross-entropy, and gather backward rather than one source. |
| post-change DSV3 cross-version certificate | `520203` / `520570`, independent 32-GPU allocations | `dsv3-routing-scatter-cross-version-520203-520570.json`, SHA256 `8148959a084b…` | 6,080/6,080 events exact; zero divergences; 64 recomputes, 1,920 collective events, zero pending collectives, and zero missing hierarchical DP reductions per tree. |
| post-change Nemotron cross-version certificate | `520204` / `520561`, independent 32-GPU allocations | `nemotron-routing-scatter-cross-version-520204-520561.json`, SHA256 `dc492dfc3bf9…` | 9,664/9,664 events exact; zero divergences; 192 recomputes, 2,304 collective events, zero pending collectives, and zero missing hierarchical DP reductions per tree. |
| post-change native/deterministic ABBA (GB200) | `520625`, fixed 32-GPU allocation | `perf-nemotron-route-gap-520625.out`, SHA256 `99a9aa6104f…` | median-of-leg medians: native 56.4 ms, deterministic 59.4 ms (+5.3%). Forward pair +8.8%; reverse pair +1.6%. This reaches the aggressive target on one allocation but remains order/allocation-sensitive. |
| fused routing focused suite (GB200) | `520748`; final-source rerun `520897`, 4 GPUs | `submit_logs/test_det_routing_fused_520897.out`, SHA256 `cb73fc1977d0…` | 39 passed on every rank: bf16/fp32 DSV3 and Nemotron shapes match scatter exactly in outputs, boolean maps, and gradients; CPU fallback and CUDA-graph forward/backward replay pass. |
| fused routing construction ABBA (GB200) | `520763`, 1 rank on a 4-GPU allocation | `submit_logs/bench_det_routing_fused_520763.out`, SHA256 `20aea5beed85…` | bf16/fp32 × 16/512/4096 tokens for DSV3 (256 experts, top-k 8) and Nemotron (512 experts, top-k 22): all 12 cells repeat exactly; fused/scatter speedup is 4.21–4.63× forward and 1.66–1.87× forward+backward. |
| fused DSV3 cross-version certificate | `520570` / `520784`, independent 32-GPU allocations | `dsv3-fused-cross-version-520570-520784.json`, SHA256 `0e0cf6575a32…` | 6,080/6,080 events exact; zero divergences; 64 recomputes, 1,920 collective events, zero pending collectives, and zero missing hierarchical DP reductions per tree. |
| fused Nemotron cross-version certificate | `520561` / `520785`, independent 32-GPU allocations | `nemotron-fused-cross-version-520561-520785.json`, SHA256 `9cc61f201005…` | 9,664/9,664 events exact; zero divergences; 192 recomputes, 2,304 collective events, zero pending collectives, and zero missing hierarchical DP reductions per tree. |
| production routing `scatter_`/fused ABBA (GB200) | `520786`, fixed 32-GPU allocation | `perf-nemotron-fused-scatter-520786.out`, SHA256 `2980c4e33c93…` | median-of-leg medians 67.1→65.75 ms (-2.0%); forward pair -4.8%, reverse pair +1.1%; all 14-step loss, sequence-aux-loss, and grad-norm sequences are identical. |
| post-fusion native/deterministic ABBA (GB200) | `520827`, fixed 32-GPU allocation | `perf-nemotron-fused-gap-520827.out`, SHA256 `c82242660b4d…` | median-of-leg medians: native 56.1 ms, deterministic 64.65 ms (+15.2%); forward pair +14.5%, reverse pair +16.0%. This confirms that closing routing does not close the remaining TE/attention/loss/gather gap on every allocation. |
| post-fusion rank-scoped Nemotron profile (GB200) | `520828`, 32 GPUs | `profile-nemotron-fused-gap-520828/leaderboard.txt`, SHA256 `28f7a16c0038…`; `index-put-attribution.json`, SHA256 `bbad652beebd…` | `_DeterministicRoutingBackward` totals 0.376 ms; aggregate `index_put_` is 3.025 ms across eight ranges, all attributed to vocab cross-entropy or gather backward and none to routing. Latest deltas: MLP forward +3.38 ms, self-attention +1.32 ms, Mamba backward +2.79 ms, gather backward +1.75 ms, and vocab-CE backward +1.50 ms over the captured steps. |
| selected-gather focused suite (GB200) | `520936`, 4 GPUs | `submit_logs/test_det_routing_fused_520936.out`, SHA256 `dbd5a5c38ca7…` | 50 passed on every rank: exact DSV3/Nemotron selected-score outputs and gradients, CPU fallback, invalid-shape coverage, routing integration, and CUDA-graph forward/backward replay. |
| selected-gather ABBA microbenchmark (GB200) | `520946`, 1 rank on a 4-GPU allocation | `submit_logs/bench_det_selected_gather_520946.out`, SHA256 `c04e4f557e30…` | bf16/fp32 × 16/512/4096 tokens for DSV3 (256/top-8) and Nemotron (512/top-22): all 12 cells repeat exactly. Fused forward+backward is 1.48–2.23× faster; forward-only is 0.60–0.65× as fast, so the fast path is gated on grad-enabled inputs. |
| selected-gather DSV3 cross-version certificate | `520784` / `520961`, independent 32-GPU allocations | `cross-version-520784-520961.json`, SHA256 `53bcccecda5e…` | 6,080/6,080 events exact; zero divergences; 64 recomputes, 1,920 collective events, zero pending collectives, and zero missing hierarchical DP reductions. |
| selected-gather Nemotron cross-version certificate | `520785` / `520962`, independent 32-GPU allocations | `cross-version-520785-520962.json`, SHA256 `640cdffe5a33…` | 9,664/9,664 events exact; zero divergences; 192 recomputes, 2,304 collective events, zero pending collectives, and zero missing hierarchical DP reductions. |
| post-gather native/deterministic ABBA (GB200) | `520963`, fixed 32-GPU allocation | `perf-nemotron-fused-gap-520963.out`, SHA256 `0b6f8f6dfe69…` | median-of-leg medians: native 54.85 ms, deterministic 60.3 ms (+9.9%); forward pair +7.8%, reverse pair +12.2%. Both deterministic legs have identical 14-step loss, sequence-aux-loss, and grad-norm sequences. Allocation sensitivity remains. |
| post-gather rank-scoped Nemotron profile (GB200) | `520964`, 32 GPUs | `profile-nemotron-fused-gap-520964/leaderboard.txt`, SHA256 `824e5e986958…`; `index-put-attribution.json`, SHA256 `95082aec60c7…` | `_DeterministicSelectedGatherBackward` totals 0.290 ms and `GatherBackward0` is absent. The six residual `index_put_` ranges total 1.551 ms and are all vocab CE. Latest deltas: MLP forward +2.05 ms, Mamba backward +1.84 ms, vocab-CE backward +1.46 ms, and core attention +0.45 ms. |
| gather-only production ABBA (GB200) | `521026`, fixed 32-GPU allocation | `perf-nemotron-selected-gather-521026.out`, SHA256 `8d604b452bb8…` | baseline 64.3 ms vs candidate 66.45 ms (+3.3% median-of-leg medians); forward pair +5.0%, reverse pair +1.7%; all loss, sequence-aux-loss, and grad-norm sequences identical. Candidate rejected. |
| CE selected-update microbenchmark (GB200) | `521031`, 1 rank on a 4-GPU allocation | `submit_logs/bench_det_ce_write_521031.out`, SHA256 `374e486b39d5…` | 16/512/4096 rows × vocab shards 128/8192: all six backward cells byte-exact and 13.64–15.54× faster. Replacing the four forward mask writes with `where` was only 0.66–0.68× as fast, so forward remains unchanged. |
| CE selected-update focused suite (GB200) | `521106`, 1 GPU | `submit_logs/test_det_ce_focused_521106.out`, SHA256 `23e3a460e124…` | 11 passed: bf16/fp32, label smoothing on/off, CPU fallback, invalid shapes, and CUDA-graph replay. |
| combined gather+CE production ABBA (GB200) | `521127`, fixed 32-GPU single-domain allocation | `perf-nemotron-selected-gather-521127.out`, SHA256 `f3cc5d330ccd…` | baseline 61.8 ms vs combined candidate 63.8 ms (+3.2%); forward pair +7.4%, reverse pair -0.6%; all training sequences identical. This second negative gate triggered removal of gather. |
| combined gather+CE rank profile (GB200) | `521128`, 32 GPUs | `profile-nemotron-fused-gap-521128/leaderboard.txt`, SHA256 `a1d22973fa0d…`; `index-put-attribution.json`, SHA256 `81678748e602…` | CE backward indexed writes are absent; only four forward mask writes remain (0.178 ms total). Broader TE/Mamba deltas were strongly allocation-sensitive; use the paired ABBA for the retention decision. |
| fused CE backward microbenchmark (GB200) | `521310`, 1 rank on a 4-GPU allocation | `submit_logs/bench_fused_det_ce_521310.out`, SHA256 `ed992271e9ca…` | 16/512/4,096 rows × vocab shards 128/8,192 × label smoothing on/off: all 12 cells are byte-exact. Fusing selected subtract, optional smoothing, and output-gradient scaling is 3.34–20.41× faster than the original update+scale and 1.25–2.92× faster than the selected-only candidate. |
| fused CE focused suite (GB200) | `521463`, 1 GPU | `submit_logs/test_det_ce_focused_521463.out`, SHA256 `08bc00be3091…` | 27 passed: direct bf16/fp32 kernels, label smoothing on/off, 16/512 rows and 128/8,192 columns, CPU fallback, invalid shapes, CUDA-graph replay, and full CE integration for batch 1/3 and vocab 128/257. |
| fused CE DSV3 cross-version certificate | `520784` / `521405`, independent 32-GPU allocations | `cross-version-520784-521405.json`, SHA256 `9bf601725f2a…`; certification SHA256 `5f363c150cdf…` | 6,080/6,080 events exact; zero divergences; 64 recomputes, 1,920 collective events, zero pending collectives, and zero missing hierarchical DP reductions. |
| fused CE Nemotron cross-version certificate | `520785` / `521406`, independent 32-GPU allocations | `cross-version-520785-521406.json`, SHA256 `c52c72350ef7…`; certification SHA256 `7d52f590e209…` | 9,664/9,664 events exact; zero divergences; 192 recomputes, 2,304 collective events, zero pending collectives, and zero missing hierarchical DP reductions. |
| fused CE rank-scoped Nemotron profile (GB200) | `521407`, 32 GPUs | `profile-nemotron-fused-gap-521407/leaderboard.txt`, SHA256 `21f11d76841e…`; `index-put-attribution.json`, SHA256 `b913575972b5…` | `_VocabParallelCrossEntropyBackward` is 0.749 vs 1.257 ms (-0.508 ms, -40.4%) over two captured steps. No CE-backward indexed writes remain; the six residual writes are four small CE-forward mask writes and two gather-backward writes. |
| fused CE extended production ABBA (GB200) | `521404`, fixed 32-GPU allocation | `perf-nemotron-ce-extended-521404.out`, SHA256 `9f7d57ba2ccb…` | Iterations 11–50: baseline/candidate medians are 67.5/66.9 ms forward-order (-0.9%) and 65.45/62.45 ms reverse-order (-4.6%); median-of-leg medians 66.475→64.675 ms (-2.7%). All 50 loss, sequence-aux-loss, and grad-norm values match across all four legs. |
| deterministic embedding-backward candidate (GB200) | `521744`, 1 rank on a 4-GPU allocation | `submit_logs/bench_det_embedding_521744.out`, SHA256 `c15faf510804…` | 128/8,192 vocab rows, 16/512/2,048 tokens, hidden 1,024/2,048/4,096, bf16/fp32, and uniform/hot duplicate distributions: all 12 cells are byte-exact. Stable-sort + fixed-order segmented reduction is only 0.61–0.69× as fast as the current deterministic direct-index backward, so the candidate is rejected. |
| refreshed dense GPT baseline (GB200) | HSG `3614788`, 4 GPUs | `leaderboard.txt`, SHA256 `219315ec31a4…`; deterministic SQLite SHA256 `f86562fce98d…`; console SHA256 `0db28bc895ba…` | Step 7 is 119.0/104.7 ms deterministic/native (1.14×). Over the capture, vocab-CE backward is 17.808/5.373 ms (+12.435 ms), self-attention +21.351 ms, and core attention +13.898 ms. |
| refreshed dense GPT baseline (GB300) | AWS-CMH `699350`, 4 GPUs | `leaderboard.txt`, SHA256 `4c6ac53f4adbc…`; deterministic SQLite SHA256 `0b3d71954b48…`; console SHA256 `bb5d81955c33…` | Step 7 is 130.2/107.0 ms deterministic/native (1.22×). Vocab-CE backward is 18.713/5.183 ms (+13.531 ms); 16 CE-backward `index_put_` ranges total 9.431 ms. Self-attention is +25.194 ms and core attention +17.189 ms. |
| strided fused-CE microbenchmark (GB200) | AWS-DFW `522493`, 1 rank on a 4-GPU allocation | `submit_logs/bench_strided_det_ce_522493.out`, SHA256 `9ad4293378f0…` | 512 rows × 128/8,192 columns plus 4,096 rows × 128 columns, crossed with bf16/fp32 and smoothing on/off: all 12 cells are byte-exact. Direct 2D-strided loads are 14.41–16.52× faster than the deterministic indexed-write fallback. |
| strided fused-CE focused suite (GB200) | AWS-DFW `522494`, 1 GPU | `submit_logs/test_det_ce_focused_522494.out`, SHA256 `9322de841b1a…` | 35 passed: contiguous/strided bf16/fp32 kernels, smoothing, fallback, invalid shapes, strided CUDA-graph replay, and full CE integration with explicit strided batch-3 loss gradients. |
| contiguous-copy strided-CE experiment (GB300) | AWS-CMH `699565`, fixed 4-GPU allocation | `perf-dense-ce-warm-699565.out`, SHA256 `75eb627ee192…` | After a 200-step thermal warmup, baseline/candidate median-of-leg medians are 56.35/53.10 ms (-5.8%), but the order pairs disagree (+8.4%/-17.1%). All 50 loss and grad-norm values match. This noisy, allocation-adding design was superseded by direct stride-aware loads. |
| stride-aware fused-CE rank profile (GB300) | AWS-CMH `699916`, 4 GPUs | console SHA256 `62fb7fc85fb7…`; deterministic SQLite SHA256 `d313825bf678…`; `index-put-attribution.json`, SHA256 `a9843d931a0e…` | Vocab-CE backward is 7.001/6.014 ms deterministic/native (+0.987 ms) over the capture; deterministic time is 62.6% below the refreshed baseline's 18.713 ms. No CE-backward indexed writes remain. Step 7 is 123.2/107.1 ms (1.15×); self-attention/core-attention remain +22.743/+15.620 ms. |
| Nsight kernel-identity attribution (GB300) | AWS-CMH retained profile `699916`, 4 GPUs; HSG tests `3632378` | `determinism-nsys-kernels-699916/deterministic.json` / `nondeterministic.json`, SHA256 `9187b0e27912…` / `66ed6e4a937f…`; HSG log SHA256 `2b5a417d25c4…`; tool SHA256 `99b0a70cb55e…` | All eight final-source tests pass. The tool maps 32 nested CE-backward NVTX ranges to 160/128 unique deterministic/native launches without double-counting nested targets. Deterministic mode selects `_cross_entropy_backward` and no index-elementwise kernels; native mode selects two distinct index-elementwise kernel identities and not `_cross_entropy_backward`. Summed kernel activity is 0.209/0.372 ms; it is attribution evidence, not wall-clock time. |
| stride-aware fused-CE production ABBA (GB300) | AWS-CMH `699912`, fixed 4-GPU allocation | `perf-dense-ce-warm-699912.out`, SHA256 `0dd8057cfe1a…` | After a 200-step thermal warmup, iterations 11–50 improve 63.25→52.05 ms in forward order (-17.7%) and 66.8→61.2 ms in reverse order (-8.4%); median-of-leg medians improve 65.025→56.625 ms (-12.9%). All 50 loss and grad-norm values match across all four legs. |
| final-source DSV3 EP32 certificate | AWS-DFW `522520`, 32 GPUs | `det-dsv3-fused-ep32-522520/certification.json`, SHA256 `62b506f4c16a…` | Two launches match 6,080/6,080 events with zero divergences, 64 exact recomputes, 1,920 collective events, zero pending collectives, and zero missing hierarchical DP reductions. |
| final-source Nemotron EP32 certificate | AWS-DFW `522521`, 32 GPUs | `det-nemotron-fused-ep32-522521/certification.json`, SHA256 `b1f8b2688382…` | Two launches match 9,664/9,664 events with zero divergences, 192 exact recomputes, 2,304 collective events, zero pending collectives, and zero missing hierarchical DP reductions. |
| checked-in weekly DSV3 EP32 certificate | AWS-DFW `522695`, 32 GPUs across four NVL72 domains | `det-weekly-cert-dsv3-522695/determinism-certification/dsv3/certification.json`, SHA256 `9e887e0d821a…` | The weekly runner matches 6,080/6,080 events with zero divergences, 64 exact recomputes, 1,920 collective events, zero pending collectives, and zero missing hierarchical DP reductions. |
| checked-in weekly Nemotron EP32 certificate | AWS-DFW `522696`, 32 GPUs in one NVL72 domain | `det-weekly-cert-nemotron-522696/determinism-certification/nemotron/certification.json`, SHA256 `fbb673910cfe…` | The weekly runner matches 9,664/9,664 events with zero divergences, 192 exact recomputes, 2,304 collective events, zero pending collectives, and zero missing hierarchical DP reductions. |
| exact router expert-bias count/update suite | Draco `10475888`, 8 H100 GPUs | console SHA256 `3d32a7971763…`; manifest SHA256 `7762d0dc0f61…` | All eight ranks pass the large-count trace, finalize update, and EP8 production-router cases. The near-tie global counts differ by one above fp32's exact range but map to the same fp32 value; the int64 SUM and integer-space comparison still produce the correct opposite bias-update signs. |
| expert-bias-gated DSV3 + Nemotron EP32 certificates | AWS-DFW `539206`, same 32-GPU allocation | DSV3/Nemotron certification SHA256 `1221e09002b9…` / `98a78a304897…`; manifest SHA256 `03ba6c436edb…` | The checked-in runner requires `moe.router_expert_bias.` and matches 6,720/6,720 DSV3 events plus 10,304/10,304 Nemotron events. Each side has 128 expert-bias collective events, zero divergences or pending collectives, exact recomputes, and no DP reductions missing ordered or hierarchical fp32 accumulation. |
| hardened trace certifier | HSG `3617620`; AWS-DFW weekly jobs `522695` / `522696` | HSG log SHA256 `cfe1b0944ee6…`; `certification-hardened.json`, SHA256 `4b21c868f245…` / `66fe9463c6a7…` | All 16 final-source certifier/comparator tests pass. The stricter schema, file-identity, sequence, boundary, iteration-error, and ordered collective-balance checks preserve both production verdicts: DSV3 6,080/6,080 and Nemotron 9,664/9,664 events exact across 64 files each, with zero unmatched collectives and zero iteration errors. |
| full-width A2A model matrix | Draco `10438654`, 8 H100 GPUs | `model-a2a-10438654/verification.json`, SHA256 `f808fe963e1c…`; pytest log SHA256 `24d2beab4b1e…` | All eight ranks pass all 12 DeepSeek top-k-8 and Nemotron top-k-6 cells (96/96 per-cell pass reports), including TP2×EP4, FSDP8×EP4, and FSDP8. GPU step `10438654.0` is `COMPLETED 0:0`; the batch wrapper alone is failed because its post-check regex omitted pytest's `33 deselected` field. |
| strict serialized-training-log gate | HSG `3616466`; AWS-DFW `522790`; weekly logs `522695` / `522696` | `bridge-ultra-current/log-tests-3616466.out`, SHA256 `aa6f7fe42ea1…`; weekly reports SHA256 `6bda80c6d699…` / `7e1cbe6f8142…` | The final 10/10 focused tests pass, including a colon-bearing interleaved Slurm rank suffix with its own pipe-delimited memory metric. Both two-run weekly logs contain two exact records for each of two iterations; all 20 nonvolatile serialized metric values per model match. This is an external-loop integration gate, not a tensor-level certificate. |
| local optimizer-state trace | HSG `3616865` / `3616898`, 4 GPUs | `optimizer-state-trace-tests-3616865.out` / `optimizer-state-trace-tests-3616898.out`, SHA256 `305373558100…` / `d9ff9c57175a…` | The full 18-test trace file passes on every rank, and final chained-optimizer coverage passes 2/2 focused tests per rank. The opt-in hashes local main parameters plus direct tensor/scalar state before and after the step using stable ordinals; it adds no communication and is disabled by default. |
| synchronous TP collective trace | HSG `3617018` / final-source `3617133`, 4 GPUs; mapping regression `3617143`, 8 GPUs | `tp-trace-tests-3617018.out` / `full-trace-tests-3617133.out` / `tp-mappings-tests-3617143.out`, SHA256 `7a02f8827841…` / `fcd4b95dcd5b…` / `68688420008b…` | The focused forward/recompute/backward test passes on all four ranks, then the full 20-test trace file passes on every rank. The seven legacy mapping tests pass on all eight ranks across two nodes. It fingerprints all-reduce plus first/last-dimension all-gather and reduce-scatter, and covers size-one zero-stride autograd gradients. |
| core TP linear collective trace | HSG focused `3617204` / full `3617239`, 4 GPUs; layer regression `3617240`, 8 GPUs | `tp-linear-trace-tests-3617204.out` / `full-trace-tests-3617239.out` / `tp-layers-tests-3617240.out`, SHA256 `a57bb228c848…` / `834bb84b0e2b…` / `f148bed8acc0…` | The focused test passes on all four ranks, the expanded 21-test trace file passes on every rank, and all three legacy layer cases pass on all eight ranks across two nodes. Async outputs are hashed only after the original work-handle wait; checkpoint recompute is labeled explicitly. |
| TE/output-discard recompute trace | HSG `3632592`, 4 GPUs | `recompute-trace-tests-3632592.out`, SHA256 `0cbe18944114…`; source manifest SHA256 `55657144c167…` | The final 24-test trace file passes on every rank and every Slurm step completes `0:0`. A real CUDA/TE autograd checkpoint plus `CheckpointWithoutOutput` both emit paired exact forward/recompute output hashes with semantic collective phases. This covers the TE low-precision checkpoint boundary, not internal FP8/FP4 quantizer state. |
| current-MCore Nemotron-3-Ultra Bridge dry run | HSG `3615897`, one GB200 node | `bridge-ultra-current/dryrun/ConfigContainer.yaml`, SHA256 `74ced7ff2eba…` | The 96-GPU TP2×PP3×EP32 recipe resolves current MCore config successfully with deterministic mode, all-to-all dispatch, DDP overlap, and ordered hierarchical fp32 reduce-scatter (logical group size 4). This validates configuration only; the paired training gate is separate. |
| recovered Bridge hierarchy plumbing | HSG `3632202`, one GB200 node | `bridge-recovery-test/job-3632202.log`, SHA256 `061fb09951f5…`; source manifest SHA256 `68ab1b4dc165…` | All 31 focused Bridge process-group tests pass. The two new cases verify that Bridge commit `5793e80a` forwards the four-rank deterministic DP hierarchy to supported MCore revisions and fails instead of silently ignoring it on unsupported revisions. The production file SHA256 `98e49c6a2503…` is byte-identical to the retained 96-GPU-qualified source. |
| current-MCore full Nemotron-3-Ultra Bridge pair | HSG `3616252`, fixed 24-node/96-GPU GB200 allocation | `bridge-ultra-current/pair-3616252/comparison-final.json`, SHA256 `c8d0912328f3…` | Two sequential five-step launches match all 60 nonvolatile serialized metric values exactly; resolved configs differ only by artifact path. This is the full-model smoke preceding job `3616801`'s 50-step qualification. The validated production patch, recovered as Bridge commit `5793e80a`, was required to propagate the configured hierarchical group size into MCore process-group initialization, and late cleanup ranks exceeded the harness's 60-second grace. The top-level batch failure is from the pre-fix comparator; both training steps are `COMPLETED 0`. |
| retained 50-step Nemotron-3-Ultra Bridge qualification | HSG `3616801`, fixed 24-node/96-GPU GB200 allocation | `bridge-ultra-current/pair-3616801/comparison.json`, SHA256 `2674e83f9f63…`; source manifest `1aa87ec17b8e…`; placement `bd219b3b40f…` | Two sequential deterministic non-profiled launches match all 50 iterations and all 600 nonvolatile serialized metric values exactly. Configs differ only by `save_config_filepath`; ranks 0–95 map four per host across 24 retained nodes; zero skipped/NaN iterations; both training steps and the top-level batch complete `0:0`. The validated source, recovered as Bridge commit `5793e80a`, materializes the configured four-rank hierarchical DP groups. |
| TE 2.17 attention determinism source audit (GB300 image) | AWS-CMH `699987` | `inspect-te-attention-699987.out`, SHA256 `abff0bf5bdfb…` | TE derives `deterministic` from `NVTE_ALLOW_NONDETERMINISTIC_ALGO` or torch global state, filters backend eligibility, and documents deterministic FlashAttention ≥2.4.1 plus conditional fused sub-backends 0/1; fused sub-backend 2 is non-deterministic. |
| forced-fused dropout-0.1 diagnostic (GB300) | AWS-CMH `700020`, 4 GPUs | `det-attn-backend-700020.out`, SHA256 `d7211a9035f1…` | Expected failure: TE debug reports the exact dense `sbh3d`, bf16, causal, seq-256, dropout-0.1 input has no deterministic fused backend. This rules out a global force-fused override. |
| fixed-Flash attention profile (GB300) | AWS-CMH `699999`, 4 GPUs | console SHA256 `38568432111d…`; deterministic SQLite SHA256 `d4b3343c5171…` | With Flash forced in both modes, attention forward/core attention are -3.894/-0.672 ms (no deterministic penalty within profile noise), while `FlashAttnFuncBackward` is +6.034 ms (+37.4%). Step 7 is 130.1/115.5 ms (1.13×). |
| dropout-0 attention backend matrix (GB300) | AWS-CMH fused `700029`, auto `700030`, 4 GPUs each | console SHA256 `5c6bcfe71b52…` / `e14fddbf33f6…`; deterministic SQLite SHA256 `4e7d4b343a68…` / `54dd4ac03c37…` | Auto selects fused attention in both modes. Forcing fused does not improve the paired ratio in these placements: 1.18× forced versus 1.13× auto. In auto, fused-attention backward is +3.740 ms (+24.7%), LayerNormLinear backward +12.123 ms, and compiled backward +7.666 ms over the capture. |

The original DSV3 baseline `.nsys-rep` files were overwritten by the two
follow-up cleanup experiments; its console leaderboard is retained and hashed.
The dense raw reports and the final routing-candidate reports remain in
`det_perf_out/` and `det_perf_moe_out/`. The top-k-8 A/B raw reports remain in
`det-profile-topk8-results*/`. NVTX values below are sums over profiled steps
5–7; they are not per-step latency and ranges may overlap.

### Measured baseline — dense GPT (TP2, 8×H100, draco, det vs nondet)

From PR #5041's `perf_breakdown.sh` (`pretrain_gpt.py`, 4 layers, mock data,
profiled steps 5–7). **Step-time ratio = 1.11× (+10.7%)** — already under the
1.25× gate for this dense config. The det-vs-nondet divergence concentrates in an
**allocate-and-fill + indexed-write pattern** (CPU-range sums over 3 steps):

| op-level range | det ms | nondet ms | Δ ms | Δ % |
| --- | --- | --- | --- | --- |
| `aten::fill_` | 146.6 | 3.8 | **+142.8** | +3752% |
| `aten::empty` | 126.3 | 33.6 | +92.6 | +275% |
| `aten::empty_strided` | 91.5 | 11.6 | +79.9 | +686% |
| `aten::empty_like` | 53.7 | 16.7 | +37.0 | +221% |
| `aten::_index_put_impl_` | 27.9 | 5.5 | +22.3 | +403% |
| `aten::index_put_` | 21.7 | 6.7 | +15.0 | +223% |
| `aten::arange` | 14.4 | 3.9 | +10.5 | +271% |

Named backward contributors:

| backward range | det ms | nondet ms | Δ ms | Δ % |
| --- | --- | --- | --- | --- |
| `_VocabParallelCrossEntropyBackward` | 27.4 | 10.7 | +16.7 | **+156%** |
| `FlashAttnFuncBackward` (deterministic FAG) | 36.5 | 26.4 | +10.0 | +38% |
| `IndexBackward0` (index_add/put bwd) | 8.5 | 0 | +8.5 | det-only |
| `_LayerNormLinearBackward` | 141.3 | 118.5 | +22.8 | +19% |

**Read:** the `fill_`/`empty*`/`index_put_` cluster is the `torch.zeros(...)` +
`index_add_`/`index_put_` deterministic allocate-and-fill pattern. Even in dense
GPT (no MoE) it dominates — driven by the deterministic vocab-embedding backward,
the `_VocabParallelCrossEntropy` deterministic backward, and torch
deterministic-algo zeroed workspaces.

### Measured — DSV3-style MoE + MLA (TP2 × EP4, group-limited routing, 8×H100)

Same harness with `--multi-latent-attention` + MoE/EP + group routing
(`perf_breakdown_moe.sh`). **Step-time ratio = 1.2536× (det 228.4ms / nondet
182.2ms)** — displayed as 1.25× but just over the gate. MoE **doubles** the dense
overhead and confirms the MoE routing/permute deterministic paths are major
contributors:

| op-level range | det ms | nondet ms | Δ ms | Δ % | source |
| --- | --- | --- | --- | --- | --- |
| `aten::fill_` | 400.5 | 23.8 | **+376.7** | +1582% | `torch.zeros(...)` before index ops |
| `aten::index_put_` | 305.8 | 6.5 | **+299.3** | +4624% | aggregate included explicit routing writes; later nesting also identifies loss/gather call sites |
| `aten::_index_put_impl_` | 332.0 | 22.5 | +309.5 | +1373% | (same aggregate) |
| `aten::empty` | 425.1 | 90.7 | +334.4 | +369% | det workspace alloc |
| `aten::arange` | 156.5 | 28.8 | +127.7 | +444% | `rows=arange` for index_put `moe_utils.py:837` |
| `aten::index_add_` | 88.2 | 2.3 | +85.9 | +3729% | unpermute `moe_utils.py:536` |
| `aten::scatter_add_` | 76.0 | 6.6 | +69.4 | +1052% | (still present in det) |
| `aten::gather_backward` | 80.9 | 6.1 | +74.8 | +1229% | permute gather bwd |
| `aten::max` / `aten::min` | 51.7 / 44.8 | 1.1 / 0 | +50.6 / +44.8 | huge | det reductions |

Forward: `_forward_mlp.mlp` **+251ms (+34%)** dominates (the MoE block); MLA
`self_attention` +77ms (+18%). Backward: `GatherBackward0` **+1168%**,
`IndexSelectBackward0` +555% (the permute `index_select` bwd), `_GroupedLinearBackward`
+20%, `_VocabParallelCrossEntropyBackward` +168%, `IndexAddBackward0` det-only.

### ⚠ Dispatch-time vs wall-clock — a measured correction

The op-level table above is **CPU NVTX-range time** (op dispatch), which overlaps
with GPU execution and cannot be added up as wall-clock time. Two experiments
showed that the largest dispatch deltas were not the limiting wall-clock work on
this proxy:

- **Removing the redundant unpermute `torch.zeros`** (`moe_utils.py:526-536`)
  was bit-exact but measured 235.1 / 181.3 ms (1.30×).
- Adding the **routing `index_put_` → `scatter` candidate** (`moe_utils.py:834`,
  verified bit-exact and non-raising in production det mode) reduced the
  `index_put_`/`fill_`/`arange` dispatch ranges, but measured 232.2 / 186.4 ms
  (1.25×). Neither change produced a repeatable step-time improvement in these
  H100 runs; the later production GB200 ABBA measures a modest 1.8% routing win.

The remaining module/autograd deltas identify areas to isolate, but they are also
overlapping NVTX ranges rather than an additive wall-clock decomposition:

| range | det | nondet | Δ | note |
| --- | --- | --- | --- | --- |
| `_forward_mlp.mlp` (fwd) | ~1003 | ~741 | **+262 (+35%)** | MoE block: deterministic grouped GEMM + index-add unpermute work |
| `_forward_attention.self_attention` (fwd) | ~520 | ~433 | **+87 (+20%)** | MLA attention det kernels |
| `GatherBackward0` | 83 | 6.6 | **+76 (+1151%)** | permute gradient scatter-add (det reduction) |
| `IndexSelectBackward0` | 52 | 7.9 | +44 (+558%) | permute gradient scatter-add; optimized below |
| `_GroupedLinearBackward` | 118 | 100 | +18 (+18%) | grouped GEMM wgrad |
| `_VocabParallelCrossEntropyBackward` | 28 | 11 | +17 (+154%) | indexed grad scatter-subtract |
| `IndexAddBackward0` | 12.6 | 0 | +12.6 (det-only) | unpermute gradient |

### Optimized token-permute backward — isolated and end-to-end evidence

The fixed-order Triton backward in `deterministic_index_select.py` directly
targets the `IndexSelectBackward0` range for dropless top-k ≥4 and hidden ≥2048.
Two same-node orderings compared parent `dabf6c007` with optimized `895a780a2`:

| metric (steps 5–7 range sum unless noted) | parent | optimized | observed change |
| --- | --- | --- | --- |
| index-select backward autograd range | `IndexSelectBackward0` 49.53–50.34 ms | `_DeterministicIndexSelectBackward` 14.41–14.97 ms | **-70.3% to -70.9%** |
| `aten::index_add_` | 87.64–88.17 ms | 43.95–44.03 ms | -49.8% to -50.2% |
| deterministic step time (iteration 7) | 238.3 / 242.8 ms | 237.4 / 241.4 ms | -0.4% / -0.6% |
| nondeterministic control (iteration 7) | 194.2 / 189.1 ms | 195.0 / 192.3 ms | +0.4% / +1.7% drift |
| det/nondet ratio | 1.23× / 1.28× | 1.22× / 1.26× | -0.01× / -0.02× |

The targeted backward improvement repeats cleanly. The total deterministic step
time also moves in the right direction in both orderings, but its 0.4–0.6% change
is comparable to control drift, so it is not yet a statistically isolated
end-to-end speedup. One run clears the 1.25× gate and one remains at 1.26×.

### Measured — Nemotron hybrid EP32 paired attribution (32×GB200)

Job `519928` ran three 14-iteration modes serially on the same cross-domain
allocation (`nvl72d038` + `nvl72d039`), with the same fused Mamba+attention+MoE
model and fixed Mamba config. Medians below use steady iterations 6–14; the
single 456 ms system outlier in the full-deterministic leg does not affect its
median (and the corresponding trimmed mean is 79.8 ms).

| mode | median step | delta vs native | attribution |
| --- | --- | --- | --- |
| fast kernels + native DP reduce-scatter | 72.5 ms | — | paired baseline |
| fast kernels + ordered fp32 DP reduce-scatter | 75.1 ms | **+3.6%** | cross-domain deterministic DP cost |
| full deterministic mode | 79.9 ms | **+10.2%** | DP plus deterministic TE/MoE branches |

Thus the ordered DP path accounts for about 2.6 ms and the remaining
deterministic TE/MoE branches for about 4.8 ms on this proxy. A separate paired
Mamba-only attribution in job `519888` measured 75.5 ms normal, 75.0 ms with
deterministic workspaces plus autotuning, and 74.5 ms with deterministic
workspaces plus the fixed config. Mamba is not the current step-time gap. The
job's unused full-deterministic leg correctly rejected an intentionally retained
`NVTE_ALLOW_NONDETERMINISTIC_ALGO=1`; the three completed Mamba legs precede
that launcher error.

### Measured — production hierarchical DP attribution (32×GB200)

Job `520235` ran four 14-iteration modes serially on one four-domain
allocation. Medians use steady iterations 6–14. The DP-only modes keep the fast
kernel settings and change only the gradient reduce-scatter; the final mode
enables the complete deterministic stack plus the production hierarchy.

| mode | median step | delta vs native | attribution |
| --- | --- | --- | --- |
| fast kernels + native DP reduce-scatter | 55.6 ms | — | paired baseline |
| fast kernels + flat ordered fp32 DP | 57.8 ms | **+4.0%** | previous deterministic DP path |
| fast kernels + hierarchical ordered fp32 DP | 55.4 ms | **-0.4%** | production hierarchy closes the DP penalty within run noise |
| full deterministic + hierarchical DP | 66.4 ms | **+19.4%** | remaining deterministic TE/MoE kernel cost |

The hierarchy increases peak allocated memory from 3,284 to 3,424 MiB in this
proxy because it materializes a locally permuted send buffer. The cross-
allocation DSV3 and Nemotron certificates above verify that this speedup does
not change the training trace across placements.

Job `520311` then compared the full deterministic flat and hierarchical paths
directly on the same node placement: native was 57.2 ms, flat deterministic was
66.5 ms (+16.3%), and hierarchical deterministic was 66.7 ms (+16.6%). Thus the
hierarchy removes the exposed DP-only penalty but does not improve full-stack
step time in this proxy; gradient communication is overlapped beneath the
remaining deterministic TE/MoE work.

### Measured — rank-scoped production Nemotron profile (32×GB200)

Job `520457` used `tools/determinism/profile_rank.py` so that global rank 0 ran
under Nsight Systems while the other 31 ranks executed normally. Over the
captured steps, deterministic MLP forward was +24.86 ms, `aten::index_put_` was
+11.16 ms, `aten::fill_` was +20.33 ms, and `aten::arange` was +3.53 ms. The
source path contained two dense routing writes per invocation: full
`zeros_like` tensors followed by explicit `index_put_(accumulate=False)` calls
for routing probabilities and the routing map.

Because top-k expert indices are unique within a token row, job `520492` tested
in-place `scatter_` as the collision-free replacement under
`torch.use_deterministic_algorithms(True)`. All bf16/fp32 output and source-grad
hashes matched the `index_put_` reference at 16, 512, and 4,096 tokens; isolated
forward latency improved by 1.76–1.80×. The production ABBA in job `520526`
then moved the median-of-leg medians from 65.7 to 64.5 ms (-1.8%). Its forward
pair improved 4.8%, while its reverse pair regressed 1.2%, so the model-level
benefit is real but small relative to allocation/overlap variance. Report the
operator result and the ABBA result separately.

Post-change job `520565` shows why the model-level gain is smaller than the
isolated ratio: deterministic `scatter_` still lowers the probability write
through PyTorch's generic `index_put_` implementation. NVTX containment in the
exported SQLite attributes four ≈1.0–1.3 ms indexed writes to routing-probability
scatter, two to vocab-cross-entropy backward, two to gather backward, and four
small writes to cross-entropy forward. The remaining direct opportunity at that
point was a fused collision-free routing kernel that creates probabilities and
the boolean map without the generic deterministic scatter machinery.

Job `520625` measured the post-change full-stack gap with native/deterministic/
deterministic/native ABBA ordering on one allocation. Median-of-leg medians were
56.4 ms native and 59.4 ms deterministic (+5.3%); the forward pair was +8.8%
and the reverse pair +1.6%. This is the first allocation to reach the aggressive
≈5% target, but the pair spread and the earlier +16–19% results prohibit treating
it as a topology-independent guarantee. The direct old/new ABBA from job
`520526` remains the attribution for this code change: -1.8%.

Job `520763` bypassed the remaining generic deterministic scatter with a
collision-free Triton kernel that initializes probabilities and the boolean map
in one row program and gathers selected gradients in backward. Across DSV3 and
Nemotron shapes, all 12 bf16/fp32 cells matched scatter byte-for-byte and the
fused path was 4.21–4.63× faster forward and 1.66–1.87× faster
forward+backward. Direct production ABBA job `520786` moved median-of-leg
medians from 67.1 to 65.75 ms (-2.0%): its forward pair improved 4.8% while the
reverse pair regressed 1.1%, again exposing overlap/order variance. DSV3 job
`520784` and Nemotron job `520785` match the prior scatter build across
independent 32-GPU allocations at 6,080/6,080 and 9,664/9,664 events.
Post-fusion native/deterministic ABBA job `520827` measures 56.1/64.65 ms
(+15.2%) with consistent +14.5%/+16.0% order pairs. Together with the +5.3%
allocation in job `520625`, this confirms that the remaining full-stack gap is
still topology/allocation-sensitive and is not routing alone.

Rank-scoped profile job `520828` first isolated the fused-routing residual:
four small vocab-CE forward writes, two vocab-CE backward writes, and two
gather-backward writes. The selected-gather experiment removed the latter two
in job `520964`, but failed the production gate and was removed. With only the
fused CE backward applied, current-source job `521407` contains six residual
`aten::index_put_` ranges: four small CE-forward mask writes and the two
gather-backward writes. No CE-backward indexed write remains, and
`_VocabParallelCrossEntropyBackward` improves from 1.257 to 0.749 ms (-40.4%)
over the two captured steps. `GatherBackward0` is now the largest directly
attributed indexed-write target at +1.621 ms.

### WS3 targets, by measured impact

1. **Remaining deterministic TE/MoE kernel set** — allocation-sensitive from
   +3.0 ms (+5.3%) in job `520625` to +8.55 ms (+15.2%) post-fusion in job
   `520827`, with older +9.5–10.8 ms (+16–19%) results in jobs
   `520235`/`520311`. Preserve ABBA ordering and report both pairs; a single
   placement is not a production guarantee.
2. **Cross-domain ordered fp32 reduce-scatter — closed for the EP32 proxies.**
   The production fixed-logical-rank hierarchy reduces isolated latency from
   3.110–3.118 ms to 1.324–1.346 ms in job `520160`, removes the paired step-time
   penalty in the DP-only modes of job `520235`, and remains exact across
   independent DSV3 and Nemotron allocations. Job `520311` shows no full-stack
   speedup because communication is hidden by other deterministic kernels.
   Retain the hierarchy as a performance and correctness gate.
3. **Routing probability scatter — closed for the target shapes.** The fused
   collision-free probability+map kernel is exact across both model traces and
   removes the generic indexed-write path. It is 4.21–4.63× faster forward in
   isolation and improves median-of-leg production step time by 2.0%. Retain
   the fallback, CUDA-graph test, and two-allocation certificates as gates.
4. **Expert-bias selected-score gather — still open.** The collision-free
   candidate was exact, 1.48–2.23× faster forward+backward in isolation, and
   reduced the profiled range from `GatherBackward0` +1.75 ms to 0.29 ms. It was
   nevertheless 3.3% slower in gather-only production ABBA and the combined
   gather+CE candidate was 3.2% slower. The candidate is removed; pursue a
   fused gather+normalization/routing design that survives the full-step gate.
5. **Vocab-parallel cross-entropy local backward — closed.** A fused,
   collision-free one-class-per-row Triton pass performs the selected subtract,
   optional smoothing, and output-gradient scaling, including direct logical
   loads from the training loss's 2D-strided gradient. It is 3.34–20.41× faster
   for contiguous inputs and 14.41–16.52× faster for strided inputs in
   isolation. The refreshed dense profile reduces deterministic CE backward
   18.713→7.001 ms (-62.6%), and the warmed production ABBA improves 12.9% with
   both order pairs positive. Forward mask writes remain because the measured
   `where` alternative regressed. TP collective ordering remains a separate
   correctness/performance gap.
6. **Attention deterministic kernels** — refreshed dense GB200/GB300 profiles
   exposed both backend selection and intrinsic deterministic-kernel cost. With
   dropout 0.1, deterministic fused attention is unsupported and auto selects
   Flash while native selects fused. Fixing both modes to Flash removes the
   forward delta, but deterministic Flash backward remains +6.034 ms (+37.4%).
   With production-like dropout 0, auto selects fused in both modes; forcing
   fused does not improve the paired ratio, and fused backward remains
   +3.740 ms (+24.7%). There is no safe MCore backend override. The kernel
   optimization belongs in TE;
   MCore should retain the auto selector and model-level certificates.
7. **Deterministic grouped GEMM** — `_GroupedLinearBackward` ranged from
   +2.27 ms in job `520565` to -0.76 ms in job `520828`; re-isolate before
   changing the kernel choice.
8. **Mamba profile delta** — backward +1.84 ms in job `520964` (+7.31 ms in
   `520565`), but paired Mamba-only attribution in job `519888` was step-time
   neutral. Re-isolate before changing the fixed-config/workspace path.

### Low-risk cleanups and measured outcomes

- ✅ **Applied:** remove the redundant second `torch.zeros` in the det unpermute
  branch (`moe_utils.py`) — pure dead-allocation removal, bit-exact.
- ✅ **Applied:** routing `index_put_` → collision-free in-place `scatter_`
  (`moe_utils.py:834`). The optimized path removes the explicit row `arange`,
  two Python `index_put_` calls, and the logits-dtype-to-bool map conversion. It is
  byte-exact for outputs and gradients, 1.76–1.80× faster in isolation, and
  improves the production ABBA median-of-leg medians by 1.8%. The full-step
  result is overlap/order-sensitive, and PyTorch still implements the source
  scatter through deterministic `index_put_`; retain the operator benchmark and
  model certificate rather than treating 1.8% as a universal speedup.
- ✅ **Applied:** deterministic `scatter_` → fused collision-free probability
  and boolean-map construction (`moe/ops/deterministic_routing.py`). The kernel
  removes generic indexed-write dispatch and dense backward metadata; backward
  is a selected-entry gather. Unsupported inputs retain the scatter fallback.
  DSV3/Nemotron output, map, gradient, CUDA-graph, and cross-allocation trace
  gates all pass; isolated and production gains are reported separately above.
- ❌ **Rejected after production ABBA:** generic deterministic gather backward
  → collision-free selected-score backward. The candidate was exact and faster
  in isolation/profile, but gather-only and combined ABBAs regressed 3.3% and
  3.2%, respectively. The source path remains regular `torch.gather`.
- ✅ **Applied:** vocab-CE deterministic `index_put_` selected subtract plus
  separate smoothing/output scaling → one fused collision-free Triton pass per
  token row. The kernel has no atomics or reductions, directly supports the
  common 2D-strided loss gradient without a copy, retains a PyTorch fallback,
  supports label smoothing and CUDA graphs, and leaves the measured-slower
  forward mask alternative untouched. The original extended production ABBA
  improves by 2.7%; the dense strided gate improves by 12.9%, with exact
  training sequences in both.

## Verification backlog (the ⚠ rows)

Run `BitExactRunner` (PR #5041) with `deterministic_mode` toggled, plus the
`RacingStreams` / `CudaSleepJitter` stressors, to confirm/deny each row.

**Verified (WS2 proxies, draco 8×H100, fw-final pipe.50600619):** the
`test_deepseek_model.py` (MLA + group-limited MoE routing) and
`test_nemotron_hybrid_model.py` (MoE-in-hybrid) proxies pass bit-exact across
EP≤4 / TP / FSDP / PP / VPP (12/12 cells). This promotes: group-limited top-k,
aux-loss routing map, grouped GEMM, EP all-to-all (small EP), MLA, and
MoE-inside-hybrid.

**Verified (optimized A2A guard, AWS-DFW GB200):** job `516181` ran the
DeepSeek top-k-8 and Nemotron top-k-6 A2A presets at hidden 2048; each rank
reported 8 passed / 4 expected 8-GPU skips. Job `516190` re-ran the existing
top-k-2 presets after fixing the harness to preserve their declared 8-expert
topology, with the same 8 passed / 4 skipped result on each rank. The helper's
fallback and dtype/shape matrix passed 20/20 cases per rank in job `516124`.
Draco H100 job `10438654` then closed the skipped full-width cells: all eight
ranks passed all 12 A2A model cells, including TP2×EP4, FSDP8×EP4, and FSDP8.
The strict retained report validates 96/96 per-cell `PASSED` records with no
failed node or terminal summaries (report SHA256 `f808fe963e1c…`, pytest log
SHA256 `24d2beab4b1e…`).

**Verified (capacity and Sinkhorn routing):** AWS-DFW GB200 job `516442`
reported 12 passed / 9 expected 8-GPU skips per rank. Draco H100 job `10429794`
then passed all 21 selected cells on each of 8 ranks. Together these cover both
capacity drop policies, padded and unpadded A2A dispatch, and the Sinkhorn
router-local scatter across EP≤4 / TP / FSDP / PP / VPP.

**Verified (torch norm and DSA masks):** AWS-DFW GB200 job `516499` and Draco
H100 job `10429898` each passed all 7 selected cases on every rank (4 and 8
ranks, respectively). The matrix covers PyTorch LayerNorm/RMSNorm forward and
backward at hidden 128/2048, DSA indexer-loss mask construction in forward and
recomputed manual backward (dense and sparse loss), and unfused sparse-attention
mask construction plus input gradients.

**Verified (structured EP all-to-all trace):** AWS-DFW GB200 job `516965`
passed all 21 focused trace/dump tests per rank, including synchronous and
NCCL-stream all-to-all plus activation-recompute and backward propagation;
AWS-CMH GB300 job `696891` passed the same final suite per rank. AWS-DFW job
`516924` matched 984/984 events across two independent DSV3-style TP2×EP2 runs;
288 were hashed collective boundary events spanning original forward, recompute,
and backward.

**Verified (structured pipeline P2P trace):** the final 22-test suite passed per
rank in AWS-DFW GB200 job `517022`, AWS-CMH GB300 job `696943`, and Draco H100
job `10430851`. AWS-DFW job `517040` matched 2,320/2,320 events across two
independent DSV3-style PP2×VPP2×EP2 runs; 384 were synchronous or overlapped P2P
boundary events completed through the schedule's existing waits.

**Verified (structured DP reduction and parameter-gather trace):** the expanded
29-test suite passed per rank in AWS-DFW GB200 job `517130`, AWS-CMH GB300 job
`697037`, and Draco H100 job `10431044`. AWS-DFW job `517136` matched
1,352/1,352 events across two independent DSV3-style TP2×EP2 distributed-
optimizer runs. The comparison included 48 DP reduction/gather boundary events,
24 completed output hashes, and zero pending collectives.

**Verified (EP32, ordered DP, and scaled recompute):** AWS-DFW GB200 jobs
`518849` and `518850` independently certified final DSV3-style EP32 runs, with
6,080/6,080 semantic events matching both within and across allocations.
Follow-up job `519887` extended that topology to four optimizer updates per
launch and matched 12,416/12,416 events, including 128 recomputes and 4,096
collective boundaries with zero pending operations.
Nemotron-style fused-Mamba+attention+MoE EP32 jobs `518541` and `518542` matched
9,664/9,664 events within each allocation and across allocations. Each trace
tree contained 192 matching recomputes, 1,152 completed collective hashes, no
pending operations, and 384 DP reductions using ordered fp32 accumulation. Job
`519253` passed the final 48-test focused suite on every AWS-DFW GB200 rank;
AWS-CMH GB300 job `697567` passed the same 48 tests on every rank.

**Still open:** a checked-in weekly schedule/dashboard and same-mode 192/3,072-
GPU controls for the full Megatron-Bridge recipe, plus topology-independent
native TP floating-point reductions and non-distributed-optimizer DP all-reduce.
The retained 96-GPU 50-step qualification is job `3616801`; the new A2A
presets' 8-GPU cells are closed by Draco job `10438654` above.
