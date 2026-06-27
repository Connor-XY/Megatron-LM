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
| Routing map/probs | `moe_utils.py:834-849` | scatter vs index_put | 🔵 | `index_put_(accumulate=False)` ×2 | `scatter` ×2 | `are_deterministic_algorithms_enabled()` | code | `index_put_` +299.3 ms CPU NVTX-range time over 3 steps | An out-of-place `scatter` candidate is bit-exact and removes dispatch overhead, but did not improve proxy step time; held pending production-scale evidence. |
| Top-k expert select | `moe_utils.py` | `torch.topk(sorted=...)` | 🔵 | `sorted=True`, including no-grad checkpoint forward | `sorted=is_grad_enabled()` | deterministic algorithms | code+**test** | `cub::DeviceRadixSort` 6.36 vs 1.23 ms over 3 steps (+5.13 ms) | Fixes sigmoid top-k probability drift between original forward and grad-enabled recompute. Investigate a faster stable-select path. |
| Group-limited top-k mask | `moe_utils.py:590-645` | `scatter_(1, group_idx, 1)` | 🟢 | unique group indices ⇒ det fwd | — (no branch) | always | code+**test** | small | **Verified** by the DSV3 proxy across EP≤4/TP/FSDP/PP/VPP and by the final EP32 certificates (`518849`/`518850`). |
| Aux-loss routing map | `moe_utils.py:888-901` | `scatter` | 🟢 | unique indices ⇒ det fwd | — (no branch) | always | code+**test** | small | **Verified** via DSV3 proxy and final EP32 certificates with `seq_aux_loss` enabled. |
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
| FlashAttention backward | TE; `extensions/transformer_engine.py:1697-1703` | atomic dQ/dK/dV accumulation | 🔵🟡 | TE deterministic FA kernels | TE atomic FA | `deterministic_mode` asserts `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` | doc+code | dense proxy: +10.0 ms (+38%) over 3 profiled steps | **Deterministic FAG** (Longcat/DSV4): independent accum buffers + global deterministic sum. |
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
| Vocab-parallel embedding | `tensor_parallel/layers.py:299-303` | indexing vs `F.embedding` | 🔵 | direct `weight[idx]` (det bwd) | `F.embedding` (atomic bwd) | `self.deterministic_mode` | doc+code | TBD | Comment: "F.embedding has non-deterministic backward". |
| RoPE / position emb | `models/common/embeddings/rotary_pos_embedding.py` | trig | 🟢 | — | — | — | doc | — | — |
| Column/Row parallel linear (GEMM) | `tensor_parallel/layers.py` | cuBLAS GEMM | 🟡 | fixed cuBLAS workspace | — | `CUBLAS_WORKSPACE_CONFIG` | doc | — | — |
| Async TP all-gather/all-reduce | `tensor_parallel/layers.py:544/565/577`, `mappings.py:452` | `async_op=True` | 🟡 | `wait()` re-imposes order | overlap | `tp_comm_overlap` | doc | — | `tp_comm_overlap` force-off in det mode. |
| Pipeline P2P send/receive | `pipeline_parallel/p2p_communication.py` | `isend` / `irecv` / batched P2P | 🟡 | existing wait pins readiness before use | overlap / VPP | schedule | code+**test** | hash tracing is debug-only | Structured trace fingerprints sends and completed receives without adding waits; PP2×VPP2×EP2 passed two-run comparison. |

### Loss

| Op | File:line | Primitive | Det? | Det path | Non-det path | Selected by | Evidence | Perf Δ | Gap / TODO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vocab-parallel cross-entropy | `tensor_parallel/cross_entropy.py:119-156` | 3× all-reduce + fp32 | 🟡 | no topology-independent TP reduction yet | native NCCL | env | code | MoE proxy: +17.9 ms (+168%) over 3 profiled steps | `NCCL_ALGO=Ring` does not pin the physical ring across allocations. TP>1 cross-allocation certification and an ordered SUM path remain open; indexed backward is also a perf target. |
| Fused cross-entropy | `cross_entropy_loss_fusion` | fused kernel | 🔴 | — (forbidden) | fused | asserted off | code | — | Open: can it be made deterministic? (perf opportunity). |

### Backward / grad reduction / optimizer

| Op | File:line | Primitive | Det? | Det path | Non-det path | Selected by | Evidence | Perf Δ | Gap / TODO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Grad bucket all-reduce / reduce-scatter | `distributed/param_and_grad_buffer.py` | floating-point NCCL collective | 🔵 | ordered fp32 reduce-scatter for deterministic distributed optimizer | native NCCL | deterministic args | code+**test** | hash tracing is debug-only | Native floating-point collectives are allocation-topology-sensitive even with Ring. Non-distributed-optimizer all-reduce remains a gap. |
| fp32-accum reduce-scatter | `distributed/reduce_scatter_with_fp32_accumulation.py` | all-to-all + ordered `sum(fp32)` | 🟢 | rank-indexed ordered fp32 sum | native RS | forced in deterministic distributed optimizer | code+**test** | TBD | One optimizer instance and no collective AVG are enforced. Final DSV3 EP32 jobs `518849`/`518850` matched 6,080/6,080 trace events across independent allocations. |
| Distributed-optimizer param all-gather | `distributed/param_and_grad_buffer.py` | NCCL all-gather | 🟢 | rank-indexed byte copies | — | always | code+**test** | hash tracing is debug-only | No floating-point reduction; structured trace fingerprints sync and overlapped gathers without adding a collective or wait. |
| Distributed optimizer param order | `optimizer/distrib_optimizer.py:1094` | shard mapping | 🟢 | "preserving deterministic ordering across ranks" | — | always | code | — | — |
| Grad clip global norm | `optimizer/clip_grads.py`, `distributed/deterministic_collectives.py` | SUM reduction | 🔵 | rank-ordered all-gather + local sum | native all-reduce | deterministic algorithms | code+**test** | TBD | Closed a one-ulp scalar divergence that otherwise changed every optimizer parameter. Intended only for small statistics. |
| Reported loss / MoE metrics | `training.py`, `training/utils/common_utils.py`, `transformer/moe/moe_logging.py` | SUM / AVG reductions | 🔵 | rank-ordered all-gather + local sum/divide | native all-reduce | deterministic algorithms | code+**test** | TBD | Keeps reported loss and expert-bias/load-balancing metrics exact across allocations. |

### SSM / Mamba (hybrid models)

| Op | File:line | Primitive | Det? | Det path | Non-det path | Selected by | Evidence | Perf Δ | Gap / TODO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Triton autotune | `megatron/determinism_env.py`, `ssm/ops/determinism.py`, external `mamba_ssm.utils.determinism` | autotune config select | 🔵 | one fixed config (`TRITON_CACHE_AUTOTUNING=0`) | timing-selected config | early deterministic env | code+**test** | TBD | Cold cached-autotune selected different reduction tilings across launches; env setup must precede kernel-module import. |
| Tiled reduction workspace | `ssm/ops/determinism.py:106-123` | `zeros`+`sum` vs `empty` | 🔵 | ordered tile sum | unordered | `use_deterministic_mode()` | code | mem+ | Extra memory for tiled reduction. |
| Gated-delta-rule kernel | `ssm/gated_delta_net.py:213-216` | torch vs FLA fused | 🔵 | `torch_chunk_gated_delta_rule` | FLA fused | `deterministic_mode` | code | TBD | — |
| Causal conv1d | `ssm/gated_delta_net.py:430-446` | `F.conv1d` vs FLA | 🔵 | `F.conv1d` (+transpose) | `causal_conv1d` | `deterministic_mode` | code | TBD | — |
| Mamba-2 combined training | `ssm/mamba_mixer.py`, external `mamba_ssm` | fused causal conv + selective scan | 🟢🟡 | fused path with deterministic workspaces and fixed Triton config | same path with timing autotune / atomic reductions | early env + torch deterministic algorithms | code+**test** | production-scale delta pending | Fused EP32 jobs `518541`/`518542` matched 9,664/9,664 events within and across allocations without launcher-supplied Mamba/Triton env. Disabling only the fused path did not fix cold-cache autotune drift; pinning config did. |
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
| `aten::index_put_` | 305.8 | 6.5 | **+299.3** | +4624% | routing map `moe_utils.py:834-843` |
| `aten::_index_put_impl_` | 332.0 | 22.5 | +309.5 | +1373% | (same) |
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
  (1.25×). Neither change produced a repeatable step-time improvement.

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

### WS3 targets, by measured impact

1. **Remaining deterministic scatter-add backward work** — `GatherBackward0`
   remains ≈+75 ms over 3 steps, and `IndexAddBackward0` remains det-only. The
   optimized index-select path closes its guarded slice, but top-k <4 and hidden
   <2048 retain the PyTorch fallback.
2. **MoE MLP range** (+262 ms over 3 steps) — isolate deterministic grouped GEMM
   (`NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`) from token unpermute before changing it.
3. **Deterministic grouped GEMM** wgrad (`_GroupedLinearBackward` +18%).
4. **MLA / attention deterministic kernels** (+20% module-range time).
5. **`_VocabParallelCrossEntropy` deterministic backward** (+154%; all models).

### Free cleanups found along the way (correct but ~wall-clock-neutral on the proxy)

- ✅ **Applied:** remove the redundant second `torch.zeros` in the det unpermute
  branch (`moe_utils.py`) — pure dead-allocation removal, bit-exact.
- 🔬 **Held (candidate):** routing `index_put_` → `scatter` (`moe_utils.py:834`).
  Proven bit-exact + non-raising in det mode, removes a `torch.zeros`+`arange`+
  `index_put_`, but wall-clock-neutral on the 256-token proxy. Re-evaluate at
  production scale (Tier-B / large `num_tokens × num_experts`) before changing a
  deliberate determinism branch.

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
Nemotron-style fused-Mamba+attention+MoE EP32 jobs `518541` and `518542` matched
9,664/9,664 events within each allocation and across allocations. Each trace
tree contained 192 matching recomputes, 1,152 completed collective hashes, no
pending operations, and 384 DP reductions using ordered fp32 accumulation. Job
`519253` passed the final 48-test focused suite on every AWS-DFW GB200 rank;
AWS-CMH GB300 job `697567` passed the same 48 tests on every rank.

**Still open:** a retained weekly gate for the full Megatron-Bridge recipe;
topology-independent native TP floating-point reductions and non-distributed-
optimizer DP all-reduce; and the 8-GPU cells for the new
A2A presets. AWS-DFW and AWS-CMH expose four GPUs per node to this fixture; HSG
was unreachable during this run.
