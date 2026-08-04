# GTP + 1F1B EP Overlap — Reproduction Manifest & Findings

Shareable handoff for the M≥3 stall on Nemotron-3 a55b. Everything needed to reproduce, plus what
is established, what is refuted, and what is still open.

Date: 2026-08-03. Contact: Yan Xu (yxu1).

---

## 1. TL;DR

1F1B EP-comm overlap (`--overlap-moe-expert-parallel-comm --delay-wgrad-compute`) on the a55b
hybrid recipe **runs at M ≤ 2 and deadlocks during iteration 2 at M ≥ 3**, where
`M = GBS / (MBS × DP)` is the microbatch count.

The deadlock is **GPU memory exhaustion**, not a defect in the overlap machinery:

- cuda-gdb on a live hang: **30 of 32 nodes** have resident `ncclDevKernel_AllGather_RING_LL`,
  GPU memory at **283.2 / 284.2 GiB = 99.6%**.
- At that occupancy NCCL cannot obtain collective buffers, so the AllGather never completes.
- The HybridEP `expecting X got Y` timeouts that dominate the logs are a **symptom**: HybridEP's
  device-side spin times out alongside the stuck NCCL call.
- The same overlap runs **5/5 iterations clean on DeepSeek-V3** at PP4/VPP4/EP64/M=8.

**The stall is recipe-specific, not an EP-overlap defect.**

---

## 2. Exact stack

### Container image

```
gitlab-master.nvidia.com/xren/nemo_megatron_perf_optimization:mcore-moe-pytorch26.04-texren7b2742cd-hybridep42144303-nccl2.30u1719b2bbb-arm
```

Tag decodes as: PyTorch 26.04 · TE `7b2742cd` (xren) · HybridEP `42144303` · NCCL
`2.30u1719b2bbb` · aarch64.

Observed inside the container:

| component | version |
|---|---|
| PyTorch | 2.12.0a0+0291f960b6.nv26.4.48445190 |
| Megatron-Core | 0.19.0 |
| Transformer Engine | 2.19.0.dev0 (overridden — see below) |
| deep_ep | **1.2.1+4214430** |
| NCCL | 2.30.5a3+cuda13.2 |
| Nsight Systems | 2026.2.1.210 |

`deep_ep 1.2.1+4214430` = DeepEP `hybrid-ep` branch commit
[`42144303752422ade37f24bca9e2dde12df70e09`](https://github.com/deepseek-ai/DeepEP/commit/42144303752422ade37f24bca9e2dde12df70e09)
(2026-05-20). **This is PRE-#682.**

### Transformer Engine (overrides the image's TE)

| item | value |
|---|---|
| base commit | **`f1d5f8d5222b5789351dd6dc145253a7ac727b78`** |
| source tarball | `transformer-engine-f1d5f8d-latest-gtp-delay-wgrad-cudnnfe126-src.tar.gz` |
| contents | TE main @ f1d5f8d + GTP + delay-wgrad + cuDNN-frontend 1.26 |
| build env | `NVTE_FRAMEWORK=pytorch NVTE_CUDA_ARCHS=103a NVTE_WITH_NCCL_EP=0 MAX_JOBS=32` |
| build cmd | `python -m pip wheel --no-build-isolation --no-deps -w $WHEEL_DIR .` |
| wheel | `transformer_engine-2.19.0.dev0-cp312-cp312-linux_aarch64.whl` |
| wheel sha256 | `dcf3d308e6aa9116daf9dd01b5b599d150be8ed6b64085afbc26cd187499b630` |
| build script | `build_te_main_f1d5f8_gtp_cudnnfe126_cmh2.sbatch` |

> `NVTE_CUDA_ARCHS=103a` is **GB300 / sm103a only**. This wheel will not load on GB200 (sm100a) —
> it fails with "no kernel image is available for execution on the device". That rules out hsg,
> dfw, and aga as hosts.

Also installed: `nvidia_cudnn_frontend-1.26.0-cp312-cp312-manylinux_2_27_aarch64...whl`,
sha256 `ee50df3468f672aa31402fde4c911ad545d08e1d5e133e15bc55c3679d9fd9ca`.

### Megatron-Core

| item | value |
|---|---|
| archive | `megatron-gtp-dense-ag-high-priority-v1-src-v2.tar.gz` |
| sha256 | `93cf1ded2c95f694196d562d797874d49408bd8fd193492b6108a7e2343489fb` |
| version | Megatron-Core **0.19.0** |

> **Reproducibility gap:** the archive is a source tarball with `.git` stripped, so there is **no
> upstream commit hash** for the exact mcore we run. Anyone reproducing must use this tarball.
> Its sha256 is the only identity it has.

### Findings branch (docs only)

| item | value |
|---|---|
| fork | `git@github.com:Connor-XY/Megatron-LM.git` (remote `origin`) |
| branch | `yx/gtp-1f1b-overlap-findings` |
| HEAD | `5c011f8be` |

This branch holds **write-ups only**. The launchers and overlays live on the cluster (§6); the
runtime is the pinned tarball above, not this checkout.

### Cluster

**cmh** (`aws-cmh-slurm-1`), GB300, NVL72, 4 GPUs/node, ~284 GiB/GPU.
`--account=nemotron_sw_pre --partition=batch --qos=normal`.
cmh is the **only** validated host (see §7 for why others are out).

---

## 3. Model configuration

Nemotron-3 a55b hybrid (Mamba + MoE), `pretrain_hybrid.py`:

```
--hybrid-layer-pattern MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEM*E   # 54 layers
--spec megatron.core.models.hybrid.hybrid_layer_specs hybrid_stack_spec
--hidden-size 8192 --num-attention-heads 64 --num-query-groups 8
--mamba-num-heads 256 --ffn-hidden-size 5120 --seq-length 8192
--num-experts 512 --moe-router-topk 22 --moe-shared-expert-intermediate-size 10240
--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend hybridep
--moe-hybridep-num-sms 32 --moe-permute-fusion --moe-router-force-load-balancing
--expert-model-parallel-size 64 --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1
--tensor-parallel-num-weight-shards 64 --expert-tensor-parallel-num-weight-shards 2
--high-priority-stream-groups ep gtp_remat expt_gtp_remat tp
--cuda-graph-impl full_iteration
--moe-paged-stash --moe-expert-rank-capacity-factor 2
--moe-paged-stash-buffer-size-factor-cuda 1.6 --moe-paged-stash-buffer-size-factor-cpu 0
--recompute-granularity selective --recompute-modules moe_act
--fp8-format e4m3 --fp8-recipe mxfp8 --fp8-param-gather --reuse-grad-buf-for-mxfp8-param-ag
--use-nccl-ub --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather
--mock-data --tokenizer-type NullTokenizer --vocab-size 500000
```

Overlap arm adds `--overlap-moe-expert-parallel-comm --delay-wgrad-compute`.

Relevant env (set by the inner launcher):

```
NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=64
USE_MNNVL=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,graph_capture_record_stream_reuse:True
NCCL_DEBUG=WARN
```

> `cuda_graph_impl = full_iteration` **is active.** The `enable_cuda_graph = False` in the config
> dump is a legacy flag that only exists to be translated into `cuda_graph_impl`
> (`arguments.py:593-612`) — it is not an indication that graphs are off.

---

## 4. Minimal reproduction

**M=3, GBS=384, MBS=1, 32 nodes (128 GPUs, DP=128), 5 iterations, profiler off.**
80× cheaper than the M=240 case originally reported, and highly deterministic.

```
GBS=384 TRAIN_ITERS=5 PROFILE_ENABLE=0 STASH_CUDA=1.6 PROFILE_VARIANT=1f1b
```

Expected: **exactly 1 iteration completes**, then `HYBRID-EP ALLGATHER TIMEOUT:SM n [0]:expecting
X got Y` lines begin and the job hangs until wall clock.

Reproduced identically in a same-allocation A/B (job 2802548): stock arm **125 mismatch lines**
(shortfall histogram 63×1, 62×2, across 30 SMs), matching the original run in a different
allocation weeks earlier.

> **Do not judge by mismatch line count.** A hung job emits those lines continuously until killed,
> so the count measures hang *duration*, not severity — the same arm was observed at 62, then 125,
> then 187, then 483. Read it as **zero vs non-zero**. The stable signal is the shortfall *size*
> (1–7) and the **iteration count** (must clear iteration 2).

---

## 5. Evidence

### 5.1 The cliff is exact

`combined_1f1b.py` runs `for i in range(num_microbatches - 1)`, so overlapped steps = M−1.

| M | GBS | overlapped steps | result |
|---|---|---|---|
| 1 | 128 | 0 | clean (−0.065%, parity) |
| 2 | 256 | 1 | clean (**+0.699%**, SE 0.269, t 2.60, p≈0.029, n=13) |
| 3 | 384 | 2 | **hangs** |
| 4, 8, 64, 240 | … | … | hangs |

### 5.2 cuda-gdb on a live hang (job 2810315, 32 dumps)

```
Kernel Parent Dev Grid   Status  GridDim  BlockDim  Invocation
*    0    -   0 228130   Active  (8,1,1)  (640,1,1) ncclDevKernel_AllGather_RING_LL()
```

- **30 / 32 nodes** show that as the resident Active kernel.
- GPU memory **283,2xx / 284,208 MiB = 99.6%** on every node, utilisation 100%.
- `ag_nvl_kernel` and `device_sync_kernel` appear only in backtraces, never as the resident
  kernel.

### 5.3 py-spy on the same hang (job 2811225, 32 dumps)

Ranks are in **different schedule phases**:

| frame | ranks |
|---|---|
| `default_backward_func` / `_backward` / `backward_impl` | **17** |
| `forward_impl` / `_forward` | **15** |
| `reload_paged_tensors` (paged_stash.py:526) | 8 |
| `paged_stash_group_commit` (paged_stash.py:878) | 8 |

Blocked frame is `b_layer.backward()` inside the overlapped step
(`model_chunk_schedule_plan.py:270` / `:565`). A collective cannot complete when participants sit
at different points.

### 5.4 The paged stash allocates ~80 GiB once, at iteration 2

`allocate_stash_buffers` appears **exactly 3 times per run, all immediately after iteration 1**:

| shape | dtype | size |
|---|---|---|
| `[7136896, 2048]` | uint8 | 13.6 GiB |
| `[6920640, 5120]` | bf16 | **66.0 GiB** |
| `[6920640, 1]` | fp32 | 0.03 GiB |
| | | **≈ 79.6 GiB (28% of the GPU)** |

This is a **lazy, one-time** allocation sized from iteration 1's statistics. It explains why
iteration 1 always completes and every failure mode lands at iteration 2.

### 5.5 The stash is a mitigation, not the leak

Removing `--moe-paged-stash` (job 2812232, config verified `moe_paged_stash ... False`) made
memory **worse**:

```
NCCL WARN Failed to CUDA call   include/alloc.h:497 (ncclCudaCallocAsyncDebug)
[rank89]: ncclUnhandledCudaError: Call to CUDA function failed.
```

NCCL could not allocate. Per #megatron-core-developments, the paged stash exists specifically to
**make MoE modules graphable under full-iteration CUDA graphs**; without it the routed-expert
activations live as ordinary tensors, which costs more.

> **Detection note:** NCCL reports OOM as `Failed to CUDA call` / `ncclUnhandledCudaError`, *not*
> "out of memory". Grepping for the latter returns clean and is how this was missed for several
> hours. `nvidia-smi` occupancy is the reliable check; torch's `reserved` figure (203 GB) excludes
> CUDA-graph pools, NCCL buffers and the stash.

### 5.6 DSv3 control — the overlap itself is fine (job 2811756)

DeepSeek-V3 proxy, TP1/PP4/**VPP4**/EP64, 16 layers, seq 4096, GBS 512 → **M=8**, 64 nodes.
Config dump verified: `overlap_moe_expert_parallel_comm True`, `delay_wgrad_compute True`,
`virtual_pipeline_model_parallel_size 4`, `EFFECTIVE_CDMC=32`.

**5/5 iterations, 0 mismatches**, loss 13.21 → 11.86 → 6.49 → 4.44 → 2.14, steady state
~1.2–1.6 s/iter.

DSv3 differs from a55b on several axes at once — **no paged stash, no CUDA graphs**, bf16 vs
mxfp8, no GTP, 16 vs 54 layers, PP4 vs PP1 — so it narrows the cause to the recipe without
pinpointing the ingredient.

### 5.7 What the overlap actually does (M=2 nsys, job 113079, matched pair)

Identical launch counts in both arms — 1F1B changes *when* kernels run, not *what* runs.

| A2A overlaps with… | baseline | overlap |
|---|---|---|
| **GEMM (compute)** | **0.0%** | **37.6%** |
| AllGather | 8.0% | 25.6% |
| ReduceScatter | 34.8% | 42.0% |

Per-kernel, on identical launch counts: AllGather +11.1%, ReduceScatter +9.5%, A2A +7.3%,
`nvjet` GEMM **+22%**, `device_sync_kernel` +53%, `paged_stash_pop_kernel` **+122%**.

Σ kernel time +9.5% while wall time −1.3%; concurrency 1.38× → 1.53×.

**Interpretation:** overlap moves A2A from 0% to 37.6% hidden behind compute (the win), but also
raises comm-on-comm overlap, and A2A/AG/RS all contend for the same NVLink bandwidth. The GEMM
slowing 22% shows the contention includes **SM occupancy**, not just bandwidth. Net at M=2:
+0.699%.

---

## 6. Artifacts (all on cmh)

Staging: `/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/yxu1/gtp-profile-staging`
Runs:    `/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/yxu1/gtp-profile-runs`

| file | sha256 |
|---|---|
| `megatron-gtp-dense-ag-high-priority-v1-src-v2.tar.gz` | `93cf1ded…3489fb` |
| `te-main-f1d5f8-gtp-wheels-v5-fe126/…whl` | `dcf3d308…7499b630` |
| `cudnn-frontend-1.26.0/…whl` | `ee50df34…9fd9ca` |
| `gtp_hybridep_stash_cmh1.sbatch` (stock inner) | `aa66737f…82675de` |
| `gtp_hybridep_rootcause_cmh1.sbatch` | `1d63155e…eaeca6c6` |
| `gtp_dsv3_cmh1.sbatch` (DSv3 inner) | `9e63d776…639ed20dc` |
| `gtp_hybridep_stashgate_cmh1.sbatch` (STASH=0/1 gate) | `28812632…513eb8a6` |
| `gtp_hybridep_stash_cmh1_dispgate.sbatch` (DISPATCHER gate) | `b6c60d4b…3a71641a` |

Debug wrapper: `gtp_cudagdb_rank_wrapper.sh` — runs python as a **child, not `exec`**, because
yama `ptrace_scope=1` only permits tracing descendants; the wrapper must be the parent to attach
cuda-gdb / py-spy. Dumps `info cuda kernels` + `py-spy dump --locals` from LOCAL_RANK 0 on every
node after `GDB_DELAY_SEC`.

Hang snapshots: `gtp-profile-runs/cudagdb-2810315/` (32 files),
`gtp-profile-runs/ncclspy-2811225/` (32 files, with Python stacks).

M=2 nsys profiles (paired arms, matched capture windows):
`abba-113078-rep{1,2}-*/latest-*.nsys-rep`, `abba-113079-rep{1,2}-*/latest-*.nsys-rep`
(rank 0 only). SQLite exports in `gtp-profile-runs/m2-exposure/`.

---

## 7. Refuted — please don't re-run these

| hypothesis | test | result |
|---|---|---|
| staging-buffer race between overlapped steps | `comm.wait_stream(comp)`, CG on (2795066) / off (2797933) | 61 / 63 mismatches — inert. The count is read via **pinned host memory** and HybridEP is device-initiated, so a host `wait_stream` cannot order it |
| insufficient work queues | `CDMC=32`, verified `EFFECTIVE_CDMC=32` (2802283) | still hangs |
| uneven token counts across ranks | `moe_hybridep_pad_uneven_dispatch_inputs=1` (2798200, 2802548) | mismatches → 0 but **IMA** on `GTP_WEIGHT_REMAT_GROUP`; CG-independent |
| the custom NVLink allgather | `enable_custom_allgather=False` → NCCL AG (2805681, 2805631), marker verified on 127/126 of 128 ranks | **still hangs**, now silently |
| HybridEP's device barrier | `device_side_sync_{dispatch,combine}_api=False` (2806745), marker on 103 ranks | still hangs, allgather message returns |
| paged stash is the leak | `--moe-paged-stash` removed (2812232) | **worse** — NCCL alloc failure |
| memory headroom via `STASH_CUDA` 1.6→1.03 | (2810996) | **invalid lever** — that is the buffer *size* factor (avg-based, +3% headroom), not a memory dial; buffer shrank `[6920640,5120]`→`[4455168,5120]` and hit an IMA |
| shared per-dispatch dispatcher state | analysis | structurally identical at M=2, which is clean |

Only `CUDA_LAUNCH_BLOCKING=1` suppresses the hang (5/5 iterations) — blanket serialisation at
~100× cost. It **masks** the race; it is not a fix.

**Pattern worth internalising:** silencing one HybridEP handshake just moves the report to the
next one. They are all downstream detectors of a stuck NCCL call.

---

## 8. Gotchas that cost us time

- **`--virtual-pipeline-model-parallel-size` does not exist** in this Megatron. That string is the
  *config field* quoted in the assertion text. The CLI flags are
  `--num-virtual-stages-per-pipeline-rank` / `--num-layers-per-virtual-pipeline-stage`. Passing the
  wrong one exits argparse with **code 2 and only a usage dump** — no exception text, so
  error-greps find nothing.
- **EP overlap requires VPP when PP>1**: *"If enabling EP A2A overlap,
  virtual_pipeline_model_parallel_size must be specified when pipeline_model_parallel_size > 1."*
  This is why no functional test case combines PP>1 with `--overlap-moe-expert-parallel-comm`.
  a55b is PP1, so it never trips.
- **`NCCL_DEBUG` set in the driver is overridden** — the inner exports `NCCL_DEBUG=WARN` at line
  166. Set it in the `RANK_WRAPPER`, which runs last, per rank, inside the container.
- **Flag dependency chain**: `moe_paged_stash` **requires** `moe_expert_rank_capacity_factor`, and
  that requires `hybridep`/`ncclep`. So the a55b recipe **cannot use the deepep backend** without
  dismantling its memory strategy.
- **`sacct` is useless for job IDs in the 111k–113k range** — those are from cmh2 (decommissioned)
  and the IDs have been reused on cmh. Use the log files.
- **Hosts:** cmh only. hsg/dfw/aga are GB200 and cannot load the sm103a TE wheel; aga additionally
  fails on a cutlass/cuDNN-frontend skew. lyris has a `gb300` partition and is viable but needs
  interactive MFA.

---

## 9. Open questions

1. **Which allocation consumes the last ~0.4%?** The stash's 79.6 GiB is the proximate trigger,
   but the aggregate footprint (full-iteration CG pools including the optimizer, GTP weight-remat
   across 64 shards, mxfp8 param-gather buffers, NCCL buffers for **28 process groups**, live
   activations for M microbatches) is what makes it not fit.
2. **Does the stash buffer size scale with M?** Only M=3 sizes were captured; the M=2 runs are from
   the decommissioned cmh2. This determines whether the stash is a fixed large cost or an
   M-dependent one.
3. **Which of the 28 NCCL process groups owns the stuck AllGather?** `GTP_WEIGHT_REMAT_GROUP` is
   the prime suspect (it is the PG named in the padding-run IMA); `--overlap-param-gather` is the
   alternative. NCCL `SUBSYS=INIT,COLL` tracing at the wrapper level would settle it.
4. **Does DeepEP #682 help?** Untested against this hang. #682 both replaces the custom allgather
   with NCCL *and* "enhanced the program to default hang instead of silently going forward"
   (Gao Deng), so post-#682 hanging is partly *intended*. Testing needs a GB300 host plus a
   post-#682 image; shiqingf has `pyt26.04-temain4adad4c2-hybridep94a9f8f6-arm.sqsh` on aga.
5. **Is `STASH_CUDA=-1.1` (actual-max sizing) enough?** The sign convention is
   *positive = avg-based, negative = actual-max*; we run +1.6 (avg +60%). Actual-max +10% may be
   both safer and smaller. Job 2812592 queued.

---

## 10. Related Slack

- `#megatron-core-developments` `1784071007.468499` — Gao Deng: the timeout "might cause silent
  data corruption without hanging"; #682 changes it to hang loudly. On cause: *"Earlier we saw OOM
  which lead to this hybridep timeout… Might need to inspect the hang point."*
- `#megatron-core-developments` `1784226839.136269` — Gautham Kollu: paged stash is what makes MoE
  graphable under full-iteration CG; ~10% default overhead; falls back to non-graphed compute if
  buffers don't fit.
- `#swdl-dlalgo-michalm-staff` `1778173504.255929` — Jeremi Piotrowski: token-drop reruns correlate
  with failures (1253 vs 182, 6.9×); fix was raising `moe_expert_rank_capacity_factor`.
- `#gnani-internal` `1784622225.895709` — identical one-element mismatches fixed with token-level
  `moe_expert_capacity_factor=1.25` + `moe_pad_expert_input_to_capacity=true` (knobs we have never
  enabled; distinct from our per-rank `moe_expert_rank_capacity_factor=2`).
