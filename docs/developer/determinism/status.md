# Determinism in Megatron-LM — Current Status

> **Audience:** Megatron-LM / Megatron-Core developers working on deterministic
> training. For the *user-facing* "how do I turn it on" guide, see
> `docs/user-guide/deterministic-training.md` (added by PR #5041).
>
> **Companion docs in this directory**
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
reduction (atomic adds, multi-block reductions, async collectives whose
completion order varies, allocator-driven kernel autotuning) can produce
different round-off between runs. Determinism therefore means pinning reduction
order everywhere it matters.

## 3. Targets and motivation

Customers are moving determinism from a debug/CI nicety to a **production / hero-run
requirement** (deterministic replay of loss spikes, checkpoint-resume that follows
the original trajectory, and a proof point that the GPU stack produces correct
math — mitigating SDC/SDE concerns). The blocker to enabling it by default is the
performance penalty.

| Milestone | Determinism overhead (step time) |
| --- | --- |
| Current baseline (mcore) | **~15%** overall; Nemotron-3-Ultra **+17.4%**, DSV3 **+8.5%** |
| Adoption target | **< 10%** (would let us default `deterministic_mode=True`) |
| Aggressive target | **≈ 5%** (per Meituan Longcat: deterministic FAG + ScatterAdd + grouped GEMM + fused GemmAdd) |

Profiling attributes most of the gap to a handful of ops — see
[`op-catalog.md`](./op-catalog.md) "Hotspots". The headline offenders from the
det-vs-nondet nsys leaderboard are `aten::fill_`, `aten::empty`,
`aten::index_put_`, `scatter_add`, and `cub::DeviceRadixSort` (MoE top-k path).
An earlier same-allocation Nemotron EP32 proxy measured **+10.2%** (79.9 vs
72.5 ms median): ordered fp32 DP reduction contributed +3.6%. With the
production hierarchy, job `520235` measures 55.4 ms for hierarchical-DP-only
versus 55.6 ms native and 57.8 ms flat-DP-only, closing the measured DP penalty.
Full deterministic mode with the hierarchy is 66.4 ms (**+19.4%**) on that
allocation, so the remaining and currently allocation-sensitive gap is in the
deterministic TE/MoE branches rather than Mamba or DP reduction. Direct
full-mode A/B job `520311` confirms that conclusion: flat deterministic is
66.5 ms and hierarchical deterministic is 66.7 ms on the same placement.
The routing `index_put_`→collision-free `scatter_` change improves
median-of-leg medians by 1.8% in direct ABBA job `520526`. Post-change
native/deterministic ABBA job `520625` measures 56.4/59.4 ms (+5.3%), reaching
the aggressive target on that allocation; its forward and reverse pairs are
+8.8% and +1.6%, so the result remains order/allocation-sensitive rather than a
production-wide guarantee. The follow-up fused collision-free routing kernel
removes generic deterministic scatter dispatch. Job `520763` measures
4.21–4.63× faster forward and 1.66–1.87× faster forward+backward across DSV3
and Nemotron shapes; direct production ABBA job `520786` improves
median-of-leg medians by another 2.0% (forward pair -4.8%, reverse pair +1.1%).
Post-fusion native/deterministic ABBA job `520827` still measures +15.2%
(56.1/64.65 ms) on a different allocation, with +14.5%/+16.0% order pairs.
Thus routing is improved and independently attributed, but the full target is
not consistently closed.

## 4. Control plane — how determinism is turned on

### 4.1 Upstream state

PR #5041 is closed and there is currently no determinism PR against
`NVIDIA/main`. This branch still contains the stacked implementation and tests
described below; none of them are an upstream guarantee.

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
  metrics) use rank-ordered all-gather plus a fixed local sum. `NCCL_ALGO=Ring`
  selects an algorithm but does not pin the physical ring across allocations.

Flash attention is permitted when Transformer Engine honors
`NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`; the bit-exact suite covers supported
configurations.

## 5. Enforced limitations in deterministic mode

These features are currently **incompatible** with deterministic mode:

| Feature | Where enforced | Reason |
| --- | --- | --- |
| `cross_entropy_loss_fusion` | `arguments.py:1502` / `determinism.py` assert | Fused CE kernel is non-deterministic |
| Flex MoE dispatcher (DeepEP/HybridEP) | `determinism.py` switches to standard all-to-all; `TransformerConfig` rejects any surviving flex config | External fused dispatch/combine kernels do not yet have a fixed-order cross-allocation certificate |
| `tp_comm_overlap` (async TP) | `determinism.py` (forces off) | Async NCCL collective ordering varies |
| Multiple distributed-optimizer instances | `determinism.py` assert | The cross-instance floating-point reduction is not ordered |
| `ddp_average_in_collective` | `determinism.py` assert | AVG must follow the rank-ordered SUM, not occur inside NCCL |
| Megatron-FSDP | `determinism.py` assert | Its native all-reduce/reduce-scatter path does not yet provide fixed-rank accumulation |
| Packed sequence (`thd`) in gated-delta-net | `ssm/gated_delta_net.py:314` assert | No deterministic packed-seq SSM path |

> **Open question (tracked in the roadmap docs):** it is not fully established
> *whether* CE-fusion and tp_comm_overlap can be made deterministic. Resolving
> this is part of Workstream 3 (perf) — lifting a limitation is often cheaper than
> a slow fallback.

## 6. Surface area — the det/non-det branches that exist

The kernel-level determinism surface in `megatron/core` is concentrated in a
small set of reductions and dispatch choices. They are fully enumerated in
[`op-catalog.md`](./op-catalog.md); the load-bearing ones:

- **MoE unpermute** (`moe_utils.py`,
  `moe/ops/deterministic_index_select.py`): dropless, unpadded deterministic A2A
  with top-k ≥4 and hidden ≥2,048 uses a fixed-order Triton segment sum with a
  collision-free gather backward;
  unsupported shapes retain deterministic `index_add_`, and normal mode uses
  `scatter_add_`. The optimized path is CUDA-graph safe and exactly preserves
  the former accumulation order. GB200 job `539787` measures a 1.63–4.48×
  isolated speedup with 42–47% lower peak allocation; H100 profile `10478093`
  passes the 1.25× step gate, and final EP32 job `539876` remains byte-exact
  against the pre-optimization DeepSeek-V3 and Nemotron traces.
- **MoE routing map/probs** (`moe_utils.py`,
  `moe/ops/deterministic_routing.py`): deterministic mode uses one row-wise
  Triton kernel to initialize dense probabilities and the boolean map and write
  unique top-k entries, with a selected-entry gather backward. It has no
  atomics or reductions. Unsupported devices, dtypes, layouts, or missing
  Triton retain collision-free in-place `scatter_`; normal mode keeps the
  out-of-place `scatter` path.
- **Flex MoE dispatch** (`transformer/moe/token_dispatcher.py`): DeepEP and
  HybridEP fuse permutation, communication, and combine inside external kernels
  whose token-arrival/reduction order is not controlled or traced by MCore.
  Deterministic CLI and Bridge overrides therefore select the certified standard
  all-to-all dispatcher, while direct configs that retain `flex` fail closed.
  AWS-CMH job `721485` passed the MCore guard on every one of four ranks plus
  both Bridge override/validator tests; job `721541` passed the target
  DeepSeek-V3 recipe switch (`COMPLETED 0:0`; pytest SHA256 `30bafb911e8c…`).
- **MoE top-k under activation checkpointing** (`moe_utils.py`): deterministic
  mode keeps `sorted=True` in both no-grad forward and grad-enabled recompute.
- **Router expert-bias counts** (`transformer/moe/router.py`,
  `transformer/moe/moe_utils.py`): int64 local accumulation plus an exact
  TP×CP×DP integer SUM and integer-space mean comparison prevent large-batch
  counts and near-tie bias decisions from crossing fp32's exact integer range;
  the existing collective is also visible in structured traces.
- **Vocab embedding fwd** (`tensor_parallel/layers.py:299`): direct `weight[idx]`
  (det backward) vs `F.embedding` (non-det backward).
- **Vocab-parallel CE backward** (`tensor_parallel/cross_entropy.py`,
  `tensor_parallel/deterministic_cross_entropy.py`): a collision-free Triton
  pass fuses the selected-class subtract, smoothing, and output scaling. It
  loads contiguous or 2D-strided loss gradients in logical row order; unusual
  layouts retain the equivalent PyTorch fallback.
- **Mamba/SSM** (`ssm/ops/determinism.py`, `ssm/mamba_mixer.py`,
  `ssm/gated_delta_net.py`): fixed Triton config + tiled-workspace reductions;
  the fast fused Mamba path remains enabled; gated-delta-net uses torch
  fallbacks and retains its packed-seq restriction.
- **DP and small-stat reductions** (`distributed/param_and_grad_buffer.py`,
  `distributed/deterministic_collectives.py`): rank-indexed all-to-all or
  all-gather followed by a fixed local fp32 sum.
- **Final TP/PP gradient synchronization**
  (`distributed/finalize_model_grads.py`,
  `distributed/deterministic_collectives.py`): deterministic floating SUM/AVG
  keeps native collectives for groups of at most two, where there is at most one
  addition per element, and uses bounded 64 MiB chunks with all-to-all, a fixed
  logical-rank fp32 sum, optional post-sum division, and all-gather for larger
  groups. The trace fingerprints exact inputs/outputs and records which path ran.
  Draco job `10477843` passed all 11 SUM/AVG, dtype, chunking, and trace cases on
  every H100 rank. Independent 16-GPU GB200 TP2×PP2×DP4 jobs `539519` and
  `539562` match 1,856/1,856 events across allocations and select the native
  two-rank exact path. Independent TP4×DP4 jobs `539611` and `539639` also match
  2,432/2,432 events across allocations and record 32 fixed-logical-rank
  final-gradient reductions per launch. Post-change EP32 jobs `539669` and
  `539674` preserve the full DeepSeek-V3 and Nemotron certificates across
  independent allocations: 6,720/6,720 and 10,304/10,304 events, respectively.
- **Megatron-FSDP gradient reduction**
  (`distributed/fsdp/src/megatron_fsdp/param_and_grad_buffer.py`) remains an
  unsupported cross-allocation path: it launches native SUM/AVG/premultiplied
  all-reduce and reduce-scatter from both synchronous and overlapped pipelines
  and does not consume the ordered-fp32 flag. The checked-in FSDP8 proxy cells
  only repeat within one initialized topology. This does not affect the Ultra
  production launcher, which uses the standard distributed optimizer. The
  fail-closed MCore and Bridge validation tests passed in AWS-CMH job `719072`
  (`COMPLETED 0:0`; console SHA256 `1cb105e7ecf0…`, evidence-manifest SHA256
  `47638ad5d151…`).
- **TE attention** (`extensions/transformer_engine.py`): asserts
  `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` when `deterministic_mode` is on, then TE
  filters Flash/Fused/Unfused backends by input-specific deterministic support.
  The opt-in structured trace records the backend/sub-backend selected after a
  real TE forward, plus configured selector, layer, forward/recompute phase,
  mask, QKV layout, and tensor metadata. A version-incompatible private
  diagnostic state emits `te.attention.backend.unavailable` instead of guessing.
  The backend must not be globally forced: dropout, layout, dtype, mask, and
  architecture change which deterministic kernels are eligible.
- **Inference/RL scheduling** (`dynamic_engine.py:607`,
  `data_parallel_inference_coordinator.py:181`, `rl/rl_utils.py:678`): sort by a
  stable key instead of completion order to remove timing jitter.

## 7. Validation status (tests)

- **Module-level (PR #5041, OPEN):** `tests/unit_tests/determinism/` —
  `BitExactRunner` runs a "cell" twice (restoring RNG between runs) and asserts
  bit-identical outputs **and** gradients across
  GPTModel × HybridModel × TransformerBlock, crossed with TP/EP/FSDP/PP/VPP
  composites and FP8 (tensorwise/delayed/mxfp8) / FP4 (nvfp4) recipes. Stress
  utilities `RacingStreams` and `CudaSleepJitter` perturb scheduling to surface
  latent collective-ordering races.
- **Perf gate (PR #5041, OPEN):**
  `tests/performance_tests/shell_test_utils/determinism/` runs `pretrain_gpt.py`
  det + nondet under nsys, prints a per-NVTX-range leaderboard
  (`print_nsys_leaderboard.py`), and **fails if det/nondet step-time ratio >
  1.25×**. Routed via `recipes/h100/determinism-perf.yaml`.
- **Model-level proxies (WS2 Tier A — added):** `BitExactRunner`-based proxies
  for the two target architectures:
  - `tests/unit_tests/determinism/correctness/test_deepseek_model.py` — DSV3-style
    MLA + fine-grained MoE with **group-limited routing** (the `group_limited_topk`
    path) + sigmoid/expert-bias + grouped GEMM.
  - `tests/unit_tests/determinism/correctness/test_nemotron_hybrid_model.py` —
    Nemotron-3-Ultra-style **MoE-inside-hybrid** (Mamba + attention + MoE) under EP.

  Both pass bit-exact across EP≤4 / TP / FSDP / PP / VPP (12/12 cells, draco
  8×H100). CI auto-discovers them via the `determinism/correctness/**/*.py` glob
  in `unit-tests.yaml`. **Note:** the branch pins `nvidia-resiliency-ext==0.6.0`;
  images older than that need an in-container upgrade to import mcore.
  The production-shape A2A presets are also closed at the full fixture width:
  Draco H100 job `10438654` passed all 12 DeepSeek top-k-8 and Nemotron top-k-6
  cells on every one of eight ranks (96/96 per-cell pass reports), including
  TP2×EP4, FSDP8×EP4, and pure FSDP8. The retained strict verification report
  has SHA256 `f808fe963e1c…`; the pytest log has SHA256 `24d2beab4b1e…`.
  The GPU test step is `COMPLETED 0:0`; the top-level batch status is `FAILED`
  only because its post-check regex omitted pytest's `33 deselected` field.
  The FSDP cells are same-process-group repeatability tests; Megatron-FSDP's
  native floating reductions still require an ordered implementation and an
  independent-allocation certificate.
- **Operator-level gaps (added):**
  `test_torch_norm.py` directly certifies PyTorch LayerNorm and RMSNorm backward,
  bypassing TE/Apex so the fallback is genuinely exercised. `test_dsa_paths.py`
  certifies all three DSA sparse-mask scatter sites through indexer-loss forward,
  recomputed manual backward, and unfused attention backward. AWS-DFW job
  `516499` passed 7/7 cases on every GB200 rank; Draco job `10429898` passed 7/7
  on every H100 rank.
- **Typed runtime trace (added):** `megatron/core/determinism_trace.py` writes a
  versioned, rank-sharded semantic event stream for selected iterations.
  `train_step` records forward/backward and optimizer boundaries plus optional
  exact named wgrad/updated-parameter hashes. An additional opt-in records local
  main parameters and direct tensor/scalar optimizer state before and after the
  step using stable optimizer/group/parameter ordinals, localizing Adam moment
  divergence without adding communication. Megatron and Transformer Engine
  activation checkpointing plus `CheckpointWithoutOutput` record paired
  forward/recompute fingerprints and report their within-run identity. The
  TE attention wrapper records the actual selected backend/sub-backend after
  each traced forward and emits a separate unavailable event when the installed
  TE diagnostic contract cannot be read. The shared MoE all-to-all wrapper
  records semantically named
  dispatch/combine inputs and outputs in original forward, activation
  recompute, and backward. Pipeline P2P tracing records sends and completed
  receives at the schedule's existing synchronous, batched, or overlapped wait
  boundary without adding communication or an extra wait. DP bucket tracing
  covers synchronous/overlapped all-reduce and reduce-scatter plus synchronous/
  overlapped distributed-optimizer parameter all-gather. Standard synchronous TP
  mapping collectives now fingerprint all-reduce and first/last-dimension
  all-gather/reduce-scatter payloads with explicit forward, recompute, and
  backward identities. Core TP linear tracing covers the synchronous forward
  gather and wraps backward async gather/all-reduce/reduce-scatter work so
  outputs are recorded at the existing waits. Rank-process visibility captures
  autograd-worker launches; `iteration.end.pending_collectives` reports
  operations that outlive the selected window instead of writing to a closed
  trace. End-of-backward finalization records deterministic TP SUM/AVG, PP
  embedding/replicated-parameter reductions, and token-count synchronization at
  their existing synchronous completion points, including implementation and
  process-group size. AWS-CMH job `719032` passed the
  focused real-NCCL trace test on all four GB200 ranks with every Slurm step
  `COMPLETED 0:0` (console SHA256 `32c8d9751876…`, evidence-manifest SHA256
  `022414bbd4c2…`).
  `tools/determinism/compare_traces.py` aligns rank traces by semantic event
  identity instead of PP/VPP arrival order. It now orders differences within a
  rank by the earliest recorded local sequence and reports both left/right
  sequence numbers, so lexical event names cannot hide an earlier causal
  boundary. AWS-CMH job `721836` passed all 17 comparator/certifier tests on
  each of four ranks with every Slurm step `COMPLETED 0:0` (console SHA256
  `b70cda722423…`, sbatch SHA256 `513c34d7c5f2…`). `certify_traces.py` additionally
  enforces rank/iteration coverage, deterministic runtime state, recompute
  identity, completed collective hashes, zero pending collectives, requested
  semantic event/collective surfaces, actual TE backend visibility, and ordered
  DP accumulation before comparing two trees. It
  rejects unsupported/incomplete schemas, mixed per-file identities, sequence
  gaps, malformed runtime/iteration boundaries, explicit iteration errors, and
  unmatched collective begin/end events. The hardened certifier replayed the
  retained weekly DSV3 and Nemotron trees without changing their verdicts:
  6,080/6,080 and 9,664/9,664 events remain exact, respectively. HSG job
  `3617620` passed all 16 final-source certifier/comparator tests (log SHA256
  `cfe1b0944ee6…`).
  The final 22-test focused suite
  passed on every rank in AWS-DFW GB200 job `517022`, AWS-CMH GB300 job `696943`,
  and Draco H100 job `10430851`. Two independent DSV3-style TP2×EP2 launches
  matched all 984 events on GB200 (`516924`) and all 1,968 events on H100
  (`10430682`), including exact all-to-all fingerprints in forward, recompute,
  and backward. AWS-DFW job `517040` then exercised PP2×VPP2×EP2 with overlapped
  P2P and matched all 2,320 events across 8 rank/iteration shards, including 384
  P2P boundary events. The expanded 29-test suite passed on every rank in GB200
  job `517130`, GB300 job `697037`, and H100 job `10431044`. AWS-DFW job `517136`
  matched 1,352/1,352 events across two distributed-optimizer runs, including 48
  DP boundary events and zero pending collectives. Every completed model run
  reported byte-identical recompute outputs. HSG job `3616865` passed the
  expanded 18-test trace file on every rank, including CUDA hashing of optimizer
  parameters and moments (log SHA256 `305373558100…`); final chained-optimizer
  coverage passed 2/2 focused tests per rank in job `3616898` (log SHA256
  `d9ff9c57175a…`). HSG job `3617018` then passed the focused synchronous TP
  collective trace on all four ranks (log SHA256 `7a02f8827841…`), and job
  `3617133` passed the final-source 20-test trace file on every rank (log SHA256
  `fcd4b95dcd5b…`). Two-node job `3617143` also passed all seven legacy TP
  mapping tests on all eight ranks (log SHA256 `68688420008b…`). The core TP
  linear focused test then passed on every rank in job `3617204` (log SHA256
  `a57bb228c848…`); job `3617239` passed the expanded 21-test trace file on all
  four ranks (log SHA256 `834bb84b0e2b…`), and job `3617240` passed all three
  legacy layer cases on all eight ranks (log SHA256 `f148bed8acc0…`). HSG job
  `3632592` passed the final 24-test trace file on all four ranks, including a
  real CUDA/TE autograd checkpoint and output-discarding recompute; every Slurm
  step completed `0:0` (log SHA256 `0cbe18944114…`, source-manifest SHA256
  `55657144c167…`).
- **Static sensitive-operation audit (added):**
  `tools/determinism/audit_sensitive_ops.py` parses the source without importing
  Megatron and emits stable file/function candidates for floating collective
  reductions, rank-indexed collectives, indexed reductions/writes, ordering
  operations, and determinism controls. It is deliberately a review queue, not
  an automatic verdict; the audit exposed the previously conflated
  Megatron-FSDP reduction surface. AWS-CMH job `719179` passed all three
  focused tests and audited a clean source snapshot into 318 candidates (82
  floating reductions, 79 rank-indexed collectives, 12 indexed reductions, 74
  indexed writes/gathers, 59 ordering calls, and 12 controls); every Slurm step
  completed `0:0` (report SHA256 `c06b713309ba…`, evidence-manifest SHA256
  `7c984304b982…`). Draco job `10476456` passes the expanded four-test suite
  and filters 40 known non-tensor name collisions from that same clean source
  surface: queue/NVSHMEM `put`, async `gather`, and module-level `embedding`.
  The resulting 278 candidates retain 82 floating reductions, 79 rank-indexed
  collectives, 3 indexed reductions, 43 indexed writes/gathers, 59 ordering
  calls, and 12 controls (report SHA256 `d16379e25326…`, manifest SHA256
  `9c55455b1f70…`).
  The final branch now supports `--git-tracked-only`, so local scratch files do
  not contaminate this inventory. At MCore `a5b284629`, the tracked report has
  284 candidates across 88 files: 82 floating reductions, 81 rank-indexed
  collectives, 4 indexed reductions, 44 indexed writes/gathers, 60 ordering
  calls, and 13 controls. The six additions since `d76d4b600` are exactly the
  fixed-order final-gradient all-to-all/all-gather plus its control branch and
  the fixed-order unpermute argsort/gather/fallback; both semantic operations
  are classified in the catalog above. AWS-DFW job `540456` passed all five
  focused scanner tests (`COMPLETED 0:0`; pytest SHA256 `2fdd72bfa913…`). The
  retained tracked report has SHA256 `7c27d7021da5…`. The catalog now records an
  explicit disposition for all 88 audited files and embeds the canonical
  284-operation fingerprint `bc95302609a9…`; `--verify-catalog` fails on a
  changed operation identity, count, file set, or missing disposition. Exact-source
  AWS-CMH job `721643` passed all six focused tests plus the full catalog gate,
  and every Slurm step completed `0:0` (console SHA256 `5fe2c3de94a5…`, sbatch
  SHA256 `1328269cbb30…`).
- **Strict external-recipe log gate (added):**
  `tools/determinism/compare_training_logs.py` compares every logged iteration
  and every nonvolatile serialized metric, requires loss and grad norm by
  default, and rejects missing or conflicting duplicate records. Its focused
  suite passed its original 8/8 tests in AWS-DFW job `522790`, then 9/9 in HSG
  job `3616185` and the final 10/10 in HSG job `3616466` (log SHA256
  `aa6f7fe42ea1…`). Applied to the retained weekly logs from `522695` and
  `522696`, it found two identical records for each of both iterations and
  matched all 20 nonvolatile values per model; report SHA256 values are
  `6bda80c6d699…` and `7e1cbe6f8142…`. This gate is intended for external
  loops that do not yet enter the structured-trace context; equal printed
  metrics remain weaker evidence than exact tensor hashes.
- **Rank-scoped Nsight profiling (added):**
  `tools/determinism/profile_rank.py` runs exactly one global `torchrun` rank
  under nsys while every peer executes the Python training command directly.
  It uses `--capture-range-end=stop` so ending the CUDA-profiler capture does
  not terminate the wrapped rank and strand its distributed peers. AWS-DFW job
  `520457` validated the launcher on the 32-GPU Nemotron topology and localized
  a measurable forward delta to the former explicit routing
  `index_put_`/`arange` path. Post-change job `520565` then showed that the
  residual aggregate `index_put_` time spans routing-probability scatter, vocab
  cross-entropy, and gather backward; it is not one monolithic routing cost.
  `tools/determinism/attribute_nsys_ranges.py` makes that SQLite containment
  analysis reusable for any NVTX range substring. Post-fusion job `520828`
  measures `_DeterministicRoutingBackward` at 0.376 ms over the capture and only
  eight residual `index_put_` ranges totaling 3.025 ms; containment assigns all
  eight to vocab cross-entropy or gather backward and none to routing.
  Experimental post-gather job `520964` removes both gather indexed writes:
  `_DeterministicSelectedGatherBackward` totals 0.290 ms, and all six residual
  `index_put_` ranges (1.551 ms) belong to vocab cross-entropy. The gather
  candidate was later removed after two negative production ABBAs. In the
  current-source profile `521407`, the fused CE backward removes both CE
  backward indexed writes and improves `_VocabParallelCrossEntropyBackward`
  from 1.257 to 0.749 ms (-40.4%). The six residual writes are four small CE
  forward masks and two gather-backward writes. The attribution tool now also
  canonicalizes sequence/operator IDs and aggregates large match sets by a
  selectable parent depth. On the same profile it reduces 802 `aten::fill_`
  events (16.767 ms) to ranked groups; at depth 2 the leading groups are
  `empty_like` (14.0%), `_to_copy` (12.0%), and `zeros` (8.6%), showing that the
  aggregate fill delta is distributed rather than one removable allocation.
  AWS-DFW job `522195` passes all six attribution-tool tests.
  Kernel attribution on the retained AWS-CMH `699916` SQLite pair then maps 32
  nested CE-backward ranges to 160/128 unique deterministic/native launches.
  It identifies `_cross_entropy_backward` only in deterministic mode and two
  index-elementwise kernel identities only in native mode; report SHA256 values
  are `9187b0e27912…` and `66ed6e4a937f…`. The final eight-test suite passes in
  HSG job `3632378` (`COMPLETED 0:0`; log SHA256 `2b5a417d25c4…`).
  Fresh dense profiles on HSG GB200 (`3614788`) and AWS-CMH GB300 (`699350`)
  then exposed a batch-2 vocab-CE fallback: deterministic CE backward was
  17.808 and 18.713 ms, respectively. With direct 2D-strided gradient loads,
  AWS-CMH job `699916` reduces it to 7.001 ms (-62.6% from the refreshed
  baseline); its 80 residual `index_put_` ranges belong to embeddings, CE
  forward masks, or generic index backward, with none under CE backward. The
  step-7 deterministic/native ratio is 1.15×, while self-attention and core
  attention remain the leading +22.743/+15.620 ms ranges.
  A TE 2.17 source/runtime audit (`699987`/`700020`) then separates selection
  from kernel cost. At dropout 0.1, deterministic fused attention is unavailable
  for the dense input, so auto correctly selects Flash versus native fused.
  Fixed-Flash job `699999` removes the forward/core-attention delta but leaves
  Flash backward +6.034 ms (+37.4%). At dropout 0, auto job `700030` selects
  fused attention in both modes; forcing it in job `700029` does not improve
  the paired ratio. Fused backward remains +3.740 ms (+24.7%), followed by
  LayerNormLinear backward +12.123 ms and compiled backward +7.666 ms over the
  capture. There is no safe branch-side backend override; the attention kernel
  gap is in TE.
- **Scaled MCore certification (WS2 Tier B):** AWS-DFW GB200 jobs `518849` and
  `518850` ran the final DSV3-style 32-GPU EP32 distributed-optimizer topology. Each
  two-launch comparison matched 6,080/6,080 semantic events, and the two
  independent allocations also matched exactly; reported LM loss, sequence
  auxiliary loss, and grad norm matched as well. The Slurm wrappers exited 1
  only after comparison because their temporary summary snippet referenced an
  undefined local variable; the retained comparison JSON and strict certifier
  both pass. Follow-up job `519887` extended the same topology to four optimizer
  updates per launch: the strict certifier matched 12,416/12,416 events, 128
  recomputes, and 4,096 collective boundaries with zero pending operations. Its
  wrapper likewise failed only in a stale post-check that expected 64 trace
  files instead of the correct 128; the comparison and certifier both pass.
  Nemotron-style hybrid EP32 jobs
  `518541` and `518542` kept the fused memory-efficient Mamba path and matched
  9,664/9,664 events within each allocation and
  across allocations, including 192 exact activation recomputes, 1,152 completed
  collective hashes, and zero pending collectives per trace tree. Their launcher
  did not export Mamba/Triton determinism variables, which also verifies the
  early `pretrain_hybrid.py` bootstrap. The final 48-test focused suite passed
  on every rank in AWS-DFW GB200 job `519253` and AWS-CMH GB300 job `697567`.
  HSG login timed out during both verification attempts.
- **Production hierarchical DP certification:** the fixed-logical-rank hierarchy
  is now wired through production DDP and exercised with asynchronous gradient
  overlap. DSV3 EP32 jobs `520170`/`520203` matched 6,080/6,080 events across
  independent allocations; Nemotron EP32 jobs `520171`/`520204` matched
  9,664/9,664. The strict reports require the hierarchical fp32 path for every
  DP reduction larger than two ranks and report zero missing reductions, zero
  pending collectives, and zero semantic divergences. Groups of at most two do
  not need a hierarchy because they contain at most one floating-point addition.
- **Small-group target-topology and TE-backend certification:** AWS-CMH jobs `723347`
  and `723417` independently certify the native two-rank reduce-scatter path at
  TP2×EP16. DSV3 matches 12,864/12,864 events and Nemotron matches
  17,984/17,984 both within and across allocations; group-2 reductions report
  `native_two_rank_exact`, while group-16 reductions retain the hierarchy.
  Primitive job `723271` matches native and ordered paths in all 288 randomized
  fp16/bf16/fp32 sync/async cells, and job `723241` measures native 1.56–1.89×
  faster at 8/64/160 MiB. Later exact-source trace job `723738` then matches
  13,120/13,120 DSV3 and 18,240/18,240 Nemotron events with 256 actual-backend
  events per run and zero unavailable events: DSV3 MLA selects unfused under
  auto, while Nemotron selects fused `NVTE_F16_arbitrary_seqlen`. AWS-DFW
  GB200 probe `544156` independently records that fused sub-backend under TE
  2.17. The TP1 target path also had large expert-DP groups of size one
  unnecessarily entering all-to-all plus fp32 sum. Job `723921` measures the
  no-op identity 23.5–42.4×
  faster on the 32–160 MiB target buckets. Cleaned final-source job `724148`
  then certifies DSV3 and Nemotron at TP1×EP32 with 6,848/6,848 and
  10,432/10,432 exact events and every expert reduction labeled
  `single_rank_identity` (certificate SHA256 `2e45875de123…` /
  `a6771a4d3f44…`). Final-source focused job `724146` passes all 36 CPU/CUDA,
  alias/copy, sync/async, dtype, and certifier cases per rank, including the
  fp16/bf16/fp32 native guard and a float64 regression that must retain explicit
  fp32 accumulation.
  Same-allocation DSV3 ABBA job `723987` improves median-of-leg medians from
  53.625 to 52.85 ms (-1.45%); both order pairs improve (2.72% and 0.19%), and
  all 12 serialized loss values match across all four legs.
- **Post-routing-optimization certification:** focused AWS-DFW job `520515`
  verifies bf16/fp32 × sigmoid/softmax outputs and gradients against the former
  `index_put_` reference on every rank; job `520692` extends the focused suite
  to the rank profiler and Nsight attribution tool (12 passed on every rank).
  Job `520569` reports 114 passed and 4 skipped on every rank for the expanded
  hierarchy suite. Candidate DSV3 job
  `520570` matches baseline allocation `520203` at 6,080/6,080 events, and
  candidate Nemotron job `520561` matches baseline allocation `520204` at
  9,664/9,664 events. Both cross-version reports have zero divergences, zero
  pending collectives, exact recompute coverage, and no missing hierarchical
  DP reductions.
- **Fused-routing certification:** AWS-DFW jobs `520748` and final-source rerun
  `520897` pass 39 focused tests on every rank, including exact DSV3/Nemotron
  forward, boolean-map, and backward comparisons plus CPU fallback and
  CUDA-graph replay. Job `520763`
  repeats exact hashes for all 12 bf16/fp32 model-shape cells. DSV3 job `520784`
  matches the prior scatter build (`520570`) at 6,080/6,080 events on an
  independent 32-GPU allocation; Nemotron job `520785` matches `520561` at
  9,664/9,664. Both reports have zero divergence, zero pending collectives,
  exact recompute coverage, and no missing hierarchical DP reductions.
- **Selected-score gather experiment (rejected):** AWS-DFW job `520936` passes 50
  focused tests on every rank, including exact DSV3/Nemotron outputs and
  gradients, routing integration, fallbacks, and CUDA-graph replay. Job
  `520946` repeats exact hashes in all 12 model-shape cells and measures
  1.48–2.23× forward+backward speedup. DSV3 job `520961` matches pre-change
  build `520784` at 6,080/6,080 events; Nemotron job `520962` matches `520785`
  at 9,664/9,664, with zero divergence, exact recomputes and collectives, and no
  missing hierarchical reductions. Despite that correctness and profiler win,
  gather-only production job `521026` regressed 3.3%, and combined gather+CE job
  `521127` regressed 3.2%; the gather candidate is therefore not applied.
- **Vocab-CE fused local-backward certification:** AWS-DFW job `521463` passes
  all 27 focused cases across bf16/fp32, label smoothing, batch 1/3, vocab
  128/257, fallback, invalid shapes, full CE integration, and CUDA graphs.
  Microbenchmark job `521310` is byte-exact in all 12 shape/smoothing cells and
  is 3.34–20.41× faster than the original selected update plus output scaling.
  DSV3 job `521405` matches pre-change build `520784` at 6,080/6,080 events;
  Nemotron job `521406` matches `520785` at 9,664/9,664, with zero divergence,
  exact recomputes and collectives, and no missing hierarchical reductions.
  Profile `521407` measures the CE backward at -0.508 ms (-40.4%) and finds no
  CE-backward indexed writes. Extended 50-step production ABBA `521404`
  improves the median-of-leg medians 66.475→64.675 ms (-2.7%); both order pairs
  improve and all loss, sequence-aux-loss, and grad-norm values match. The
  measured-slower forward `where` candidate is not applied.
- **Strided vocab-CE hardening:** refreshed dense profiles exposed that the
  training loss gradient is non-contiguous at micro-batch size 2, so the first
  fused implementation fell back to deterministic `index_put_`. HSG GB200 job
  `3614788` measures vocab-CE backward at 17.808/5.373 ms and AWS-CMH GB300 job
  `699350` at 18.713/5.183 ms; the latter contains 16 CE-backward indexed writes
  totaling 9.431 ms. The final kernel loads the common 2D-strided layout
  directly without an allocation or copy. AWS-DFW jobs `522493`/`522494` are
  byte-exact in all 12 benchmark cells and pass all 35 focused cases, including
  strided CUDA-graph replay and full batch-3 CE integration; the isolated
  speedup over the indexed-write fallback is 14.41–16.52×. A contiguous-copy
  experiment (`699565`) had a favorable median-of-leg result but contradictory
  +8.4%/-17.1% order pairs, so that allocation-adding design was not retained.
  The final stride-aware production ABBA (`699912`) improves both order pairs
  (-17.7%/-8.4%) and the median-of-leg medians 65.025→56.625 ms (-12.9%);
  all four 50-step loss and grad-norm sequences are identical.
  Final-source DSV3 EP32 job `522520` matches 6,080/6,080 semantic events with
  64 exact recomputes, 1,920 collective events, zero pending collectives, and
  no missing hierarchical DP reductions. Nemotron EP32 job `522521` likewise
  matches 9,664/9,664 events with 192 exact recomputes, 2,304 collective events,
  zero pending collectives, and no missing hierarchical reductions.
- **Checked-in weekly MCore gate:**
  `tests/functional_tests/shell_test_utils/determinism/run_model_certification.sh`
  now owns the two-run 32-GPU launch and strict certificate, while
  `tests/test_utils/recipes/gb200/determinism-certification.yaml` registers the
  DSV3 and Nemotron rows as weekly L3 GB200 workloads. Exact-runner AWS-DFW
  validation passed in jobs `522695` and `522696`. DSV3 matched 6,080/6,080
  events across a four-domain allocation (report SHA256 `9e887e0d821a…`);
  Nemotron matched 9,664/9,664 on a single NVL72 domain (report SHA256
  `fbb673910cfe…`). Both reports have exact recomputes, zero pending
  collectives, and zero multi-rank DP reductions missing hierarchical fp32
  accumulation. This gates the scaled MCore proxies; it is not a substitute for
  the full Megatron-Bridge recipe.
- **Exact router expert-bias counts and gate:** Draco H100 job `10475888`
  passes the large-count trace, finalize-time update, and production EP8 router
  cases on all eight ranks (console SHA256 `3d32a7971763…`, manifest SHA256
  `7762d0dc0f61…`). The near-tie case differs by one token after the all-reduce
  while both counts map to the same fp32 value, proving that the int64 SUM and
  integer-space mean comparison preserve the intended opposite bias updates.
  AWS-DFW GB200 job `539206` then runs the checked-in weekly DSV3 and Nemotron
  certificates sequentially on one 32-GPU allocation with the expert-bias
  trace prefix required. DSV3 matches 6,720/6,720 events and Nemotron matches
  10,304/10,304; each report contains 128 expert-bias collective events, exact
  recomputes, zero divergences or pending collectives, and no DP reductions
  missing ordered or hierarchical fp32 accumulation (report SHA256
  `1221e09002b9…` / `98a78a304897…`, manifest SHA256 `03ba6c436edb…`).
- **Vocab-embedding backward experiment (rejected):** AWS-DFW job `521744`
  compares the current deterministic direct-index backward with a stable-sort,
  fixed-order segmented reduction across 12 shape, dtype, and duplicate-token
  cells. Every output and weight gradient is byte-exact, but the candidate is
  only 0.61–0.69× as fast. The source remains `weight[masked_input]`; the older
  dense/Hopper `IndexBackward0` hotspot should be refreshed before another
  implementation is attempted.
- **Full Megatron-Bridge evidence (96-GPU qualification complete):** branch
  `zhiyul/nemotron-3-ultra-perf-recipe` records exact 96-GPU results across eight
  allocations, 5/7 exact 192-GPU trials, and a 3,072-GPU det+nsys versus
  det-without-nsys divergence from iteration 3. The latter lacks a same-mode
  control, and these historical raw logs were not independently available in
  this checkout. A current-MCore HSG smoke pair in job `3616252` then ran the
  full Nemotron-3-Ultra 550B TP2×PP3×EP32 recipe twice, sequentially on the same
  24-node/96-GPU allocation. Both launches reached all five iterations, and the
  corrected strict comparator matched all 60 nonvolatile serialized metric
  values exactly (report SHA256 `c8d0912328f3…`). The resolved configs differ
  only in `save_config_filepath`. Both Slurm training steps report
  `COMPLETED 0`, although `srun --wait=60` killed late cleanup ranks after the
  first rank exited; the retained harness should use at least 180 seconds. The
  top-level batch record is `FAILED 1` because the pre-fix comparator consumed
  appended rank-memory fields; rerunning the corrected comparator on the
  retained logs produced the exact report above. Current MCore also exposed a
  Bridge integration gap: its DDP config sets the logical hierarchical group
  size, but Bridge did not pass that value into
  `initialize_model_parallel`. Bridge commit `5793e80a` recovers the validated
  production source byte-for-byte, propagates that value, and creates the
  requested fixed logical hierarchy. HSG job `3632202` passed all 31 focused
  Bridge process-group tests, including forwarding and unsupported-MCore
  coverage (log SHA256 `061fb09951f5…`).

  Bridge commits `7ef03732` and `23e7fc8c` then close the remaining launcher
  control-plane mismatch. Deterministic config validation rejects Tree NCCL,
  multiple distributed-optimizer instances, and collective AVG; disables TP
  overlap; and enables ordered fp32 reduce-scatter. The performance environment
  also sets the two early Mamba controls. Both Ultra launchers now enter the
  common deterministic override path, use the fixed four-rank DP hierarchy,
  and disable Triton autotuning only in deterministic runs. AWS-CMH job
  `718578` passed 5 config-safety, 1 environment-plugin, 2 deterministic-recipe,
  and 4 dynamic launcher tests; every Slurm step completed `0:0` (log SHA256
  `7660df732eda…`, manifest SHA256 `8a37a8ef86a2…`). The dynamic launcher tests
  execute all four fake Slurm runs and prove the det/non-det config separation,
  artifact retention, same-mode failure gate, and nsys-only diagnostic policy.
  They also exposed and fixed the empty-array Bash 3.2 failure in the single-run
  launcher and collective AVG inherited by the deterministic Llama wrappers.
  Bridge commit `e09d3ef6` adds the separate allocation-internal certificate
  harness required by the scale gate. It executes two non-profiled runs
  sequentially, requires exact node/rank placement, normalizes and compares
  resolved configs, invokes the strict metric comparator, and retains source,
  allocation, placement, config, log, accounting, and hash evidence. Its four
  fail-closed contract cases pass on both macOS and AWS-CMH Linux.

  HSG job `3616801` then qualified the full recipe for 50 steps: two
  deterministic, non-profiled launches ran sequentially on the same
  24-node/96-GPU allocation and every Slurm record, including both hour-long
  training steps and the top-level batch, completed `0:0`. The strict comparator
  matched all 50 iterations and all 600 nonvolatile serialized metric values
  (12 per iteration; report SHA256 `2674e83f9f63…`). The configs are deeply
  identical after removing only `logger.save_config_filepath`; placement covers
  ranks 0–95 exactly, four per host across the retained 24-node list. Both runs
  report zero skipped and zero NaN iterations. Source revisions, deterministic
  environment, configs, logs, comparison, placement, and their hashes are
  retained under `bridge-ultra-current/pair-3616801/`. The only log noise is
  three/five TCPStore watchdog warnings during post-training process cleanup,
  after iteration 50; there are no rank errors or tracebacks. This is a clean
  qualification artifact, but the full Bridge recipe still needs a checked-in
  weekly schedule/dashboard and same-mode controls at 192 and 3,072 GPUs.

## 8. Known gaps (feeding the roadmap)

1. **Full Bridge evidence is not yet CI-gated.** The checked-in weekly L3 GB200
   rows retain strict EP32 MCore certificates for both model proxies, but the
   full Bridge recipe's clean 50-step same-allocation 96-GPU qualification is a
   retained cluster artifact, not yet a checked-in weekly schedule/dashboard.
   Bridge commit `5793e80a` propagates
   `reduce_scatter_hierarchical_group_size` into MCore process-group
   initialization; commits `7ef03732` and `23e7fc8c` align Bridge validation,
   environment setup, and the Ultra launchers with that certified MCore path,
   while `e09d3ef6` checks in the allocation-internal certificate harness. The
   remaining scale gate should use that harness to repeat the same two
   deterministic, non-profiled jobs sequentially in one allocation, compare all
   50 iterations with `compare_training_logs.py`, and retain both logs, both
   resolved configs, the comparison JSON, source revisions, rank-to-host
   placement, and hashes. Run that control weekly at 192 GPUs (48 GB200 nodes).
   At 3,072 GPUs (768 nodes), run the same control under a reservation before
   comparing nsys-on with nsys-off; instrumentation comparisons are diagnostics,
   not substitutes for the no-nsys/no-nsys reproducibility control.
2. **The production perf target is not consistently closed.** The production
   fixed-logical-rank hierarchy removes the measured DP penalty in job `520235`:
   hierarchical-DP-only is 55.4 ms versus 55.6 ms native and 57.8 ms flat
   ordered. It is exact across independent DSV3 and Nemotron allocations. Full
   deterministic mode was +16–19% in jobs `520235`/`520311`; the direct
   full-mode A/B was 66.5 ms flat versus 66.7 ms hierarchical. The routing
   scatter cleanup contributes 1.8% in direct old/new ABBA, and the fused
   collision-free kernel contributes another 2.0% in direct scatter/fused ABBA.
   Post-scatter native/deterministic job `520625` reached +5.3% on one allocation
   (56.4/59.4 ms), post-fusion job `520827` measured +15.2% on another
   (56.1/64.65 ms), and post-gather job `520963` measured +9.9% (54.85/60.3 ms)
   with +7.8%/+12.2% order pairs. Topology/allocation
   variance remains too large to declare the target closed globally. The
   exposed DP and routing improvements can be hidden by communication overlap.
   After the strided CE fix, the refreshed dense step-7 ratio is 1.15× with
   default dropout and 1.13× with dropout 0. The attention matrix rules out a
   global backend override: TE already chooses the input-compatible backend,
   while deterministic Flash/Fused backward, TE LayerNormLinear, and compiled
   BDA remain measured kernel costs. Expert-score gather and grouped GEMM remain
   MoE targets. Vocab-CE's local backward is closed, while TP collective
   ordering above two ranks remains open. The target TP2 distributed-optimizer
   path now uses native exact reduce-scatter and is 1.56–1.89× faster than the
   former ordered helper without weakening the two-allocation certificate. The
   TP1×EP32 expert-DP identity now avoids three unnecessary large reductions per
   rank/iteration, is 23.5–42.4× faster in the synchronized primitive gate, and
   improves same-allocation DSV3 median step time by 1.45% with both ABBA order
   pairs positive.
   Mamba's fixed config/workspaces show no measurable
   slowdown in the paired attribution. The
   hierarchy also increases peak allocated memory in this proxy by about 140
   MiB because it materializes the locally permuted send buffer. Each further
   optimization must retain the two-allocation certificate.
3. **First-divergence tooling still has uncovered runtime surfaces.**
   `compare_dumps.py` localizes existing activation/param/wgrad/dgrad dumps, and
   the structured runtime trace now covers phase ordering, Megatron recompute
   identity, optimizer boundary scalars, exact wgrad/parameter hashes, runtime
   library/env settings, allocator backend, and MoE EP all-to-all inputs and
   outputs, pipeline P2P sends/receives, DP gradient reduction, and distributed-
   optimizer parameter gather, synchronous TP mapping reduction/gather payloads,
   core TP linear synchronous/async collective payloads, and opt-in local
   optimizer parameters and moment state. Nsight SQLite attribution now maps
   selected NVTX ranges to the exact causally launched CUDA kernel identities,
   deduplicating launches covered by nested ranges. The runtime event now records
   installed TE/Triton/Mamba/FlashAttention versions and the TE-attention,
   Mamba, and Triton selection controls without importing optional packages;
   AWS-CMH job `722022` passed the full trace, comparator, and certifier suite
   (47 tests on each of four ranks; every Slurm step `COMPLETED 0:0`, console
   SHA256 `0232e9690b7b…`, sbatch SHA256 `ab8db5925dd9…`). TP
   userbuffer payloads, TE internal FP8/FP4 quantizer state, and selection state
   for auto-dispatching libraries other than TE attention are still missing.
   TE checkpoint inputs/outputs, forward/recompute identity, and the actual
   attention backend/sub-backend are now covered. Floating-point reductions
   inside TP mappings, TP linears, and
   vocab-parallel cross-entropy, plus the non-distributed-optimizer DP
   all-reduce, still lack topology-independent paths.

## 9. References

- Determinism roadmap & meeting notes (internal Google Docs).
- Customer technical reports: Meituan Longcat (deterministic FAG / ScatterAdd /
  grouped GEMM / fused GemmAdd), MSFT, DeepSeek-V4.
- Thinking Machines, "Defeating nondeterminism in LLM inference."
- PyTorch reproducibility & `torch.use_deterministic_algorithms` docs.
