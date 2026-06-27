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
  fusion, validates `NCCL_ALGO` (excluding `Tree`), forces
  `tp_comm_overlap=False`, and finally enables torch deterministic algorithms.
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
| `tp_comm_overlap` (async TP) | `determinism.py` (forces off) | Async NCCL collective ordering varies |
| Multiple distributed-optimizer instances | `determinism.py` assert | The cross-instance floating-point reduction is not ordered |
| `ddp_average_in_collective` | `determinism.py` assert | AVG must follow the rank-ordered SUM, not occur inside NCCL |
| Packed sequence (`thd`) in gated-delta-net | `ssm/gated_delta_net.py:314` assert | No deterministic packed-seq SSM path |

> **Open question (tracked in the roadmap docs):** it is not fully established
> *whether* CE-fusion and tp_comm_overlap can be made deterministic. Resolving
> this is part of Workstream 3 (perf) — lifting a limitation is often cheaper than
> a slow fallback.

## 6. Surface area — the det/non-det branches that exist

The kernel-level determinism surface in `megatron/core` is concentrated in a
small set of reductions and dispatch choices. They are fully enumerated in
[`op-catalog.md`](./op-catalog.md); the load-bearing ones:

- **MoE unpermute** (`moe_utils.py:517`): `index_add_` (det, CUDA-graph safe) vs
  `scatter_add_` (fast).
- **MoE routing map/probs** (`moe_utils.py`,
  `moe/ops/deterministic_routing.py`): deterministic mode uses one row-wise
  Triton kernel to initialize dense probabilities and the boolean map and write
  unique top-k entries, with a selected-entry gather backward. It has no
  atomics or reductions. Unsupported devices, dtypes, layouts, or missing
  Triton retain collision-free in-place `scatter_`; normal mode keeps the
  out-of-place `scatter` path.
- **MoE top-k under activation checkpointing** (`moe_utils.py`): deterministic
  mode keeps `sorted=True` in both no-grad forward and grad-enabled recompute.
- **Vocab embedding fwd** (`tensor_parallel/layers.py:299`): direct `weight[idx]`
  (det backward) vs `F.embedding` (non-det backward).
- **Mamba/SSM** (`ssm/ops/determinism.py`, `ssm/mamba_mixer.py`,
  `ssm/gated_delta_net.py`): fixed Triton config + tiled-workspace reductions;
  the fast fused Mamba path remains enabled; gated-delta-net uses torch
  fallbacks and retains its packed-seq restriction.
- **DP and small-stat reductions** (`distributed/param_and_grad_buffer.py`,
  `distributed/deterministic_collectives.py`): rank-indexed all-to-all or
  all-gather followed by a fixed local fp32 sum.
- **TE attention** (`extensions/transformer_engine.py:1697`): asserts
  `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` when `deterministic_mode` is on.
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
  exact named wgrad/updated-parameter hashes; Megatron activation checkpointing
  records paired forward/recompute fingerprints and reports their within-run
  identity. The shared MoE all-to-all wrapper records semantically named
  dispatch/combine inputs and outputs in original forward, activation
  recompute, and backward. Pipeline P2P tracing records sends and completed
  receives at the schedule's existing synchronous, batched, or overlapped wait
  boundary without adding communication or an extra wait. DP bucket tracing
  covers synchronous/overlapped all-reduce and reduce-scatter plus synchronous/
  overlapped distributed-optimizer parameter all-gather. Rank-process visibility
  captures autograd-worker launches; `iteration.end.pending_collectives` reports
  operations that outlive the selected window instead of writing to a closed trace.
  `tools/determinism/compare_traces.py` aligns rank traces by semantic event
  identity instead of PP/VPP arrival order. `certify_traces.py` additionally
  enforces rank/iteration coverage, deterministic runtime state, recompute
  identity, completed collective hashes, zero pending collectives, requested
  semantic surfaces, and ordered DP accumulation before comparing two trees.
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
  reported byte-identical recompute outputs.
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
  forward masks and two gather-backward writes.
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
  multi-rank DP reduction and report zero missing reductions, zero pending
  collectives, and zero semantic divergences. Single-rank expert-DP reductions
  remain on the flat path because they perform no inter-rank reduction.
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
- **Full Megatron-Bridge evidence (not yet a gate):** branch
  `zhiyul/nemotron-3-ultra-perf-recipe` records exact 96-GPU results across eight
  allocations, 5/7 exact 192-GPU trials, and a 3,072-GPU det+nsys versus
  det-without-nsys divergence from iteration 3. The latter lacks a same-mode
  control, and these historical raw logs were not independently available in
  this checkout. The remaining deliverable is a reproducible weekly gate and
  dashboard on the current MCore commit.

## 8. Known gaps (feeding the roadmap)

1. **Scaled evidence is not yet CI-gated.** The EP32 MCore certificates close the
   immediate coverage gap, but the full Bridge recipe still needs a weekly gate,
   retained artifacts, and a same-mode control at 192/3,072 GPUs.
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
   exposed DP and routing improvements can be hidden by communication overlap,
   which leaves expert-score gather backward, attention, and deterministic
   grouped GEMM as the primary measured kernel targets. Vocab-CE's local
   backward is closed, while TP collective ordering remains open. Mamba's fixed
   config/workspaces show no measurable slowdown in the paired attribution. The
   hierarchy also increases peak allocated memory in this proxy by about 140
   MiB because it materializes the locally permuted send buffer. Each further
   optimization must retain the two-allocation certificate.
3. **First-divergence tooling still has uncovered runtime surfaces.**
   `compare_dumps.py` localizes existing activation/param/wgrad/dgrad dumps, and
   the structured runtime trace now covers phase ordering, Megatron recompute
   identity, optimizer boundary scalars, exact wgrad/parameter hashes, runtime
   library/env settings, allocator backend, and MoE EP all-to-all inputs and
   outputs, pipeline P2P sends/receives, DP gradient reduction, and distributed-
   optimizer parameter gather. TP reduction/gather payloads, optimizer moment
   state, TE FP8/FP4 recompute, and actual selected kernel identities are still
   missing. Native floating-point TP reductions and the non-distributed-optimizer
   DP all-reduce also lack topology-independent paths.

## 9. References

- Determinism roadmap & meeting notes (internal Google Docs).
- Customer technical reports: Meituan Longcat (deterministic FAG / ScatterAdd /
  grouped GEMM / fused GemmAdd), MSFT, DeepSeek-V4.
- Thinking Machines, "Defeating nondeterminism in LLM inference."
- PyTorch reproducibility & `torch.use_deterministic_algorithms` docs.
