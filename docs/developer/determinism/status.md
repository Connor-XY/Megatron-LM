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

## 4. Control plane — how determinism is turned on

### 4.1 Today (on `main`)

`--deterministic-mode` is handled inline in `validate_args`
(`megatron/training/arguments.py:1499-1508`):

1. Asserts `not use_flash_attn` — **flash-attn is forbidden** today.
2. Asserts `not cross_entropy_loss_fusion` — fused CE is non-deterministic.
3. Validates `NCCL_ALGO ∈ {Tree, Ring, CollnetDirect, CollnetChain, ^NVLS}`
   (the env var must already be exported by the launcher).
4. Calls `torch.use_deterministic_algorithms(True)`.

It does **not** set env vars for you and does **not** force `tp_comm_overlap` off.

The config flag is `model_parallel_config.py:153` `deterministic_mode: bool = False`,
threaded into `TransformerConfig`; library code reads either that flag or
`torch.are_deterministic_algorithms_enabled()`.

### 4.2 After PR #5041 (`megatron/training/determinism.py`)

PR #5041 extracts the logic into a reusable module (mirroring Megatron-Bridge) so
tests and profiling scripts can opt in without an `args` Namespace:

- **`set_determinism_env_vars()`** — `os.environ.setdefault` of:
  `NCCL_ALGO=Ring`, `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Uses `setdefault` so a launcher-exported
  value wins (these are captured by NCCL/cuBLAS/TE at *first use*, so they must be
  set before the first kernel).
- **`apply_determinism_to_args(args)`** — asserts CE fusion off, validates
  `NCCL_ALGO` (**now excludes `Tree`** — its reduction order is not
  user-controllable), forces `tp_comm_overlap=False` (with a `warn_rank_0`),
  calls `torch.use_deterministic_algorithms(True)`.

Two behavior changes vs. today worth flagging:

- **Flash-attn is now permitted** under `--deterministic-mode`. TE FlashAttention
  is deterministic on supported configs when `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`;
  the bit-exact suite covers it. (Old code rejected it outright.)
- **`tp_comm_overlap` is force-disabled** rather than left to the user.

## 5. Enforced limitations in deterministic mode

These features are currently **incompatible** with deterministic mode:

| Feature | Where enforced | Reason |
| --- | --- | --- |
| `cross_entropy_loss_fusion` | `arguments.py:1502` / `determinism.py` assert | Fused CE kernel is non-deterministic |
| `tp_comm_overlap` (async TP) | `determinism.py` (forces off) | Async NCCL collective ordering varies |
| Flash-attn (today only) | `arguments.py:1501` assert | Relaxed by PR #5041 (TE FA is deterministic) |
| Packed sequence (`thd`) in gated-delta-net | `ssm/gated_delta_net.py:314` assert | No deterministic packed-seq SSM path |

> **Open question (tracked in the roadmap docs):** it is not fully established
> *whether* CE-fusion and tp_comm_overlap can be made deterministic. Resolving
> this is part of Workstream 3 (perf) — lifting a limitation is often cheaper than
> a slow fallback.

## 6. Surface area — the det/non-det branches that exist

The kernel-level determinism surface in `megatron/core` is **small**: there are
only ~5 `torch.are_deterministic_algorithms_enabled()` call sites plus a handful
of `config.deterministic_mode` branches. They are fully enumerated in
[`op-catalog.md`](./op-catalog.md); the load-bearing ones:

- **MoE unpermute** (`moe_utils.py:517`): `index_add_` (det, CUDA-graph safe) vs
  `scatter_add_` (fast).
- **MoE routing map/probs** (`moe_utils.py:823`): `index_put_(accumulate=False)`
  vs `scatter`.
- **Vocab embedding fwd** (`tensor_parallel/layers.py:299`): direct `weight[idx]`
  (det backward) vs `F.embedding` (non-det backward).
- **Mamba/SSM** (`ssm/ops/determinism.py`, `ssm/gated_delta_net.py:213/314/430`):
  cheapest-autotune + tiled-workspace reduction; torch `chunk_gated_delta_rule` /
  `F.conv1d` fallbacks; packed-seq forbidden.
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
  recompute, and backward.
  `tools/determinism/compare_traces.py` aligns rank traces by semantic event
  identity instead of PP/VPP arrival order. AWS-DFW job `516965` passed all 21
  focused trace/dump tests on every GB200 rank, including synchronous,
  NCCL-stream, recomputed, and backward all-to-all paths; AWS-CMH job `696891`
  passed the same final suite on every GB300 rank. AWS-DFW job `516924` then ran
  two independent deterministic DSV3-style TP2×EP2 training launches with full
  activation recompute and matched all 984 events across 8 rank/iteration shards,
  including 16 checkpoint pairs and 288 collective events spanning all three
  execution phases. The base tracer was also validated on H100: Draco job
  `10430400` passed 19/19 focused tests per rank and job `10430403` matched all
  1,392 events across 16 shards, including 32 checkpoint pairs. Exact H100
  confirmation of the collective extension remains pending. Every completed
  model run reported byte-identical recompute outputs.
- **E2E full-recipe (WS2 Tier B — pending):** the real nemotron-3-ultra recipe
  lives in **Megatron-Bridge** (`zhiyul/nemotron-3-ultra-perf-recipe`); the
  weekly multi-node e2e + wandb dashboard is the remaining tier — it is the only
  way to reach **EP>16**, where DSV3 non-determinism empirically appears.

## 8. Known gaps (feeding the roadmap)

1. **No e2e determinism coverage** → Workstream 2: scaled CI-gating proxies in
   mcore (nemotron-3-ultra, DSV3) + full recipes in Megatron-Bridge (weekly).
2. **~15% perf overhead** concentrated in MoE scatter/unpermute, FlashAttention
   backward (FAG), grouped GEMM, and top-k radix sort → Workstream 3.
3. **First-divergence tooling still has uncovered runtime surfaces.**
   `compare_dumps.py` localizes existing activation/param/wgrad/dgrad dumps, and
   the structured runtime trace now covers phase ordering, Megatron recompute
   identity, optimizer boundary scalars, exact wgrad/parameter hashes, runtime
   library/env settings, allocator backend, and MoE EP all-to-all inputs and
   outputs. Pipeline P2P and TP/DP reduction/gather payloads, optimizer moment
   state, TE FP8/FP4 recompute, and actual selected kernel identities are still
   missing. DSV3's known EP>16 + PP/VPP divergence therefore still needs the
   scaled e2e recipe plus the remaining collective instrumentation to localize
   fully.
## 9. References

- Determinism roadmap & meeting notes (internal Google Docs).
- Customer technical reports: Meituan Longcat (deterministic FAG / ScatterAdd /
  grouped GEMM / fused GemmAdd), MSFT, DeepSeek-V4.
- Thinking Machines, "Defeating nondeterminism in LLM inference."
- PyTorch reproducibility & `torch.use_deterministic_algorithms` docs.
