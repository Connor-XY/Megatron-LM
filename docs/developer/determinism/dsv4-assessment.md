---
orphan: true
---

# DeepSeek-V4 determinism — assessment and gaps

> **Audience:** developers scoping determinism for DeepSeek-V4. Terms are in the
> [glossary](./glossary.md); the branch map is in [training-path.md](./training-path.md);
> the per-op verdicts are in [op-catalog.md](./op-catalog.md). This page answers
> "can we do DSV4 determinism, how effective is it, and what does it cost?" and
> records what is still missing.

## 1. What DSV4 is, and what is on this branch

DeepSeek-V4 = the DeepSeek-V3 architecture (Multi-Latent Attention + fine-grained
MoE + Multi-Token Prediction) **plus DeepSeek Sparse Attention (DSA)**: a
lightning indexer scores tokens and top-k selection sparsifies core attention.
The production DSV4 stack additionally uses **CSA** (compressed sparse attention),
**hyper-connections (mHC)**, and **fused DSA kernels**.

| DSV4 component | On this branch (`yx/determinism-ws2`) | On `NVIDIA/dev` |
| --- | --- | --- |
| MLA + fine-grained MoE (the DSV3 core) | ✔ (certified — see DSV3 proxy) | ✔ |
| DSA (lightning indexer + top-k sparse attention) | ✔ **pure-torch unfused path only** (`experimental_attention_variant/dsa.py`) | ✔ + fused `dsa_kernels.py` |
| Fused DSA kernels | ✖ (absent) | ✔ (`--no-dsa-kernel-fusion` toggle) |
| CSA (compressed sparse attention) | ✖ (absent) | ✔ (`csa.py`, `deepseek_v4_hybrid_attention.py`) |
| Hyper-connections (mHC) | ✖ (absent) | ✔ (`transformer/hyper_connection.py`, `fusions/fused_mhc_kernels.py`) |
| MTP | ✔ (config/spec present) | ✔ |
| **Determinism suite** (`tests/unit_tests/determinism/`, tools, docs) | ✔ (this workstream) | ✖ **absent upstream** |

The two facts that shape everything below: **DSA is the only DSV4-specific
component present here**, and **the determinism suite exists only on this
branch**. Supporting "DSV4 determinism on dev" is therefore a *port both
directions* problem, not a duplication problem (§6).

## 2. Can we do DSV4 determinism? — yes, for the DSA surface present here

DSA's determinism-sensitive operations, all in
`megatron/core/transformer/experimental_attention_variant/dsa.py`, and their
verdicts:

| DSA op | Where | Determinism | Note |
| --- | --- | --- | --- |
| Top-k token selection | `fused_qk_topk_naive` (`:321`, `.topk`) | ✔ | Unique indices; the same `sorted` top-k determinism as MoE routing. |
| Sparse mask writes ×3 | `:214` (indexer-loss fwd), `:385` (manual bwd), `:950` (unfused attn) | ✔ | Unique-index `scatter_(-1, topk_indices, 0)` — collision-free, no accumulation. |
| Indexer KL-loss score reduction (TP) | `:234` (fwd), `:419` (bwd) | ◆▲ | Native `torch.distributed.all_reduce(group=tp)`. Bit-exact at **TP ≤ 2** (one addition per element, topology-independent); an **open cross-allocation gap at TP > 2** — same class as the other native TP floating reductions cataloged as gaps. |
| Attention / index score GEMM + softmax | `torch.bmm(..float())`, `F.softmax(..fp32)` (`:202/225/941/960`) | ✔ | fp32 accumulate; deterministic under the pinned cuBLAS workspace. |
| Indexer rotation | `rotate_activation` → `hadamard_transform` (`:46`) | ✔ | Orthogonal transform; deterministic whether the CUDA extension or a torch equivalent. Requires the `fast_hadamard_transform` package (see §5). |
| Indexer-loss gradient injection | `DSAIndexerLossAutoScaler` (`:579`), `FusedDSAIndexerLoss` (`:510`) | ✔ | Aux-loss-autoscaler pattern: the indexer-loss gradient is injected into the main backward whenever the model output's backward runs, so the loss path (including `:419`) is on the autograd graph even without summing the scalar into the training loss. |

There are **no atomics, `cumsum`, or `argsort`** in the DSA path — the surface is
narrower than the MoE hot zone. The only genuinely new determinism concern DSV4
adds over DSV3 is the indexer TP all-reduce at TP > 2 (above).

## 3. Coverage — existing vs added by this work

**Already present (do not duplicate):**
- `tests/unit_tests/determinism/correctness/test_dsa_paths.py` — op-level bit-exact
  coverage of the three `scatter_` mask sites, the indexer-loss manual backward,
  and the unfused sparse attention, at TP=1.
- The `op-catalog.md` DSA row (Inference/experimental section).

**Added by this work:**
- `tests/unit_tests/determinism/configs.py` — `deepseek_v4_base()` (DSV3 base +
  DSA indexer config), `DEEPSEEK_V4_CONFIGS`, `DEEPSEEK_V4_PARALLELISM_CONFIGS`.
- `tests/unit_tests/determinism/correctness/test_deepseek_v4_model.py` —
  **model-level** DSA determinism: a full `GPTModel` built through the same
  `get_transformer_block_with_experimental_attention_variant_spec` dispatch
  production uses, run twice under `BitExactRunner` across the EP / TP×EP cells
  (`ep2`, `ep4`, `tp2-ep2`, `tp2-ep4`). The `tp2-ep2` cell is the first
  distributed exercise of the DSA indexer TP all-reduce (§2). PP cells are
  excluded (DSA+PP hangs on this branch — §4/§6); skips cleanly where
  `fast_hadamard_transform` is absent.
- `tests/functional_tests/shell_test_utils/determinism/run_model_certification.sh` —
  a `dsv4` mode (DSV3 EP32 certificate args + DSA indexer flags) so the weekly
  32-GPU trace certificate can cover DSA at scale.

## 4. Effectiveness and performance

Method matches the rest of the suite: bit-exact `BitExactRunner` matrix for
effectiveness; paired det-vs-nondet nsys leaderboard for performance. Runs on
8×H100 (draco) with the real `fast_hadamard_transform` kernel.

- **Effectiveness (bit-exact matrix): confirmed — 4/4 EP cells pass.**
  `test_deepseek_v4_model.py` produced bit-identical outputs and gradients across
  `ep2`, `ep4`, `tp2-ep2`, and `tp2-ep4` — two full-model runs each, under
  deterministic mode. The `tp2-ep2` cell exercises the DSA indexer top-k
  selection, all three sparse-mask `scatter_` sites, the unfused sparse
  attention, **and** the indexer-loss TP all-reduce (bit-exact at TP=2 by the
  single-addition argument). This is the first model-level, distributed DSA
  determinism evidence; op-level coverage (`test_dsa_paths.py`, TP=1) already
  existed.
- **Scale certification — DSV4 is bit-identical at 32 GPUs (EP32).** The
  `dsv4` mode of `run_model_certification.sh` ran two independent 8-node ×
  4-GPU launches (TP1, EP32, one MoE + DSA layer, activation recompute,
  hierarchical fp32 DP reduction) and compared their structured traces with
  `certify_traces.py`: **0 divergences over 7,360 compared events**, recompute
  and collective surfaces balanced, no pending collectives. The DSV3 EP32
  certificate (no DSA) passes identically. This is the same machinery and scale
  as the existing weekly nemotron/DSV3 certificates.
  - *Cert-invariant note:* the DSV3 cert requires a
    `te.attention.backend.selected` trace event; DSA uses an **unfused torch
    attention** path and emits no TE-attention-backend event, so that one
    requirement is dropped for `dsv4` (it would otherwise fail a
    provably-bit-identical run on a missing event, not a divergence). All other
    invariants — MoE EP / router-bias / DP collective surfaces and the ordered
    fp32 DP reduction — are still required and pass.
- **DSA + pipeline parallelism does not run here (functional, not determinism).**
  The `pp2` / `pp2-vpp2` cells **hang** on NCCL P2P setup: DSA's
  `DSAIndexerLossLoggingHelper` on this branch does not negotiate per-stage
  tracker sizes across pipeline stages, so one stage issues a collective the
  other does not. Non-DSA proxies pass PP in the same runner, so this is a DSA
  bug (fixed by the dev DSA rewrite), not a determinism divergence. The proxy
  therefore excludes PP cells and sets `supports_pp=False`; re-enable after the
  dev tracker fix lands (§6).
- **Performance (DSA determinism overhead):** _PENDING — det-vs-nondet step-time
  ratio on the DSV4-precursor config (MLA + MoE + DSA, TP2×EP4), to be compared
  with the DSV3 (no-DSA) baseline of 1.25×. Expectation: DSA's unfused torch
  attention is compute-heavy in both modes, so its determinism delta should be
  small relative to the MoE contribution; the number establishes whether that
  holds._

## 5. Onboarding blocker — `fast_hadamard_transform`

DSA's indexer asserts the `fast_hadamard_transform` CUDA extension is installed
(`dsa.py:46`). It is **not** in the base `fw-final` CI image, and the PyPI sdist
is broken (missing `csrc/*.cpp`; the wheel URL 404s). It builds cleanly from the
GitHub source (`git+https://github.com/Dao-AILab/fast-hadamard-transform.git`)
**with `--no-deps`** — without that flag pip drags a second torch into the target
directory and the extension fails at import with an ABI `undefined symbol`. For
CI, the extension should be baked into the determinism/DSA image rather than
built per-job. The model test therefore uses `pytest.importorskip` so it runs the
real kernel where present and skips elsewhere instead of failing.

## 6. What is missing for "DSV4 determinism on dev"

Ordered by what blocks a credible DSV4 determinism claim:

1. **Port the determinism suite to `dev` first.** `NVIDIA/dev` has the full DSV4
   model but **no** `tests/unit_tests/determinism/`, tools, or docs. None of this
   determinism coverage exists upstream. The suite (and the `dsv4` cell) must land
   on `dev` — via the merged PR #5041 lineage plus this branch's additions —
   before DSV4-on-dev can be certified at all.
2. **mHC (hyper-connections) determinism — uncovered, and it has real risk.**
   The Sinkhorn-Knopp normalization runs an autotuned Triton kernel
   (`fused_mhc_kernels.py`, `@triton.autotune` over `num_warps`), so autotuner
   choice can change reduction order — the same cold-autotune hazard that caused
   the Nemotron Mamba divergence. Plus residual-stream `sum` reductions and a
   native↔Triton/cuTile backend switch. The real DSV4 recipe sets
   `use_fused_mhc: true` and `NVTE_ALLOW_NONDETERMINISTIC_ALGO=1` (runs
   non-deterministic). mHC needs the pin-config-early + fixed-backend treatment
   and its own bit-exact cell.
3. **CSA determinism — uncovered.** Compressed sparse attention and its CP-layout
   kernels are a new surface with no determinism analysis yet.
4. **Fused DSA kernels — uncertified.** This branch certifies the *unfused* torch
   DSA path. The dev fused `dsa_kernels.py` path (default on GB200) needs its own
   exactness comparison against the unfused reference before it can be trusted in
   deterministic mode.
5. **DSA + pipeline parallelism hangs on this branch (port the dev fix).** The
   `DSAIndexerLossLoggingHelper` does not negotiate per-stage tracker sizes
   across pipeline stages, so PP>1 deadlocks on NCCL P2P setup (confirmed:
   `pp2`/`pp2-vpp2` cells hang; non-DSA PP cells pass). The dev DSA rewrite adds
   PP/MTP-safe lazy tracker growth and cross-PP size negotiation. Until that
   lands, DSA determinism coverage is EP/TP-only, and the weekly DSV4 certificate
   (TP=1, EP32) sidesteps it but does not certify DSA under PP.
6. **DSA indexer TP all-reduce at TP > 2** — native `all_reduce` (§2); route
   through the ordered deterministic-collective path or document it as a bounded
   gap. Neither current certificate reaches it (the EP32 cert uses TP=1; the
   proxy `tp2-ep2` cell is bit-exact by the single-addition argument only).
6. **A DSV4 functional/certification recipe with the real image.** The `dsv4`
   certification mode added here needs a GB200 recipe entry and an image that
   carries `fast_hadamard_transform` (§5).

## 7. Bottom line

DSV4 determinism is **tractable for the DSA surface** and this branch now has
op-level, model-level, and (scaffolded) scale-level coverage for it. The DSA path
is narrow and mostly collision-free, so the incremental determinism risk over
DSV3 is small and well-localized (the TP > 2 indexer reduction). The **larger
work is not DSA at all** — it is (a) getting the determinism suite onto `dev`
where the full DSV4 model lives, and (b) covering **mHC and CSA**, which are the
components with genuinely new, higher-risk reductions.
