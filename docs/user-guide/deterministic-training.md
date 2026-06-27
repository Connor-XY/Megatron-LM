<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Deterministic Training

Deterministic training guarantees that two runs with identical inputs produce identical outputs at every step. Useful for debugging regressions and for reproducibility studies.

Pass `--deterministic-mode` to any Megatron training entry point (e.g. `pretrain_gpt.py`):

```bash
python pretrain_gpt.py \
  --deterministic-mode \
  <other args ...>
```

When enabled, Megatron applies the env vars and config overrides below via `megatron.training.determinism.apply_determinism_to_args` (called from `validate_args`).

## Environment variables

The cuBLAS / Transformer Engine / NCCL defaults use `os.environ.setdefault`,
so a user-supplied value wins. The Mamba controls are forced to their safe
values because timing-selected cold-cache Triton configs are not reproducible.
All five must take effect before their libraries or kernel modules initialize;
`pretrain_hybrid.py` performs a lightweight argv bootstrap before importing
torch or the hybrid model, then `apply_determinism_to_args` repeats the setup
during validation as defense in depth.

| Variable | Value | Reason |
|---|---|---|
| `NCCL_ALGO` | `Ring` | Conservative algorithm choice; it avoids tree-order variability but does not by itself guarantee the same floating-point rank order across allocations |
| `NVTE_ALLOW_NONDETERMINISTIC_ALGO` | `0` | Forces Transformer Engine to use deterministic algorithms |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` | Disables cuBLAS heuristic workspace selection |
| `MAMBA_DETERMINISTIC` | `1` | Enables deterministic Mamba reduction workspaces |
| `TRITON_CACHE_AUTOTUNING` | `0` | Selects one fixed Triton config at import instead of timing all configs against a cold cache |

If you override `NCCL_ALGO`, the value must be a comma-separated subset of `{Ring, CollnetDirect, CollnetChain, ^NVLS}`. `Tree` is intentionally excluded: its intra-node chain reduction order is not user-controllable, and the inter-node tree topology can vary across runs without a pinned topology file. `^NVLS` is accepted (banning NVLS is a legitimate user choice on hardware that exposes it); the user is responsible for validating the fallback on their environment. Even `Ring` can map ranks to a different physical ring on a new allocation. Megatron therefore uses rank-ordered reductions for deterministic distributed-optimizer gradients and small floating-point statistics instead of treating `NCCL_ALGO` as a complete guarantee.

## Config overrides

Applied to the parsed `args` Namespace in `apply_determinism_to_args`:

| Flag | Behavior under `--deterministic-mode` |
|---|---|
| `--cross-entropy-loss-fusion` | Must be off (asserted; fused CE is non-deterministic) |
| `--tp-comm-overlap` | Forced off (the overlap path uses non-deterministic NCCL collectives) |
| `--ddp-reduce-scatter-with-fp32-accumulation` | Forced on with the distributed optimizer; rank-ordered all-to-all plus local fp32 accumulation removes allocation-topology-dependent reduction order |
| `--ddp-average-in-collective` | Must be off with the distributed optimizer; averaging is applied outside the ordered sum |
| `--num-distributed-optimizer-instances` | Must be 1; the multi-instance reduction path is not yet certified |
| `torch.use_deterministic_algorithms` | Set to `True` |

Flash attention is permitted: Transformer Engine's flash-attention backend is deterministic when `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` (see the [Transformer Engine docs](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html)).

Mamba training retains the fused memory-efficient path. Its deterministic
reduction workspaces are enabled with `MAMBA_DETERMINISTIC=1`, while disabling
cold-cache timing autotuning pins the Triton reduction tiling. This setup must
happen before importing the Mamba kernels; use the standard
`pretrain_hybrid.py --deterministic-mode` entry point or call
`megatron.determinism_env.set_determinism_env_vars()` equally early in a custom
launcher.

## Verifying determinism

The bit-exact correctness suite lives at `tests/unit_tests/determinism/correctness/`. It parametrizes over model presets (GPT-like, Llama-like, Hybrid/Mamba) × parallelism cells (TP, PP, VPP, EP, FSDP, and composites) and asserts that two runs of the same configuration produce bit-identical outputs and gradients. FP8 / FP4 recipes (`tensorwise`, `delayed`, `mxfp8`, `nvfp4`) are covered by `tests/unit_tests/determinism/correctness/test_fp8_determinism.py`; the Blackwell-only recipes are capability-skipped on Hopper.

The cost of `--deterministic-mode` is measured outside pytest by an nsys-driven per-NVTX-range breakdown: `tests/performance_tests/shell_test_utils/determinism/run_nsys_breakdown.sh` wraps any training entry point (e.g. `pretrain_gpt.py --profile`) under nsys for a det-vs-nondet comparison, and `tests/performance_tests/shell_test_utils/determinism/print_nsys_leaderboard.py` joins the two CSVs into a side-by-side table. The CI invocation lives at `tests/test_utils/recipes/h100/determinism-perf.yaml`.

For multi-node certification, enable tensor hashes for a bounded trace window in
two independent launches, then run `tools/determinism/certify_traces.py` over
the two trace roots. The certifier validates runtime state, coverage, recompute
identity, collective completion and hashes, ordered DP accumulation, and exact
semantic trace equality; it exits nonzero for either a missing invariant or a
numerical divergence.
