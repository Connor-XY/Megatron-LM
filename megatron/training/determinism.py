# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Reusable helpers for enabling bit-exact-reproducible execution.

Mirrors the split that ``megatron-bridge`` uses for the same purpose:

* :func:`set_determinism_env_vars` — lightweight env setup, re-exported from
  :mod:`megatron.determinism_env`, that must happen BEFORE CUDA libraries and
  Triton kernel modules are imported. Equivalent to bridge's
  ``PerfEnvPlugin._set_determinism_env_vars`` (``scripts/performance/perf_plugins.py``).
* :func:`apply_determinism_to_args` — config-level overrides applied to a
  parsed ``args`` Namespace. Equivalent to bridge's
  ``apply_determinism_overrides`` (``recipes/utils/determinism_utils.py``)
  but works on the ``args`` produced by Megatron-LM's argparser.

Callers:

* CLI training (``pretrain_*.py``) goes through ``validate_args`` which
  calls :func:`apply_determinism_to_args` when ``--deterministic-mode`` is
  passed.
* Test suite (``tests/unit_tests/determinism/``) and standalone profiling
  scripts call :func:`set_determinism_env_vars` directly at import time —
  they don't have an ``args`` Namespace.
"""

from __future__ import annotations

import os

import torch

from megatron.determinism_env import set_determinism_env_vars


def apply_determinism_to_args(args) -> None:
    """Apply deterministic-mode overrides to a parsed-args Namespace.

    Idempotent. Performs (in this order):

    1. Asserts ``cross_entropy_loss_fusion`` is off (fused CE is non-deterministic).
    2. Sets env vars via :func:`set_determinism_env_vars`. A user-supplied
       cuBLAS/TE/NCCL value survives its setdefault; Mamba deterministic mode
       and cold-cache Triton autotuning are forced to safe values.
    3. Replaces uncertified fused MoE dispatch with the standard all-to-all dispatcher.
    4. Forces ``tp_comm_overlap=False`` (non-deterministic NCCL collectives).
    5. Uses rank-ordered FP32 accumulation for distributed-optimizer gradient reduction.
    6. Calls ``torch.use_deterministic_algorithms(True)``.

    The argument assertion runs FIRST so a malformed args Namespace fails
    fast without leaving the process in a half-deterministic state (env
    vars set but torch global state untouched).
    """
    # 1. Validate args first — direct attribute access so a malformed
    #    Namespace (missing the field entirely) fails loudly rather than
    #    silently passing the check.
    assert (
        not args.cross_entropy_loss_fusion
    ), "Cross Entropy Fusion is currently not deterministic."
    assert not args.use_megatron_fsdp, (
        "Deterministic mode does not support --use-megatron-fsdp: its gradient "
        "all-reduce/reduce-scatter path does not yet provide fixed-rank accumulation."
    )

    if args.moe_token_dispatcher_type == "flex":
        from megatron.training.utils import warn_rank_0

        warn_rank_0(
            "Switching the uncertified flex MoE dispatcher to alltoall for deterministic mode."
        )
        args.moe_token_dispatcher_type = "alltoall"
        args.moe_flex_dispatcher_backend = None
        args.moe_shared_expert_overlap = False

    # NB: ``--use-flash-attn`` is intentionally NOT rejected under
    # --deterministic-mode. FlashAttention is deterministic on supported
    # configs (modern TE + Hopper/Blackwell); the bit-exact correctness
    # suite covers it across recipes (see
    # ``tests/unit_tests/determinism/correctness/test_fp8_determinism.py``
    # and the bf16 parallelism matrix). Backend choice is left to TE's
    # default selection or to the launcher via ``NVTE_*_ATTN`` env vars.

    # 2. NCCL_ALGO sanity check on a launcher-supplied value. Tree is
    #    intentionally excluded: its intra-node chain reduction order is not
    #    user-controllable and multi-node inter-tree topology can vary
    #    across runs without a pinned topology file, so we cannot vouch
    #    for it as deterministic. ``^NVLS`` is kept because banning NVLS
    #    on hardware that exposes it is a legitimate user choice; the user
    #    is responsible for picking a fallback that's deterministic on
    #    their stack. Comma-separated lists are valid NCCL syntax.
    accepted_tokens = {"Ring", "CollnetDirect", "CollnetChain", "^NVLS"}
    nccl_algo = os.environ.get("NCCL_ALGO")
    if nccl_algo is not None:
        tokens = [t.strip() for t in nccl_algo.split(",") if t.strip()]
        assert tokens and all(t in accepted_tokens for t in tokens), (
            f"NCCL_ALGO={nccl_algo!r} must be a comma-separated subset of "
            f"{sorted(accepted_tokens)}."
        )

    # 3. Apply env vars. ``setdefault`` preserves any launcher-set value
    #    that just passed validation; the default of ``Ring`` only kicks
    #    in when nothing was set.
    set_determinism_env_vars()

    # 4. Override tp_comm_overlap. ``warn_rank_0`` (not print_rank_0) so the
    #    override is capturable via ``pytest.warns`` / ``-W error``.
    if args.tp_comm_overlap:
        # Lazy import — warn_rank_0 lives in training.utils which has heavier
        # dependencies than this module.
        from megatron.training.utils import warn_rank_0

        warn_rank_0("Disabling tp_comm_overlap for deterministic mode.")
        args.tp_comm_overlap = False

    # 5. NCCL Ring does not fix the physical ring across allocations. A different ring changes
    #    floating-point accumulation order even when every rank enters with identical gradients.
    #    The ordered all-to-all + local FP32 sum is the deterministic distributed-optimizer path.
    if args.use_distributed_optimizer:
        assert args.num_distributed_optimizer_instances == 1, (
            "Deterministic distributed-optimizer gradient reduction does not support "
            "multiple optimizer instances."
        )
        assert not args.ddp_average_in_collective, (
            "Deterministic distributed-optimizer gradient reduction does not support "
            "--ddp-average-in-collective."
        )
        if not args.ddp_reduce_scatter_with_fp32_accumulation:
            from megatron.training.utils import warn_rank_0

            warn_rank_0(
                "Enabling ddp_reduce_scatter_with_fp32_accumulation for deterministic mode."
            )
            args.ddp_reduce_scatter_with_fp32_accumulation = True

    # 6. Torch global state last — all assertions have already passed.
    torch.use_deterministic_algorithms(True)
    # Experimental profiling toggle: use_deterministic_algorithms(True) also zero-fills
    # uninitialized memory, which is a large share of the deterministic overhead. Skipping
    # the fill has stayed bit-exact in validation. Env-gated so the default is unchanged;
    # set DET_NO_FILL=1 to measure the fill-off path.
    if os.environ.get("DET_NO_FILL", "").startswith("1"):
        import torch.utils.deterministic as torch_det_utils

        if hasattr(torch_det_utils, "fill_uninitialized_memory"):
            # This is a settable bool flag, not a function.
            torch_det_utils.fill_uninitialized_memory = False
