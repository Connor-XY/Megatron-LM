# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.determinism_env import set_determinism_env_vars
from megatron.training import utils as training_utils
from megatron.training.determinism import apply_determinism_to_args


def _args(**overrides):
    values = {
        "cross_entropy_loss_fusion": False,
        "tp_comm_overlap": False,
        "use_megatron_fsdp": False,
        "use_distributed_optimizer": True,
        "num_distributed_optimizer_instances": 1,
        "ddp_average_in_collective": False,
        "ddp_reduce_scatter_with_fp32_accumulation": False,
        "moe_token_dispatcher_type": "alltoall",
        "moe_flex_dispatcher_backend": None,
        "moe_shared_expert_overlap": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_distributed_optimizer_uses_ordered_reduce_scatter(monkeypatch):
    args = _args()
    use_deterministic_algorithms = Mock()
    warn_rank_0 = Mock()
    monkeypatch.setattr(torch, "use_deterministic_algorithms", use_deterministic_algorithms)
    monkeypatch.setattr(training_utils, "warn_rank_0", warn_rank_0)

    apply_determinism_to_args(args)

    assert args.ddp_reduce_scatter_with_fp32_accumulation is True
    warn_rank_0.assert_called_once_with(
        "Enabling ddp_reduce_scatter_with_fp32_accumulation for deterministic mode."
    )
    use_deterministic_algorithms.assert_called_once_with(True)


@pytest.mark.parametrize(
    "overrides, message",
    [
        (
            {"num_distributed_optimizer_instances": 2},
            "does not support multiple optimizer instances",
        ),
        ({"ddp_average_in_collective": True}, "does not support --ddp-average-in-collective"),
    ],
)
def test_unsupported_distributed_optimizer_modes_fail_closed(monkeypatch, overrides, message):
    monkeypatch.setattr(torch, "use_deterministic_algorithms", Mock())

    with pytest.raises(AssertionError, match=message):
        apply_determinism_to_args(_args(**overrides))


def test_non_distributed_optimizer_does_not_enable_reduce_scatter(monkeypatch):
    args = _args(use_distributed_optimizer=False)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", Mock())

    apply_determinism_to_args(args)

    assert args.ddp_reduce_scatter_with_fp32_accumulation is False


def test_megatron_fsdp_fails_closed(monkeypatch):
    monkeypatch.setattr(torch, "use_deterministic_algorithms", Mock())

    with pytest.raises(AssertionError, match="does not support --use-megatron-fsdp"):
        apply_determinism_to_args(_args(use_megatron_fsdp=True))


def test_flex_dispatcher_switches_to_certified_alltoall(monkeypatch):
    args = _args(
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="hybridep",
        moe_shared_expert_overlap=True,
    )
    monkeypatch.setattr(torch, "use_deterministic_algorithms", Mock())
    warn_rank_0 = Mock()
    monkeypatch.setattr(training_utils, "warn_rank_0", warn_rank_0)

    apply_determinism_to_args(args)

    assert args.moe_token_dispatcher_type == "alltoall"
    assert args.moe_flex_dispatcher_backend is None
    assert args.moe_shared_expert_overlap is False
    warn_rank_0.assert_any_call(
        "Switching the uncertified flex MoE dispatcher to alltoall for deterministic mode."
    )


def test_transformer_config_rejects_deterministic_flex_dispatcher():
    with pytest.raises(ValueError, match="does not support the flex MoE dispatcher"):
        TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            num_moe_experts=2,
            moe_token_dispatcher_type="flex",
            deterministic_mode=True,
        )


def test_mamba_determinism_disables_cold_cache_autotuning(monkeypatch):
    monkeypatch.setenv("MAMBA_DETERMINISTIC", "0")
    monkeypatch.setenv("TRITON_CACHE_AUTOTUNING", "1")

    set_determinism_env_vars()

    assert os.environ["MAMBA_DETERMINISTIC"] == "1"
    assert os.environ["TRITON_CACHE_AUTOTUNING"] == "0"
