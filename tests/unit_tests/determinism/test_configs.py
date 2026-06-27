# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for determinism configuration helpers."""

from tests.unit_tests.determinism.configs import merge_moe_overrides, moe_overrides


def test_moe_overrides_enable_generic_topology():
    overrides = moe_overrides(tp=1, ep=2)

    assert overrides["num_moe_experts"] == 4
    assert overrides["moe_router_topk"] == 2
    assert overrides["expert_model_parallel_size"] == 2


def test_moe_overrides_preserve_model_topology():
    overrides = moe_overrides(tp=2, ep=4, enable_moe=False)

    assert overrides == {
        "sequence_parallel": True,
        "tensor_model_parallel_size": 2,
        "expert_model_parallel_size": 4,
    }


def test_merge_moe_overrides_preserves_specialized_topology():
    merged = merge_moe_overrides(
        {"num_moe_experts": 8, "moe_router_topk": 2},
        {"num_moe_experts": 16, "moe_router_topk": 8},
        tp=2,
        ep=4,
    )

    assert merged == {
        "num_moe_experts": 16,
        "moe_router_topk": 8,
        "sequence_parallel": True,
        "tensor_model_parallel_size": 2,
        "expert_model_parallel_size": 4,
    }
