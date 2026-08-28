# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.transformer.moe import moe_utils
from megatron.core.transformer.moe.moe_utils import switch_load_balancing_loss_func


class _FusedAuxLossSpy:
    """Stands in for Transformer Engine's fused kernel and records whether it ran."""

    def __init__(self):
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        return torch.zeros((), dtype=torch.float32)


@pytest.fixture
def aux_loss_inputs():
    num_tokens, num_experts, topk = 16, 4, 2
    torch.manual_seed(1234)
    probs = torch.rand(num_tokens, num_experts, dtype=torch.float32)
    tokens_per_expert = torch.full((num_experts,), num_tokens * topk // num_experts)
    return probs, tokens_per_expert, num_tokens, topk, num_experts


@pytest.mark.parametrize("deterministic", [True, False])
def test_fused_aux_loss_is_skipped_under_deterministic_algorithms(
    monkeypatch, aux_loss_inputs, deterministic
):
    """The fused kernel reduces over tokens in an order the scheduler picks, so it is not
    bit-reproducible. Deterministic mode must take the torch reduction instead."""
    probs, tokens_per_expert, num_tokens, topk, num_experts = aux_loss_inputs
    spy = _FusedAuxLossSpy()
    monkeypatch.setattr(moe_utils, "HAVE_TE", True, raising=False)
    monkeypatch.setattr(moe_utils, "fused_moe_aux_loss", spy, raising=False)

    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(deterministic)
    try:
        loss = switch_load_balancing_loss_func(
            probs,
            tokens_per_expert,
            total_num_tokens=num_tokens,
            topk=topk,
            num_experts=num_experts,
            moe_aux_loss_coeff=1e-4,
            fused=True,
        )
    finally:
        torch.use_deterministic_algorithms(previous)

    if deterministic:
        assert spy.calls == 0, "deterministic mode must not dispatch to the fused kernel"
        expected = torch.sum(probs.sum(dim=0) * tokens_per_expert) * (
            num_experts * 1e-4 / (topk * num_tokens * num_tokens)
        )
        torch.testing.assert_close(loss, expected)
    else:
        assert spy.calls == 1, "non-deterministic mode should keep using the fused kernel"


def test_unfused_aux_loss_is_bitwise_stable_across_repeats(aux_loss_inputs):
    """Guards the path deterministic mode falls back to."""
    probs, tokens_per_expert, num_tokens, topk, num_experts = aux_loss_inputs
    losses = [
        switch_load_balancing_loss_func(
            probs,
            tokens_per_expert,
            total_num_tokens=num_tokens,
            topk=topk,
            num_experts=num_experts,
            moe_aux_loss_coeff=1e-4,
            fused=False,
        )
        for _ in range(3)
    ]
    for other in losses[1:]:
        assert torch.equal(losses[0], other)
