# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.transformer.moe.moe_utils import permute
from megatron.core.transformer.moe.ops.deterministic_index_select import (
    HAVE_TRITON,
    deterministic_index_select,
)


@pytest.mark.skipif(not HAVE_TRITON, reason="Triton is required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("topk", [4, 6, 8])
@pytest.mark.parametrize("hidden_size", [2048, 2051])
def test_dropless_permute_backward_matches_deterministic_index_add(dtype, topk, hidden_size):
    num_tokens = 64
    num_experts = 8
    torch.manual_seed(1234)

    scores = torch.rand(num_tokens, num_experts, device="cuda")
    expert_indices = torch.topk(scores, topk, dim=1, sorted=False).indices
    routing_map = torch.zeros(num_tokens, num_experts, device="cuda", dtype=torch.bool).scatter_(
        1, expert_indices, True
    )
    grad_output = torch.randn(num_tokens * topk, hidden_size, device="cuda", dtype=dtype)

    previous_deterministic_mode = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        reference_input = torch.randn(
            num_tokens, hidden_size, device="cuda", dtype=dtype, requires_grad=True
        )
        candidate_input = reference_input.detach().clone().requires_grad_(True)

        reference_output, _, reference_indices, _, _ = permute(
            reference_input, routing_map, num_out_tokens=num_tokens * topk
        )
        candidate_output, _, candidate_indices, _, _ = permute(
            candidate_input, routing_map, num_out_tokens=num_tokens * topk, dropless_topk=topk
        )

        reference_grad = torch.autograd.grad(reference_output, reference_input, grad_output)[0]
        candidate_grad = torch.autograd.grad(candidate_output, candidate_input, grad_output)[0]

        assert torch.equal(reference_indices, candidate_indices)
        assert torch.equal(reference_output, candidate_output)
        assert torch.equal(reference_grad, candidate_grad)
    finally:
        torch.use_deterministic_algorithms(previous_deterministic_mode)


def test_cpu_fallback():
    input_ = torch.randn(3, 5, requires_grad=True)
    indices = torch.arange(3).repeat(4)
    output = deterministic_index_select(input_, indices, topk=4)
    output.sum().backward()

    assert output.shape == (12, 5)
    assert torch.equal(input_.grad, torch.full_like(input_, 4))


def test_rejects_nonuniform_index_count():
    input_ = torch.randn(3, 5, requires_grad=True)
    indices = torch.arange(3).repeat(3)
    with pytest.raises(ValueError, match="Expected 12 indices"):
        deterministic_index_select(input_, indices, topk=4)
