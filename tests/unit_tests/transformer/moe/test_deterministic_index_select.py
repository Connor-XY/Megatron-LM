# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.transformer.moe.moe_utils import permute, unpermute
from megatron.core.transformer.moe.ops.deterministic_index_select import (
    HAVE_TRITON,
    deterministic_index_select,
    deterministic_unpermute,
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


@pytest.mark.skipif(not HAVE_TRITON, reason="Triton is required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("topk", [4, 8, 22])
@pytest.mark.parametrize("hidden_size", [2048, 2051])
def test_dropless_unpermute_matches_deterministic_index_add(dtype, topk, hidden_size):
    num_tokens = 64
    num_experts = 32
    torch.manual_seed(1234)

    scores = torch.rand(num_tokens, num_experts, device="cuda")
    expert_indices = torch.topk(scores, topk, dim=1, sorted=False).indices
    routing_map = torch.zeros(num_tokens, num_experts, device="cuda", dtype=torch.bool).scatter_(
        1, expert_indices, True
    )
    tokens = torch.randn(num_tokens, hidden_size, device="cuda", dtype=dtype)
    _, _, sorted_indices, _, _ = permute(tokens, routing_map, num_out_tokens=num_tokens * topk)
    reference_input = torch.randn(
        num_tokens * topk, hidden_size, device="cuda", dtype=dtype, requires_grad=True
    )
    candidate_input = reference_input.detach().clone().requires_grad_(True)
    grad_output = torch.randn(num_tokens, hidden_size, device="cuda", dtype=dtype)

    previous_deterministic_mode = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        reference_output = unpermute(reference_input, sorted_indices, restore_shape=tokens.shape)
        candidate_output = unpermute(
            candidate_input, sorted_indices, restore_shape=tokens.shape, dropless_topk=topk
        )
        reference_grad = torch.autograd.grad(reference_output, reference_input, grad_output)[0]
        candidate_grad = torch.autograd.grad(candidate_output, candidate_input, grad_output)[0]

        assert torch.equal(candidate_output, reference_output)
        assert torch.equal(candidate_grad, reference_grad)
    finally:
        torch.use_deterministic_algorithms(previous_deterministic_mode)


def test_deterministic_unpermute_cpu_fallback():
    input_ = torch.arange(12, dtype=torch.float32).view(6, 2).requires_grad_(True)
    indices = torch.tensor([0, 1, 2, 0, 1, 2])
    output = deterministic_unpermute(input_, indices, num_tokens=3, topk=2)
    output.sum().backward()

    expected = torch.stack((input_[0] + input_[3], input_[1] + input_[4], input_[2] + input_[5]))
    assert torch.equal(output, expected)
    assert torch.equal(input_.grad, torch.ones_like(input_))


def test_deterministic_unpermute_rejects_nonuniform_row_count():
    input_ = torch.randn(5, 4)
    indices = torch.arange(5)
    with pytest.raises(ValueError, match="Expected 6 input rows and indices"):
        deterministic_unpermute(input_, indices, num_tokens=3, topk=2)


def test_unpermute_keeps_index_add_fallback_for_small_shapes(mocker):
    fast_path = mocker.patch("megatron.core.transformer.moe.moe_utils.deterministic_unpermute")
    input_ = torch.arange(12, dtype=torch.float32).view(6, 2)
    indices = torch.tensor([0, 1, 2, 0, 1, 2])
    previous_deterministic_mode = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        output = unpermute(input_, indices, restore_shape=torch.Size((3, 2)), dropless_topk=2)
    finally:
        torch.use_deterministic_algorithms(previous_deterministic_mode)

    fast_path.assert_not_called()
    expected = torch.stack((input_[0] + input_[3], input_[1] + input_[4], input_[2] + input_[5]))
    assert torch.equal(output, expected)


@pytest.mark.internal
@pytest.mark.skipif(not HAVE_TRITON or not torch.cuda.is_available(), reason="CUDA Triton required")
def test_deterministic_unpermute_cuda_graph_replay():
    torch.manual_seed(1234)
    num_tokens, hidden_size, topk, num_experts = 64, 2048, 8, 32
    scores = torch.rand(num_tokens, num_experts, device="cuda")
    expert_indices = torch.topk(scores, topk, dim=1, sorted=False).indices
    routing_map = torch.zeros(num_tokens, num_experts, device="cuda", dtype=torch.bool).scatter_(
        1, expert_indices, True
    )
    tokens = torch.randn(num_tokens, hidden_size, device="cuda")
    _, _, sorted_indices, _, _ = permute(tokens, routing_map, num_out_tokens=num_tokens * topk)
    values = torch.randn(
        num_tokens * topk, hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    grad_output = torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        warmup_values = values.detach().clone().requires_grad_(True)
        warmup_output = deterministic_unpermute(
            warmup_values, sorted_indices, num_tokens=num_tokens, topk=topk
        )
        warmup_output.backward(grad_output)
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    values.grad = torch.zeros_like(values)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        values.grad.zero_()
        output = deterministic_unpermute(values, sorted_indices, num_tokens=num_tokens, topk=topk)
        output.backward(grad_output)

    graph.replay()
    previous_deterministic_mode = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        expected_output = torch.zeros_like(output).index_add_(0, sorted_indices, values)
    finally:
        torch.use_deterministic_algorithms(previous_deterministic_mode)
    expected_grad = grad_output.index_select(0, sorted_indices)

    assert torch.equal(output, expected_output)
    assert torch.equal(values.grad, expected_grad)
