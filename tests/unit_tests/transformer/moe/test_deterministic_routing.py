# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.transformer.moe.ops.deterministic_routing import (
    HAVE_TRITON,
    deterministic_routing_probs_and_map,
)


@pytest.mark.internal
@pytest.mark.skipif(not HAVE_TRITON or not torch.cuda.is_available(), reason="CUDA Triton required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("num_experts, topk", [(256, 8), (512, 22)])
def test_fused_routing_matches_scatter(dtype, num_experts, topk):
    torch.manual_seed(1234)
    num_tokens = 64
    indices = torch.rand(num_tokens, num_experts, device="cuda").topk(topk, dim=1).indices
    candidate_probs = torch.randn(num_tokens, topk, device="cuda", dtype=dtype, requires_grad=True)
    reference_probs = candidate_probs.detach().clone().requires_grad_(True)
    grad_output = torch.randn(num_tokens, num_experts, device="cuda", dtype=dtype)

    candidate_dense, candidate_map = deterministic_routing_probs_and_map(
        candidate_probs, indices, num_experts
    )
    reference_dense = torch.zeros_like(candidate_dense).scatter(1, indices, reference_probs)
    reference_map = torch.zeros_like(candidate_map).scatter(1, indices, True)
    candidate_grad = torch.autograd.grad(candidate_dense, candidate_probs, grad_output)[0]
    reference_grad = torch.autograd.grad(reference_dense, reference_probs, grad_output)[0]

    assert torch.equal(candidate_dense, reference_dense)
    assert torch.equal(candidate_map, reference_map)
    assert torch.equal(candidate_grad, reference_grad)


def test_cpu_fallback():
    probs = torch.tensor([[0.25, 0.75]], requires_grad=True)
    indices = torch.tensor([[3, 1]])
    routing_probs, routing_map = deterministic_routing_probs_and_map(probs, indices, 4)
    routing_probs.sum().backward()

    assert torch.equal(routing_probs, torch.tensor([[0.0, 0.75, 0.0, 0.25]]))
    assert torch.equal(routing_map, torch.tensor([[False, True, False, True]]))
    assert torch.equal(probs.grad, torch.ones_like(probs))


def test_rejects_invalid_shapes():
    with pytest.raises(ValueError, match="matching 2D"):
        deterministic_routing_probs_and_map(torch.ones(2, 3), torch.ones(6, dtype=torch.long), 4)
    with pytest.raises(ValueError, match="topk <= num_experts"):
        deterministic_routing_probs_and_map(
            torch.ones(2, 3), torch.zeros(2, 3, dtype=torch.long), 2
        )


@pytest.mark.internal
@pytest.mark.skipif(not HAVE_TRITON or not torch.cuda.is_available(), reason="CUDA Triton required")
def test_fused_routing_cuda_graph_replay():
    torch.manual_seed(1234)
    num_tokens, num_experts, topk = 64, 512, 22
    indices = torch.rand(num_tokens, num_experts, device="cuda").topk(topk, dim=1).indices
    probs = torch.randn(num_tokens, topk, device="cuda", requires_grad=True)
    grad_output = torch.randn(num_tokens, num_experts, device="cuda")

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        warmup_probs = probs.detach().clone().requires_grad_(True)
        warmup_dense, _ = deterministic_routing_probs_and_map(warmup_probs, indices, num_experts)
        warmup_dense.backward(grad_output)
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    probs.grad = torch.zeros_like(probs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        probs.grad.zero_()
        routing_probs, routing_map = deterministic_routing_probs_and_map(
            probs, indices, num_experts
        )
        routing_probs.backward(grad_output)

    graph.replay()
    expected_probs = torch.zeros_like(routing_probs).scatter(1, indices, probs)
    expected_map = torch.zeros_like(routing_map).scatter(1, indices, True)
    expected_grad = grad_output.gather(1, indices)

    assert torch.equal(routing_probs, expected_probs)
    assert torch.equal(routing_map, expected_map)
    assert torch.equal(probs.grad, expected_grad)
