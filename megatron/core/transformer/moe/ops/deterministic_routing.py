# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Deterministic construction of dense MoE routing tensors."""

from unittest.mock import MagicMock

import torch

from megatron.core.utils import null_decorator

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False
    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()


@triton.jit
def _routing_forward(
    probs_ptr,
    indices_ptr,
    routing_probs_ptr,
    routing_map_ptr,
    num_experts: tl.constexpr,
    topk: tl.constexpr,
    expert_block_size: tl.constexpr,
    topk_block_size: tl.constexpr,
):
    """Initialize one dense row and write its collision-free selected entries."""
    row = tl.program_id(0)
    expert_offsets = tl.arange(0, expert_block_size)
    expert_mask = expert_offsets < num_experts
    output_offsets = row * num_experts + expert_offsets
    tl.store(routing_probs_ptr + output_offsets, 0.0, mask=expert_mask)
    tl.store(routing_map_ptr + output_offsets, 0, mask=expert_mask)

    # Selected entries share the same program as initialization. Keep the stores ordered
    # without requiring a second kernel or a global synchronization.
    tl.debug_barrier()
    topk_offsets = tl.arange(0, topk_block_size)
    topk_mask = topk_offsets < topk
    input_offsets = row * topk + topk_offsets
    indices = tl.load(indices_ptr + input_offsets, mask=topk_mask)
    selected_probs = tl.load(probs_ptr + input_offsets, mask=topk_mask)
    selected_offsets = row * num_experts + indices
    tl.store(routing_probs_ptr + selected_offsets, selected_probs, mask=topk_mask)
    tl.store(routing_map_ptr + selected_offsets, 1, mask=topk_mask)


@triton.jit
def _routing_backward(
    grad_routing_probs_ptr,
    indices_ptr,
    grad_probs_ptr,
    num_experts: tl.constexpr,
    topk: tl.constexpr,
    topk_block_size: tl.constexpr,
):
    """Gather selected dense gradients without reductions or atomics."""
    row = tl.program_id(0)
    topk_offsets = tl.arange(0, topk_block_size)
    topk_mask = topk_offsets < topk
    output_offsets = row * topk + topk_offsets
    indices = tl.load(indices_ptr + output_offsets, mask=topk_mask)
    grad_offsets = row * num_experts + indices
    grad_probs = tl.load(grad_routing_probs_ptr + grad_offsets, mask=topk_mask)
    tl.store(grad_probs_ptr + output_offsets, grad_probs, mask=topk_mask)


class _DeterministicRouting(torch.autograd.Function):
    """Build dense routing probabilities and a routing map in one Triton kernel."""

    @staticmethod
    def forward(
        ctx, probs: torch.Tensor, indices: torch.Tensor, num_experts: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens, topk = probs.shape
        routing_probs = torch.empty(
            (num_tokens, num_experts), dtype=probs.dtype, device=probs.device
        )
        routing_map = torch.empty((num_tokens, num_experts), dtype=torch.bool, device=probs.device)
        expert_block_size = triton.next_power_of_2(num_experts)
        topk_block_size = triton.next_power_of_2(topk)
        num_warps = min(max(expert_block_size // 128, 1), 8)
        _routing_forward[(num_tokens,)](
            probs,
            indices,
            routing_probs,
            routing_map,
            num_experts=num_experts,
            topk=topk,
            expert_block_size=expert_block_size,
            topk_block_size=topk_block_size,
            num_warps=num_warps,
        )
        ctx.save_for_backward(indices)
        ctx.num_experts = num_experts
        ctx.mark_non_differentiable(routing_map)
        return routing_probs, routing_map

    @staticmethod
    def backward(ctx, grad_routing_probs: torch.Tensor, _grad_routing_map: None):
        (indices,) = ctx.saved_tensors
        num_tokens, topk = indices.shape
        grad_probs = torch.empty(
            (num_tokens, topk), dtype=grad_routing_probs.dtype, device=grad_routing_probs.device
        )
        topk_block_size = triton.next_power_of_2(topk)
        _routing_backward[(num_tokens,)](
            grad_routing_probs.contiguous(),
            indices,
            grad_probs,
            num_experts=ctx.num_experts,
            topk=topk,
            topk_block_size=topk_block_size,
            num_warps=1,
        )
        return grad_probs, None, None


def deterministic_routing_probs_and_map(
    probs: torch.Tensor, indices: torch.Tensor, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build deterministic dense routing tensors from unique top-k entries.

    Each row in ``indices`` must contain unique values in ``[0, num_experts)``. MoE
    top-k routing guarantees this invariant, so the fused CUDA path has no write
    collisions and its backward is a gather. Unsupported inputs use PyTorch scatter.

    Args:
        probs: Selected routing probabilities shaped ``[num_tokens, topk]``.
        indices: Selected expert indices shaped ``[num_tokens, topk]``.
        num_experts: Width of the dense routing tensors.

    Returns:
        Dense routing probabilities and a boolean routing map, each shaped
        ``[num_tokens, num_experts]``.
    """
    if probs.dim() != 2 or indices.dim() != 2 or probs.shape != indices.shape:
        raise ValueError(
            f"Expected matching 2D probabilities and indices, got {probs.shape=} and "
            f"{indices.shape=}"
        )
    if probs.shape[1] < 1 or num_experts < probs.shape[1]:
        raise ValueError(
            f"Expected 1 <= topk <= num_experts, got topk={probs.shape[1]} and "
            f"num_experts={num_experts}"
        )

    supported_dtype = probs.dtype in (torch.bfloat16, torch.float16, torch.float32)
    supported_index_dtype = indices.dtype in (torch.int32, torch.int64)
    use_triton = (
        HAVE_TRITON
        and probs.is_cuda
        and indices.device == probs.device
        and probs.is_contiguous()
        and indices.is_contiguous()
        and probs.shape[0] > 0
        and supported_dtype
        and supported_index_dtype
    )
    if use_triton:
        return _DeterministicRouting.apply(probs, indices, num_experts)

    routing_probs = probs.new_zeros((probs.shape[0], num_experts))
    routing_probs.scatter_(1, indices, probs)
    routing_map = torch.zeros((probs.shape[0], num_experts), dtype=torch.bool, device=probs.device)
    routing_map.scatter_(1, indices, True)
    return routing_probs, routing_map


__all__ = ["HAVE_TRITON", "deterministic_routing_probs_and_map"]
