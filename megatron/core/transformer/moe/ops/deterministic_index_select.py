# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Deterministic backward for dropless MoE token permutation."""

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
def _fixed_order_index_select_backward(
    grad_output_ptr,
    inverse_indices_ptr,
    grad_input_ptr,
    hidden_size: tl.constexpr,
    topk: tl.constexpr,
    block_size: tl.constexpr,
):
    """Gather and sum token gradients in a fixed order."""
    token = tl.program_id(0)
    block = tl.program_id(1)
    columns = block * block_size + tl.arange(0, block_size)
    mask = columns < hidden_size

    row = tl.load(inverse_indices_ptr + token * topk)
    accumulator = tl.load(grad_output_ptr + row * hidden_size + columns, mask=mask)
    for index in tl.static_range(1, topk):
        row = tl.load(inverse_indices_ptr + token * topk + index)
        value = tl.load(grad_output_ptr + row * hidden_size + columns, mask=mask)
        # Match deterministic index_add's contribution order and rounding.
        accumulator = (accumulator + value).to(grad_input_ptr.dtype.element_ty)

    tl.store(grad_input_ptr + token * hidden_size + columns, accumulator, mask=mask)


class _DeterministicIndexSelect(torch.autograd.Function):
    """Index-select with a fixed-order dropless-MoE backward."""

    @staticmethod
    def forward(ctx, input_: torch.Tensor, indices: torch.Tensor, topk: int) -> torch.Tensor:
        ctx.input_shape = input_.shape
        ctx.topk = topk
        ctx.save_for_backward(torch.argsort(indices, stable=True))
        return input_.index_select(0, indices)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (inverse_indices,) = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input = torch.empty(
            ctx.input_shape, dtype=grad_output.dtype, device=grad_output.device
        )
        hidden_size = ctx.input_shape[1]
        block_size = 512
        grid = (ctx.input_shape[0], triton.cdiv(hidden_size, block_size))
        _fixed_order_index_select_backward[grid](
            grad_output,
            inverse_indices,
            grad_input,
            hidden_size=hidden_size,
            topk=ctx.topk,
            block_size=block_size,
            num_warps=4,
        )
        return grad_input, None, None


def deterministic_index_select(
    input_: torch.Tensor, indices: torch.Tensor, topk: int
) -> torch.Tensor:
    """Select repeated token rows with a deterministic fixed-order backward.

    This fast path is valid when every input row occurs exactly ``topk`` times
    in ``indices``, as it does for the first token permutation in dropless MoE.
    Unsupported environments retain PyTorch's regular ``index_select`` path.
    """
    if input_.dim() != 2 or indices.dim() != 1:
        raise ValueError(
            f"Expected a 2D input and 1D indices, got {input_.shape=} and {indices.shape=}"
        )
    if topk < 1:
        raise ValueError(f"Expected topk >= 1, got {topk}")

    expected_indices = input_.shape[0] * topk
    if indices.numel() != expected_indices:
        raise ValueError(
            f"Expected {expected_indices} indices for dropless topk={topk}, got {indices.numel()}"
        )

    supported_dtype = input_.dtype in (torch.bfloat16, torch.float16, torch.float32)
    if not (
        HAVE_TRITON
        and input_.is_cuda
        and input_.requires_grad
        and torch.is_grad_enabled()
        and supported_dtype
    ):
        return input_.index_select(0, indices)
    return _DeterministicIndexSelect.apply(input_, indices, topk)


__all__ = ["HAVE_TRITON", "deterministic_index_select"]
