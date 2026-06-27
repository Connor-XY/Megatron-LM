# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Collision-free deterministic backward for vocab-parallel cross entropy."""

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
def _cross_entropy_backward(
    values_ptr,
    indices_ptr,
    updates_ptr,
    grad_output_ptr,
    grad_output_stride_0,
    grad_output_stride_1,
    numel,
    smoothing_update,
    num_columns: tl.constexpr,
    grad_output_columns: tl.constexpr,
    grad_output_is_contiguous: tl.constexpr,
    block_size: tl.constexpr,
    has_smoothing: tl.constexpr,
):
    """Apply the selected-class update and output-gradient scaling in one pass."""
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < numel
    rows = offsets // num_columns
    columns = offsets - rows * num_columns
    selected_columns = tl.load(indices_ptr + rows, mask=mask)
    selected_updates = tl.load(updates_ptr + rows, mask=mask)
    if grad_output_is_contiguous:
        grad_output_offsets = rows
    else:
        grad_output_rows = rows // grad_output_columns
        grad_output_cols = rows - grad_output_rows * grad_output_columns
        grad_output_offsets = (
            grad_output_rows * grad_output_stride_0 + grad_output_cols * grad_output_stride_1
        )
    grad_output = tl.load(grad_output_ptr + grad_output_offsets, mask=mask)
    values = tl.load(values_ptr + offsets, mask=mask)
    selected_values = (values - selected_updates).to(values_ptr.dtype.element_ty)
    values = tl.where(columns == selected_columns, selected_values, values)
    if has_smoothing:
        values = (values - smoothing_update).to(values_ptr.dtype.element_ty)
    values = (values * grad_output).to(values_ptr.dtype.element_ty)
    tl.store(values_ptr + offsets, values, mask=mask)


def deterministic_cross_entropy_backward_(
    values: torch.Tensor,
    indices: torch.Tensor,
    updates: torch.Tensor,
    grad_output: torch.Tensor,
    smoothing_update: float = 0.0,
) -> torch.Tensor:
    """Apply deterministic vocab-CE gradient updates in place.

    Vocab-parallel cross entropy updates exactly one class per token row before
    scaling the full row by ``grad_output``. The Triton path applies both steps
    in one dense pass, preserving the original arithmetic order while avoiding
    generic deterministic ``index_put_`` and a second elementwise kernel.
    Unsupported inputs retain the equivalent PyTorch operations.
    """
    if values.dim() != 2 or indices.dim() != 1 or updates.dim() != 1:
        raise ValueError(
            f"Expected 2D values and 1D indices/updates, got {values.shape=}, "
            f"{indices.shape=}, {updates.shape=}, and {grad_output.shape=}"
        )
    if (
        values.shape[0] != indices.numel()
        or indices.shape != updates.shape
        or indices.numel() != grad_output.numel()
    ):
        raise ValueError(
            f"Expected one index, update, and output gradient per row, got "
            f"{values.shape[0]} rows, {indices.numel()} indices, {updates.numel()} updates, "
            f"and {grad_output.numel()} output gradients"
        )

    supported_dtype = values.dtype in (torch.bfloat16, torch.float16, torch.float32)
    supported_index_dtype = indices.dtype in (torch.int32, torch.int64)
    supported_grad_output_layout = grad_output.is_contiguous() or grad_output.dim() == 2
    use_triton = (
        HAVE_TRITON
        and values.is_cuda
        and indices.device == values.device
        and updates.device == values.device
        and grad_output.device == values.device
        and values.is_contiguous()
        and indices.is_contiguous()
        and updates.is_contiguous()
        and values.shape[0] > 0
        and updates.dtype == values.dtype
        and grad_output.dtype == values.dtype
        and supported_dtype
        and supported_index_dtype
        and supported_grad_output_layout
    )
    if use_triton:
        # Loss masking/scaling can produce a strided [sequence, batch] gradient.
        # Load that common 2D layout directly so the fused path does not add a
        # per-backward allocation/copy. Other non-contiguous layouts retain the
        # generic PyTorch fallback below.
        if grad_output.dim() == 2:
            grad_output_columns = grad_output.shape[1]
            grad_output_stride_0, grad_output_stride_1 = grad_output.stride()
        else:
            grad_output_columns = 1
            grad_output_stride_0 = grad_output_stride_1 = 1
        block_size = 256
        _cross_entropy_backward[(triton.cdiv(values.numel(), block_size),)](
            values,
            indices,
            updates,
            grad_output,
            grad_output_stride_0,
            grad_output_stride_1,
            values.numel(),
            smoothing_update,
            num_columns=values.shape[1],
            grad_output_columns=grad_output_columns,
            grad_output_is_contiguous=grad_output.is_contiguous(),
            block_size=block_size,
            has_smoothing=smoothing_update != 0.0,
            num_warps=4,
        )
        return values

    rows = torch.arange(values.shape[0], device=values.device)
    values[rows, indices] -= updates
    if smoothing_update:
        values.sub_(smoothing_update)
    values.mul_(grad_output.reshape(-1).unsqueeze(dim=-1))
    return values


__all__ = ["HAVE_TRITON", "deterministic_cross_entropy_backward_"]
