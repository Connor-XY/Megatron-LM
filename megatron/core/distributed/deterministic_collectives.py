# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Deterministic collective helpers for training statistics and gradients."""

import torch

_DEFAULT_MAX_CHUNK_BYTES = 64 * 1024 * 1024


def all_reduce_with_fp32_accumulation(
    tensor: torch.Tensor,
    op: torch.distributed.ReduceOp,
    group: torch.distributed.ProcessGroup | None,
    *,
    max_chunk_bytes: int = _DEFAULT_MAX_CHUNK_BYTES,
) -> torch.Tensor:
    """All-reduce a floating tensor in fixed logical-rank order with bounded workspace.

    Each chunk is reduce-scattered with an all-to-all, accumulated in fp32 in process-group
    rank order, and all-gathered. Unlike a native floating-point all-reduce, the sum order is
    independent of the physical NCCL topology. Chunking bounds temporary memory for model-sized
    gradients; the final chunk is zero-padded when its size is not divisible by the group size.

    Args:
        tensor: Floating-point tensor to reduce in place.
        op: ``SUM`` or ``AVG``.
        group: Process group whose logical rank order defines accumulation order.
        max_chunk_bytes: Approximate upper bound for each input/output communication chunk.

    Returns:
        The input ``tensor`` after the in-place reduction.
    """
    if op not in (torch.distributed.ReduceOp.SUM, torch.distributed.ReduceOp.AVG):
        raise ValueError("Only SUM and AVG are supported")
    if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"Unsupported dtype for fp32 accumulation: {tensor.dtype}")
    if max_chunk_bytes <= 0:
        raise ValueError("max_chunk_bytes must be positive")

    world_size = torch.distributed.get_world_size(group=group)
    if world_size == 1 or tensor.numel() == 0:
        return tensor

    flat_tensor = tensor.contiguous().view(-1)
    element_size = flat_tensor.element_size()
    chunk_numel = max(world_size, max_chunk_bytes // element_size)
    chunk_numel -= chunk_numel % world_size

    for start in range(0, flat_tensor.numel(), chunk_numel):
        valid_numel = min(chunk_numel, flat_tensor.numel() - start)
        padded_numel = ((valid_numel + world_size - 1) // world_size) * world_size
        input_chunk = flat_tensor.narrow(0, start, valid_numel)
        if padded_numel == valid_numel:
            send_chunk = input_chunk
        else:
            send_chunk = torch.zeros(
                padded_numel, dtype=flat_tensor.dtype, device=flat_tensor.device
            )
            send_chunk[:valid_numel].copy_(input_chunk)

        received = torch.empty_like(send_chunk)
        torch.distributed.all_to_all_single(received, send_chunk, group=group)
        reduced_shard = received.view(world_size, -1).sum(dim=0, dtype=torch.float32)
        if op == torch.distributed.ReduceOp.AVG:
            reduced_shard.div_(world_size)

        output_shard = reduced_shard.to(dtype=flat_tensor.dtype)
        gathered = torch.empty_like(send_chunk)
        torch.distributed.all_gather_into_tensor(gathered, output_shard, group=group)
        input_chunk.copy_(gathered[:valid_numel])

    if not tensor.is_contiguous():
        tensor.copy_(flat_tensor.view(tensor.shape))
    return tensor


def all_reduce_sum(
    tensor: torch.Tensor, group: torch.distributed.ProcessGroup | None
) -> torch.Tensor:
    """Sum ``tensor`` across ranks with fixed rank-order accumulation in deterministic mode.

    NCCL Ring fixes the collective algorithm but can choose a different physical ring across
    allocations. For floating-point sums that changes accumulation order. Gathering rank-ordered
    inputs and reducing them locally removes that topology dependency. This path is intended for
    small statistics such as gradient norms and reported losses, not model-sized tensors.
    """
    world_size = torch.distributed.get_world_size(group=group)
    if world_size == 1:
        return tensor

    if not torch.are_deterministic_algorithms_enabled():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=group)
        return tensor

    flat_tensor = tensor.contiguous().view(-1)
    gathered = torch.empty(
        world_size * flat_tensor.numel(), dtype=tensor.dtype, device=tensor.device
    )
    torch.distributed.all_gather_into_tensor(gathered, flat_tensor, group=group)
    tensor.copy_(gathered.view(world_size, -1).sum(dim=0).view_as(tensor))
    return tensor


def all_reduce_avg(
    tensor: torch.Tensor, group: torch.distributed.ProcessGroup | None
) -> torch.Tensor:
    """Average ``tensor`` across ranks with topology-independent accumulation.

    Keep the native AVG collective outside deterministic mode. In deterministic mode, use the
    same fixed rank layout as :func:`all_reduce_sum`, then divide once after the reduction.
    """
    if not torch.are_deterministic_algorithms_enabled():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.AVG, group=group)
        return tensor

    world_size = torch.distributed.get_world_size(group=group)
    all_reduce_sum(tensor, group=group)
    tensor.div_(world_size)
    return tensor
