# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Deterministic collective helpers for small training statistics."""

import torch


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
