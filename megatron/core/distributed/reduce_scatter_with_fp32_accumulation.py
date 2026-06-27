# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.


from typing import Any

import torch


class _ReduceScatterWithFP32AccumulationWorkHandle:
    """Work handle to return to user when using reduce_scatter_with_fp32_accumulation with
    async_op=True."""

    def __init__(
        self,
        all_to_all_handle: Any,
        all_to_all_output_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        world_size: int,
    ):
        """Initialize WorkHandle object."""
        self.all_to_all_handle = all_to_all_handle
        self.all_to_all_output_tensor = all_to_all_output_tensor
        self.output_tensor = output_tensor
        self.world_size = world_size

    def wait(self):
        """Wait until communication (and associated computation) is completed."""
        # Wait for communication to complete if needed.
        if self.all_to_all_handle is not None:
            self.all_to_all_handle.wait()

        # Accumulate into a fp32 sum.
        output_tensor_in_fp32 = torch.sum(
            self.all_to_all_output_tensor.view((self.world_size, -1)), dim=0, dtype=torch.float32
        )
        assert output_tensor_in_fp32.dtype == torch.float32

        # Copy downcasted sum into output_tensor.
        self.output_tensor.copy_(output_tensor_in_fp32)


class _HierarchicalReduceScatterWithFP32AccumulationWorkHandle:
    """Finish a two-level fixed-rank reduce-scatter after its local exchange."""

    def __init__(
        self,
        local_handle: Any,
        local_send: torch.Tensor,
        local_recv: torch.Tensor,
        output_tensor: torch.Tensor,
        local_group_size: int,
        inter_group: torch.distributed.ProcessGroup,
    ):
        self.local_handle = local_handle
        self.local_send = local_send
        self.local_recv = local_recv
        self.output_tensor = output_tensor
        self.local_group_size = local_group_size
        self.inter_group = inter_group

    def wait(self):
        """Complete the fixed local sum, inter-group exchange, and fixed final sum."""
        if self.local_recv is None:
            return
        if self.local_handle is not None:
            self.local_handle.wait()

        num_local_groups = self.inter_group.size()
        shard_numel = self.output_tensor.numel()
        local_partial = self.local_recv.view(
            self.local_group_size, num_local_groups, shard_numel
        ).sum(dim=0, dtype=torch.float32)

        inter_recv = torch.empty_like(local_partial)
        torch.distributed.all_to_all_single(
            output=inter_recv, input=local_partial, group=self.inter_group, async_op=False
        )
        output_tensor_in_fp32 = inter_recv.view(num_local_groups, shard_numel).sum(
            dim=0, dtype=torch.float32
        )
        self.output_tensor.copy_(output_tensor_in_fp32)

        self.local_handle = None
        self.local_send = None
        self.local_recv = None


def reduce_scatter_with_fp32_accumulation(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    op: torch.distributed.ReduceOp,
    group: torch.distributed.ProcessGroup,
    async_op: bool,
    hierarchical_groups: (
        tuple[torch.distributed.ProcessGroup, torch.distributed.ProcessGroup] | None
    ) = None,
):
    """Reduce-scatter with FP32 accumulation.

    Collects input_tensor in lower precision using an all-to-all, then locally accumulates in FP32
    precision, then downcasts the final sum into output_tensor.


    Args:
        output_tensor (torch.Tensor): Output tensor with reduce-scattered output (only the shard).
        input_tensor (torch.Tensor): Input tensor that needs to be reduce-scattered.
        op (torch.distributed.ReduceOp): Only torch.distributed.ReduceOp.SUM is supported.
        group (torch.distributed.ProcessGroup): Process group to use for reduce-scatter.
        async_op (bool): Return a work handle when true; complete before returning when false.
        hierarchical_groups (tuple, optional): Fixed local and inter groups whose sizes multiply
            to the parent group size. When provided, reduce locally first, exchange one fp32
            partial per destination over the inter group, and finish with a fixed group-order sum.
    """
    # Make sure arguments conform to the implementation.
    assert op == torch.distributed.ReduceOp.SUM

    # Get world_size.
    if group is None:
        world_size = torch.distributed.get_world_size()
    else:
        world_size = group.size()

    # Make sure input_tensor size is divisible by world size.
    assert input_tensor.numel() % world_size == 0

    if hierarchical_groups is not None:
        local_group, inter_group = hierarchical_groups
        local_group_size = local_group.size()
        assert local_group_size * inter_group.size() == world_size
        shard_numel = input_tensor.numel() // world_size

        local_send = (
            input_tensor.view(inter_group.size(), local_group_size, shard_numel)
            .permute(1, 0, 2)
            .contiguous()
        )
        local_recv = torch.empty_like(local_send)
        local_handle = torch.distributed.all_to_all_single(
            output=local_recv, input=local_send, group=local_group, async_op=async_op
        )
        reduce_scatter_handle = _HierarchicalReduceScatterWithFP32AccumulationWorkHandle(
            local_handle, local_send, local_recv, output_tensor, local_group_size, inter_group
        )
        if async_op:
            return reduce_scatter_handle
        reduce_scatter_handle.wait()
        return None

    # Call all_to_all (every rank should have their respective gradient shards collected from
    # all ranks). We also create a tensor for the all-to-all output (the all-to-all collective
    # cannot be performed in-place).
    all_to_all_output_tensor = torch.empty_like(input_tensor)
    all_to_all_handle = torch.distributed.all_to_all_single(
        output=all_to_all_output_tensor, input=input_tensor, group=group, async_op=async_op
    )

    # Create a work handle to finish communication and reduction.
    reduce_scatter_handle = _ReduceScatterWithFP32AccumulationWorkHandle(
        all_to_all_handle, all_to_all_output_tensor, output_tensor, world_size
    )
    if async_op:
        # Return work handle; consumers can call .wait() to ensure communication and associated
        # reduction complete.
        return reduce_scatter_handle
    else:
        # Wait on work handle.
        reduce_scatter_handle.wait()
