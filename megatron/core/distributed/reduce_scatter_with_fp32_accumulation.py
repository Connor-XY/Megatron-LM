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


class _SingleRankReduceScatterWorkHandle:
    """Preserve the async wait contract for an identity reduce-scatter."""

    def __init__(self, completion_event: torch.cuda.Event | None):
        self.completion_event = completion_event

    def wait(self):
        """Make the current stream wait for an earlier asynchronous device copy, if any."""
        if self.completion_event is not None:
            self.completion_event.wait()
            self.completion_event = None


def _native_two_rank_reduce_scatter_is_exact(input_tensor: torch.Tensor, world_size: int) -> bool:
    """Return whether native two-rank SUM preserves this helper's fp32-accumulation result."""
    return world_size == 2 and input_tensor.dtype in (torch.float16, torch.bfloat16, torch.float32)


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

    A one-rank reduction is the identity. A two-rank reduction has exactly one addition per output
    element, so the native collective is already independent of physical topology and produces
    the same rounded output as adding the two lower-precision inputs in FP32 before downcasting.
    Larger flat groups collect input_tensor with an all-to-all and then accumulate in FP32
    precision before downcasting the final sum into output_tensor.


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
    assert output_tensor.numel() == input_tensor.numel() // world_size

    if world_size == 1:
        # A one-rank reduce-scatter is the identity. Production distributed-optimizer output is a
        # view of the input buffer, so this normally avoids every kernel and allocation. Keep the
        # general helper correct for distinct tensors and preserve cross-stream async semantics.
        completion_event = None
        if not output_tensor.is_set_to(input_tensor):
            output_tensor.copy_(input_tensor.view_as(output_tensor))
            if async_op and output_tensor.is_cuda:
                completion_event = torch.cuda.Event()
                completion_event.record()
        if async_op:
            return _SingleRankReduceScatterWorkHandle(completion_event)
        return None

    if _native_two_rank_reduce_scatter_is_exact(input_tensor, world_size):
        # With two ranks each output element has exactly two operands, hence one addition. There
        # is no reduction-tree order for the physical topology to change. The native path also
        # preserves the original synchronous/asynchronous return contract while avoiding the
        # all-to-all receive buffer and explicit local reduction.
        return torch.distributed.reduce_scatter_tensor(
            output_tensor, input_tensor, op=op, group=group, async_op=async_op
        )

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
