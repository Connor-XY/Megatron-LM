# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.


import pytest
import torch

from megatron.core import parallel_state

# Import our reduce_scatter implementation and shard_buffer (used for
# checks in the test).
from megatron.core.distributed.param_and_grad_buffer import (
    reduce_scatter_with_fp32_accumulation,
    shard_buffer,
)
from tests.unit_tests.test_utilities import Utils


def get_non_matching_values(tensor1_shard, tensor2_shard):
    mask = torch.isclose(tensor1_shard, tensor2_shard)
    indices = (~mask).nonzero()
    return indices, tensor1_shard[indices], tensor2_shard[indices]


class TestReduceScatterWithFP32Accumulation:
    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel()
        cls.hierarchical_groups = None
        if Utils.world_size >= 4 and Utils.world_size % 2 == 0:
            cls.hierarchical_groups, _ = parallel_state.create_hierarchical_groups(
                Utils.rank,
                list(range(Utils.world_size)),
                [2, Utils.world_size // 2],
                create_gloo_process_groups=False,
                group_desc="TEST_ORDERED_REDUCE_SCATTER",
            )

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("async_op", [True, False])
    @pytest.mark.parametrize("baseline_reduce_scatter_in_fp32", [True, False])
    @pytest.mark.parametrize("hierarchical", [True, False])
    def test_reduce_scatter_with_fp32_accumulation(
        self, async_op: bool, baseline_reduce_scatter_in_fp32: bool, hierarchical: bool
    ):
        if hierarchical and self.hierarchical_groups is None:
            pytest.skip("Hierarchical test requires an even world size of at least four")
        num_tests = 20
        rank = Utils.rank
        world_size = Utils.world_size
        for _ in range(num_tests):
            # Initialize input tensors.
            tensor1 = torch.rand(100000, device='cuda', dtype=torch.bfloat16)
            tensor2 = tensor1.clone()

            # Make sure the two APIs are *identical*.
            kwargs = {
                "op": torch.distributed.ReduceOp.SUM,
                "group": None,
                "async_op": async_op,
                "hierarchical_groups": self.hierarchical_groups if hierarchical else None,
            }

            # Reduce-scatter with all-to-alls.
            args = [
                shard_buffer(tensor1, world_size)[rank],
                tensor1,
            ]  # Output tensor is view into original input.
            handle = reduce_scatter_with_fp32_accumulation(*args, **kwargs)
            if async_op:
                assert handle is not None
                handle.wait()
            tensor1_shard = shard_buffer(tensor1, world_size)[rank]

            if baseline_reduce_scatter_in_fp32:
                tensor2 = tensor2.float()

            # Reduce-scatter with reduce-scatter API.
            args = [
                shard_buffer(tensor2, world_size)[rank],
                tensor2,
            ]  # Output tensor is view into original input.
            kwargs.pop("hierarchical_groups")
            handle = torch.distributed.reduce_scatter_tensor(*args, **kwargs)
            if async_op:
                assert handle is not None
                handle.wait()
            tensor2_shard = shard_buffer(tensor2, world_size)[rank]
            if baseline_reduce_scatter_in_fp32:  # Cast result back to bfloat16.
                tensor2_shard = tensor2_shard.bfloat16()

            # Compare results: results should match when doing FP32 reduction and not match when
            # doing direct BF16 reduction. We only look at relevant shard of tensor1 and tensor2.
            assert (
                torch.allclose(tensor1_shard, tensor2_shard) == baseline_reduce_scatter_in_fp32
            ), f"{get_non_matching_values(tensor1_shard, tensor2_shard)}"
