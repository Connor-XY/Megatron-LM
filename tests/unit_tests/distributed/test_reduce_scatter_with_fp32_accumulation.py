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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("async_op", [False, True])
def test_two_rank_reduction_uses_native_exact_collective(monkeypatch, async_op, dtype):
    class TwoRankGroup:
        @staticmethod
        def size():
            return 2

    expected_handle = object() if async_op else None
    recorded = {}

    def fake_reduce_scatter(output, input_, *, op, group, async_op):
        recorded.update(output=output, input=input_, op=op, group=group, async_op=async_op)
        return expected_handle

    monkeypatch.setattr(torch.distributed, "reduce_scatter_tensor", fake_reduce_scatter)
    output = torch.empty(2, dtype=dtype)
    input_ = torch.empty(4, dtype=dtype)
    group = TwoRankGroup()

    result = reduce_scatter_with_fp32_accumulation(
        output, input_, op=torch.distributed.ReduceOp.SUM, group=group, async_op=async_op
    )

    assert result is expected_handle
    assert recorded["output"] is output
    assert recorded["input"] is input_
    assert recorded["op"] == torch.distributed.ReduceOp.SUM
    assert recorded["group"] is group
    assert recorded["async_op"] is async_op


def test_two_rank_unsupported_dtype_keeps_fp32_accumulation(monkeypatch):
    class TwoRankGroup:
        @staticmethod
        def size():
            return 2

    def fail_native_reduce_scatter(*args, **kwargs):
        raise AssertionError("float64 must retain the explicit fp32-accumulation path")

    def fake_all_to_all(output, input, *, group, async_op):
        assert isinstance(group, TwoRankGroup)
        assert not async_op
        output.copy_(input)
        return None

    monkeypatch.setattr(torch.distributed, "reduce_scatter_tensor", fail_native_reduce_scatter)
    monkeypatch.setattr(torch.distributed, "all_to_all_single", fake_all_to_all)
    input_ = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    output = torch.empty(2, dtype=torch.float64)

    result = reduce_scatter_with_fp32_accumulation(
        output, input_, op=torch.distributed.ReduceOp.SUM, group=TwoRankGroup(), async_op=False
    )

    assert result is None
    torch.testing.assert_close(
        output, input_.view(2, -1).sum(dim=0, dtype=torch.float32).to(torch.float64), rtol=0, atol=0
    )


@pytest.mark.parametrize("async_op", [False, True])
@pytest.mark.parametrize("aliased", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_single_rank_reduction_is_identity_without_collective(
    monkeypatch, async_op, aliased, device
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for the device-event path")

    class SingleRankGroup:
        @staticmethod
        def size():
            return 1

    def fail_collective(*args, **kwargs):
        raise AssertionError("single-rank reduce-scatter must not launch a collective")

    monkeypatch.setattr(torch.distributed, "reduce_scatter_tensor", fail_collective)
    monkeypatch.setattr(torch.distributed, "all_to_all_single", fail_collective)
    input_ = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64, device=device)
    output = shard_buffer(input_, 1)[0] if aliased else torch.empty_like(input_)
    if aliased:
        assert output.is_set_to(input_)

    result = reduce_scatter_with_fp32_accumulation(
        output,
        input_,
        op=torch.distributed.ReduceOp.SUM,
        group=SingleRankGroup(),
        async_op=async_op,
    )

    if async_op:
        assert result is not None
        result.wait()
    else:
        assert result is None
    if device == "cuda":
        torch.cuda.synchronize()
    torch.testing.assert_close(output, input_, rtol=0, atol=0)
