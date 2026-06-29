# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.distributed.deterministic_collectives import (
    all_reduce_avg,
    all_reduce_sum,
    all_reduce_with_fp32_accumulation,
)
from tests.unit_tests.test_utilities import Utils


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_all_reduce_sum_uses_rank_ordered_accumulation():
    Utils.initialize_model_parallel()
    deterministic_algorithms_enabled = torch.are_deterministic_algorithms_enabled()
    try:
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        rank_values = torch.arange(1, world_size + 1, dtype=torch.float32, device="cuda")
        if world_size >= 4:
            rank_values[:4] = torch.tensor(
                [1.0e20, 1.0, -1.0e20, 3.0], dtype=torch.float32, device="cuda"
            )
        expected = rank_values.sum()
        local_value = rank_values[rank].clone()

        torch.use_deterministic_algorithms(True)
        result = all_reduce_sum(local_value, group=torch.distributed.group.WORLD)

        assert result is local_value
        torch.testing.assert_close(local_value, expected, rtol=0, atol=0)

        expected_avg = expected / world_size
        local_value = rank_values[rank].clone()
        result = all_reduce_avg(local_value, group=torch.distributed.group.WORLD)

        assert result is local_value
        torch.testing.assert_close(local_value, expected_avg, rtol=0, atol=0)
    finally:
        torch.use_deterministic_algorithms(deterministic_algorithms_enabled)
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    ("op", "average"),
    [(torch.distributed.ReduceOp.SUM, False), (torch.distributed.ReduceOp.AVG, True)],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_large_all_reduce_uses_bounded_rank_ordered_chunks(dtype, op, average):
    Utils.initialize_model_parallel()
    try:
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        numel = world_size * 3 + 1
        offsets = torch.arange(numel, dtype=torch.float32, device="cuda") % 4
        tensor = (offsets + rank + 1).to(dtype)
        expected = offsets * world_size + world_size * (world_size + 1) / 2
        if average:
            expected.div_(world_size)
        expected = expected.to(dtype)

        result = all_reduce_with_fp32_accumulation(
            tensor,
            op=op,
            group=torch.distributed.group.WORLD,
            max_chunk_bytes=world_size * tensor.element_size() * 2,
        )

        assert result is tensor
        torch.testing.assert_close(tensor, expected, rtol=0, atol=0)
    finally:
        Utils.destroy_model_parallel()
