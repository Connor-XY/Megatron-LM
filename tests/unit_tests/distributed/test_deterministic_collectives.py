# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.distributed.deterministic_collectives import all_reduce_avg, all_reduce_sum
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
