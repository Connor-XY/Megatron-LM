# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.tensor_parallel.deterministic_cross_entropy import (
    HAVE_TRITON,
    deterministic_cross_entropy_backward_,
)


@pytest.mark.internal
@pytest.mark.skipif(not HAVE_TRITON or not torch.cuda.is_available(), reason="CUDA Triton required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("num_rows, num_columns", [(16, 128), (512, 8192)])
@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
@pytest.mark.parametrize("strided_grad_output", [False, True])
def test_fused_backward_matches_index_put(
    dtype, num_rows, num_columns, label_smoothing, strided_grad_output
):
    torch.manual_seed(1234)
    source = torch.randn(num_rows, num_columns, device="cuda", dtype=dtype)
    indices = torch.randint(num_columns, (num_rows,), device="cuda")
    updates = torch.randint(2, (num_rows,), device="cuda", dtype=torch.int32).to(dtype)
    if strided_grad_output:
        grad_output = torch.randn(2, num_rows // 2, device="cuda", dtype=dtype).t()
        assert not grad_output.is_contiguous()
    else:
        grad_output = torch.randn(num_rows, device="cuda", dtype=dtype)
    smoothing_update = label_smoothing / num_columns
    rows = torch.arange(num_rows, device="cuda")
    reference = source.clone()
    candidate = source.clone()

    previous_deterministic_mode = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        reference[rows, indices] -= updates
        if smoothing_update:
            reference.sub_(smoothing_update)
        reference.mul_(grad_output.reshape(-1).unsqueeze(-1))
        deterministic_cross_entropy_backward_(
            candidate, indices, updates, grad_output, smoothing_update=smoothing_update
        )
    finally:
        torch.use_deterministic_algorithms(previous_deterministic_mode)

    assert torch.equal(candidate, reference)


def test_fused_backward_cpu_fallback():
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    indices = torch.tensor([1, 0])
    updates = torch.tensor([0.5, 1.5])
    grad_output = torch.tensor([[2.0], [3.0]])

    output = deterministic_cross_entropy_backward_(values, indices, updates, grad_output)

    assert output is values
    assert torch.equal(values, torch.tensor([[2.0, 3.0], [4.5, 12.0]]))


def test_fused_backward_rejects_invalid_shapes():
    with pytest.raises(ValueError, match="2D values and 1D indices/updates"):
        deterministic_cross_entropy_backward_(
            torch.ones(4), torch.zeros(4), torch.ones(4), torch.ones(4)
        )
    with pytest.raises(ValueError, match="one index, update, and output gradient per row"):
        deterministic_cross_entropy_backward_(
            torch.ones(4, 8), torch.zeros(3, dtype=torch.long), torch.ones(3), torch.ones(3)
        )


@pytest.mark.internal
@pytest.mark.skipif(not HAVE_TRITON or not torch.cuda.is_available(), reason="CUDA Triton required")
def test_fused_backward_cuda_graph_replay():
    torch.manual_seed(1234)
    num_rows, num_columns = 512, 8192
    source = torch.randn(num_rows, num_columns, device="cuda")
    values = source.clone()
    indices = torch.randint(num_columns, (num_rows,), device="cuda")
    updates = torch.randint(2, (num_rows,), device="cuda", dtype=torch.int32).float()
    grad_output = torch.randn(2, num_rows // 2, device="cuda").t()
    assert not grad_output.is_contiguous()

    warmup_values = source.clone()
    deterministic_cross_entropy_backward_(warmup_values, indices, updates, grad_output)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        values.copy_(source)
        deterministic_cross_entropy_backward_(values, indices, updates, grad_output)

    graph.replay()
    expected = source.clone()
    expected[torch.arange(num_rows, device="cuda"), indices] -= updates
    expected.mul_(grad_output.reshape(-1).unsqueeze(-1))

    assert torch.equal(values, expected)
