# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Bit-exact coverage for the PyTorch normalization fallback.

Most model-level determinism presets use Transformer Engine or Apex norms.
This test instantiates ``WrappedTorchNorm`` directly so those backends cannot
mask a regression in PyTorch LayerNorm or RMSNorm backward.
"""

import pytest
import torch

from megatron.core.transformer.torch_norm import WrappedTorchNorm
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.determinism.utils import assert_bit_exact, collect_grads


def _build_config(normalization: str, hidden_size: int) -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        ffn_hidden_size=2 * hidden_size,
        num_attention_heads=8,
        normalization=normalization,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        deterministic_mode=True,
    )


@pytest.mark.internal
@pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
@pytest.mark.parametrize("hidden_size", [128, 2048])
def test_torch_norm_backward_is_bit_exact(normalization, hidden_size):
    """Repeat the fallback forward/backward at small and production-like widths."""
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(1234)

    config = _build_config(normalization, hidden_size)
    norm = WrappedTorchNorm(config=config, hidden_size=hidden_size).cuda().to(torch.bfloat16)
    input_base = torch.randn(32, 4, hidden_size, device="cuda", dtype=torch.bfloat16)
    grad_output = torch.randn_like(input_base)

    def run_once():
        norm.zero_grad(set_to_none=True)
        x = input_base.detach().clone().requires_grad_(True)
        output = norm(x)
        (output.float() * grad_output.float()).sum().backward()
        grads = collect_grads([norm])
        grads["input"] = x.grad.detach().clone()
        return output.detach().clone(), grads

    output_a, grads_a = run_once()
    torch.cuda.synchronize()
    output_b, grads_b = run_once()
    assert_bit_exact(output_a, grads_a, output_b, grads_b)
