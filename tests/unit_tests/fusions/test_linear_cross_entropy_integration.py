# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.linear_cross_entropy import LinearCrossEntropyModule
from megatron.core.transformer.multi_token_prediction import process_mtp_loss
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def test_hybrid_model_selects_linear_cross_entropy_output_layer():
    Utils.initialize_model_parallel(1, 1)
    try:
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=3,
            hidden_size=256,
            num_attention_heads=4,
            use_cpu_initialization=True,
            cross_entropy_loss_fusion=True,
            cross_entropy_fusion_impl="linear",
        )
        model = HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=100,
            max_sequence_length=4,
            hybrid_layer_pattern="M*-",
        )

        assert model.fuse_linear_cross_entropy
        assert isinstance(model.output_layer, LinearCrossEntropyModule)
    finally:
        Utils.destroy_model_parallel()


@torch.no_grad()
def test_mtp_dispatches_linear_cross_entropy_without_materializing_logits():
    config = TransformerConfig(
        mtp_num_layers=2,
        num_layers=4,
        hidden_size=64,
        num_attention_heads=8,
        use_cpu_initialization=True,
        cross_entropy_loss_fusion=True,
        cross_entropy_fusion_impl="linear",
    )
    seq_len = 4
    batch_size = 2
    vocab_size = 16
    hidden_states = torch.randn(
        (1 + config.mtp_num_layers) * seq_len, batch_size, config.hidden_size
    )
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    loss_mask = torch.ones(batch_size, seq_len)
    output_weight = torch.nn.Parameter(torch.randn(vocab_size, config.hidden_size))
    fused_calls = []

    def output_layer(
        hidden,
        weight=None,
        runtime_gather_output=None,
        output_cross_entropy_loss=False,
        labels=None,
    ):
        del runtime_gather_output
        assert output_cross_entropy_loss
        assert labels is not None
        fused_calls.append(tuple(labels.shape))
        return (hidden @ weight.T).sum(dim=-1).transpose(0, 1)

    result = process_mtp_loss(
        hidden_states=hidden_states,
        labels=labels,
        loss_mask=loss_mask,
        output_layer=output_layer,
        output_weight=output_weight,
        runtime_gather_output=None,
        is_training=False,
        compute_language_model_loss=lambda *_: (_ for _ in ()).throw(
            AssertionError("the materialized-logits loss path must not run")
        ),
        config=config,
    )

    assert result.shape == (seq_len, batch_size, config.hidden_size)
    assert fused_calls == [(batch_size, seq_len)] * config.mtp_num_layers
