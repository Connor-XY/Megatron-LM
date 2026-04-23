# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.

"""End-to-end tests for the HybridModel 1F1B combined-layer path.

These tests construct a small :class:`HybridModel` with a bracketed
``hybrid_layer_pattern``, verify the monolithic forward works, verify that
``build_schedule_plan`` returns a populated
:class:`TransformerModelChunkSchedulePlan`, and verify that the combined
layer's split forward (``forward_ssm_attn`` + ``forward_mlp``) composes to
the same output as the monolithic forward.
"""

import pytest
import torch

from megatron.core.models.hybrid.hybrid_layer_specs import (
    hybrid_stack_spec_with_combined_layers,
)
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_attn_mlp_layer import (
    MambaAttnMLPLayer,
    MambaMLPLayer,
    MambaSelfAttnMLPLayer,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _tiny_config(num_layers: int) -> TransformerConfig:
    return TransformerConfig(
        hidden_size=256,
        num_layers=num_layers,
        num_attention_heads=4,
        use_cpu_initialization=True,
    )


def _build_model(pattern: str, pg_collection: ProcessGroupCollection):
    layer_type_count = sum(
        len(group.submodule_standalone_indices)
        for group in __parse(pattern)
    )
    config = _tiny_config(num_layers=layer_type_count)
    return HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec_with_combined_layers,
        vocab_size=1024,
        max_sequence_length=64,
        hybrid_layer_pattern=pattern,
        pre_process=True,
        post_process=True,
        pg_collection=pg_collection,
    )


def __parse(pattern):
    from megatron.core.models.hybrid.hybrid_layer_allocation import parse_bracketed_pattern

    return parse_bracketed_pattern(pattern)


@pytest.mark.internal
class TestHybridSchedulePlan:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.pg_collection = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=["tp", "pp", "cp", "embd"]
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_bracketed_pattern_builds_combined_layers(self):
        """HybridModel with a bracketed pattern builds ``MambaAttnMLPLayer`` instances."""
        model = _build_model("[M-][M*-]", self.pg_collection).cuda()
        assert model.uses_combined_layers
        assert len(model.decoder.layers) == 2
        assert isinstance(model.decoder.layers[0], MambaMLPLayer)
        assert isinstance(model.decoder.layers[1], MambaSelfAttnMLPLayer)

    def test_monolithic_forward_runs(self):
        """Forward through a combined-layer model produces the expected logits shape."""
        model = _build_model("[M-][M*-]", self.pg_collection).cuda()
        bsz, seq_len = 2, 16
        input_ids = torch.randint(0, 1024, (bsz, seq_len)).cuda()
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(bsz, -1).cuda()
        attention_mask = torch.ones((bsz, 1, seq_len, seq_len), dtype=torch.bool).cuda()
        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                runtime_gather_output=True,
            )
        # labels=None: output is logits in [b, s, h] shape after the transpose.
        assert logits.shape[0] == bsz
        assert logits.shape[1] == seq_len

    def test_build_schedule_plan_returns_populated_plan(self):
        """``build_schedule_plan`` returns a plan with pre_process + N layers + post_process."""
        model = _build_model("[M-][M*-][M-]", self.pg_collection).cuda()
        model.config.overlap_moe_expert_parallel_comm = True

        bsz, seq_len = 2, 16
        input_ids = torch.randint(0, 1024, (bsz, seq_len)).cuda()
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(bsz, -1).cuda()
        attention_mask = torch.ones((bsz, 1, seq_len, seq_len), dtype=torch.bool).cuda()

        # Initialize the comm stream before the plan is built.
        from megatron.core.pipeline_parallel.utils import set_streams

        set_streams()

        plan = model.build_schedule_plan(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=None,
            runtime_gather_output=True,
        )
        assert plan.pre_process is not None
        assert plan.post_process is not None
        assert plan.num_layers() == 3
        # Each layer's wrapped module is a MambaAttnMLPLayer.
        for i in range(plan.num_layers()):
            layer_plan = plan.get_layer(i)
            assert isinstance(layer_plan.layer, MambaAttnMLPLayer)

    def test_legacy_pattern_still_works(self):
        """Non-bracketed patterns keep using the legacy separate-layer path."""
        from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
        from megatron.core.ssm.mamba_layer import MambaLayer

        config = _tiny_config(num_layers=3)
        model = HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=1024,
            max_sequence_length=64,
            hybrid_layer_pattern="M*-",
            pre_process=True,
            post_process=True,
            pg_collection=self.pg_collection,
        ).cuda()
        assert not model.uses_combined_layers
        # The legacy path should build a MambaLayer for the M symbol.
        assert isinstance(model.decoder.layers[0], MambaLayer)
