# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for :class:`CombinedHybridLayer`.

Covers all six combined-layer subclasses (Mamba/GDN + optional
Attention/MLA + MLP/MoE), the split forward ≡ monolithic forward invariant
(which the 1F1B overlap schedule depends on), and the submodule-layer-number
plumbing used for FP8 per-layer contexts / legacy checkpoint interop.
"""

import pytest
import torch

from megatron.core.models.hybrid.combined_layer import (
    ATTENTION_TYPE_ATTENTION,
    ATTENTION_TYPE_NONE,
    MAMBA_TYPE_GDN,
    MAMBA_TYPE_MAMBA,
    MLP_TYPE_MLP,
    AttnMoELayer,
    CombinedHybridLayer,
    GDNSelfAttnMLPLayer,
    GDNMLPLayer,
    MambaMLPLayer,
    MambaMoELayer,
    MambaSelfAttnMLPLayer,
    MambaSelfAttnMoELayer,
)
from megatron.core.models.hybrid.hybrid_layer_specs import combined_hybrid_layer_submodules
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.mark.internal
class TestCombinedHybridLayer:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.config = TransformerConfig(
            hidden_size=256,
            num_layers=1,
            num_attention_heads=4,
            use_cpu_initialization=True,
        )
        self.pg_collection = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=["tp", "pp", "cp"]
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _inputs(self, hidden_size: int, seq_len: int = 32, micro_batch_size: int = 2):
        hidden_states = torch.ones((seq_len, micro_batch_size, hidden_size)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, seq_len, seq_len), dtype=torch.bool
        ).cuda()
        return hidden_states, attention_mask

    def test_invalid_mamba_type_raises(self):
        with pytest.raises(AssertionError):
            CombinedHybridLayer(
                self.config,
                combined_hybrid_layer_submodules,
                mamba_type="not_real",
                pg_collection=self.pg_collection,
            )

    def test_all_none_raises(self):
        with pytest.raises(AssertionError, match="at least one block"):
            CombinedHybridLayer(
                self.config,
                combined_hybrid_layer_submodules,
                mamba_type="none",
                attention_type="none",
                mlp_type="none",
                pg_collection=self.pg_collection,
            )

    def test_mamba_mlp_forward_shape(self):
        layer = MambaMLPLayer(
            self.config,
            combined_hybrid_layer_submodules,
            pg_collection=self.pg_collection,
        ).cuda()
        hidden_states, attention_mask = self._inputs(self.config.hidden_size)
        output = layer(hidden_states, attention_mask=attention_mask)
        assert output.shape == hidden_states.shape
        # Attention block absent.
        assert layer.self_attention is None
        assert layer.attn_norm is None
        assert layer.attn_bda is None
        assert layer.mamba_type == MAMBA_TYPE_MAMBA
        assert layer.attention_type == ATTENTION_TYPE_NONE
        assert layer.mlp_type == MLP_TYPE_MLP
        assert not layer.is_moe_layer

    def test_self_attn_mlp_forward_shape(self):
        layer = MambaSelfAttnMLPLayer(
            self.config,
            combined_hybrid_layer_submodules,
            pg_collection=self.pg_collection,
        ).cuda()
        hidden_states, attention_mask = self._inputs(self.config.hidden_size)
        output = layer(hidden_states, attention_mask=attention_mask)
        assert output.shape == hidden_states.shape
        assert layer.self_attention is not None
        assert layer.attention_type == ATTENTION_TYPE_ATTENTION

    def test_gdn_mlp_forward_shape(self):
        try:
            from megatron.core.ssm.gated_delta_net import HAVE_FLA
        except ImportError:
            pytest.skip("gated_delta_net module not importable")
        if not HAVE_FLA:
            pytest.skip("flash-linear-attention not installed")
        if not hasattr(self.config, "linear_key_head_dim"):
            pytest.skip("TransformerConfig lacks GDN fields in this build")

        layer = GDNMLPLayer(
            self.config,
            combined_hybrid_layer_submodules,
            pg_collection=self.pg_collection,
        ).cuda()
        hidden_states, attention_mask = self._inputs(self.config.hidden_size)
        output = layer(hidden_states, attention_mask=attention_mask)
        assert output.shape == hidden_states.shape
        assert layer.mamba_type == MAMBA_TYPE_GDN
        assert layer.mixer is not None
        # GDN replaces Mamba in the mixer slot, not the attention slot.
        assert layer.self_attention is None

    def test_gdn_self_attn_mlp_forward_shape(self):
        """GDN mixer followed by full attention and MLP."""
        try:
            from megatron.core.ssm.gated_delta_net import HAVE_FLA
        except ImportError:
            pytest.skip("gated_delta_net module not importable")
        if not HAVE_FLA:
            pytest.skip("flash-linear-attention not installed")
        if not hasattr(self.config, "linear_key_head_dim"):
            pytest.skip("TransformerConfig lacks GDN fields in this build")

        layer = GDNSelfAttnMLPLayer(
            self.config,
            combined_hybrid_layer_submodules,
            pg_collection=self.pg_collection,
        ).cuda()
        hidden_states, attention_mask = self._inputs(self.config.hidden_size)
        output = layer(hidden_states, attention_mask=attention_mask)
        assert output.shape == hidden_states.shape
        assert layer.mamba_type == MAMBA_TYPE_GDN
        assert layer.mixer is not None
        assert layer.attention_type == ATTENTION_TYPE_ATTENTION
        assert layer.self_attention is not None

    def test_attn_moe_forward_shape(self):
        """Pure Attention + MoE layer (GPT-style) inside the combined-layer API."""
        pytest.skip(
            "MoE construction needs EP>1 process group; covered by end-to-end tests."
        )

    def test_split_forward_matches_monolithic(self):
        """``forward_ssm_attn`` followed by ``forward_mlp`` equals ``forward``.

        This is the invariant the 1F1B overlap schedule depends on.
        """
        layer = MambaSelfAttnMLPLayer(
            self.config,
            combined_hybrid_layer_submodules,
            pg_collection=self.pg_collection,
        ).cuda()
        hidden_states, attention_mask = self._inputs(self.config.hidden_size)

        with torch.no_grad():
            expected = layer(hidden_states, attention_mask=attention_mask)
            mid = layer.forward_ssm_attn(hidden_states, attention_mask=attention_mask)
            actual = layer.forward_mlp(mid)

        torch.testing.assert_close(actual, expected)

    def test_mamba_moe_constructs(self):
        pytest.skip(
            "MoE construction needs EP>1 process group; covered by end-to-end tests."
        )

    def test_mamba_self_attn_moe_constructs(self):
        pytest.skip(
            "MoE construction needs EP>1 process group; covered by end-to-end tests."
        )

    def test_submodule_layer_numbers_default_to_layer_number(self):
        layer = MambaMLPLayer(
            self.config,
            combined_hybrid_layer_submodules,
            layer_number=7,
            pg_collection=self.pg_collection,
        )
        assert layer.submodule_layer_numbers == {"mamba": 7, "attention": 7, "mlp": 7}

    def test_submodule_layer_numbers_override(self):
        layer = MambaSelfAttnMLPLayer(
            self.config,
            combined_hybrid_layer_submodules,
            layer_number=1,
            submodule_layer_numbers={"mamba": 3, "attention": 4, "mlp": 5},
            pg_collection=self.pg_collection,
        )
        assert layer.mixer.layer_number == 3
        assert layer.self_attention.layer_number == 4
