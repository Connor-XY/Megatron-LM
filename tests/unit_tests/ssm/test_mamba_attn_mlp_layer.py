# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_attn_mlp_layer import (
    ATTENTION_TYPE_GATED_DELTA_NET,
    ATTENTION_TYPE_NONE,
    ATTENTION_TYPE_SELF_ATTENTION,
    MambaAttnMLPLayer,
    MambaAttnMLPLayerSubmodules,
    MambaGDNMLPLayer,
    MambaMLPLayer,
    MambaSelfAttnMLPLayer,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _build_submodules(attention_type: str) -> MambaAttnMLPLayerSubmodules:
    """Build a submodules dataclass by pulling specs out of ``hybrid_stack_spec``.

    Reuses the byte-identical specs used by the legacy (separate-layer) path so
    the combined layer is numerically equivalent to a Mamba + Attn + MLP
    sequence built from the same specs.
    """
    mamba_submods = hybrid_stack_spec.submodules.mamba_layer.submodules
    mlp_submods = hybrid_stack_spec.submodules.mlp_layer.submodules

    submods = MambaAttnMLPLayerSubmodules(
        mamba_norm=mamba_submods.norm,
        mamba_mixer=mamba_submods.mixer,
        mamba_bda=mamba_submods.mamba_bda,
        pre_mlp_layernorm=mlp_submods.pre_mlp_layernorm,
        mlp=mlp_submods.mlp,
        mlp_bda=mlp_submods.mlp_bda,
    )

    if attention_type == ATTENTION_TYPE_SELF_ATTENTION:
        attn_submods = hybrid_stack_spec.submodules.attention_layer.submodules
        submods.attn_norm = attn_submods.input_layernorm
        submods.attention = attn_submods.self_attention
        submods.attn_bda = attn_submods.self_attn_bda
    elif attention_type == ATTENTION_TYPE_GATED_DELTA_NET:
        gdn_submods = hybrid_stack_spec.submodules.gdn_layer.submodules
        submods.attn_norm = gdn_submods.input_layernorm
        submods.attention = gdn_submods.self_attention
        submods.attn_bda = gdn_submods.self_attn_bda

    return submods


@pytest.mark.internal
class TestMambaAttnMLPLayer:

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

    def _build_inputs(self, hidden_size: int, seq_len: int = 32, micro_batch_size: int = 2):
        hidden_states = torch.ones((seq_len, micro_batch_size, hidden_size)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, seq_len, seq_len), dtype=torch.bool
        ).cuda()
        return hidden_states, attention_mask

    def test_invalid_attention_type_raises(self):
        with pytest.raises(ValueError):
            MambaAttnMLPLayer(
                self.config,
                _build_submodules(ATTENTION_TYPE_NONE),
                attention_type="not_a_real_type",
                pg_collection=self.pg_collection,
            )

    def test_mamba_mlp_forward_shape(self):
        layer = MambaMLPLayer(
            self.config,
            _build_submodules(ATTENTION_TYPE_NONE),
            pg_collection=self.pg_collection,
        ).cuda()
        hidden_states, attention_mask = self._build_inputs(self.config.hidden_size)
        output = layer(hidden_states, attention_mask=attention_mask)
        assert output.shape == hidden_states.shape
        # Attention block is completely absent.
        assert layer.attention is None
        assert layer.attn_norm is None
        assert layer.attn_bda is None
        assert layer.attention_type == ATTENTION_TYPE_NONE

    def test_self_attn_forward_shape(self):
        layer = MambaSelfAttnMLPLayer(
            self.config,
            _build_submodules(ATTENTION_TYPE_SELF_ATTENTION),
            pg_collection=self.pg_collection,
        ).cuda()
        hidden_states, attention_mask = self._build_inputs(self.config.hidden_size)
        output = layer(hidden_states, attention_mask=attention_mask)
        assert output.shape == hidden_states.shape
        assert layer.attention is not None
        assert layer.attention_type == ATTENTION_TYPE_SELF_ATTENTION

    def test_gdn_forward_shape(self):
        # GDN requires the `flash-linear-attention` package and GDN-specific
        # config fields (``linear_key_head_dim`` etc.). Skip if either is
        # unavailable so the rest of the suite still runs.
        try:
            from megatron.core.ssm.gated_delta_net import HAVE_FLA
        except ImportError:
            pytest.skip("gated_delta_net module not importable")
        if not HAVE_FLA:
            pytest.skip("flash-linear-attention not installed")
        if not hasattr(self.config, "linear_key_head_dim"):
            pytest.skip("TransformerConfig lacks GDN fields in this build")

        layer = MambaGDNMLPLayer(
            self.config,
            _build_submodules(ATTENTION_TYPE_GATED_DELTA_NET),
            pg_collection=self.pg_collection,
        ).cuda()
        hidden_states, attention_mask = self._build_inputs(self.config.hidden_size)
        output = layer(hidden_states, attention_mask=attention_mask)
        assert output.shape == hidden_states.shape
        assert layer.attention is not None
        assert layer.attention_type == ATTENTION_TYPE_GATED_DELTA_NET

    def test_split_forward_matches_monolithic(self):
        """forward_ssm_attn + forward_mlp must equal the monolithic forward.

        This is the core invariant the 1F1B overlap schedule relies on: the
        scheduler drives the two halves independently on different streams, and
        the observable output has to be identical to the non-split path.
        """
        layer = MambaSelfAttnMLPLayer(
            self.config,
            _build_submodules(ATTENTION_TYPE_SELF_ATTENTION),
            pg_collection=self.pg_collection,
        ).cuda()
        hidden_states, attention_mask = self._build_inputs(self.config.hidden_size)

        with torch.no_grad():
            # Monolithic
            expected = layer(hidden_states, attention_mask=attention_mask)
            # Split
            mid = layer.forward_ssm_attn(hidden_states, attention_mask=attention_mask)
            actual = layer.forward_mlp(mid)

        torch.testing.assert_close(actual, expected)

    def test_submodule_layer_numbers_default_to_layer_number(self):
        """When no map is passed the three submodules inherit ``layer_number``."""
        layer = MambaMLPLayer(
            self.config,
            _build_submodules(ATTENTION_TYPE_NONE),
            layer_number=7,
            pg_collection=self.pg_collection,
        )
        assert layer.submodule_layer_numbers == {"mamba": 7, "attention": 7, "mlp": 7}

    def test_submodule_layer_numbers_override(self):
        """Explicit ``submodule_layer_numbers`` flow through to inner modules."""
        layer = MambaSelfAttnMLPLayer(
            self.config,
            _build_submodules(ATTENTION_TYPE_SELF_ATTENTION),
            layer_number=1,
            submodule_layer_numbers={"mamba": 3, "attention": 4, "mlp": 5},
            pg_collection=self.pg_collection,
        )
        assert layer.mixer.layer_number == 3
        assert layer.attention.layer_number == 4
        # MLPs without set_layer_number just keep the default; both paths are
        # acceptable since some MLP impls (dense MLP) have no layer_number.
        if hasattr(layer.mlp, "layer_number") and layer.mlp.layer_number is not None:
            assert layer.mlp.layer_number == 5
