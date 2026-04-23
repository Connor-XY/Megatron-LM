# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the Mamba fine-grained callables.

These tests exercise the callable construction and dense-path forward parity
against the monolithic ``MambaAttnMLPLayer.forward``. MoE-path coverage (which
needs a full MoELayer build) is deferred to the integration tests added in
Steps 4-5 of the hybrid 1F1B work.
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.gpt.fine_grained_callables import build_layer_callables
from megatron.core.models.hybrid.fine_grained_callables import (
    build_mamba_attn_mlp_layer_callables,
)
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_attn_mlp_layer import (
    ATTENTION_TYPE_NONE,
    ATTENTION_TYPE_SELF_ATTENTION,
    MambaAttnMLPLayerSubmodules,
    MambaMLPLayer,
    MambaSelfAttnMLPLayer,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _build_submodules(attention_type: str) -> MambaAttnMLPLayerSubmodules:
    """Mirror the helper in ``test_mamba_attn_mlp_layer`` to pull real specs."""
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
    return submods


class _FakeNode:
    """Stand-in for ``TransformerLayerNode`` used to drive the callables.

    Only the attributes actually read by the callables are populated.
    """

    def __init__(self, *, attention_mask=None):
        self.chunk_state = SimpleNamespace(
            attention_mask=attention_mask,
            rotary_pos_emb=None,
            rotary_pos_cos=None,
            rotary_pos_sin=None,
            packed_seq_params=None,
            sequence_len_offset=None,
        )
        self.layer_state = SimpleNamespace()

    def detach(self, t):
        # The real TransformerLayerNode records the tensor for backward; we only
        # need the detached tensor here because the tests are no-grad.
        d = t.detach()
        d.requires_grad = t.requires_grad
        return d


@pytest.mark.internal
class TestMambaFineGrainedCallables:

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

    def test_dense_returns_five_callables(self):
        layer = MambaMLPLayer(
            self.config,
            _build_submodules(ATTENTION_TYPE_NONE),
            pg_collection=self.pg_collection,
        ).cuda()
        forward_funcs, backward_dw = build_mamba_attn_mlp_layer_callables(layer)
        assert len(forward_funcs) == 5
        attn_func, dispatch_func, mlp_func, combine_func, mtp_post_process_func = forward_funcs
        assert callable(attn_func)
        assert callable(mlp_func)
        # Dense layer: dispatch/combine are the "raise" stubs and MTP is None.
        assert mtp_post_process_func is None
        with pytest.raises(NotImplementedError):
            dispatch_func(None, None, None)
        with pytest.raises(NotImplementedError):
            combine_func(None, None)
        # Dense layer with no attention: backward_dw['attn'] is None.
        assert backward_dw["attn"] is None
        assert backward_dw["mlp"] is layer.mlp

    def test_dense_split_forward_matches_monolithic(self):
        """attn_callable + mlp_callable ≡ layer.forward on the dense path."""
        layer = MambaSelfAttnMLPLayer(
            self.config,
            _build_submodules(ATTENTION_TYPE_SELF_ATTENTION),
            pg_collection=self.pg_collection,
        ).cuda()

        seq_len, bsz = 32, 2
        hidden_states = torch.ones((seq_len, bsz, self.config.hidden_size)).cuda()
        attention_mask = torch.ones((bsz, 1, seq_len, seq_len), dtype=torch.bool).cuda()

        forward_funcs, _ = build_mamba_attn_mlp_layer_callables(layer)
        attn_func, _, mlp_func, _, _ = forward_funcs

        node = _FakeNode(attention_mask=attention_mask)

        with torch.no_grad():
            expected = layer(hidden_states, attention_mask=attention_mask)
            mid = attn_func(node, hidden_states)
            actual = mlp_func(node, mid)

        torch.testing.assert_close(actual, expected)

    def test_build_layer_callables_dispatches_to_mamba(self):
        """The shared dispatcher routes ``MambaAttnMLPLayer`` to our builder."""
        layer = MambaMLPLayer(
            self.config,
            _build_submodules(ATTENTION_TYPE_NONE),
            pg_collection=self.pg_collection,
        ).cuda()

        # Sanity: both builders return the same shape so the dispatch result
        # is structurally identical to what we would get by calling the Mamba
        # builder directly.
        direct_funcs, direct_bw = build_mamba_attn_mlp_layer_callables(layer)
        via_dispatcher_funcs, via_dispatcher_bw = build_layer_callables(layer)
        assert len(direct_funcs) == len(via_dispatcher_funcs) == 5
        assert set(direct_bw.keys()) == set(via_dispatcher_bw.keys())

    def test_self_attn_layer_registers_backward_dw_wrapper(self):
        """When attention is present the layer gets a backward_dw_wrapper."""
        layer = MambaSelfAttnMLPLayer(
            self.config,
            _build_submodules(ATTENTION_TYPE_SELF_ATTENTION),
            pg_collection=self.pg_collection,
        ).cuda()
        _, backward_dw = build_mamba_attn_mlp_layer_callables(layer)
        assert backward_dw["attn"] is not None
        assert backward_dw["attn"] is layer.backward_dw_wrapper
