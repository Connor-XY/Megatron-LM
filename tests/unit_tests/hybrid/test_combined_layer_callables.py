# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for :mod:`combined_layer_callables`.

Exercises callable construction and the dense-path ``attn_callable``
collapses-whole-layer invariant. MoE-path coverage needs a full MoE build
and lives in the end-to-end tests alongside the schedule plan.
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.gpt.fine_grained_callables import build_layer_callables
from megatron.core.models.hybrid.combined_layer import MambaMLPLayer, MambaSelfAttnMLPLayer
from megatron.core.models.hybrid.combined_layer_callables import (
    _CombinedBackwardDWWrapper,
    build_combined_hybrid_layer_callables,
)
from megatron.core.models.hybrid.hybrid_layer_specs import combined_hybrid_layer_submodules
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class _FakeNode:
    """Minimal stand-in for ``TransformerLayerNode`` that drives the callables.

    Populates only the attributes read by the combined-layer callables in
    their dense-path (no MoE) code path. For the dense path the whole layer
    runs in the attn slot; the mlp slot is an identity pass-through.
    """

    def __init__(self, model, *, attention_mask=None):
        self.chunk_state = SimpleNamespace(
            model=model,
            attention_mask=attention_mask,
            rotary_pos_emb=None,
            rotary_pos_cos=None,
            rotary_pos_sin=None,
            packed_seq_params=None,
            sequence_len_offset=None,
        )
        self.layer_state = SimpleNamespace()
        # Schedule metadata consumed by the dense ``attn_callable`` to decide
        # whether to apply ``decoder.final_norm`` inline.
        self.is_last_layer = False
        self.is_mtp = False

    def detach(self, t):
        d = t.detach()
        d.requires_grad = t.requires_grad
        return d


class _FakeDecoder:
    """Stand-in for ``HybridStack`` -- only ``final_norm`` is read."""

    final_norm = None


class _FakeModel:
    """Stand-in for :class:`HybridModel` -- only ``decoder.final_norm`` is read."""

    def __init__(self):
        self.decoder = _FakeDecoder()


@pytest.mark.internal
class TestCombinedLayerCallables:

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
            combined_hybrid_layer_submodules,
            pg_collection=self.pg_collection,
        ).cuda()
        forward_funcs, backward_dw = build_combined_hybrid_layer_callables(layer)
        assert len(forward_funcs) == 5
        attn_func, dispatch_func, mlp_func, combine_func, mtp_post_process_func = forward_funcs
        assert callable(attn_func)
        assert callable(mlp_func)
        assert mtp_post_process_func is None
        with pytest.raises(NotImplementedError):
            dispatch_func(None, None, None)
        with pytest.raises(NotImplementedError):
            combine_func(None, None)
        # Dense path uses a no-op backward_dw wrapper; mlp bwd_dw is unused.
        assert isinstance(backward_dw["attn"], _CombinedBackwardDWWrapper)
        assert backward_dw["mlp"] is None

    def test_dense_attn_slot_runs_whole_layer(self):
        """For the dense path, ``attn_func`` produces the full-layer output.

        The ``mlp_func`` is an identity pass-through on the dense path because
        the dense MLP already ran inside ``attn_func``.
        """
        layer = MambaSelfAttnMLPLayer(
            self.config,
            combined_hybrid_layer_submodules,
            pg_collection=self.pg_collection,
        ).cuda()

        seq_len, bsz = 32, 2
        hidden_states = torch.ones((seq_len, bsz, self.config.hidden_size)).cuda()
        attention_mask = torch.ones((bsz, 1, seq_len, seq_len), dtype=torch.bool).cuda()

        forward_funcs, _ = build_combined_hybrid_layer_callables(layer)
        attn_func, _, mlp_func, _, _ = forward_funcs

        node = _FakeNode(_FakeModel(), attention_mask=attention_mask)

        with torch.no_grad():
            expected = layer(hidden_states, attention_mask=attention_mask)
            # Dense attn slot returns the post-MLP output directly.
            attn_out = attn_func(node, hidden_states)
            actual = mlp_func(node, attn_out)

        torch.testing.assert_close(actual, expected)

    def test_build_layer_callables_dispatches_to_combined(self):
        """The shared GPT dispatcher routes ``CombinedHybridLayer`` to us."""
        layer = MambaMLPLayer(
            self.config,
            combined_hybrid_layer_submodules,
            pg_collection=self.pg_collection,
        ).cuda()
        direct = build_combined_hybrid_layer_callables(layer)
        via = build_layer_callables(layer)
        assert len(direct[0]) == len(via[0]) == 5
        assert set(direct[1].keys()) == set(via[1].keys())

    def test_no_op_backward_dw_releases_reference(self):
        """The combined backward_dw wrapper is a no-op that drops the layer ref."""
        layer = MambaMLPLayer(
            self.config,
            combined_hybrid_layer_submodules,
            pg_collection=self.pg_collection,
        ).cuda()
        _, backward_dw = build_combined_hybrid_layer_callables(layer)
        wrapper = backward_dw["attn"]
        assert wrapper.layer is layer
        wrapper.backward_dw()
        assert wrapper.layer is None
