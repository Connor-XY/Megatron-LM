# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.

"""Checkpoint interoperability tests for the combined-layer path.

These tests verify that the sharded state dict keys produced by the
combined-layer path align with the keys produced by the legacy separate-layer
path, so a checkpoint saved under one can be loaded by the other.

The full save/load round-trip (writing to disk via MCore's dist_checkpointing)
needs a distributed process group; the tests here validate the key layout via
direct ``sharded_state_dict`` invocation.
"""

import pytest
import torch

from megatron.core.models.hybrid.hybrid_layer_specs import (
    hybrid_stack_spec,
    hybrid_stack_spec_with_combined_layers,
)
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.process_groups_config import ProcessGroupCollection
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


@pytest.mark.internal
class TestCombinedCheckpointRoundtrip:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.pg_collection = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=["tp", "pp", "cp", "embd"]
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _sharded_keys(self, model):
        """Return the set of sharded-state-dict keys (prefixes of the decoder)."""
        sd = model.sharded_state_dict()
        return {k for k in sd.keys() if k.startswith("decoder.layers.")}

    def test_combined_layer_emits_legacy_compat_keys(self):
        """Combined-layer ``[M*-]`` emits keys under 3 standalone indices."""
        config = _tiny_config(num_layers=3)
        model = HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec_with_combined_layers,
            vocab_size=1024,
            max_sequence_length=64,
            hybrid_layer_pattern="[M*-]",
            pre_process=True,
            post_process=True,
            pg_collection=self.pg_collection,
        )

        keys = self._sharded_keys(model)
        # The hybrid_stack_spec fuses norms into ``linear_qkv`` /
        # ``linear_fc1`` (TELayerNormColumnParallelLinear) and ``in_proj``,
        # so the standalone ``norm`` / ``input_layernorm`` /
        # ``pre_mlp_layernorm`` attributes are IdentityOp with no parameters.
        # The checkpoint key layout therefore tests the weighted submodules:
        #   Mamba -> decoder.layers.0.mixer.*
        #   Attn  -> decoder.layers.1.self_attention.*
        #   MLP   -> decoder.layers.2.mlp.*
        prefixes_seen = set()
        for k in keys:
            parts = k.split(".")
            # e.g. ['decoder', 'layers', '1', 'self_attention', 'linear_qkv', 'weight']
            if len(parts) >= 4:
                prefixes_seen.add((parts[2], parts[3]))

        expected_subset = {
            ("0", "mixer"),
            ("1", "self_attention"),
            ("2", "mlp"),
        }
        missing = expected_subset - prefixes_seen
        assert not missing, f"missing expected sharded keys for subset {missing}"

    def test_combined_key_layout_matches_legacy(self):
        """Combined-layer and legacy keys agree for every submodule."""
        # Use the exact same pattern spelled two ways.
        config = _tiny_config(num_layers=3)
        legacy_model = HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=1024,
            max_sequence_length=64,
            hybrid_layer_pattern="M*-",
            pre_process=True,
            post_process=True,
            pg_collection=self.pg_collection,
        )
        combined_model = HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec_with_combined_layers,
            vocab_size=1024,
            max_sequence_length=64,
            hybrid_layer_pattern="[M*-]",
            pre_process=True,
            post_process=True,
            pg_collection=self.pg_collection,
        )

        legacy_keys = self._sharded_keys(legacy_model)
        combined_keys = self._sharded_keys(combined_model)

        # Legacy keys are a full ground truth. Combined should be a superset of
        # the legacy keys (it may emit a few extra due to TE's internal
        # bookkeeping but the core weight keys must match).
        missing_in_combined = legacy_keys - combined_keys
        assert not missing_in_combined, (
            f"Combined-layer state dict is missing {len(missing_in_combined)} "
            f"keys present in the legacy state dict. Sample: "
            f"{list(missing_in_combined)[:5]}"
        )
