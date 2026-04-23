# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.

"""Checkpoint interoperability tests for the combined-layer path.

The combined layer supports two sharded-state-dict layouts:

1. **Nested** (default): keys live under the combined-layer slot, e.g.
   ``decoder.layers.0.mixer.*``, ``decoder.layers.0.self_attention.*``, ...
2. **Legacy-compat** (``legacy_sharded_state_dict=True`` on the layer, plumbed
   through by :class:`HybridStack`): each submodule's keys are emitted under
   the standalone layer index it *would* have had in the separate-layer
   hybrid stack, so checkpoints round-trip between the two paths.

This test file covers the legacy-compat layout against the reference produced
by the legacy separate-layer path built from the same spec.
"""

import pytest
import torch

from megatron.core.models.hybrid.combined_layer import CombinedHybridLayer
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
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
        sd = model.sharded_state_dict()
        return {k for k in sd.keys() if k.startswith("decoder.layers.")}

    def test_nested_layout_uses_combined_slot_indices(self):
        """Default layout keeps all submodules under the combined-layer slot."""
        config = _tiny_config(num_layers=3)
        model = HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=1024,
            max_sequence_length=64,
            hybrid_layer_pattern="[M*-]",
            pre_process=True,
            post_process=True,
            pg_collection=self.pg_collection,
        )
        # [M*-] is a single combined layer at slot 0. Every submodule key
        # should live under ``decoder.layers.0.``.
        keys = self._sharded_keys(model)
        slot_indices = {k.split(".")[2] for k in keys}
        assert slot_indices == {"0"}

    def test_legacy_layout_emits_standalone_indices(self):
        """Enabling ``legacy_sharded_state_dict`` spreads keys across 3 slots."""
        config = _tiny_config(num_layers=3)
        model = HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=1024,
            max_sequence_length=64,
            hybrid_layer_pattern="[M*-]",
            pre_process=True,
            post_process=True,
            pg_collection=self.pg_collection,
        )
        # Flip the combined layer into legacy-compat mode (after construction
        # because :class:`HybridModel` does not expose the flag as a top-level
        # argument; downstream training frameworks that need interop set it on
        # the layer directly).
        combined_layer = model.decoder.layers[0]
        assert isinstance(combined_layer, CombinedHybridLayer)
        # Give each submodule role an explicit standalone index (M=0, *=1, -=2).
        combined_layer.submodule_layer_numbers = {"mamba": 1, "attention": 2, "mlp": 3}
        combined_layer.legacy_sharded_state_dict = True

        keys = self._sharded_keys(model)
        slot_indices = {k.split(".")[2] for k in keys}
        # Submodules now live under slots 0 (mamba), 1 (attention), 2 (mlp).
        assert {"0", "1", "2"}.issubset(slot_indices)

        # The mixer's key should live under the mamba slot (idx 0).
        mixer_keys = [k for k in keys if ".mixer." in k]
        assert mixer_keys, "mixer keys missing"
        for k in mixer_keys:
            assert k.startswith("decoder.layers.0."), k

        # self_attention keys under idx 1.
        attn_keys = [k for k in keys if ".self_attention." in k]
        assert attn_keys, "self_attention keys missing"
        for k in attn_keys:
            assert k.startswith("decoder.layers.1."), k

        # mlp keys under idx 2.
        mlp_keys = [k for k in keys if ".mlp." in k and ".mlp_bda" not in k]
        assert mlp_keys, "mlp keys missing"
        for k in mlp_keys:
            assert k.startswith("decoder.layers.2."), k

    def test_legacy_layout_matches_separate_layer_keys(self):
        """Legacy layout keys are a superset of the separate-layer reference.

        Builds a 3-layer ``"M*-"`` separate-layer model and the equivalent
        ``"[M*-]"`` combined model; after enabling legacy layout on the
        combined model, every weighted key in the reference must also appear
        in the combined model's sharded_state_dict.
        """
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
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=1024,
            max_sequence_length=64,
            hybrid_layer_pattern="[M*-]",
            pre_process=True,
            post_process=True,
            pg_collection=self.pg_collection,
        )
        cl = combined_model.decoder.layers[0]
        cl.submodule_layer_numbers = {"mamba": 1, "attention": 2, "mlp": 3}
        cl.legacy_sharded_state_dict = True

        legacy_keys = self._sharded_keys(legacy_model)
        combined_keys = self._sharded_keys(combined_model)
        missing = legacy_keys - combined_keys
        assert not missing, (
            f"Combined-layer legacy state dict is missing {len(missing)} keys "
            f"present in the reference. Sample: {sorted(missing)[:5]}"
        )
