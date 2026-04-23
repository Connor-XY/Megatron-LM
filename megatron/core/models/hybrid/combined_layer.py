# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.

"""Combined hybrid layer: Mamba/GDN + optional Attention + MLP/MoE in one module.

The combined layer bundles up to three functional blocks into a single
``decoder.layers[i]`` entry, matching the overall shape of GPT's
:class:`TransformerLayer`. The existing architecture-agnostic 1F1B fine-grained
schedule plan therefore applies to hybrid models without model-specific
branches: MoE All-to-All dispatch/combine can run on the comm stream while
another microbatch's mixer/attention compute runs on the compute stream.

Composition is controlled by three independent selectors passed to the
constructor:

* ``mamba_type``     -- ``'mamba'`` | ``'gdn'`` | ``'none'``  (the "mixer" slot)
* ``attention_type`` -- ``'attention'`` | ``'mla'`` | ``'none'``
* ``mlp_type``       -- ``'mlp'`` | ``'moe'`` | ``'none'``

The spec (``CombinedHybridLayerSubmodules``) provides every possible submodule;
the layer only builds the ones matching its selectors. Concrete subclasses
(:class:`MambaMLPLayer`, :class:`MambaSelfAttnMoELayer`, etc.) pin the
combination so ``type(layer).__name__`` tells you exactly what runs.

Classifying GDN as a sequence mixer (not an "attention" alternative) is
deliberate: a GDN layer replaces Mamba in the first slot, so mixer +
attention + MLP/MoE layers can be expressed directly. The attention slot is
reserved for SelfAttention / MLA when either is needed on top of the mixer.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import torch
from torch import Tensor

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_layer import LayerNormBuilder
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module
from megatron.core.utils import deprecate_inference_params, make_viewless_tensor

# Valid values for the three selectors. Exposed as module-level constants so
# callers and tests can introspect them without string magic.
MAMBA_TYPE_MAMBA = "mamba"
MAMBA_TYPE_GDN = "gdn"
MAMBA_TYPE_NONE = "none"
VALID_MAMBA_TYPES = (MAMBA_TYPE_MAMBA, MAMBA_TYPE_GDN, MAMBA_TYPE_NONE)

ATTENTION_TYPE_ATTENTION = "attention"
ATTENTION_TYPE_MLA = "mla"
ATTENTION_TYPE_NONE = "none"
VALID_ATTENTION_TYPES = (ATTENTION_TYPE_ATTENTION, ATTENTION_TYPE_MLA, ATTENTION_TYPE_NONE)

MLP_TYPE_MLP = "mlp"
MLP_TYPE_MOE = "moe"
MLP_TYPE_NONE = "none"
VALID_MLP_TYPES = (MLP_TYPE_MLP, MLP_TYPE_MOE, MLP_TYPE_NONE)


@dataclass
class CombinedHybridLayerSubmodules:
    """Building blocks for :class:`CombinedHybridLayer`.

    All possible submodules are declared here; the layer's constructor picks
    which to actually instantiate based on ``mamba_type`` / ``attention_type``
    / ``mlp_type``. Unused specs are ignored.
    """

    # --- Sequence mixer (``mamba_type`` picks between ``mamba_mixer`` and ``gdn_mixer``) ---
    mamba_norm: LayerNormBuilder = IdentityOp
    mamba_mixer: Union[ModuleSpec, type] = IdentityOp
    gdn_mixer: Union[ModuleSpec, type] = IdentityOp
    mamba_bda: Union[ModuleSpec, type] = IdentityFuncOp

    # --- Attention (``attention_type`` picks between ``self_attention`` and ``mla_attention``) ---
    attn_norm: LayerNormBuilder = IdentityOp
    self_attention: Union[ModuleSpec, type] = IdentityOp
    mla_attention: Union[ModuleSpec, type] = IdentityOp
    attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    # --- MLP / MoE (``mlp_type`` picks between ``mlp`` and ``moe``) ---
    pre_mlp_layernorm: LayerNormBuilder = IdentityOp
    mlp: Union[ModuleSpec, type] = IdentityOp
    moe: Union[ModuleSpec, type] = IdentityOp
    mlp_bda: Union[ModuleSpec, type] = IdentityFuncOp

    # Sharded-state-dict key remapping (rarely used, matches TransformerLayer
    # / MambaLayer pattern).
    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)


class CombinedHybridLayer(MegatronModule):
    """Combined Mamba/GDN + optional Attention + MLP/MoE decoder layer.

    Shape ``[s, b, h]`` in, same shape out. See the module docstring for the
    selector semantics. At least one of the three blocks must be enabled.

    Args:
        config: TransformerConfig.
        submodules: :class:`CombinedHybridLayerSubmodules`; any building block
            this layer does not use is ignored.
        mamba_type: ``'mamba'`` | ``'gdn'`` | ``'none'``.
        attention_type: ``'attention'`` | ``'mla'`` | ``'none'``.
        mlp_type: ``'mlp'`` | ``'moe'`` | ``'none'``.
        layer_number: 1-indexed combined-layer position in the decoder.
        hidden_dropout: Residual-path dropout; defaults to
            ``config.hidden_dropout``.
        pg_collection: Process groups used by inner blocks.
        pp_layer_offset: Global layer offset for PP-aware submodule sharding.
        submodule_layer_numbers: Optional ``{"mamba": i, "attention": j,
            "mlp": k}`` mapping giving each inner block the layer number it
            would have had in the legacy separate-layer hybrid stack. Used by
            FP8 per-layer contexts and for producing legacy-compatible
            sharded-checkpoint keys when ``legacy_sharded_state_dict=True``.
            Defaults to ``layer_number`` for every role.
        legacy_sharded_state_dict: When True, :meth:`sharded_state_dict` emits
            keys under the per-submodule standalone indices in
            ``submodule_layer_numbers`` so a checkpoint saved from a legacy
            separate-layer hybrid model can be loaded into this combined
            layer (and vice versa). When False (default), uses the simpler
            nested layout ``<prefix>.<submodule>.<...>`` which matches the
            layer's own module tree.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: CombinedHybridLayerSubmodules,
        mamba_type: str = MAMBA_TYPE_MAMBA,
        attention_type: str = ATTENTION_TYPE_NONE,
        mlp_type: str = MLP_TYPE_MLP,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        pp_layer_offset: int = 0,
        submodule_layer_numbers: Optional[Dict[str, int]] = None,
        legacy_sharded_state_dict: bool = False,
    ) -> None:
        super().__init__(config=config)
        assert pg_collection is not None, "pg_collection must be provided"
        assert mamba_type in VALID_MAMBA_TYPES, (
            f"mamba_type must be one of {VALID_MAMBA_TYPES}, got {mamba_type!r}"
        )
        assert attention_type in VALID_ATTENTION_TYPES, (
            f"attention_type must be one of {VALID_ATTENTION_TYPES}, got {attention_type!r}"
        )
        assert mlp_type in VALID_MLP_TYPES, (
            f"mlp_type must be one of {VALID_MLP_TYPES}, got {mlp_type!r}"
        )
        assert (
            mamba_type != MAMBA_TYPE_NONE
            or attention_type != ATTENTION_TYPE_NONE
            or mlp_type != MLP_TYPE_NONE
        ), "combined layer must enable at least one block"

        self.config = config
        self.submodules_config = submodules
        self.layer_number = layer_number
        self.hidden_dropout = config.hidden_dropout if hidden_dropout is None else hidden_dropout
        self.pg_collection = pg_collection
        self.tp_group = pg_collection.tp

        self.mamba_type = mamba_type
        self.attention_type = attention_type
        self.mlp_type = mlp_type
        self.legacy_sharded_state_dict = legacy_sharded_state_dict

        # Per-submodule "legacy" layer numbers. These are the indices each
        # block would have had in the separate-layer hybrid stack; used for
        # FP8 per-layer contexts and (optionally) checkpoint interop.
        self.submodule_layer_numbers = {
            "mamba": layer_number,
            "attention": layer_number,
            "mlp": layer_number,
        }
        if submodule_layer_numbers is not None:
            self.submodule_layer_numbers.update(submodule_layer_numbers)

        self.bias_dropout_add_exec_handler = torch.enable_grad

        # ---- Mamba / GDN block ----
        self.mamba_norm = None
        self.mixer = None
        self.mamba_bda = None
        if mamba_type == MAMBA_TYPE_MAMBA:
            self.mamba_norm = submodules.mamba_norm(config, config.hidden_size)
            self.mixer = build_module(
                submodules.mamba_mixer,
                config,
                d_model=config.hidden_size,
                layer_number=self.submodule_layer_numbers["mamba"],
                pg_collection=pg_collection,
                pp_layer_offset=pp_layer_offset,
            )
            self.mamba_bda = build_module(submodules.mamba_bda)
        elif mamba_type == MAMBA_TYPE_GDN:
            self.mamba_norm = submodules.mamba_norm(config, config.hidden_size)
            self.mixer = build_module(
                submodules.gdn_mixer,
                config,
                layer_number=self.submodule_layer_numbers["mamba"],
                pg_collection=pg_collection,
            )
            self.mamba_bda = build_module(submodules.mamba_bda)

        # ---- Attention block ----
        self.attn_norm = None
        self.self_attention = None
        self.attn_bda = None
        if attention_type != ATTENTION_TYPE_NONE:
            self.attn_norm = submodules.attn_norm(config, config.hidden_size)
            attn_spec = (
                submodules.self_attention
                if attention_type == ATTENTION_TYPE_ATTENTION
                else submodules.mla_attention
            )
            attn_kwargs: Dict[str, Any] = {
                "config": config,
                "layer_number": self.submodule_layer_numbers["attention"],
                "pg_collection": pg_collection,
            }
            if config.context_parallel_size > 1 and config.cp_comm_type is not None:
                if isinstance(config.cp_comm_type, list):
                    attn_kwargs["cp_comm_type"] = config.cp_comm_type[
                        self.submodule_layer_numbers["attention"] - 1
                    ]
                else:
                    attn_kwargs["cp_comm_type"] = config.cp_comm_type
            if pp_layer_offset is not None:
                attn_kwargs["pp_layer_offset"] = pp_layer_offset
            self.self_attention = build_module(attn_spec, **attn_kwargs)
            self.attn_bda = build_module(submodules.attn_bda)

        # ---- MLP / MoE block ----
        self.pre_mlp_layernorm = None
        self.mlp = None
        self.mlp_bda = None
        if mlp_type != MLP_TYPE_NONE:
            self.pre_mlp_layernorm = submodules.pre_mlp_layernorm(config, config.hidden_size)
            mlp_spec = submodules.mlp if mlp_type == MLP_TYPE_MLP else submodules.moe
            mlp_kwargs: Dict[str, Any] = {"config": config}
            if mlp_type == MLP_TYPE_MOE:
                mlp_kwargs["pg_collection"] = pg_collection
            else:
                mlp_kwargs["tp_group"] = pg_collection.tp
            self.mlp = build_module(mlp_spec, **mlp_kwargs)
            if hasattr(self.mlp, "set_layer_number"):
                self.mlp.set_layer_number(self.submodule_layer_numbers["mlp"])
            self.mlp_bda = build_module(submodules.mlp_bda)

        self.is_moe_layer = isinstance(self.mlp, MoELayer)

        # Recompute / offload knobs consumed by the schedule callables. The
        # combined-layer path does not yet support selective pre-MLP norm
        # recompute; the flag stays False so the MoE callable skips that
        # branch. Same reasoning for the offload flag.
        self.recompute_pre_mlp_layernorm = False
        self.offload_mlp_norm = False

    # ------------------------------------------------------------------
    # Forward halves -- entry points used by the 1F1B schedule callables.
    # The full ``forward`` below composes them for the non-overlap path.
    # ------------------------------------------------------------------

    def forward_ssm_attn(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        *,
        inference_params: Optional[Any] = None,
    ) -> Tensor:
        """Run the mixer block (Mamba/GDN) then the optional attention block.

        This is the compute-heavy half of the layer in the 1F1B overlap split.
        """
        inference_context = deprecate_inference_params(inference_context, inference_params)

        if self.mamba_type != MAMBA_TYPE_NONE:
            residual = hidden_states
            if self.config.fp32_residual_connection:
                residual = residual.float()
            hidden_states = hidden_states.to(dtype=self.config.params_dtype)
            mixer_norm_out = apply_module(self.mamba_norm)(hidden_states)
            mixer_out_with_bias = self.mixer(
                mixer_norm_out,
                inference_context=inference_context,
                packed_seq_params=packed_seq_params,
            )
            with self.bias_dropout_add_exec_handler():
                hidden_states = self.mamba_bda(
                    training=self.training, fused=self.config.bias_dropout_fusion
                )(mixer_out_with_bias, residual, self.hidden_dropout)

        if self.attention_type != ATTENTION_TYPE_NONE:
            attn_norm_out = apply_module(self.attn_norm)(hidden_states)
            if isinstance(attn_norm_out, tuple):
                # Fused residual RMSNorm returns (output, residual).
                if len(attn_norm_out) != 2:
                    raise ValueError(
                        "When attn_norm returns a tuple it must have 2 elements "
                        f"(output, residual); got {len(attn_norm_out)}."
                    )
                attn_norm_out, residual = attn_norm_out
            else:
                residual = hidden_states
            if self.config.fp32_residual_connection:
                residual = residual.float()

            attention_output_with_bias = self.self_attention(
                attn_norm_out,
                attention_mask=attention_mask,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                rotary_pos_cos_sin=rotary_pos_cos_sin,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
            )
            with self.bias_dropout_add_exec_handler():
                hidden_states = self.attn_bda(self.training, self.config.bias_dropout_fusion)(
                    attention_output_with_bias, residual, self.hidden_dropout
                )

        return hidden_states

    def forward_mlp(self, hidden_states: Tensor) -> Tensor:
        """Run the MLP/MoE block + residual BDA.

        For the 1F1B overlap path with an MoE block, the scheduler bypasses
        this method and drives dispatch / experts / combine individually.
        This helper is used by :meth:`forward` and by the dense-path schedule
        callable.
        """
        if self.mlp_type == MLP_TYPE_NONE:
            return hidden_states

        pre_mlp_out = apply_module(self.pre_mlp_layernorm)(hidden_states)
        if isinstance(pre_mlp_out, tuple):
            if len(pre_mlp_out) != 2:
                raise ValueError(
                    "When pre_mlp_layernorm returns a tuple it must have 2 elements "
                    f"(output, residual); got {len(pre_mlp_out)}."
                )
            pre_mlp_out, residual = pre_mlp_out
        else:
            residual = hidden_states
        if self.config.fp32_residual_connection:
            residual = residual.float()

        mlp_output_with_bias = self.mlp(pre_mlp_out)
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )
        return make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        sequence_len_offset: Optional[Tensor] = None,
        *,
        inference_params: Optional[Any] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tensor:
        """Monolithic forward: run the mixer+attention block then the MLP block."""
        hidden_states = self.forward_ssm_attn(
            hidden_states,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            attention_bias=attention_bias,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            inference_params=inference_params,
        )
        return self.forward_mlp(hidden_states)

    # ------------------------------------------------------------------
    # Sharded state dict
    # ------------------------------------------------------------------

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """Sharded state dict in either the layer-nested or legacy layout.

        When ``legacy_sharded_state_dict=True``, keys are emitted under the
        *standalone* layer indices carried by ``submodule_layer_numbers`` with
        TransformerLayer-compatible submodule names, so checkpoints
        interoperate with the legacy separate-layer hybrid path:

            prefix="decoder.layers.0."  (combined-layer slot)
              -> decoder.layers.{mamba_idx-1}.{mamba_norm|mixer|mamba_bda}.*
              -> decoder.layers.{attn_idx-1}.{input_layernorm|self_attention|
                                              self_attn_bda}.*
              -> decoder.layers.{mlp_idx-1}.{pre_mlp_layernorm|mlp|mlp_bda}.*

        Otherwise (default), returns the simpler nested layout using the
        layer's own attribute names.
        """
        if not self.legacy_sharded_state_dict:
            sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
            prefixed_map = {
                f"{prefix}{k}": f"{prefix}{v}"
                for k, v in self.submodules_config.sharded_state_dict_keys_map.items()
            }
            if prefixed_map:
                apply_prefix_mapping(sharded_state_dict, prefixed_map)
            return sharded_state_dict

        return self._legacy_compat_sharded_state_dict(prefix, sharded_offsets, metadata)

    def _legacy_compat_sharded_state_dict(
        self,
        prefix: str,
        sharded_offsets: tuple,
        metadata: Optional[dict],
    ) -> ShardedStateDict:
        """Emit keys keyed by standalone submodule indices (legacy layout)."""
        import torch.nn as nn

        from megatron.core.transformer.utils import sharded_state_dict_default

        if not prefix.endswith("."):
            raise ValueError(
                f"Legacy sharded_state_dict expects a prefix ending in '.' "
                f"(e.g. 'decoder.layers.0.'); got {prefix!r}."
            )
        parts = prefix[:-1].rsplit(".", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            raise ValueError(
                f"Legacy sharded_state_dict expects a prefix of the form "
                f"'<stem>.<idx>.'; got {prefix!r}."
            )
        stem = parts[0] + "."

        def _submod(submodule, sub_prefix):
            if not isinstance(submodule, nn.Module):
                # Callable BDAs (e.g. :func:`get_bias_dropout_add`) have no
                # parameters -- contribute nothing to the checkpoint.
                return {}
            return sharded_state_dict_default(
                submodule, sub_prefix, sharded_offsets, metadata, tp_group=self.tp_group
            )

        result: ShardedStateDict = {}

        if self.mamba_type != MAMBA_TYPE_NONE:
            idx = self.submodule_layer_numbers["mamba"] - 1
            pfx = f"{stem}{idx}."
            result.update(_submod(self.mamba_norm, f"{pfx}mamba_norm."))
            result.update(_submod(self.mixer, f"{pfx}mixer."))
            result.update(_submod(self.mamba_bda, f"{pfx}mamba_bda."))

        if self.attention_type != ATTENTION_TYPE_NONE:
            idx = self.submodule_layer_numbers["attention"] - 1
            pfx = f"{stem}{idx}."
            result.update(_submod(self.attn_norm, f"{pfx}input_layernorm."))
            result.update(_submod(self.self_attention, f"{pfx}self_attention."))
            result.update(_submod(self.attn_bda, f"{pfx}self_attn_bda."))

        if self.mlp_type != MLP_TYPE_NONE:
            idx = self.submodule_layer_numbers["mlp"] - 1
            pfx = f"{stem}{idx}."
            result.update(_submod(self.pre_mlp_layernorm, f"{pfx}pre_mlp_layernorm."))
            result.update(_submod(self.mlp, f"{pfx}mlp."))
            result.update(_submod(self.mlp_bda, f"{pfx}mlp_bda."))

        if self.submodules_config.sharded_state_dict_keys_map:
            apply_prefix_mapping(
                result, dict(self.submodules_config.sharded_state_dict_keys_map)
            )
        return result


# ---------------------------------------------------------------------------
# Concrete subclasses -- pin the composition at construction time so the
# class name describes the layer and the spec stays uniform.
# ---------------------------------------------------------------------------


class MambaMLPLayer(CombinedHybridLayer):
    """Mamba + MLP (no attention)."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("mamba_type", MAMBA_TYPE_MAMBA)
        kwargs.setdefault("attention_type", ATTENTION_TYPE_NONE)
        kwargs.setdefault("mlp_type", MLP_TYPE_MLP)
        super().__init__(*args, **kwargs)


class MambaMoELayer(CombinedHybridLayer):
    """Mamba + MoE (no attention)."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("mamba_type", MAMBA_TYPE_MAMBA)
        kwargs.setdefault("attention_type", ATTENTION_TYPE_NONE)
        kwargs.setdefault("mlp_type", MLP_TYPE_MOE)
        super().__init__(*args, **kwargs)


class MambaSelfAttnMLPLayer(CombinedHybridLayer):
    """Mamba + SelfAttention + MLP."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("mamba_type", MAMBA_TYPE_MAMBA)
        kwargs.setdefault("attention_type", ATTENTION_TYPE_ATTENTION)
        kwargs.setdefault("mlp_type", MLP_TYPE_MLP)
        super().__init__(*args, **kwargs)


class MambaSelfAttnMoELayer(CombinedHybridLayer):
    """Mamba + SelfAttention + MoE (the DeepSeek-style hybrid)."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("mamba_type", MAMBA_TYPE_MAMBA)
        kwargs.setdefault("attention_type", ATTENTION_TYPE_ATTENTION)
        kwargs.setdefault("mlp_type", MLP_TYPE_MOE)
        super().__init__(*args, **kwargs)


class GDNMLPLayer(CombinedHybridLayer):
    """GatedDeltaNet + MLP (GDN replaces Mamba in the mixer slot)."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("mamba_type", MAMBA_TYPE_GDN)
        kwargs.setdefault("attention_type", ATTENTION_TYPE_NONE)
        kwargs.setdefault("mlp_type", MLP_TYPE_MLP)
        super().__init__(*args, **kwargs)


class GDNSelfAttnMLPLayer(CombinedHybridLayer):
    """GatedDeltaNet + SelfAttention + MLP."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("mamba_type", MAMBA_TYPE_GDN)
        kwargs.setdefault("attention_type", ATTENTION_TYPE_ATTENTION)
        kwargs.setdefault("mlp_type", MLP_TYPE_MLP)
        super().__init__(*args, **kwargs)


class GDNSelfAttnMoELayer(CombinedHybridLayer):
    """GatedDeltaNet + SelfAttention + MoE."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("mamba_type", MAMBA_TYPE_GDN)
        kwargs.setdefault("attention_type", ATTENTION_TYPE_ATTENTION)
        kwargs.setdefault("mlp_type", MLP_TYPE_MOE)
        super().__init__(*args, **kwargs)


# Backward-compatible alias for the earlier class name. GDN is the mixer; no
# Mamba block is instantiated by this layer.
MambaGDNMLPLayer = GDNMLPLayer


class AttnMoELayer(CombinedHybridLayer):
    """Pure Attention + MoE (no mixer) -- GPT-style MoE inside a hybrid stack."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("mamba_type", MAMBA_TYPE_NONE)
        kwargs.setdefault("attention_type", ATTENTION_TYPE_ATTENTION)
        kwargs.setdefault("mlp_type", MLP_TYPE_MOE)
        super().__init__(*args, **kwargs)


__all__ = [
    "CombinedHybridLayer",
    "CombinedHybridLayerSubmodules",
    "MambaMLPLayer",
    "MambaMoELayer",
    "MambaSelfAttnMLPLayer",
    "MambaSelfAttnMoELayer",
    "GDNMLPLayer",
    "GDNSelfAttnMLPLayer",
    "GDNSelfAttnMoELayer",
    "MambaGDNMLPLayer",
    "AttnMoELayer",
    "MAMBA_TYPE_MAMBA",
    "MAMBA_TYPE_GDN",
    "MAMBA_TYPE_NONE",
    "ATTENTION_TYPE_ATTENTION",
    "ATTENTION_TYPE_MLA",
    "ATTENTION_TYPE_NONE",
    "MLP_TYPE_MLP",
    "MLP_TYPE_MOE",
    "MLP_TYPE_NONE",
]
