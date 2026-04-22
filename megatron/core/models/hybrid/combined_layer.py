# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Combined hybrid layer: Mamba/GDN + optional Attention + MLP/MoE in one module.

Each `CombinedHybridLayer` instance aggregates a Mamba-family (or transformer-only)
block with an MLP/MoE block, matching the structure of `TransformerLayer` so the
same 1F1B fine-grained schedule plan (attn/dispatch/mlp/combine) can be reused.

The layer's composition is fully determined by its constructor arguments — the
spec provides all possible submodules and the layer chooses which to instantiate.
This design keeps the code readable: the class name tells you exactly what the
layer computes.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Union

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
from megatron.core.utils import deprecate_inference_params


@dataclass
class CombinedHybridLayerSubmodules:
    """All possible building blocks for a combined hybrid layer.

    The layer's constructor selects which submodules to actually instantiate
    based on the `mamba_type`, `attention_type`, and `mlp_type` arguments.
    Unused submodules (e.g. `standard_attention` when `attention_type='gdn'`)
    are simply not built.
    """

    # --- Mamba family: one of {'mamba', 'gdn', 'none'} ---
    mamba_norm: LayerNormBuilder = IdentityOp
    mamba_mixer: Union[ModuleSpec, type] = IdentityOp
    gdn_mixer: Union[ModuleSpec, type] = IdentityOp
    mamba_bda: Union[ModuleSpec, type] = IdentityFuncOp

    # --- Attention: one of {'attention', 'mla', 'none'} ---
    attn_norm: LayerNormBuilder = IdentityOp
    self_attention: Union[ModuleSpec, type] = IdentityOp
    mla_attention: Union[ModuleSpec, type] = IdentityOp
    attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    # --- MLP family: one of {'mlp', 'moe', 'none'} ---
    pre_mlp_layernorm: LayerNormBuilder = IdentityOp
    mlp: Union[ModuleSpec, type] = IdentityOp
    moe: Union[ModuleSpec, type] = IdentityOp
    mlp_bda: Union[ModuleSpec, type] = IdentityFuncOp

    # sharded-state-dict key remapping (rarely used, matches MambaLayer pattern)
    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)


class CombinedHybridLayer(MegatronModule):
    """Combined Mamba/GDN + optional Attention + MLP/MoE layer.

    This layer packs up to three functional blocks into one ``decoder.layers[i]``
    entry so the existing 1F1B fine-grained overlap schedule (designed for
    TransformerLayer) applies uniformly:

    * **Mamba block** (optional): ``mamba_norm → MambaMixer → bias_dropout_add``
    * **Attention block** (optional): ``attn_norm → SelfAttention → bias_dropout_add``
    * **MLP/MoE block** (optional): ``pre_mlp_layernorm → MLP or MoE → bias_dropout_add``

    The three selectors are independent:

    ==================  =====================================================
    ``mamba_type``      ``'mamba'`` · ``'gdn'`` · ``'none'``
    ``attention_type``  ``'attention'`` · ``'mla'`` · ``'none'``
    ``mlp_type``        ``'mlp'`` · ``'moe'`` · ``'none'``
    ==================  =====================================================

    At least one block must be enabled. Typical combinations:

    * ``mamba=mamba, attn=none, mlp=mlp``  → Mamba+MLP
    * ``mamba=mamba, attn=attention, mlp=mlp`` → Mamba + SelfAttn + MLP
    * ``mamba=mamba, attn=none, mlp=moe`` → Mamba + MoE
    * ``mamba=none, attn=attention, mlp=moe`` → dense-attn + MoE (GPT-like)
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: CombinedHybridLayerSubmodules,
        mamba_type: str = 'mamba',
        attention_type: str = 'none',
        mlp_type: str = 'mlp',
        layer_number: int = 1,
        pg_collection: Optional[ProcessGroupCollection] = None,
        pp_layer_offset: int = 0,
    ):
        super().__init__(config)

        assert pg_collection is not None, "pg_collection must be provided"
        assert mamba_type in {'mamba', 'gdn', 'none'}, f"bad mamba_type={mamba_type!r}"
        assert attention_type in {'attention', 'mla', 'none'}, (
            f"bad attention_type={attention_type!r}"
        )
        assert mlp_type in {'mlp', 'moe', 'none'}, f"bad mlp_type={mlp_type!r}"
        assert (
            mamba_type != 'none' or attention_type != 'none' or mlp_type != 'none'
        ), "combined layer must enable at least one block"

        self.config = config
        self.submodules_config = submodules
        self.layer_number = layer_number
        self.hidden_dropout = config.hidden_dropout
        self.pg_collection = pg_collection

        self.mamba_type = mamba_type
        self.attention_type = attention_type
        self.mlp_type = mlp_type

        self.bias_dropout_add_exec_handler = torch.enable_grad

        # ---- Mamba / GDN block ----
        self.mamba_norm = None
        self.mixer = None
        self.mamba_bda = None
        if mamba_type == 'mamba':
            self.mamba_norm = submodules.mamba_norm(config, config.hidden_size)
            self.mixer = build_module(
                submodules.mamba_mixer,
                config,
                d_model=config.hidden_size,
                layer_number=layer_number,
                pg_collection=pg_collection,
                pp_layer_offset=pp_layer_offset,
            )
            self.mamba_bda = build_module(submodules.mamba_bda)
        elif mamba_type == 'gdn':
            self.mamba_norm = submodules.mamba_norm(config, config.hidden_size)
            self.mixer = build_module(
                submodules.gdn_mixer,
                config,
                layer_number=layer_number,
                pg_collection=pg_collection,
            )
            self.mamba_bda = build_module(submodules.mamba_bda)

        # ---- Attention block ----
        self.attn_norm = None
        self.self_attention = None
        self.attn_bda = None
        if attention_type in ('attention', 'mla'):
            self.attn_norm = submodules.attn_norm(config, config.hidden_size)
            attn_spec = (
                submodules.self_attention
                if attention_type == 'attention'
                else submodules.mla_attention
            )
            self.self_attention = build_module(
                attn_spec,
                config=config,
                layer_number=layer_number,
                pg_collection=pg_collection,
            )
            self.attn_bda = build_module(submodules.attn_bda)

        # ---- MLP / MoE block ----
        self.pre_mlp_layernorm = None
        self.mlp = None
        self.mlp_bda = None
        if mlp_type in ('mlp', 'moe'):
            self.pre_mlp_layernorm = submodules.pre_mlp_layernorm(config, config.hidden_size)
            mlp_spec = submodules.mlp if mlp_type == 'mlp' else submodules.moe
            mlp_kwargs = {'config': config}
            if mlp_type == 'moe':
                mlp_kwargs['pg_collection'] = pg_collection
            else:
                mlp_kwargs['tp_group'] = pg_collection.tp
            self.mlp = build_module(mlp_spec, **mlp_kwargs)
            self.mlp_bda = build_module(submodules.mlp_bda)

        self.is_moe_layer = isinstance(self.mlp, MoELayer)

    # ------------------------------------------------------------------
    # Forward halves: these are the entry points used by the 1F1B
    # schedule callables. The full `forward` below composes them for the
    # non-overlap path.
    # ------------------------------------------------------------------

    def forward_ssm_attn(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
    ) -> Tensor:
        """SSM (Mamba/GDN) block followed by optional Attention block.

        This is the "compute-heavy" half of the layer for 1F1B overlap. Returns
        the hidden_states after both blocks have been applied; a subsequent call
        to :meth:`forward_mlp` completes the layer.
        """
        if self.mamba_type != 'none':
            residual = hidden_states
            if self.config.fp32_residual_connection:
                residual = residual.float()
            hidden_states = hidden_states.to(dtype=self.config.params_dtype)
            h = apply_module(self.mamba_norm)(hidden_states)
            mixer_out_with_bias = self.mixer(
                h,
                inference_context=inference_context,
                packed_seq_params=packed_seq_params,
            )
            with self.bias_dropout_add_exec_handler():
                hidden_states = self.mamba_bda(
                    training=self.training, fused=self.config.bias_dropout_fusion
                )(mixer_out_with_bias, residual, self.hidden_dropout)

        if self.attention_type != 'none':
            residual = hidden_states
            if self.config.fp32_residual_connection:
                residual = residual.float()
            h = apply_module(self.attn_norm)(hidden_states)
            attention_output_with_bias = self.self_attention(
                h,
                attention_mask=attention_mask,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
            )
            with self.bias_dropout_add_exec_handler():
                hidden_states = self.attn_bda(
                    training=self.training, fused=self.config.bias_dropout_fusion
                )(attention_output_with_bias, residual, self.hidden_dropout)

        return hidden_states

    def forward_mlp(
        self,
        hidden_states: Tensor,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tensor:
        """MLP or MoE block (pre_norm + compute + BDA).

        Note: for MoE layers, the 1F1B schedule bypasses this method entirely
        and instead drives the MoE dispatch / experts / combine stages
        individually. :meth:`forward` (the non-overlap path) uses this helper
        for both dense and MoE cases so the layer is usable standalone.
        """
        if self.mlp_type == 'none':
            return hidden_states

        residual = hidden_states
        if self.config.fp32_residual_connection:
            residual = residual.float()
        pre_mlp_output = apply_module(self.pre_mlp_layernorm)(hidden_states)
        mlp_out_with_bias = self.mlp(pre_mlp_output)
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(
                training=self.training, fused=self.config.bias_dropout_fusion
            )(mlp_out_with_bias, residual, self.hidden_dropout)
        return hidden_states

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
    ) -> Tensor:
        """Default (non-overlap) forward: run SSM/Attn then MLP sequentially."""
        inference_context = deprecate_inference_params(inference_context, inference_params)

        hidden_states = self.forward_ssm_attn(
            hidden_states,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )
        hidden_states = self.forward_mlp(
            hidden_states,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
        )
        return hidden_states

    def sharded_state_dict(
        self,
        prefix: str = '',
        sharded_offsets: tuple = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """Standard nested sharded state dict.

        Keys are ``<prefix>.<submodule>.<...>`` where ``<submodule>`` is one of
        ``mamba_norm``, ``mixer``, ``attn_norm``, ``self_attention``,
        ``pre_mlp_layernorm``, ``mlp``. Omitting a block simply omits its keys.
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        if self.submodules_config.sharded_state_dict_keys_map:
            apply_prefix_mapping(
                sharded_state_dict,
                {f'{prefix}{k}': f'{prefix}{v}' for k, v in
                 self.submodules_config.sharded_state_dict_keys_map.items()},
            )
        return sharded_state_dict


# ----------------------------------------------------------------------
# Concrete subclasses — pin the composition at construction time so
# `type(layer).__name__` is informative and the constructor signature is
# minimal for callers.
# ----------------------------------------------------------------------


class MambaMLPLayer(CombinedHybridLayer):
    """Mamba + MLP (no attention)."""

    def __init__(self, config, submodules, **kwargs):
        super().__init__(
            config, submodules, mamba_type='mamba', attention_type='none', mlp_type='mlp',
            **kwargs,
        )


class MambaMoELayer(CombinedHybridLayer):
    """Mamba + MoE (no attention)."""

    def __init__(self, config, submodules, **kwargs):
        super().__init__(
            config, submodules, mamba_type='mamba', attention_type='none', mlp_type='moe',
            **kwargs,
        )


class MambaSelfAttnMLPLayer(CombinedHybridLayer):
    """Mamba + SelfAttention + MLP."""

    def __init__(self, config, submodules, **kwargs):
        super().__init__(
            config, submodules, mamba_type='mamba', attention_type='attention', mlp_type='mlp',
            **kwargs,
        )


class MambaSelfAttnMoELayer(CombinedHybridLayer):
    """Mamba + SelfAttention + MoE (the DeepSeek-style hybrid)."""

    def __init__(self, config, submodules, **kwargs):
        super().__init__(
            config, submodules, mamba_type='mamba', attention_type='attention', mlp_type='moe',
            **kwargs,
        )


class MambaGDNMLPLayer(CombinedHybridLayer):
    """Mamba + GatedDeltaNet + MLP — wait, GDN replaces Mamba, not adds to it.

    Kept as an explicit class to match the design doc naming, but note that
    'GDN' is itself a Mamba-family mixer, so ``mamba_type='gdn'`` is used.
    """

    def __init__(self, config, submodules, **kwargs):
        super().__init__(
            config, submodules, mamba_type='gdn', attention_type='none', mlp_type='mlp',
            **kwargs,
        )


class AttnMoELayer(CombinedHybridLayer):
    """Pure Attention + MoE (no Mamba) — GPT-style MoE layer inside the hybrid stack."""

    def __init__(self, config, submodules, **kwargs):
        super().__init__(
            config, submodules, mamba_type='none', attention_type='attention', mlp_type='moe',
            **kwargs,
        )


# Re-export for easy import
__all__ = [
    'CombinedHybridLayer',
    'CombinedHybridLayerSubmodules',
    'MambaMLPLayer',
    'MambaMoELayer',
    'MambaSelfAttnMLPLayer',
    'MambaSelfAttnMoELayer',
    'MambaGDNMLPLayer',
    'AttnMoELayer',
]
