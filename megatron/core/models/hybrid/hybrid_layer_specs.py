# Copyright (c) 2023-2026, NVIDIA CORPORATION. All rights reserved.

from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TELinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.gpt.moe_module_specs import (
    get_inference_optimized_moe_spec,
    get_moe_module_spec,
)
from megatron.core.models.hybrid.hybrid_block import HybridStack, HybridStackSubmodules
from megatron.core.ssm.gated_delta_net import GatedDeltaNet, GatedDeltaNetSubmodules
from megatron.core.ssm.mamba_attn_mlp_layer import (
    MambaAttnMLPLayerSubmodules,
    MambaGDNMLPLayer,
    MambaMLPLayer,
    MambaSelfAttnMLPLayer,
)
from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules
from megatron.core.ssm.mamba_mixer import MambaMixer, MambaMixerSubmodules
from megatron.core.ssm.mlp_layer import MLPLayer
from megatron.core.tensor_parallel import (
    InferenceColumnParallelLinear,
    InferenceLayerNormColumnParallelLinear,
    InferenceRowParallelLinear,
)
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexer,
    DSAIndexerSubmodules,
    DSAttention,
    DSAttentionSubmodules,
)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSubmodules,
)
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionBlock,
    MultiTokenPredictionBlockSubmodules,
    MultiTokenPredictionLayer,
    MultiTokenPredictionLayerSubmodules,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import (
    MoETransformerLayer,
    TransformerLayer,
    TransformerLayerSubmodules,
)

# This should be private and should not be used outside of this file.
moe = get_moe_module_spec(
    use_te=True,
    num_experts=8,  # Can be any positive integer (must not be None).
    moe_grouped_gemm=True,
)

# Inference-optimized MoE spec
moe_inference = get_inference_optimized_moe_spec()


# MTP block spec - provides norms and projection only.
# Inner layers are built by MultiTokenPredictionLayer using nested HybridStack
_hybrid_mtp_block_spec = ModuleSpec(
    module=MultiTokenPredictionBlock,
    submodules=MultiTokenPredictionBlockSubmodules(
        layer_specs=[
            ModuleSpec(
                module=MultiTokenPredictionLayer,
                submodules=MultiTokenPredictionLayerSubmodules(
                    enorm=TENorm,
                    hnorm=TENorm,
                    eh_proj=TEColumnParallelLinear,
                    mtp_model_layer=None,  # Built via pattern + hybrid_submodules
                    layer_norm=TENorm,
                ),
            )
        ]
    ),
)


hybrid_stack_spec = ModuleSpec(
    module=HybridStack,
    submodules=HybridStackSubmodules(
        mamba_layer=ModuleSpec(
            module=MambaLayer,
            submodules=MambaLayerSubmodules(
                mixer=ModuleSpec(
                    module=MambaMixer,
                    submodules=MambaMixerSubmodules(
                        in_proj=TELayerNormColumnParallelLinear, out_proj=TERowParallelLinear
                    ),
                ),
                mamba_bda=get_bias_dropout_add,
            ),
        ),
        gdn_layer=ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                self_attention=ModuleSpec(
                    module=GatedDeltaNet,
                    submodules=GatedDeltaNetSubmodules(
                        in_proj=TELayerNormColumnParallelLinear,
                        out_norm=TENorm,
                        out_proj=TERowParallelLinear,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
            ),
        ),
        # Started with spec from gpt_layer_specs.py (with MLP removed)
        # Using the TE spec because we had problems getting the non-TE spec
        # working
        attention_layer=ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                self_attention=ModuleSpec(
                    module=SelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=SelfAttentionSubmodules(
                        linear_qkv=TELayerNormColumnParallelLinear,
                        core_attention=TEDotProductAttention,
                        linear_proj=TERowParallelLinear,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
            ),
        ),
        dsa_layer=ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=TENorm,
                self_attention=ModuleSpec(
                    module=MLASelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=MLASelfAttentionSubmodules(
                        linear_q_proj=TEColumnParallelLinear,
                        linear_q_down_proj=TELinear,
                        linear_q_up_proj=TEColumnParallelLinear,
                        linear_kv_down_proj=TELinear,
                        linear_kv_up_proj=TEColumnParallelLinear,
                        core_attention=ModuleSpec(
                            module=DSAttention,
                            submodules=DSAttentionSubmodules(
                                indexer=ModuleSpec(
                                    module=DSAIndexer,
                                    submodules=DSAIndexerSubmodules(
                                        linear_wq_b=TELinear,
                                        linear_wk=TELinear,
                                        k_norm=TENorm,
                                        linear_weights_proj=TELinear,
                                    ),
                                )
                            ),
                        ),
                        linear_proj=TERowParallelLinear,
                        q_layernorm=IdentityOp,
                        kv_layernorm=IdentityOp,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
            ),
        ),
        # Started with spec from gpt_layer_specs.py
        # Using the TE spec because we had problems getting the non-TE spec
        # working
        mlp_layer=ModuleSpec(
            module=MLPLayer,
            submodules=TransformerLayerSubmodules(
                mlp=ModuleSpec(
                    module=MLP,
                    submodules=MLPSubmodules(
                        linear_fc1=TELayerNormColumnParallelLinear, linear_fc2=TERowParallelLinear
                    ),
                ),
                mlp_bda=get_bias_dropout_add,
            ),
        ),
        moe_layer=ModuleSpec(
            module=MoETransformerLayer,
            submodules=TransformerLayerSubmodules(
                pre_mlp_layernorm=TENorm, mlp=moe, mlp_bda=get_bias_dropout_add
            ),
        ),
        mtp_block_spec=_hybrid_mtp_block_spec,
    ),
)


hybrid_inference_stack_spec = ModuleSpec(
    module=HybridStack,
    submodules=HybridStackSubmodules(
        mamba_layer=ModuleSpec(
            module=MambaLayer,
            submodules=MambaLayerSubmodules(
                mixer=ModuleSpec(
                    module=MambaMixer,
                    submodules=MambaMixerSubmodules(
                        in_proj=InferenceLayerNormColumnParallelLinear,
                        out_proj=InferenceRowParallelLinear,
                    ),
                ),
                mamba_bda=get_bias_dropout_add,
            ),
        ),
        # Started with spec from gpt_layer_specs.py (with MLP removed)
        # Using the TE spec because we had problems getting the non-TE spec
        # working
        attention_layer=ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                self_attention=ModuleSpec(
                    module=SelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=SelfAttentionSubmodules(
                        linear_qkv=InferenceLayerNormColumnParallelLinear,
                        core_attention=TEDotProductAttention,
                        linear_proj=InferenceRowParallelLinear,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
            ),
        ),
        dsa_layer=ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=TENorm,
                self_attention=ModuleSpec(
                    module=MLASelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=MLASelfAttentionSubmodules(
                        linear_q_proj=TEColumnParallelLinear,
                        linear_q_down_proj=TELinear,
                        linear_q_up_proj=TEColumnParallelLinear,
                        linear_kv_down_proj=TELinear,
                        linear_kv_up_proj=TEColumnParallelLinear,
                        core_attention=ModuleSpec(
                            module=DSAttention,
                            submodules=DSAttentionSubmodules(
                                indexer=ModuleSpec(
                                    module=DSAIndexer,
                                    submodules=DSAIndexerSubmodules(
                                        linear_wq_b=TELinear,
                                        linear_wk=TELinear,
                                        k_norm=TENorm,
                                        linear_weights_proj=TELinear,
                                    ),
                                )
                            ),
                        ),
                        linear_proj=InferenceRowParallelLinear,
                        q_layernorm=IdentityOp,
                        kv_layernorm=IdentityOp,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
            ),
        ),
        # Started with spec from gpt_layer_specs.py
        # Using the TE spec because we had problems getting the non-TE spec
        # working
        mlp_layer=ModuleSpec(
            module=MLPLayer,
            submodules=TransformerLayerSubmodules(
                mlp=ModuleSpec(
                    module=MLP,
                    submodules=MLPSubmodules(
                        linear_fc1=InferenceLayerNormColumnParallelLinear,
                        linear_fc2=InferenceRowParallelLinear,
                    ),
                ),
                mlp_bda=get_bias_dropout_add,
            ),
        ),
        moe_layer=ModuleSpec(
            # Use inference-optimized MoE layer for end-to-end CUDA graph support
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                pre_mlp_layernorm=TENorm, mlp=moe_inference, mlp_bda=get_bias_dropout_add
            ),
        ),
        mtp_block_spec=ModuleSpec(
            module=MultiTokenPredictionBlock,
            submodules=MultiTokenPredictionBlockSubmodules(
                layer_specs=[
                    ModuleSpec(
                        module=MultiTokenPredictionLayer,
                        submodules=MultiTokenPredictionLayerSubmodules(
                            enorm=TENorm,
                            hnorm=TENorm,
                            eh_proj=InferenceColumnParallelLinear,
                            mtp_model_layer=None,  # Built via pattern + hybrid_submodules
                            layer_norm=TENorm,
                        ),
                    )
                ]
            ),
        ),
    ),
)


# Backward-compatible aliases
mamba_stack_spec = hybrid_stack_spec
mamba_inference_stack_spec = hybrid_inference_stack_spec


# ---------------------------------------------------------------------------
# Combined-layer specs for the 1F1B compute/communication overlap path.
#
# ``hybrid_stack_spec_with_combined_layers`` mirrors ``hybrid_stack_spec``
# above, plus populates the ``combined_*_layer`` fields with
# :class:`MambaAttnMLPLayer` variants whose inner building blocks reuse the
# same specs that the legacy separate-layer path uses. This guarantees the
# combined layer is numerically equivalent to a Mamba + (optional attn) + MLP
# sequence built from the same specs.
# ---------------------------------------------------------------------------

# Submodule specs shared across all three combined-layer variants.
# Extract the inner specs from the existing hybrid_stack_spec so any future
# tweaks (e.g. swapping TENorm) propagate to both paths.
_combined_mamba_norm = hybrid_stack_spec.submodules.mamba_layer.submodules.norm
_combined_mamba_mixer = hybrid_stack_spec.submodules.mamba_layer.submodules.mixer
_combined_mamba_bda = hybrid_stack_spec.submodules.mamba_layer.submodules.mamba_bda

_combined_pre_mlp_layernorm = (
    hybrid_stack_spec.submodules.mlp_layer.submodules.pre_mlp_layernorm
)
_combined_mlp = hybrid_stack_spec.submodules.mlp_layer.submodules.mlp
_combined_mlp_bda = hybrid_stack_spec.submodules.mlp_layer.submodules.mlp_bda

_combined_attn_norm_self = (
    hybrid_stack_spec.submodules.attention_layer.submodules.input_layernorm
)
_combined_attn_self = hybrid_stack_spec.submodules.attention_layer.submodules.self_attention
_combined_attn_bda_self = hybrid_stack_spec.submodules.attention_layer.submodules.self_attn_bda

_combined_attn_norm_gdn = hybrid_stack_spec.submodules.gdn_layer.submodules.input_layernorm
_combined_attn_gdn = hybrid_stack_spec.submodules.gdn_layer.submodules.self_attention
_combined_attn_bda_gdn = hybrid_stack_spec.submodules.gdn_layer.submodules.self_attn_bda


def _mamba_mlp_submods():
    """Submodules for Mamba + MLP (no attention) combined layer."""
    return MambaAttnMLPLayerSubmodules(
        mamba_norm=_combined_mamba_norm,
        mamba_mixer=_combined_mamba_mixer,
        mamba_bda=_combined_mamba_bda,
        pre_mlp_layernorm=_combined_pre_mlp_layernorm,
        mlp=_combined_mlp,
        mlp_bda=_combined_mlp_bda,
    )


def _mamba_self_attn_mlp_submods():
    """Submodules for Mamba + SelfAttention + MLP combined layer."""
    return MambaAttnMLPLayerSubmodules(
        mamba_norm=_combined_mamba_norm,
        mamba_mixer=_combined_mamba_mixer,
        mamba_bda=_combined_mamba_bda,
        attn_norm=_combined_attn_norm_self,
        attention=_combined_attn_self,
        attn_bda=_combined_attn_bda_self,
        pre_mlp_layernorm=_combined_pre_mlp_layernorm,
        mlp=_combined_mlp,
        mlp_bda=_combined_mlp_bda,
    )


def _mamba_gdn_mlp_submods():
    """Submodules for Mamba + GatedDeltaNet + MLP combined layer."""
    return MambaAttnMLPLayerSubmodules(
        mamba_norm=_combined_mamba_norm,
        mamba_mixer=_combined_mamba_mixer,
        mamba_bda=_combined_mamba_bda,
        attn_norm=_combined_attn_norm_gdn,
        attention=_combined_attn_gdn,
        attn_bda=_combined_attn_bda_gdn,
        pre_mlp_layernorm=_combined_pre_mlp_layernorm,
        mlp=_combined_mlp,
        mlp_bda=_combined_mlp_bda,
    )


hybrid_stack_spec_with_combined_layers = ModuleSpec(
    module=HybridStack,
    submodules=HybridStackSubmodules(
        # Legacy separate-layer specs retained for back-compat and for the
        # non-bracketed path; populated identically to ``hybrid_stack_spec``.
        mamba_layer=hybrid_stack_spec.submodules.mamba_layer,
        gdn_layer=hybrid_stack_spec.submodules.gdn_layer,
        attention_layer=hybrid_stack_spec.submodules.attention_layer,
        dsa_layer=hybrid_stack_spec.submodules.dsa_layer,
        mlp_layer=hybrid_stack_spec.submodules.mlp_layer,
        moe_layer=hybrid_stack_spec.submodules.moe_layer,
        mtp_block_spec=hybrid_stack_spec.submodules.mtp_block_spec,
        # Combined-layer specs used by the bracketed pattern path.
        combined_layer=ModuleSpec(module=MambaMLPLayer, submodules=_mamba_mlp_submods()),
        combined_attn_layer=ModuleSpec(
            module=MambaSelfAttnMLPLayer, submodules=_mamba_self_attn_mlp_submods()
        ),
        combined_gdn_layer=ModuleSpec(
            module=MambaGDNMLPLayer, submodules=_mamba_gdn_mlp_submods()
        ),
    ),
)
