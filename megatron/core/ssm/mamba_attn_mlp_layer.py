# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.

"""Combined Mamba + optional Attention + MLP layer for 1F1B overlap support.

The layer is the hybrid-model analog of GPT's :class:`TransformerLayer`: a single
decoder unit that bundles a Mamba block with an optional attention block and an
MLP/MoE block. Exposing the split into ``forward_ssm_attn`` (compute-heavy) and
``forward_mlp`` (communication-heavy for MoE) lets the existing
``TransformerModelChunkSchedulePlan`` interleave forward of microbatch N with
backward of microbatch N-1 on separate CUDA streams.
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
from megatron.core.transformer.enums import CudaGraphScope
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.module import GraphableMegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module
from megatron.core.utils import deprecate_inference_params, make_viewless_tensor

# Valid values for the ``attention_type`` constructor argument. Declared at
# module scope so callers (and tests) can introspect the allowed set.
ATTENTION_TYPE_NONE = "none"
ATTENTION_TYPE_SELF_ATTENTION = "self_attention"
ATTENTION_TYPE_GATED_DELTA_NET = "gated_delta_net"
VALID_ATTENTION_TYPES = (
    ATTENTION_TYPE_NONE,
    ATTENTION_TYPE_SELF_ATTENTION,
    ATTENTION_TYPE_GATED_DELTA_NET,
)


@dataclass
class MambaAttnMLPLayerSubmodules:
    """Submodules for :class:`MambaAttnMLPLayer`.

    The dataclass exposes every building block that any variant might need.
    ``MambaAttnMLPLayer`` selects which are actually used based on
    ``attention_type`` and whether ``mlp`` is an MoE layer.

    Mamba block (always built):
        mamba_norm: Pre-mixer layernorm.
        mamba_mixer: Selective SSM mixer (typically :class:`MambaMixer`).
        mamba_bda: Bias-dropout-add that closes the Mamba residual.

    Attention block (built iff ``attention_type != 'none'``):
        attn_norm: Pre-attention layernorm.
        attention: :class:`SelfAttention` or :class:`GatedDeltaNet` spec.
        attn_bda: Bias-dropout-add that closes the attention residual.

    MLP block (always built):
        pre_mlp_layernorm: Pre-MLP layernorm.
        mlp: Dense MLP or MoE layer.
        mlp_bda: Bias-dropout-add that closes the MLP residual.
    """

    # Mamba block
    mamba_norm: LayerNormBuilder = IdentityOp
    mamba_mixer: Union[ModuleSpec, type] = IdentityOp
    mamba_bda: Union[ModuleSpec, type] = IdentityFuncOp

    # Attention block (optional -- inspected only when attention_type != "none")
    attn_norm: LayerNormBuilder = IdentityOp
    attention: Union[ModuleSpec, type] = IdentityOp
    attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    # MLP block
    pre_mlp_layernorm: LayerNormBuilder = IdentityOp
    mlp: Union[ModuleSpec, type] = IdentityOp
    mlp_bda: Union[ModuleSpec, type] = IdentityFuncOp

    # Mapping for sharded tensor keys to be applied in sharded_state_dict.
    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)


class MambaAttnMLPLayer(GraphableMegatronModule):
    """Combined Mamba + optional Attention + MLP decoder layer.

    Input and output tensors have shape ``[s, b, h]``. The layer is
    architecture-agnostic with respect to the attention choice: the
    ``attention_type`` argument -- not the spec -- decides what runs in the
    attention slot. Concrete subclasses (:class:`MambaMLPLayer`,
    :class:`MambaSelfAttnMLPLayer`, :class:`MambaGDNMLPLayer`) fix the type so
    the spec can reference a module directly.

    Args:
        config: Transformer config.
        submodules: Building blocks for each of the three inner blocks.
        layer_number: 1-indexed combined-layer index within the decoder stack.
        attention_type: Which attention to use; one of ``VALID_ATTENTION_TYPES``.
        hidden_dropout: Dropout probability for the residual BDA. Defaults to
            ``config.hidden_dropout``.
        pg_collection: Process groups used by inner blocks.
        pp_layer_offset: Global offset of this layer in the full decoder stack,
            used by the Mamba mixer for pipeline-aware param sharding.
        submodule_layer_numbers: Optional mapping giving each submodule its
            standalone-layer index (e.g. ``{"mamba": 1, "attention": 2, "mlp": 3}``).
            This is the index the submodule *would* have had if each component
            were its own decoder layer -- needed for FP8 per-layer contexts and
            for checkpoint round-trip with the legacy separate-layer path.
            Defaults to ``layer_number`` for every submodule.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MambaAttnMLPLayerSubmodules,
        layer_number: int = 1,
        attention_type: str = ATTENTION_TYPE_NONE,
        hidden_dropout: Optional[float] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        pp_layer_offset: int = 0,
        submodule_layer_numbers: Optional[Dict[str, int]] = None,
    ) -> None:
        assert pg_collection is not None, "pg_collection must be provided for MambaAttnMLPLayer"
        if attention_type not in VALID_ATTENTION_TYPES:
            raise ValueError(
                f"attention_type must be one of {VALID_ATTENTION_TYPES}, got {attention_type!r}"
            )

        self.submodules_config = submodules
        super().__init__(config=config)

        self.config = config
        self.layer_number = layer_number
        self.attention_type = attention_type
        self.pg_collection = pg_collection
        self.tp_group = pg_collection.tp
        self.pp_layer_offset = pp_layer_offset
        self.hidden_dropout = (
            config.hidden_dropout if hidden_dropout is None else hidden_dropout
        )

        # Per-submodule "legacy" layer numbers. These are the indices each
        # submodule would have had in the separate-layer hybrid stack; used for
        # FP8 per-layer contexts and checkpoint compatibility with the legacy
        # path. Default each to layer_number when the caller hasn't provided a
        # mapping (true for standalone unit tests).
        self.submodule_layer_numbers = {
            "mamba": layer_number,
            "attention": layer_number,
            "mlp": layer_number,
        }
        if submodule_layer_numbers is not None:
            self.submodule_layer_numbers.update(submodule_layer_numbers)

        # --- Mamba block (always built) ---
        self.norm = submodules.mamba_norm(self.config, self.config.hidden_size)
        self.mixer = build_module(
            submodules.mamba_mixer,
            self.config,
            d_model=self.config.hidden_size,
            layer_number=self.submodule_layer_numbers["mamba"],
            pg_collection=pg_collection,
            pp_layer_offset=pp_layer_offset,
        )
        self.mamba_bda = build_module(submodules.mamba_bda)

        # --- Attention block (optional) ---
        if attention_type != ATTENTION_TYPE_NONE:
            self.attn_norm = submodules.attn_norm(
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )
            attention_optional_kwargs: Dict[str, Any] = {"pg_collection": pg_collection}
            if config.context_parallel_size > 1 and config.cp_comm_type is not None:
                if isinstance(config.cp_comm_type, list):
                    attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type[
                        self.submodule_layer_numbers["attention"] - 1
                    ]
                else:
                    attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type
            if pp_layer_offset is not None:
                attention_optional_kwargs["pp_layer_offset"] = pp_layer_offset
            self.attention = build_module(
                submodules.attention,
                config=self.config,
                layer_number=self.submodule_layer_numbers["attention"],
                **attention_optional_kwargs,
            )
            self.attn_bda = build_module(submodules.attn_bda)
        else:
            # Register as attributes (but not submodules) so attribute access is
            # stable and state dicts don't include empty parameters.
            self.attn_norm = None
            self.attention = None
            self.attn_bda = None

        # --- MLP block (always built) ---
        self.pre_mlp_layernorm = submodules.pre_mlp_layernorm(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        # Resolve pg / tp kwargs the same way TransformerLayer does so the MLP
        # or MoE layer sees the right groups.
        additional_mlp_kwargs: Dict[str, Any] = {}
        # Imports are local to avoid a circular dependency with the MLP module.
        from megatron.core.extensions.transformer_engine import TEFusedMLP
        from megatron.core.transformer.mlp import MLP
        from megatron.core.transformer.moe.experts import SequentialMLP, TEGroupedMLP
        from megatron.core.transformer.moe.moe_layer import MoELayer

        if isinstance(submodules.mlp, ModuleSpec):
            if submodules.mlp.module in (MoELayer, TEGroupedMLP, SequentialMLP):
                additional_mlp_kwargs["pg_collection"] = pg_collection
                if submodules.mlp.module is MoELayer:
                    # MambaAttnMLPLayer is never used for MTP in this refactor.
                    additional_mlp_kwargs["is_mtp_layer"] = False
            elif submodules.mlp.module is MLP:
                additional_mlp_kwargs["tp_group"] = pg_collection.tp
            elif TEFusedMLP is not None and submodules.mlp.module is TEFusedMLP:
                additional_mlp_kwargs["tp_group"] = pg_collection.tp
        self.mlp = build_module(submodules.mlp, config=self.config, **additional_mlp_kwargs)
        if hasattr(self.mlp, "set_layer_number"):
            self.mlp.set_layer_number(self.submodule_layer_numbers["mlp"])
        self.mlp_bda = build_module(submodules.mlp_bda)

        self.is_moe_layer = isinstance(self.mlp, MoELayer)

        self.bias_dropout_add_exec_handler = torch.enable_grad

        # Recompute / offload knobs used by the fine-grained callables. The
        # combined-layer path does not yet support selective recompute of the
        # pre-MLP layernorm or fine-grained activation offloading; the flags
        # stay False so the callables skip those branches. A future PR can
        # enable them after validating numerics and memory behavior.
        self.recompute_pre_mlp_layernorm = False
        self.offload_mlp_norm = False

    def create_mcore_cudagraph_manager(self, config):
        """Register this layer for CUDA graph capture.

        The monolithic ``forward`` path is captured with a single graph, mirroring
        :class:`MambaLayer`. The separate ``forward_ssm_attn`` / ``forward_mlp``
        sub-graph capture used by the overlap path is configured by
        :class:`HybridStack` when ``overlap_moe_expert_parallel_comm`` is on
        (see the sub-graph hooks wired up in Step 5 of the hybrid 1F1B work).
        """
        from megatron.core.transformer.cuda_graphs import CudaGraphManager

        scope = self.config.cuda_graph_scope
        if not scope or CudaGraphScope.mamba in scope:
            self.cudagraph_manager = CudaGraphManager(config)

    def mamba_state_shapes_per_request(self):
        """Return the Mamba conv and SSM state shapes per request."""
        return self.mixer.mamba_state_shapes_per_request()

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
        """Run the Mamba block, then the optional attention block.

        This is the compute-heavy half of the layer in the 1F1B overlap split.
        """
        inference_context = deprecate_inference_params(inference_context, inference_params)

        # --- Mamba block ---
        residual = hidden_states
        if self.config.fp32_residual_connection:
            residual = residual.float()

        # Match MambaLayer's precision cast; keeps fp32 residual + bf16 mixer.
        hidden_states = hidden_states.to(dtype=self.config.params_dtype)
        mamba_norm_out = apply_module(self.norm)(hidden_states)
        mixer_out_with_bias = self.mixer(
            mamba_norm_out,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
        )
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mamba_bda(
                training=self.training, fused=self.config.bias_dropout_fusion
            )(mixer_out_with_bias, residual, self.hidden_dropout)

        # --- Attention block (skipped when attention_type == "none") ---
        if self.attention is None:
            return hidden_states

        attn_norm_output = apply_module(self.attn_norm)(hidden_states)
        if isinstance(attn_norm_output, tuple):
            if len(attn_norm_output) != 2:
                raise ValueError(
                    "When attn_norm returns a tuple it must have 2 elements "
                    f"(output, residual); got {len(attn_norm_output)}."
                )
            attn_norm_output, residual = attn_norm_output
        else:
            residual = hidden_states

        if self.config.fp32_residual_connection:
            residual = residual.float()

        attention_output_with_bias = self.attention(
            attn_norm_output,
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
        """Run the MLP/MoE block and close the residual.

        This is the monolithic-dense or monolithic-MoE path. The 1F1B overlap
        path does not call this method; it invokes dispatch/experts/combine
        sub-callables separately via ``build_mamba_attn_mlp_layer_callables``.
        """
        pre_mlp_layernorm_output = apply_module(self.pre_mlp_layernorm)(hidden_states)
        if isinstance(pre_mlp_layernorm_output, tuple):
            if len(pre_mlp_layernorm_output) != 2:
                raise ValueError(
                    "When pre_mlp_layernorm returns a tuple it must have 2 elements "
                    f"(output, residual); got {len(pre_mlp_layernorm_output)}."
                )
            pre_mlp_layernorm_output, residual = pre_mlp_layernorm_output
        else:
            residual = hidden_states

        if self.config.fp32_residual_connection:
            residual = residual.float()

        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )
        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )
        return output

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
        """Monolithic forward used when 1F1B overlap is off or during inference."""
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
        hidden_states = self.forward_mlp(hidden_states)
        return hidden_states

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """Emit a legacy-compatible sharded state dict for this combined layer.

        The combined layer compresses a Mamba mixer, an optional attention,
        and an MLP into a single decoder slot, but checkpoints saved under the
        legacy separate-layer path store these as three distinct decoder
        layers. To preserve checkpoint interop, keys are emitted under the
        per-submodule *standalone* layer indices carried by
        ``self.submodule_layer_numbers``:

            prefix="decoder.layers.0."  (combined-layer index)
              -> decoder.layers.{mamba_idx-1}.{norm|mixer|mamba_bda}.*
              -> decoder.layers.{attn_idx-1}.{input_layernorm|self_attention|
                                              self_attn_bda}.*   (if present)
              -> decoder.layers.{mlp_idx-1}.{pre_mlp_layernorm|mlp|mlp_bda}.*

        Attention-block submodules are renamed to TransformerLayer's naming
        convention (``input_layernorm`` / ``self_attention`` /
        ``self_attn_bda``) so the keys line up with the legacy path.

        The ``prefix`` must end in the combined-layer index followed by a dot
        (e.g. ``"decoder.layers.0."``); the stem before the index is kept,
        and the index is replaced per submodule as above.
        """
        # Split the incoming prefix into stem + combined-layer index.
        # The contract from HybridStack is that prefix always ends with
        # ``<stem>.<combined_idx>.`` so we can split on ``.``.
        if not prefix.endswith("."):
            raise ValueError(
                f"MambaAttnMLPLayer.sharded_state_dict expects a prefix ending "
                f"in '.' (e.g. 'decoder.layers.0.'); got {prefix!r}."
            )
        parts = prefix[:-1].rsplit(".", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            raise ValueError(
                f"MambaAttnMLPLayer.sharded_state_dict expects a prefix of the "
                f"form '<stem>.<idx>.'; got {prefix!r}."
            )
        stem = parts[0] + "."  # e.g. "decoder.layers."

        def _submod_sharded_dict(submodule, sub_prefix):
            """Run sharded_state_dict on ``submodule`` with the given prefix.

            Uses the MegatronModule helper if available; otherwise falls back
            to :func:`sharded_state_dict_default` for plain ``torch.nn.Module``
            instances. Bias-dropout-add slots are plain callables (e.g.
            :func:`get_bias_dropout_add`) with no parameters, so we just skip
            them -- they contribute nothing to the checkpoint.
            """
            import torch.nn as nn

            from megatron.core.transformer.utils import sharded_state_dict_default

            if not isinstance(submodule, nn.Module):
                return {}
            return sharded_state_dict_default(
                submodule, sub_prefix, sharded_offsets, metadata, tp_group=self.tp_group
            )

        result: ShardedStateDict = {}

        # Mamba block -- keys match MambaLayer's naming.
        mamba_idx = self.submodule_layer_numbers["mamba"] - 1
        mamba_prefix = f"{stem}{mamba_idx}."
        result.update(_submod_sharded_dict(self.norm, f"{mamba_prefix}norm."))
        result.update(_submod_sharded_dict(self.mixer, f"{mamba_prefix}mixer."))
        result.update(_submod_sharded_dict(self.mamba_bda, f"{mamba_prefix}mamba_bda."))

        # Attention block -- rename to TransformerLayer naming for legacy compat.
        if self.attention is not None:
            attn_idx = self.submodule_layer_numbers["attention"] - 1
            attn_prefix = f"{stem}{attn_idx}."
            result.update(
                _submod_sharded_dict(self.attn_norm, f"{attn_prefix}input_layernorm.")
            )
            result.update(
                _submod_sharded_dict(self.attention, f"{attn_prefix}self_attention.")
            )
            result.update(_submod_sharded_dict(self.attn_bda, f"{attn_prefix}self_attn_bda."))

        # MLP block -- keys match TransformerLayer's naming.
        mlp_idx = self.submodule_layer_numbers["mlp"] - 1
        mlp_prefix = f"{stem}{mlp_idx}."
        result.update(
            _submod_sharded_dict(self.pre_mlp_layernorm, f"{mlp_prefix}pre_mlp_layernorm.")
        )
        result.update(_submod_sharded_dict(self.mlp, f"{mlp_prefix}mlp."))
        result.update(_submod_sharded_dict(self.mlp_bda, f"{mlp_prefix}mlp_bda."))

        # Optional user-provided key remaps (e.g. for custom TE state renames).
        if self.submodules_config.sharded_state_dict_keys_map:
            # Apply map keys/values as-is; they are already expected to be in
            # the final (legacy) key space.
            apply_prefix_mapping(result, dict(self.submodules_config.sharded_state_dict_keys_map))

        return result

    def _te_cuda_graph_replay(self, *args, **kwargs):
        """CUDA graph replay for this layer using the TE interface.

        TransformerEngine versions >= 1.10 allow keyword tensor arguments with
        CUDA graph, but none of them can be non-tensor python objects.
        ``inference_context`` must therefore be excluded from the input list.
        """
        assert kwargs.get("inference_context") is None, (
            "CUDA graph accepts only Tensor inputs. inference_context is excluded. "
            "For inference cuda graph, please use cuda_graph_impl=local instead."
        )
        return super()._te_cuda_graph_replay(*args, **kwargs)

    def _should_call_local_cudagraph(self, *args, **kwargs):
        """Whether to route this call through the local CUDA graph manager."""
        if (
            hasattr(self, "cudagraph_manager")
            and kwargs.get("inference_context") is None
            and not torch.is_inference_mode_enabled()
        ):
            return True
        if not self.training and (
            hasattr(self, "cudagraph_manager")
            and kwargs.get("attention_mask") is None
            and kwargs.get("inference_context") is not None
            and not self.config.cuda_graph_scope
        ):
            context = kwargs["inference_context"]
            using_cuda_graph = (
                context.is_static_batching() and context.is_decode_only()
            ) or (
                not context.is_static_batching() and context.using_cuda_graph_this_step()
            )
            return using_cuda_graph
        return False


class MambaMLPLayer(MambaAttnMLPLayer):
    """Mamba + MLP combined layer (no attention)."""

    def __init__(self, *args, **kwargs):
        kwargs["attention_type"] = ATTENTION_TYPE_NONE
        super().__init__(*args, **kwargs)


class MambaSelfAttnMLPLayer(MambaAttnMLPLayer):
    """Mamba + SelfAttention + MLP combined layer."""

    def __init__(self, *args, **kwargs):
        kwargs["attention_type"] = ATTENTION_TYPE_SELF_ATTENTION
        super().__init__(*args, **kwargs)


class MambaGDNMLPLayer(MambaAttnMLPLayer):
    """Mamba + GatedDeltaNet + MLP combined layer."""

    def __init__(self, *args, **kwargs):
        kwargs["attention_type"] = ATTENTION_TYPE_GATED_DELTA_NET
        super().__init__(*args, **kwargs)
