# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Schedule callables for :class:`CombinedHybridLayer`.

These callables decompose a combined hybrid layer into the 5-slot interface
(``attn``, ``dispatch``, ``mlp``, ``combine``, ``mtp``) expected by
:class:`TransformerLayerSchedulePlan`, so the standard 1F1B fine-grained
schedule in :mod:`megatron.core.models.common.model_chunk_schedule_plan`
interleaves combined-layer compute with MoE A2A communication automatically.

Slot assignment
---------------

* MoE layer (``[..E]``):

  ================= ===================================================
  ``attn`` (comp)   Mamba/GDN → Attention → pre-MoE norm → router →
                    dispatcher preprocess (→ ``(tokens, probs)``)
  ``dispatch`` (comm) MoE token dispatch A2A
  ``mlp`` (comp)    MoE experts
  ``combine`` (comm) MoE combine A2A + BDA
  ``mtp``           NoopScheduleNode
  ================= ===================================================

* Dense MLP layer (``[..-]``):

  ================= ===================================================
  ``attn`` (comp)   Mamba/GDN → Attention → MLP → BDA (full layer compute)
  ``dispatch`` (comm) NoopScheduleNode
  ``mlp`` (comp)    passthrough identity
  ``combine`` (comm) NoopScheduleNode
  ``mtp``           NoopScheduleNode
  ================= ===================================================

* No-MLP layer (``[M]``, ``[*]``, ``[M*]``):

  Same as the dense case with the MLP stage skipped — the whole layer
  runs inside the ``attn`` slot.
"""

from typing import Optional

import torch
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.models.hybrid.combined_layer import CombinedHybridLayer
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_layer import make_viewless_tensor
from megatron.core.typed_torch import apply_module


class _CombinedBackwardDWWrapper:
    """No-op backward_dw wrapper for combined layers.

    Mamba and MoE experts compute their weight gradients during the standard
    backward pass, so there is no separate wgrad callable to schedule. This
    wrapper exists purely to satisfy the schedule plan's
    ``backward_dw_map["attn"].backward_dw()`` interface.
    """

    def __init__(self, layer: CombinedHybridLayer):
        self.layer = layer

    def backward_dw(self):
        """Execute no separate wgrad; release layer ref to avoid memory leaks."""
        self.layer = None

    def set_graphed_backward_dw_callable(self, graphed_backward_dw_callable):
        """CUDA graph wgrad is not supported for combined layers yet."""
        # intentionally ignored


def build_combined_hybrid_layer_callables(layer: CombinedHybridLayer):
    """Return (forward_funcs, backward_dw) for a combined hybrid layer.

    The returned tuple matches the shape expected by
    ``TransformerLayerSchedulePlan._build_callable_nodes``:
    ``forward_funcs = [attn, dispatch, mlp, combine, mtp_post]`` and
    ``backward_dw = {"attn": wrapper_or_None, "mlp": module_or_None}``.
    """

    is_moe = isinstance(layer.mlp, MoELayer)
    enable_deepep = (
        layer.config.moe_token_dispatcher_type == "flex"
        and layer.config.moe_flex_dispatcher_backend == "deepep"
    )
    enable_hybridep = (
        layer.config.moe_token_dispatcher_type == "flex"
        and layer.config.moe_flex_dispatcher_backend == "hybridep"
    )

    # ------------------------------------------------------------------
    # attn slot: Mamba (optional) → Attention (optional) →
    #             (for MoE) pre_mlp_norm + router + dispatcher preprocess
    #             (for dense MLP) entire MLP block
    # ------------------------------------------------------------------
    def submodule_combined_attn_forward(node, hidden_states: Tensor):
        # 1. Mamba / GDN
        if layer.mamba_type != 'none':
            residual = hidden_states
            if layer.config.fp32_residual_connection:
                residual = residual.float()
            hidden_states = hidden_states.to(dtype=layer.config.params_dtype)
            h = apply_module(layer.mamba_norm)(hidden_states)
            mixer_out_with_bias = layer.mixer(
                h,
                inference_context=None,
                packed_seq_params=node.chunk_state.packed_seq_params,
            )
            with layer.bias_dropout_add_exec_handler():
                hidden_states = layer.mamba_bda(
                    training=layer.training, fused=layer.config.bias_dropout_fusion
                )(mixer_out_with_bias, residual, layer.hidden_dropout)

        # 2. Attention
        if layer.attention_type != 'none':
            residual = hidden_states
            if layer.config.fp32_residual_connection:
                residual = residual.float()
            h = apply_module(layer.attn_norm)(hidden_states)
            attention_output_with_bias = layer.self_attention(
                h,
                attention_mask=node.chunk_state.attention_mask,
                inference_context=None,
                rotary_pos_emb=node.chunk_state.rotary_pos_emb,
                rotary_pos_cos=node.chunk_state.rotary_pos_cos,
                rotary_pos_sin=node.chunk_state.rotary_pos_sin,
                packed_seq_params=node.chunk_state.packed_seq_params,
                sequence_len_offset=node.chunk_state.sequence_len_offset,
            )
            with layer.bias_dropout_add_exec_handler():
                hidden_states = layer.attn_bda(
                    training=layer.training, fused=layer.config.bias_dropout_fusion
                )(attention_output_with_bias, residual, layer.hidden_dropout)

        # 3. MLP prologue
        if layer.mlp_type == 'none':
            # No MLP block — apply final_norm here if this is the last layer
            final_norm = getattr(node.chunk_state.model.decoder, 'final_norm', None)
            if (
                not getattr(node, 'is_mtp', False)
                and final_norm
                and node.is_last_layer
            ):
                hidden_states = final_norm(hidden_states)
                hidden_states = make_viewless_tensor(
                    inp=hidden_states, requires_grad=True, keep_graph=True
                )
            return hidden_states

        if not is_moe:
            # Dense MLP: do the whole MLP block here (no overlap opportunity)
            residual = hidden_states
            if layer.config.fp32_residual_connection:
                residual = residual.float()
            pre_mlp = apply_module(layer.pre_mlp_layernorm)(hidden_states)
            mlp_out_with_bias = layer.mlp(pre_mlp)
            with layer.bias_dropout_add_exec_handler():
                hidden_states = layer.mlp_bda(
                    training=layer.training, fused=layer.config.bias_dropout_fusion
                )(mlp_out_with_bias, residual, layer.hidden_dropout)

            # Apply final_norm if last layer
            final_norm = getattr(node.chunk_state.model.decoder, 'final_norm', None)
            if (
                not getattr(node, 'is_mtp', False)
                and final_norm
                and node.is_last_layer
            ):
                hidden_states = final_norm(hidden_states)
                hidden_states = make_viewless_tensor(
                    inp=hidden_states, requires_grad=True, keep_graph=True
                )
            return hidden_states

        # MoE path: prepare dispatch
        pre_mlp_layernorm_output = apply_module(layer.pre_mlp_layernorm)(hidden_states)
        if isinstance(pre_mlp_layernorm_output, tuple):
            # fused residual norm returns (output, residual)
            pre_mlp_layernorm_output, hidden_states = pre_mlp_layernorm_output

        shared_expert_output = layer.mlp.shared_experts_compute(pre_mlp_layernorm_output)
        probs, routing_map = layer.mlp.route(pre_mlp_layernorm_output)
        local_tokens, probs = layer.mlp.preprocess(
            pre_mlp_layernorm_output, probs, routing_map
        )

        # Cache for combine residual + shared expert merge
        node.layer_state.residual = node.detach(hidden_states)
        if layer.mlp.use_shared_expert and not layer.mlp.shared_expert_overlap:
            node.layer_state.shared_expert_output = node.detach(shared_expert_output)

        return local_tokens, probs

    # ------------------------------------------------------------------
    # dispatch slot (comm): MoE dispatch A2A (or NoopScheduleNode upstream)
    # ------------------------------------------------------------------
    def submodule_dispatch_forward(node, local_tokens: Tensor, probs: Tensor):
        token_dispatcher = layer.mlp.token_dispatcher
        if enable_deepep or enable_hybridep:
            token_dispatcher._comm_manager.token_probs = probs
        dispatched_tokens, dispatched_probs = layer.mlp.dispatch(local_tokens, probs)
        node.layer_state.dispatched_probs = node.detach(dispatched_probs)
        return dispatched_tokens

    # ------------------------------------------------------------------
    # mlp slot (comp): experts (for MoE) or identity (for dense MLP)
    # ------------------------------------------------------------------
    def submodule_moe_forward(node, dispatched_tokens: Tensor):
        dispatched_probs = node.layer_state.dispatched_probs
        token_dispatcher = layer.mlp.token_dispatcher
        if enable_deepep or enable_hybridep:
            token_dispatcher._comm_manager.dispatched_probs = dispatched_probs
        expert_output, _ = layer.mlp.routed_experts_compute(dispatched_tokens, dispatched_probs)
        if enable_hybridep:
            tokens_per_expert = token_dispatcher._comm_manager.get_number_of_tokens_per_expert()
            node.layer_state.tokens_per_expert = tokens_per_expert
        return expert_output

    def mlp_passthrough(node, hidden_states):
        """No-op MLP for non-MoE combined layers; MLP was already done in attn."""
        return hidden_states

    # ------------------------------------------------------------------
    # combine slot (comm): MoE combine A2A + BDA (residual stream)
    # ------------------------------------------------------------------
    def submodule_combine_forward(node, expert_output: Tensor):
        residual = node.layer_state.residual
        shared_expert_output = getattr(node.layer_state, 'shared_expert_output', None)
        output = layer.mlp.combine(expert_output)
        output = layer.mlp.postprocess(output, shared_expert_output)

        mlp_output_with_bias = (output, None)
        with layer.bias_dropout_add_exec_handler():
            hidden_states = layer.mlp_bda(
                layer.training, layer.config.bias_dropout_fusion
            )(mlp_output_with_bias, residual, layer.hidden_dropout)
        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        # Cross-stream record
        node.layer_state.residual.record_stream(torch.cuda.current_stream())
        if shared_expert_output is not None:
            shared_expert_output.record_stream(torch.cuda.current_stream())

        node.layer_state.residual = None
        node.layer_state.shared_expert_output = None

        # Apply final_norm if last layer
        final_norm = getattr(node.chunk_state.model.decoder, 'final_norm', None)
        if (
            not getattr(node, 'is_mtp', False)
            and final_norm
            and node.is_last_layer
        ):
            output = final_norm(output)
            output = make_viewless_tensor(inp=output, requires_grad=True, keep_graph=True)
        return output

    def raise_not_implemented(*args, **kwargs):
        raise NotImplementedError(
            "dispatch/combine are not available for dense-MLP combined layers; "
            "the schedule should use NoopScheduleNode in those slots."
        )

    # Assemble and return
    if is_moe:
        forward_funcs = [
            submodule_combined_attn_forward,
            submodule_dispatch_forward,
            submodule_moe_forward,
            submodule_combine_forward,
            None,  # no MTP post-process
        ]
        backward_dw = {"attn": _CombinedBackwardDWWrapper(layer), "mlp": layer.mlp}
    else:
        forward_funcs = [
            submodule_combined_attn_forward,
            raise_not_implemented,
            mlp_passthrough,
            raise_not_implemented,
            None,
        ]
        backward_dw = {"attn": _CombinedBackwardDWWrapper(layer), "mlp": None}

    return forward_funcs, backward_dw


__all__ = ['build_combined_hybrid_layer_callables']
