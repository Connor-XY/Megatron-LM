# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fine-grained callables for Mamba hybrid layers.

This file is the Mamba analog of :mod:`megatron.core.models.gpt.fine_grained_callables`.
It decomposes a :class:`MambaAttnMLPLayer` into the same five scheduler slots that
GPT's :class:`TransformerLayer` uses (attn, dispatch, mlp, combine, mtp_post_process),
so a :class:`TransformerLayerSchedulePlan` can drive a Mamba hybrid model through
the 1F1B overlap schedule without architecture-specific changes.

The split maps Mamba's ``forward_ssm_attn`` (compute) and the MoE dispatch /
combine (communication) onto the compute and comm streams respectively. For
dense-MLP layers, dispatch/combine degrade to no-ops (same behavior as GPT).

See the docstring on :class:`MambaAttnMLPLayer` for the combined-layer shape.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.pipeline_parallel.utils import ScheduleNode
from megatron.core.ssm.mamba_attn_mlp_layer import MambaAttnMLPLayer
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.typed_torch import apply_module
from megatron.core.utils import make_viewless_tensor


def build_mamba_attn_mlp_layer_callables(layer: MambaAttnMLPLayer):
    """Create the five scheduler callables for a ``MambaAttnMLPLayer``.

    The returned layout matches :func:`build_transformer_layer_callables` so the
    architecture-agnostic :class:`TransformerLayerSchedulePlan` can consume it
    without branches:

    1. ``attn``      (compute): ``forward_ssm_attn`` plus, for MoE, pre-MLP
       layernorm + router + dispatch preprocess.
    2. ``dispatch``  (comm):    token dispatch All-to-All.
    3. ``mlp``       (compute): experts (MoE) or dense-MLP forward.
    4. ``combine``   (comm):    token combine All-to-All + mlp_bda.
    5. ``mtp_post_process`` (compute): always no-op -- Mamba MTP is a follow-up.

    Returns:
        Tuple ``(forward_funcs, backward_dw)`` matching GPT's convention.
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

    def submodule_attn_forward(node: ScheduleNode, hidden_states: Tensor):
        """Run Mamba + optional attention, then (for MoE) norm + route + preprocess.

        Stays on the compute stream. For dense MLP layers the return is simply
        the post-attention hidden states. For MoE layers the return contains the
        local tokens and routing probabilities ready to hand off to the
        dispatch A2A on the comm stream.

        Sub-graph CUDA graph capture across ``forward_ssm_attn`` and
        ``forward_mlp`` is a follow-up; until then, graphs are disabled on
        combined layers when ``overlap_moe_expert_parallel_comm`` is on.
        """

        def forward_func(
            hidden_states: Tensor,
            attention_mask: Optional[Tensor] = None,
            rotary_pos_emb: Optional[Tensor] = None,
            rotary_pos_cos: Optional[Tensor] = None,
            rotary_pos_sin: Optional[Tensor] = None,
            packed_seq_params: Optional[PackedSeqParams] = None,
            sequence_len_offset: Optional[Tensor] = None,
        ):
            hidden_states = layer.forward_ssm_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
            )
            if not isinstance(layer.mlp, MoELayer):
                return hidden_states, None, None, None

            # MoE fast path: compute pre-MLP norm, shared experts, router, and
            # dispatch-preprocess on the compute stream so that the subsequent
            # dispatch A2A on the comm stream is the only thing that still has
            # to run before the experts.
            if layer.recompute_pre_mlp_layernorm:
                layer.pre_mlp_norm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
                with off_interface(
                    layer.offload_mlp_norm, hidden_states, "mlp_norm"
                ) as hidden_states:
                    pre_mlp_layernorm_output = layer.pre_mlp_norm_checkpoint.checkpoint(
                        apply_module(layer.pre_mlp_layernorm), hidden_states
                    )
            else:
                with off_interface(
                    layer.offload_mlp_norm, hidden_states, "mlp_norm"
                ) as hidden_states:
                    pre_mlp_layernorm_output = apply_module(layer.pre_mlp_layernorm)(
                        hidden_states
                    )

            # When using fused residual norm (TEFusedResidualRMSNorm), layernorm
            # returns (normalized, residual). Unpack and use the fused residual
            # for the downstream BDA connection.
            if isinstance(pre_mlp_layernorm_output, tuple):
                if len(pre_mlp_layernorm_output) != 2:
                    raise ValueError(
                        f"When pre_mlp_layernorm returns a tuple it must have "
                        f"2 elements (output, residual), got {len(pre_mlp_layernorm_output)}"
                    )
                pre_mlp_layernorm_output, hidden_states = pre_mlp_layernorm_output

            shared_expert_output = layer.mlp.shared_experts_compute(pre_mlp_layernorm_output)
            probs, routing_map = layer.mlp.route(pre_mlp_layernorm_output)
            local_tokens, probs = layer.mlp.preprocess(
                pre_mlp_layernorm_output, probs, routing_map
            )
            return hidden_states, local_tokens, probs, shared_expert_output

        hidden_states, local_tokens, probs, shared_expert_output = forward_func(
            hidden_states=hidden_states,
            attention_mask=node.chunk_state.attention_mask,
            rotary_pos_emb=node.chunk_state.rotary_pos_emb,
            rotary_pos_cos=node.chunk_state.rotary_pos_cos,
            rotary_pos_sin=node.chunk_state.rotary_pos_sin,
            packed_seq_params=node.chunk_state.packed_seq_params,
            sequence_len_offset=node.chunk_state.sequence_len_offset,
        )

        if not isinstance(layer.mlp, MoELayer):
            return hidden_states

        # Detach the residual so the mlp_bda (run on the comm stream) doesn't
        # drag the attention's backward graph across stream boundaries.
        node.layer_state.residual = node.detach(hidden_states)
        if layer.mlp.use_shared_expert and not layer.mlp.shared_expert_overlap:
            node.layer_state.shared_expert_output = node.detach(shared_expert_output)

        return local_tokens, probs

    def submodule_dispatch_forward(node: ScheduleNode, local_tokens: Tensor, probs: Tensor):
        """Dispatch A2A. Runs on the comm stream; mirrors the GPT implementation."""
        token_dispatcher = layer.mlp.token_dispatcher
        if enable_deepep or enable_hybridep:
            # Update token_probs to the detached version; prevents the backward
            # graph from connecting back into the attn submodule.
            token_dispatcher._comm_manager.token_probs = probs

        dispatched_tokens, dispatched_probs = layer.mlp.dispatch(local_tokens, probs)

        # dispatched_probs is needed by the swiglu backward; stash it on
        # layer_state so the free_input logic on the forward pass doesn't
        # reclaim it.
        node.layer_state.dispatched_probs = node.detach(dispatched_probs)
        return dispatched_tokens

    def submodule_moe_forward(node: ScheduleNode, dispatched_tokens: Tensor):
        """Experts forward: dispatch-postprocess -> experts -> combine-preprocess."""
        dispatched_probs = node.layer_state.dispatched_probs
        token_dispatcher = layer.mlp.token_dispatcher
        if enable_deepep or enable_hybridep:
            token_dispatcher._comm_manager.dispatched_probs = dispatched_probs

        expert_output, _ = layer.mlp.routed_experts_compute(dispatched_tokens, dispatched_probs)

        # For HybridEP, tokens_per_expert is generated on the comm stream; keep
        # a reference alive so it is not freed before its consumers run.
        if enable_hybridep:
            tokens_per_expert = token_dispatcher._comm_manager.get_number_of_tokens_per_expert()
            node.layer_state.tokens_per_expert = tokens_per_expert

        if layer.recompute_pre_mlp_layernorm:
            layer.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(expert_output)

        return expert_output

    def submodule_combine_forward(node: ScheduleNode, output: Tensor):
        """Combine A2A + mlp_bda + (on last layer) final_layernorm."""
        residual = node.layer_state.residual
        shared_expert_output = getattr(node.layer_state, "shared_expert_output", None)
        output = layer.mlp.combine(output)
        output = layer.mlp.postprocess(output, shared_expert_output)

        mlp_output_with_bias = (output, None)
        if hasattr(layer, "cuda_graphs") and layer.cuda_graphs:
            layer.mlp.cudagraph_tensor_store.clear()
        with layer.bias_dropout_add_exec_handler():
            hidden_states = layer.mlp_bda(layer.training, layer.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, layer.hidden_dropout
            )
        if layer.offload_mlp_norm:
            hidden_states = off_interface.group_commit(
                hidden_states, name="mlp_norm", forced_released_tensors=[residual]
            )
        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        # Record residual on the compute stream so the forward pass's reference
        # doesn't outrun the comm stream's still-pending use of it.
        node.layer_state.residual.record_stream(torch.cuda.current_stream())
        if shared_expert_output is not None:
            shared_expert_output.record_stream(torch.cuda.current_stream())

        # Drop references after use to allow eager memory release.
        node.layer_state.residual = None
        node.layer_state.shared_expert_output = None

        # Apply the decoder's final_layernorm on the last layer, same as GPT.
        # ``HybridStack.final_layernorm`` is a property alias to ``final_norm``
        # so this access works for both model families without branching.
        final_layernorm = node.chunk_state.model.decoder.final_layernorm
        if not node.is_mtp and final_layernorm and node.is_last_layer:
            output = final_layernorm(output)
            output = make_viewless_tensor(inp=output, requires_grad=True, keep_graph=True)
        return output

    def mlp_wrapper(_node: ScheduleNode, hidden_states: Tensor) -> Tensor:
        """Dense-path ``mlp`` callable: run the monolithic MLP block."""
        return layer.forward_mlp(hidden_states)

    def raise_not_implemented(*_args, **_kwargs):
        """Stub for the MoE-only dispatch/combine slots on a dense layer."""
        raise NotImplementedError(
            "Dispatch/combine callables are only implemented for MoE combined layers."
        )

    # Hybrid MTP is out of scope for the combined-layer refactor; return None so
    # the schedule plan wraps it in a NoopScheduleNode.
    mtp_post_process_func = None

    # Register the backward-dw wrapper only if we have an attention submodule
    # whose weight gradients we can defer. The Mamba mixer's wgrad always runs
    # inline with the main backward pass.
    if layer.attention is not None:
        layer.init_backward_dw_wrapper()
        attn_backward_dw = layer.backward_dw_wrapper
    else:
        attn_backward_dw = None

    forward_funcs = [
        submodule_attn_forward,
        submodule_dispatch_forward if is_moe else raise_not_implemented,
        submodule_moe_forward if is_moe else mlp_wrapper,
        submodule_combine_forward if is_moe else raise_not_implemented,
        mtp_post_process_func,
    ]
    backward_dw = {"attn": attn_backward_dw, "mlp": layer.mlp}
    return forward_funcs, backward_dw
