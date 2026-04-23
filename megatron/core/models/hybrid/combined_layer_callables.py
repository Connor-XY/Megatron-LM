# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.

"""Schedule callables for :class:`CombinedHybridLayer`.

Decomposes a combined hybrid layer into the five-slot interface
(``attn``, ``dispatch``, ``mlp``, ``combine``, ``mtp_post_process``) expected
by :class:`TransformerLayerSchedulePlan`, so the standard 1F1B fine-grained
schedule plan in :mod:`megatron.core.models.common.model_chunk_schedule_plan`
drives hybrid models without model-specific branches.

Slot assignment
---------------

**MoE layer (has a ``MoELayer`` in the MLP slot):**

================= ======================================================
``attn`` (comp)   Mamba/GDN -> Attention -> pre-MoE norm -> router ->
                  dispatcher preprocess  ``-> (tokens, probs)``
``dispatch`` (comm) MoE token-dispatch All-to-All
``mlp`` (comp)    MoE experts
``combine`` (comm) MoE combine All-to-All + mlp_bda (+ optional final_norm)
``mtp``           ``None`` (NoopScheduleNode upstream)
================= ======================================================

**Dense MLP layer (no MoE):** no communication to overlap, so the whole layer
runs in the ``attn`` slot:

================= ======================================================
``attn`` (comp)   Mamba/GDN -> Attention -> dense MLP (+ optional final_norm)
``dispatch``      raises (scheduler uses NoopScheduleNode)
``mlp`` (comp)    identity pass-through
``combine``       raises (scheduler uses NoopScheduleNode)
``mtp``           ``None``
================= ======================================================

The final-layer LayerNorm (``decoder.final_norm``) is applied inline in the
callables rather than inside a dedicated PostProcessNode hook, because the
combined-layer path hands the scheduler a single ``forward`` entry per layer
and the last-layer norm has to run on the same stream as the bda that
produced its input.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.models.hybrid.combined_layer import CombinedHybridLayer
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.pipeline_parallel.utils import ScheduleNode
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.typed_torch import apply_module
from megatron.core.utils import make_viewless_tensor


class _CombinedBackwardDWWrapper:
    """No-op ``backward_dw`` wrapper for combined hybrid layers.

    Mamba mixers and MoE experts compute their weight gradients inline during
    the standard backward pass, so there's no deferred wgrad to schedule on
    the combined layer. The scheduler's ``backward_dw_callables["attn"]``
    contract still calls ``wrapper.backward_dw()``, so we provide a stub that
    releases the layer reference to avoid prolonged lifetime.

    This keeps the GPT-side :class:`_BackwardDWWrapper` (which is hard-coded
    to ``TransformerLayer.self_attention``) untouched.
    """

    def __init__(self, layer: CombinedHybridLayer):
        self.layer = layer

    def backward_dw(self):
        self.layer = None

    def set_graphed_backward_dw_callable(self, graphed_backward_dw_callable):
        """CUDA-graph wgrad is not supported for combined layers yet; ignore."""
        # intentionally unused


def _apply_final_norm(node, output: Tensor) -> Tensor:
    """Apply ``decoder.final_norm`` on the last layer (same shape as GPT)."""
    final_norm = getattr(node.chunk_state.model.decoder, "final_norm", None)
    if getattr(node, "is_mtp", False) or not final_norm or not node.is_last_layer:
        return output
    output = final_norm(output)
    return make_viewless_tensor(inp=output, requires_grad=True, keep_graph=True)


def build_combined_hybrid_layer_callables(layer: CombinedHybridLayer):
    """Return ``(forward_funcs, backward_dw)`` for a :class:`CombinedHybridLayer`.

    The returned tuple shape matches :func:`build_transformer_layer_callables`
    so :class:`TransformerLayerSchedulePlan._build_callable_nodes` consumes it
    without branches: ``forward_funcs`` is a 5-list
    ``[attn, dispatch, mlp, combine, mtp_post_process]`` and
    ``backward_dw`` is a dict ``{"attn": ..., "mlp": ...}``.
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
    # attn slot
    # ------------------------------------------------------------------
    def submodule_attn_forward(node: ScheduleNode, hidden_states: Tensor):
        """Run the combined layer's compute-heavy portion.

        For MoE layers this runs Mamba/Attn + pre-MoE norm + router + dispatch
        preprocess, stopping just before the dispatch A2A.

        For dense-MLP layers (or mamba-only / attn-only layers), there is no
        communication to overlap, so we run the *entire* layer here and let
        the remaining schedule slots be no-ops.
        """
        # 1. Mamba / GDN block
        hidden_states = layer.forward_ssm_attn(
            hidden_states,
            attention_mask=node.chunk_state.attention_mask,
            rotary_pos_emb=node.chunk_state.rotary_pos_emb,
            rotary_pos_cos=node.chunk_state.rotary_pos_cos,
            rotary_pos_sin=node.chunk_state.rotary_pos_sin,
            packed_seq_params=node.chunk_state.packed_seq_params,
            sequence_len_offset=node.chunk_state.sequence_len_offset,
        )

        # 2a. MLP slot absent -- apply final_norm if needed and return.
        if layer.mlp_type == "none":
            return _apply_final_norm(node, hidden_states)

        # 2b. Dense MLP -- run the full MLP block here.
        if not is_moe:
            hidden_states = layer.forward_mlp(hidden_states)
            return _apply_final_norm(node, hidden_states)

        # 2c. MoE path -- run pre_mlp_norm + shared experts + router +
        #     dispatch-preprocess on the compute stream, then hand
        #     ``(local_tokens, probs)`` off to the comm-stream dispatch node.
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
                pre_mlp_layernorm_output = apply_module(layer.pre_mlp_layernorm)(hidden_states)

        # Fused residual RMSNorm returns (output, residual).
        if isinstance(pre_mlp_layernorm_output, tuple):
            if len(pre_mlp_layernorm_output) != 2:
                raise ValueError(
                    f"When pre_mlp_layernorm returns a tuple it must have 2 "
                    f"elements (output, residual), got {len(pre_mlp_layernorm_output)}"
                )
            pre_mlp_layernorm_output, hidden_states = pre_mlp_layernorm_output

        shared_expert_output = layer.mlp.shared_experts_compute(pre_mlp_layernorm_output)
        probs, routing_map = layer.mlp.route(pre_mlp_layernorm_output)
        local_tokens, probs = layer.mlp.preprocess(
            pre_mlp_layernorm_output, probs, routing_map
        )

        # Stash the residual so the comm-stream ``combine`` slot can consume
        # it without dragging the attn/mamba backward graph across streams.
        node.layer_state.residual = node.detach(hidden_states)
        if layer.mlp.use_shared_expert and not layer.mlp.shared_expert_overlap:
            node.layer_state.shared_expert_output = node.detach(shared_expert_output)

        return local_tokens, probs

    # ------------------------------------------------------------------
    # dispatch slot (comm, MoE only)
    # ------------------------------------------------------------------
    def submodule_dispatch_forward(node: ScheduleNode, local_tokens: Tensor, probs: Tensor):
        token_dispatcher = layer.mlp.token_dispatcher
        if enable_deepep or enable_hybridep:
            token_dispatcher._comm_manager.token_probs = probs
        dispatched_tokens, dispatched_probs = layer.mlp.dispatch(local_tokens, probs)
        node.layer_state.dispatched_probs = node.detach(dispatched_probs)
        return dispatched_tokens

    # ------------------------------------------------------------------
    # mlp slot
    # ------------------------------------------------------------------
    def submodule_moe_forward(node: ScheduleNode, dispatched_tokens: Tensor):
        """MoE experts forward (``routed_experts_compute``)."""
        dispatched_probs = node.layer_state.dispatched_probs
        token_dispatcher = layer.mlp.token_dispatcher
        if enable_deepep or enable_hybridep:
            token_dispatcher._comm_manager.dispatched_probs = dispatched_probs

        expert_output, _ = layer.mlp.routed_experts_compute(dispatched_tokens, dispatched_probs)

        if enable_hybridep:
            tokens_per_expert = token_dispatcher._comm_manager.get_number_of_tokens_per_expert()
            node.layer_state.tokens_per_expert = tokens_per_expert

        if layer.recompute_pre_mlp_layernorm:
            layer.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(expert_output)

        return expert_output

    def mlp_passthrough(node: ScheduleNode, hidden_states: Tensor) -> Tensor:
        """Dense path: MLP already ran inside the attn slot. Identity."""
        return hidden_states

    # ------------------------------------------------------------------
    # combine slot (comm, MoE only): combine A2A + mlp_bda (+ final_norm)
    # ------------------------------------------------------------------
    def submodule_combine_forward(node: ScheduleNode, expert_output: Tensor):
        residual = node.layer_state.residual
        shared_expert_output = getattr(node.layer_state, "shared_expert_output", None)
        output = layer.mlp.combine(expert_output)
        output = layer.mlp.postprocess(output, shared_expert_output)

        mlp_output_with_bias = (output, None)
        with layer.bias_dropout_add_exec_handler():
            hidden_states = layer.mlp_bda(layer.training, layer.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, layer.hidden_dropout
            )
        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        # Cross-stream reference retention.
        node.layer_state.residual.record_stream(torch.cuda.current_stream())
        if shared_expert_output is not None:
            shared_expert_output.record_stream(torch.cuda.current_stream())
        node.layer_state.residual = None
        node.layer_state.shared_expert_output = None

        return _apply_final_norm(node, output)

    def raise_not_implemented(*_args, **_kwargs):
        raise NotImplementedError(
            "dispatch/combine are only used for MoE combined layers; the "
            "scheduler should drive dense-path slots with NoopScheduleNode."
        )

    if is_moe:
        forward_funcs = [
            submodule_attn_forward,
            submodule_dispatch_forward,
            submodule_moe_forward,
            submodule_combine_forward,
            None,  # no MTP post-process (out of scope for combined layers)
        ]
        backward_dw = {
            "attn": _CombinedBackwardDWWrapper(layer),
            "mlp": layer.mlp,
        }
    else:
        forward_funcs = [
            submodule_attn_forward,
            raise_not_implemented,
            mlp_passthrough,
            raise_not_implemented,
            None,
        ]
        backward_dw = {
            "attn": _CombinedBackwardDWWrapper(layer),
            "mlp": None,
        }

    return forward_funcs, backward_dw


__all__ = ["build_combined_hybrid_layer_callables"]
