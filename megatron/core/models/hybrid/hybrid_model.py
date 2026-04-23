# Copyright (c) 2023-2026, NVIDIA CORPORATION. All rights reserved.

import logging
from typing import Literal, Optional

import torch
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.quantization.utils import get_quant_config_or_none
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionBlock,
    mtp_on_this_rank,
    process_mtp_loss,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.utils import (
    WrappedTensor,
    deprecate_inference_params,
    is_using_quantization_scales,
    log_single_rank,
)

logger = logging.getLogger(__name__)


class HybridModel(LanguageModule):
    """Hybrid language model.

    Args:
        config (TransformerConfig): Model config
        hybrid_stack_spec (ModuleSpec): Specifies the modules to use for the various layer types
        vocab_size (int): Vocabulary size
        max_sequence_length (int): maximum size of sequence.
            This is used for positional embedding
        hybrid_layer_pattern (str): Unified hybrid layer pattern with optional MTP and
            pipeline stage boundaries.
            Format: "<main_pattern>/<mtp_pattern>/<mtp_pattern>/..."
            The main pattern may contain "|" to define pipeline stage boundaries.
            Examples:
                - "M*M*" -> main decoder only, no MTP
                - "M*M*/MM/MM" -> main="M*M*", mtp="MM", 2 depths
                - "M-M-|M-M*-|M-M-|M-M*-" -> 4 pipeline segments
        hybrid_attention_ratio (float, optional): Deprecated. Use hybrid_layer_pattern instead.
            If set to a value > 0.0 and hybrid_layer_pattern is None, a pattern will be
            generated from the ratio with a deprecation warning.
        hybrid_mlp_ratio (float, optional): Deprecated. Use hybrid_layer_pattern instead.
            If set to a value > 0.0 and hybrid_layer_pattern is None, a pattern will be
            generated from the ratio with a deprecation warning.
        hybrid_override_pattern (str, optional): Deprecated. Use hybrid_layer_pattern instead.
            If set and hybrid_layer_pattern is None, the value is copied to hybrid_layer_pattern
            with a deprecation warning.
        pre_process (bool, optional): Include embedding layer
            (used with pipeline parallelism). Defaults to True.
        post_process (bool, optional): Include an output layer (used with pipeline parallelism).
            Defaults to True.
        fp16_lm_cross_entropy (bool, optional): Defaults to False.
        parallel_output (bool, optional): Do not gather the outputs, keep them split across tensor
            parallel ranks. Defaults to True.
        share_embeddings_and_output_weights (bool, optional): When True, input embeddings and
            output logit weights are shared. Defaults to False.
        position_embedding_type (Literal[learned_absolute,rope,none], optional):  Position
            embedding type. Defaults to 'none'.
        rotary_percent (float, optional): Percent of rotary dimension to use for rotary position
            embeddings. Ignored unless position_embedding_type is 'rope'. Defaults to 1.0.
        rotary_base (int, optional): Base period for rotary position embeddings. Ignored unless
            position_embedding_type is 'rope'. Defaults to 10000.
        seq_len_interpolation_factor (Optional[float], optional): scale of linearly
            interpolating RoPE for longer sequences. The value must be a float larger than 1.0.
             Defaults to None.
        pg_collection (ProcessGroupCollection, optional): Model communication process groups.
        vp_stage (Optional[int], optional): Virtual pipeline stage index. Defaults to None.
    """

    def __init__(
        self,
        config: TransformerConfig,
        hybrid_stack_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        hybrid_layer_pattern: Optional[str] = None,
        hybrid_attention_ratio: Optional[float] = None,
        hybrid_mlp_ratio: Optional[float] = None,
        hybrid_override_pattern: Optional[str] = None,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        # Mamba with no attention has no need for position embeddings, so none is default
        position_embedding_type: Literal['learned_absolute', 'rope', 'none'] = 'none',
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        scatter_embedding_sequence_parallel: bool = True,
        seq_len_interpolation_factor: Optional[float] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
    ) -> None:
        super().__init__(config=config, pg_collection=pg_collection)

        if has_config_logger_enabled(config):
            log_config_to_disk(config, locals(), prefix=type(self).__name__)

        if self.config.use_mup and not getattr(HybridModel, "mup_warning_printed", False):
            log_single_rank(
                logger,
                logging.WARNING,
                "MuP for HybridModel is experimental and not fully validated yet.",
            )
            HybridModel.mup_warning_printed = True

        self.hybrid_stack_spec: ModuleSpec = hybrid_stack_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.hybrid_layer_pattern = hybrid_layer_pattern
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.position_embedding_type = position_embedding_type
        self.vp_stage = vp_stage
        self.disable_param_offloading = True

        # Backward compatibility for deprecated hybrid parameters
        if hybrid_override_pattern is not None:
            if self.hybrid_layer_pattern is None:
                log_single_rank(
                    logger,
                    logging.WARNING,
                    "hybrid_override_pattern has been deprecated. "
                    "Use hybrid_layer_pattern instead.",
                )
                self.hybrid_layer_pattern = hybrid_override_pattern
            else:
                raise ValueError(
                    "hybrid_override_pattern and hybrid_layer_pattern cannot both be set. "
                    "hybrid_override_pattern has been deprecated; use hybrid_layer_pattern instead."
                )
        if (hybrid_attention_ratio is not None and hybrid_attention_ratio > 0.0) or (
            hybrid_mlp_ratio is not None and hybrid_mlp_ratio > 0.0
        ):
            if hybrid_layer_pattern is not None:
                raise ValueError(
                    "hybrid_layer_pattern cannot be used together with "
                    "hybrid_attention_ratio or hybrid_mlp_ratio. "
                    "These ratios have been deprecated; use hybrid_layer_pattern alone."
                )
            log_single_rank(
                logger,
                logging.WARNING,
                "hybrid_attention_ratio and hybrid_mlp_ratio have been deprecated. "
                "Use hybrid_layer_pattern instead.",
            )
            if self.hybrid_layer_pattern is None:
                from megatron.core.models.hybrid.hybrid_layer_allocation import pattern_from_ratios

                attn_ratio = hybrid_attention_ratio if hybrid_attention_ratio else 0.0
                mlp_ratio = hybrid_mlp_ratio if hybrid_mlp_ratio else 0.0
                self.hybrid_layer_pattern = pattern_from_ratios(
                    config.num_layers, attn_ratio, mlp_ratio
                )

        # Parse unified pattern to extract main and MTP components, and
        # determine the pipeline segment for this model instance.
        from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols as _HybridSymbols
        from megatron.core.models.hybrid.hybrid_layer_allocation import (
            is_bracketed_pattern,
            parse_bracketed_pattern,
            parse_hybrid_pattern,
            select_pipeline_segment,
        )

        parsed = parse_hybrid_pattern(self.hybrid_layer_pattern)
        self.mtp_pattern = parsed.mtp_pattern
        self.mtp_num_depths = parsed.mtp_num_depths

        # Bracketed grammar triggers the combined-layer path used for 1F1B
        # compute/communication overlap on hybrid models. Detect it up front
        # so downstream code can branch on the pattern shape rather than the
        # config flag alone.
        self.uses_combined_layers = is_bracketed_pattern(parsed.main_pattern)

        # If the user supplied the legacy ``hybrid_stack_spec`` (combined-
        # layer fields default to IdentityOp) but wrote a bracketed pattern,
        # auto-substitute the spec that populates those fields. This avoids a
        # surprising error at layer-build time and keeps the import surface
        # small for users who have default specs.
        if self.uses_combined_layers:
            from megatron.core.transformer.identity_op import IdentityOp as _IdentityOp

            provided = hybrid_stack_spec.submodules
            needs_sub = (
                provided.combined_layer is _IdentityOp
                and provided.combined_attn_layer is _IdentityOp
                and provided.combined_gdn_layer is _IdentityOp
            )
            if needs_sub:
                from megatron.core.models.hybrid.hybrid_layer_specs import (
                    hybrid_stack_spec_with_combined_layers,
                )

                log_single_rank(
                    logger,
                    logging.INFO,
                    "HybridModel: bracketed hybrid_layer_pattern detected but the "
                    "provided hybrid_stack_spec has no combined-layer fields "
                    "populated. Auto-substituting hybrid_stack_spec_with_combined_layers.",
                )
                hybrid_stack_spec = hybrid_stack_spec_with_combined_layers
                self.hybrid_stack_spec = hybrid_stack_spec

        layer_type_list: Optional[list] = None
        combined_layer_groups: Optional[list] = None
        if self.uses_combined_layers:
            # Bracketed pattern: split by '|' at bracket boundaries to pick the
            # current PP/VPP segment, then parse the selected segment into
            # ``CombinedLayerGroup`` records. ``|`` is guaranteed by the parser
            # to appear only between brackets, so a plain string split is
            # unambiguous.
            main_pattern = parsed.main_pattern or ""
            segments = (
                main_pattern.split(_HybridSymbols.PIPE)
                if _HybridSymbols.PIPE in main_pattern
                else [main_pattern]
            )

            pp_size = torch.distributed.get_world_size(self.pg_collection.pp) if self.pg_collection.pp is not None else 1
            pp_rank = torch.distributed.get_rank(self.pg_collection.pp) if self.pg_collection.pp is not None else 0

            if len(segments) > 1 and len(segments) % pp_size != 0:
                raise ValueError(
                    f"Number of pipe-delimited bracket segments ({len(segments)}) must "
                    f"be evenly divisible by pipeline_model_parallel_size ({pp_size})."
                )
            if len(segments) == 1 and pp_size > 1:
                raise ValueError(
                    "Multi-stage pipeline parallelism with a bracketed "
                    "hybrid_layer_pattern requires explicit '|' stage separators. "
                    "Add '|' between bracket groups to define pipeline stage "
                    "boundaries, or disable overlap_moe_expert_parallel_comm to "
                    "use the legacy auto-split path."
                )

            vp_rel = vp_stage if vp_stage is not None else 0
            segment_index = vp_rel * pp_size + pp_rank
            if segment_index >= len(segments):
                raise ValueError(
                    f"Pipeline segment index {segment_index} (pp_rank={pp_rank}, "
                    f"vp_stage={vp_rel}) is out of range for {len(segments)} "
                    f"bracket segments in pattern {main_pattern!r}."
                )

            # layer_offset: count the standalone-equivalent layers in all
            # preceding segments, i.e. the number of mixer/attn/mlp submodules
            # that appear before this segment in the flat standalone pattern.
            # Each bracket group contributes 2 or 3 standalone layers, so we
            # reparse the preceding segments and sum.
            layer_offset = 0
            for prev_segment in segments[:segment_index]:
                prev_groups = parse_bracketed_pattern(prev_segment)
                for g in prev_groups:
                    layer_offset += len(g.submodule_standalone_indices)

            # Parse the selected segment.
            combined_layer_groups = parse_bracketed_pattern(segments[segment_index])
        else:
            layer_type_list, layer_offset = select_pipeline_segment(
                parsed.main_pattern or '',
                self.pg_collection.pp,
                vp_stage,
                first_stage_layers=self.config.num_layers_in_first_pipeline_stage,
                last_stage_layers=self.config.num_layers_in_last_pipeline_stage,
            )

        # Determine if MTP is needed (based on pattern parsing)
        self.mtp_process = (
            self.mtp_pattern is not None
            and self.mtp_num_depths > 0
            # The following forces MTP to be on the final pipeline stage. It might be more optimal
            # to split the hybrid layer pattern into pipeline stages before parsing the pattern for
            # the current pipeline stage. This could also enable MTP standalone (MTP in a pipeline
            # stage separate from loss) to be supported in the hybrid model.
            and mtp_on_this_rank(self.config, ignore_virtual=False, vp_stage=self.vp_stage)
        )

        # megatron core pipelining currently depends on model type
        # TODO: remove this dependency ?
        self.model_type = ModelType.encoder_or_decoder

        if self.pre_process or self.mtp_process:
            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=position_embedding_type,
                scatter_to_sequence_parallel=scatter_embedding_sequence_parallel,
                tp_group=self.pg_collection.tp,
            )

        # MLA (also used by DeepSeek Sparse Attention) uses its own decoupled RoPE, therefore we do
        # not build standard RoPE here when using MLA.
        if self.position_embedding_type == 'rope' and not self.config.multi_latent_attention:
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
                use_cpu_initialization=self.config.use_cpu_initialization,
                cp_group=self.pg_collection.cp,
            )

        self.decoder = build_module(
            hybrid_stack_spec,
            self.config,
            pre_process=self.pre_process,
            layer_type_list=layer_type_list,
            combined_layer_groups=combined_layer_groups,
            pp_layer_offset=layer_offset,
            post_process=self.post_process,
            dtype=config.params_dtype,
            pg_collection=self.pg_collection,
        )

        # MTP block - uses mtp_block_spec from hybrid_stack_spec.submodules
        if self.mtp_process:
            hybrid_submodules = hybrid_stack_spec.submodules
            mtp_block_spec = hybrid_submodules.mtp_block_spec
            assert mtp_block_spec is not None, (
                "MTP pattern specified but mtp_block_spec is None in hybrid_stack_spec.submodules. "
                "Ensure hybrid_stack_spec includes mtp_block_spec for MTP support."
            )

            self.mtp = MultiTokenPredictionBlock(
                config=self.config,
                spec=mtp_block_spec,
                pg_collection=self.pg_collection,
                vp_stage=self.vp_stage,
                mtp_layer_pattern=self.mtp_pattern,
                mtp_num_depths=self.mtp_num_depths,
                hybrid_submodules=hybrid_submodules,
            )

        # Output
        if post_process or self.mtp_process:
            self.output_layer = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                self.vocab_size,
                config=config,
                init_method=(
                    config.embedding_init_method
                    if config.use_mup and not self.share_embeddings_and_output_weights
                    else config.init_method
                ),
                bias=False,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=self.pre_process
                and self.share_embeddings_and_output_weights,
                tp_group=self.pg_collection.tp,
            )

        if self.pre_process or self.post_process or self.mtp_process:
            self.setup_embeddings_and_output_layer()

        for name, module in self.named_modules():
            if hasattr(module, 'finish_init'):
                quant_config = get_quant_config_or_none(name, self.config.quant_recipe)
                module.finish_init(quant_config)

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for gpt/bert'
        self.decoder.set_input_tensor(input_tensor[0])

    def preprocess_for_fine_grained_offloading(self):
        """Preprocess for fine-grained activation offloading."""
        off_interface.init_chunk_handler(
            vp_size=self.config.virtual_pipeline_model_parallel_size,
            vp_stage=self.vp_stage,
            min_offloaded_tensor_size=self.config.min_offloaded_tensor_size,
        )
        if self.disable_param_offloading:
            for param in self.decoder.parameters():
                off_interface.mark_not_offloadable(param)
            if self.mtp_process:
                for param in self.mtp.parameters():
                    off_interface.mark_not_offloadable(param)
            if self.post_process:
                for param in self.output_layer.parameters():
                    off_interface.mark_not_offloadable(param)
            self.disable_param_offloading = False

    def _preprocess(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        decoder_input: Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
        padding_mask: Optional[Tensor] = None,
    ):
        """Compute the decoder-input tensor and any positional encodings.

        Returns a 6-tuple with the same shape as :meth:`GPTModel._preprocess`
        so the GPT-authored :class:`PreProcessNode` in the shared schedule plan
        infrastructure can drive a HybridModel without code-path branches.
        Hybrid does not yet use fused-RoPE flash decode, so
        ``rotary_pos_cos`` / ``rotary_pos_sin`` are always ``None``.
        """
        in_inference_mode = inference_context is not None and not self.training

        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)

            # Clear outputs for padding tokens when using dynamic batching with
            # quantization scales, to avoid corrupting amax calculations.
            if (
                in_inference_mode
                and inference_context.is_dynamic_batching()
                and is_using_quantization_scales(self.config)
            ):
                decoder_input[inference_context.padding_slice] = 0.0
        else:
            # intermediate stage of pipeline; decoder will get hidden_states
            # from encoder.input_tensor.
            decoder_input = None

        rotary_pos_emb = None
        if self.position_embedding_type == 'rope' and not self.config.multi_latent_attention:
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_context, self.decoder, decoder_input, self.config, packed_seq_params
            )
            rotary_pos_emb = self.rotary_pos_emb(
                rotary_seq_len,
                packed_seq=packed_seq_params is not None and packed_seq_params.qkv_format == 'thd',
            )

        # Wrap decoder_input to allow the decoder (HybridStack) to delete the
        # reference held by this caller function, enabling early garbage
        # collection for inference.
        if in_inference_mode:
            decoder_input = WrappedTensor(decoder_input)

        # Flash-decode fused-RoPE is not wired up for hybrid yet; these slots
        # stay None to preserve the 6-tuple shape the shared PreProcessNode
        # expects.
        rotary_pos_cos = None
        rotary_pos_sin = None
        sequence_len_offset = None

        return (
            decoder_input,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
            padding_mask,
        )

    def _postprocess(
        self,
        hidden_states,
        input_ids,
        position_ids,
        labels,
        rotary_pos_emb,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        mtp_in_postprocess=None,
        loss_mask=None,
        decoder_input=None,
        attention_mask=None,
        inference_params=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        runtime_gather_output=None,
        extra_block_kwargs=None,
        inference_context=None,
        is_spec_decode=None,
    ):
        """Run MTP + output layer + loss on decoder hidden states.

        Signature matches :meth:`GPTModel._postprocess` so the GPT-authored
        :class:`PostProcessNode` in the shared schedule plan can drive a
        HybridModel. Args unused by hybrid (``rotary_pos_cos``,
        ``rotary_pos_sin``) are accepted and ignored.
        """
        del rotary_pos_cos  # hybrid uses standard RoPE; unused
        del rotary_pos_sin

        in_inference_mode = inference_context is not None and not self.training

        # Speculative decoding detection (mirror of GPT).
        if is_spec_decode is None:
            is_spec_decode = (
                in_inference_mode
                and inference_context.is_dynamic_batching()
                and inference_context.num_speculative_tokens > 0
            )

        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()

        mtp_forward_ran = self.mtp_process and not (in_inference_mode or is_spec_decode)
        if mtp_in_postprocess is None:
            # When ``mtp_in_postprocess`` isn't supplied (e.g. legacy callers),
            # fall back to the same condition as the original forward.
            mtp_in_postprocess = mtp_forward_ran
        if mtp_in_postprocess and not (in_inference_mode or is_spec_decode):
            hidden_states = self.mtp(
                input_ids=input_ids,
                position_ids=position_ids,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
                packed_seq_params=packed_seq_params,
                embedding=self.embedding,
                **(extra_block_kwargs or {}),
            )

        if not self.post_process:
            return hidden_states

        if self.config.mtp_num_layers is not None and self.mtp_process:
            assert self.config.mtp_num_layers > 0
            if in_inference_mode or is_spec_decode:
                self._decoder_hidden_states_cache = hidden_states
            else:
                hidden_states = process_mtp_loss(
                    hidden_states=hidden_states,
                    labels=labels,
                    loss_mask=loss_mask,
                    output_layer=self.output_layer,
                    output_weight=output_weight,
                    runtime_gather_output=runtime_gather_output,
                    is_training=self.training,
                    compute_language_model_loss=self.compute_language_model_loss,
                    config=self.config,
                    cp_group=self.pg_collection.cp,
                    packed_seq_params=packed_seq_params,
                    scale_logits_fn=self._scale_logits if self.config.use_mup else None,
                )

        sequence_parallel_override = False
        if in_inference_mode and inference_context.config.materialize_only_last_token_logits:
            if inference_context.is_static_batching():
                hidden_states = hidden_states[-1:, :, :]
            else:
                if self.output_layer.sequence_parallel:
                    hidden_states = gather_from_sequence_parallel_region(
                        hidden_states, group=self.pg_collection.tp
                    )
                    self.output_layer.sequence_parallel = False
                    sequence_parallel_override = True

                reshaped = hidden_states.squeeze(1).unsqueeze(0)
                hidden_states = inference_context.last_token_logits(reshaped).unsqueeze(1)

        logits, _ = self.output_layer(
            hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output
        )
        logits = self._scale_logits(logits)

        if sequence_parallel_override:
            assert (
                in_inference_mode
                and inference_context.is_dynamic_batching()
                and inference_context.config.materialize_only_last_token_logits
            )
            self.output_layer.sequence_parallel = True

        if labels is None:
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous()

        return self.compute_language_model_loss(labels, logits)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_context: BaseInferenceContext = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        loss_mask: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        padding_mask: Optional[Tensor] = None,
        is_spec_decode: Optional[bool] = None,
        extra_block_kwargs: Optional[dict] = None,
    ) -> Tensor:
        """Forward pass: embedding -> decoder -> output layer + loss.

        The body is split into :meth:`_preprocess` / decoder / :meth:`_postprocess`
        so the 1F1B combined schedule plan can drive the same steps on separate
        CUDA streams when ``overlap_moe_expert_parallel_comm`` is on.
        """
        if self.config.fine_grained_activation_offloading:
            self.preprocess_for_fine_grained_offloading()

        inference_context = deprecate_inference_params(inference_context, inference_params)

        in_inference_mode = inference_context is not None and not self.training
        if in_inference_mode:
            assert runtime_gather_output, "Inference must always gather TP logits"

        (
            decoder_input,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
            padding_mask,
        ) = self._preprocess(
            input_ids=input_ids,
            position_ids=position_ids,
            decoder_input=decoder_input,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
            padding_mask=padding_mask,
        )

        # Run decoder.
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
            padding_mask=padding_mask,
        )

        return self._postprocess(
            hidden_states=hidden_states,
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            mtp_in_postprocess=self.mtp_process,
            loss_mask=loss_mask,
            decoder_input=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            runtime_gather_output=runtime_gather_output,
            extra_block_kwargs=extra_block_kwargs,
            inference_context=inference_context,
            is_spec_decode=is_spec_decode,
        )

    def build_schedule_plan(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: Optional[dict] = None,
        runtime_gather_output: Optional[bool] = None,
        inference_params: Optional[BaseInferenceContext] = None,
        loss_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
    ):
        """Build a :class:`TransformerModelChunkSchedulePlan` for 1F1B overlap.

        Mirrors :meth:`GPTModel.build_schedule_plan`. The schedule plan is only
        meaningful during training; ``overlap_moe_expert_parallel_comm`` must
        be enabled and ``self.training`` must be True for the schedule plan to
        be driven by the combined-1f1b scheduler.
        """
        assert self.training or not self.config.overlap_moe_expert_parallel_comm, (
            "HybridModel.build_schedule_plan is a training-only path; "
            "overlap_moe_expert_parallel_comm must be off during inference."
        )

        if self.config.fine_grained_activation_offloading:
            self.preprocess_for_fine_grained_offloading()

        from megatron.core.models.common.model_chunk_schedule_plan import (
            TransformerModelChunkSchedulePlan,
        )

        return TransformerModelChunkSchedulePlan(
            self,
            input_ids,
            position_ids,
            attention_mask,
            decoder_input,
            labels,
            packed_seq_params,
            extra_block_kwargs,
            runtime_gather_output,
            loss_mask,
            padding_mask,
        )
