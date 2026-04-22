# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""A2A overlap tests for the *combined* hybrid layer (bracket-notation patterns).

Each bracket group in the pattern (e.g. ``[M*-][ME]``) becomes one
:class:`CombinedHybridLayer` that contains Mamba + optional Attention +
MLP/MoE blocks. The schedule plan decomposes each layer into the standard
5-slot interface so the 1F1B fine-grained overlap reuses the GPT infrastructure.

These tests verify bit-identical results between:

* Reference path: ``HybridModel.forward(**data)`` (sequential)
* A2A-overlap path: ``TransformerModelChunkSchedulePlan.run()`` with interleaved
  forward/backward across two model chunks
"""

import gc

import pytest
import torch

from megatron.core.models.common.model_chunk_schedule_plan import (
    TransformerModelChunkSchedulePlan,
)
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.pipeline_parallel.utils import set_streams
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import float16_to_fp32
from megatron.core.utils import is_te_min_version
from tests.unit_tests.a2a_overlap.utils import (
    compare_captures,
    deterministic_mode,
    get_valid_token_dispatcher_types,
)
from tests.unit_tests.test_utilities import Utils


def get_test_config(num_moe_experts=8, extra_kwargs={}):
    """TransformerConfig shared by all combined-hybrid tests."""
    return TransformerConfig(
        attention_backend="unfused",
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=4 if num_moe_experts is not None else 1,
        deterministic_mode=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        # num_layers is derived from the pattern; set to a dummy high enough value
        # so validate_args doesn't complain; the actual layers come from brackets.
        num_layers=8,
        hidden_size=256,
        add_bias_linear=False,
        num_attention_heads=4,
        ffn_hidden_size=512,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        num_moe_experts=num_moe_experts,
        moe_grouped_gemm=True,
        moe_router_dtype="fp32",
        **extra_kwargs,
    )


def build_combined_hybrid_model(config, bracket_pattern, num_layers):
    """Build a HybridModel with a bracket-notation pattern."""
    config.num_layers = num_layers  # match the number of combined layers
    seq_len = 32
    max_seq_len = 300
    ids = list(range(seq_len))

    data = {
        "input_ids": torch.tensor(ids, dtype=torch.int64).repeat((1, 1)).cuda(),
        "labels": torch.tensor(ids, dtype=torch.int64).repeat((1, 1)).cuda(),
        "position_ids": torch.tensor(ids, dtype=torch.int64).repeat((1, 1)).cuda(),
        "attention_mask": torch.ones((1, 1, seq_len, seq_len), dtype=bool).cuda(),
    }

    model = HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=128,
        max_sequence_length=max_seq_len,
        hybrid_layer_pattern=bracket_pattern,
        pre_process=True,
        post_process=True,
    )
    return model, data


class TestCombinedHybridA2AOverlapChunk:
    """Model-chunk-level A2A overlap tests for combined hybrid layers.

    Patterns chosen to exercise the main combined-layer shapes:
    * ``[M-]`` Mamba + MLP (no attention, no MoE) — overlap is trivial
    * ``[M-][ME]`` Mamba+MLP then Mamba+MoE — MoE A2A should overlap with adjacent compute
    * ``[M*E]`` Mamba + Attention + MoE — all three blocks in one layer
    """

    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=4,
        )
        set_streams()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
    @pytest.mark.parametrize("dispatcher_type", get_valid_token_dispatcher_types())
    @pytest.mark.parametrize(
        "num_layers_and_patterns",
        [
            # Two chunks, each with 2 combined layers, one MoE each
            ([2, 2], ["[M-][ME]", "[M-][ME]"]),
            # One MoE per chunk, mixed with attention
            ([2, 2], ["[M*-][M*E]", "[M*-][M*E]"]),
            # Pure Mamba+MoE in both chunks (simplest MoE overlap case)
            ([2, 2], ["[ME][ME]", "[ME][ME]"]),
        ],
    )
    def test_1f1b_schedule_combined_hybrid_model_chunk(
        self, dispatcher_type, num_layers_and_patterns
    ):
        """Two combined-hybrid model chunks: reference vs A2A overlap, bit-identical."""
        num_layers_list, patterns = num_layers_and_patterns
        microbatches = 1

        models = []
        schedule_plans = []
        ref_captures = []
        datas = []

        extra_kwargs = {"moe_token_dispatcher_type": dispatcher_type}
        if dispatcher_type == "flex":
            extra_kwargs["moe_flex_dispatcher_backend"] = "deepep"

        with deterministic_mode():
            for num_layers, pattern in zip(num_layers_list, patterns):
                config = get_test_config(extra_kwargs=extra_kwargs)
                model, data = build_combined_hybrid_model(config, pattern, num_layers)
                model.cuda()
                models.append(model)
                datas.append(data)

                schedule_plan = model.build_schedule_plan(**data)
                schedule_plans.append(schedule_plan)

                # Reference: plain forward+backward
                output_tensors = []
                for _ in range(microbatches):
                    loss = model.forward(**data)
                    loss = float16_to_fp32(loss)
                    loss.backward(torch.ones_like(loss))
                    output_tensors.append(loss)

                capture = {"outputs": output_tensors}
                for name, param in model.named_parameters():
                    capture[name] = param.grad
                ref_captures.append(capture)
                model.zero_grad()

            assert models[0].embedding is not None
            assert models[1].embedding is not None

            # A2A overlap: interleaved forward/backward across chunks
            a2a_captures = [{"outputs": []} for _ in models]

            for i in range(microbatches):
                if i > 0:
                    assert schedule_plans[0].pre_process is None
                    schedule_plans[0] = models[0].build_schedule_plan(**datas[0])
                    schedule_plans[1] = models[1].build_schedule_plan(**datas[1])

                f_input_0 = TransformerModelChunkSchedulePlan.run(schedule_plans[0], None)
                a2a_captures[0]["outputs"].append(f_input_0)

                f_input_1 = TransformerModelChunkSchedulePlan.run(
                    schedule_plans[1],
                    schedule_plans[0],
                    b_grad=torch.ones_like(f_input_0),
                )
                a2a_captures[1]["outputs"].append(f_input_1)

                TransformerModelChunkSchedulePlan.run(
                    None, schedule_plans[1], b_grad=torch.ones_like(f_input_1)
                )

            for i in range(len(models)):
                for name, param in models[i].named_parameters():
                    a2a_captures[i][name] = param.grad

            for i in range(len(ref_captures)):
                comp_res = compare_captures(ref_captures[i], a2a_captures[i], True, True)
                assert comp_res[0], f"[rank {torch.distributed.get_rank()}] {comp_res[1]}"

            # Cleanup to keep GPU memory under control across parametrizations
            for i in range(len(schedule_plans)):
                schedule_plans[i] = None
                ref_captures[i] = None
                a2a_captures[i] = None
                for k in datas[i]:
                    datas[i][k] = None
                datas[i] = None
                models[i].zero_grad()
                models[i] = None
            gc.collect()
            torch.cuda.empty_cache()
