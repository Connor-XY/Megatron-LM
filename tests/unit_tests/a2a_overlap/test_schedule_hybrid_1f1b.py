# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""Tests for A2A overlap with hybrid Mamba-Transformer models.

Verifies that the fine-grained 1F1B schedule plan produces bit-identical
results to the reference (non-overlapped) forward/backward for HybridModel
with hybrid layer patterns containing MoE layers (e.g., "ME", "MEME").
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


def get_mamba_test_config(num_layers, num_moe_experts=8, extra_kwargs={}):
    """Create a TransformerConfig suitable for hybrid Mamba+MoE A2A overlap tests."""
    config = TransformerConfig(
        attention_backend="unfused",
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=4 if num_moe_experts is not None else 1,
        deterministic_mode=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        num_layers=num_layers,
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
    return config


def build_hybrid_model(config, hybrid_layer_pattern):
    """Build a HybridModel and input data for testing."""
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
        hybrid_layer_pattern=hybrid_layer_pattern,
        pre_process=True,
        post_process=True,
    )
    return model, data


class TestHybridA2AOverlapChunk:
    """Model-chunk-level A2A overlap tests for hybrid Mamba-MoE models.

    Verifies that the full TransformerModelChunkSchedulePlan (preprocessing,
    per-layer schedule, postprocessing) produces bit-identical results to
    HybridModel.forward() for models with mixed Mamba + MoE layers.
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
        "layers_and_patterns",
        [
            ([2, 2], ["ME", "ME"]),
            ([2, 2], ["ME", "EM"]),
            ([4, 2], ["MEME", "ME"]),
        ],
    )
    def test_1f1b_schedule_hybrid_model_chunk(self, dispatcher_type, layers_and_patterns):
        """Two model chunks with hybrid Mamba-MoE: reference vs A2A overlap."""
        layers, patterns = layers_and_patterns
        microbatches = 1

        models = []
        schedule_plans = []
        ref_captures = []
        datas = []

        extra_kwargs = {"moe_token_dispatcher_type": dispatcher_type}
        if dispatcher_type == "flex":
            extra_kwargs["moe_flex_dispatcher_backend"] = "deepep"

        with deterministic_mode():
            for layer_num, pattern in zip(layers, patterns):
                config = get_mamba_test_config(num_layers=layer_num, extra_kwargs=extra_kwargs)
                model, data = build_hybrid_model(config, pattern)
                model.cuda()
                models.append(model)
                datas.append(data)

                schedule_plan = model.build_schedule_plan(**data)
                schedule_plans.append(schedule_plan)

                # --- Reference forward + backward ---
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

            # --- A2A overlap run ---
            a2a_captures = [{"outputs": []} for _ in models]

            for i in range(microbatches):
                if i > 0:
                    assert (
                        schedule_plans[0].pre_process is None
                    ), "pre_process should be released after backward"
                    schedule_plans[0] = models[0].build_schedule_plan(**datas[0])
                    schedule_plans[1] = models[1].build_schedule_plan(**datas[1])

                # 1st forward
                f_input_0 = TransformerModelChunkSchedulePlan.run(schedule_plans[0], None)
                a2a_captures[0]["outputs"].append(f_input_0)

                # Overlapped forward[1] + backward[0]
                f_input_1 = TransformerModelChunkSchedulePlan.run(
                    schedule_plans[1], schedule_plans[0], b_grad=torch.ones_like(f_input_0)
                )
                a2a_captures[1]["outputs"].append(f_input_1)

                # Last backward
                TransformerModelChunkSchedulePlan.run(
                    None, schedule_plans[1], b_grad=torch.ones_like(f_input_1)
                )

            for i in range(len(models)):
                for name, param in models[i].named_parameters():
                    a2a_captures[i][name] = param.grad

            # --- Compare ---
            for i in range(len(ref_captures)):
                comp_res = compare_captures(ref_captures[i], a2a_captures[i], True, True)
                assert comp_res[0], f"[rank {torch.distributed.get_rank()}] {comp_res[1]}"

            # --- Cleanup ---
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
