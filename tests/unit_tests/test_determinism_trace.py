# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
import os
import threading
from types import SimpleNamespace

import pytest
import torch

from megatron.core import determinism_trace as trace_module
from megatron.core.determinism_trace import (
    EventKind,
    active_trace,
    begin_collective_trace,
    begin_recompute_trace,
    collective_trace_phase,
    record_collective_result,
    record_event,
    record_optimizer_state,
    record_recompute_phase,
    record_tensor,
    trace_iteration,
)
from megatron.core.distributed.finalize_model_grads import (
    _all_reduce_with_determinism_trace,
    _broadcast_with_determinism_trace,
)
from megatron.core.extensions import transformer_engine as te_extension
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_group,
    get_tensor_model_parallel_group,
)
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.tensor_parallel.layers import (
    linear_with_frozen_weight,
    linear_with_grad_accumulation_and_async_allreduce,
)
from megatron.core.tensor_parallel.mappings import (
    all_gather_last_dim_from_tensor_parallel_region,
    all_to_all,
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_last_dim_to_tensor_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from megatron.core.tensor_parallel.random import (
    CheckpointWithoutOutput,
    checkpoint,
    get_cuda_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.moe.moe_utils import get_updated_expert_bias
from tests.unit_tests.distributed.test_param_and_grad_buffer import get_model_and_buffers
from tests.unit_tests.test_utilities import Utils


def _read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _trace_path(root, iteration):
    rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_initialized()
        else int(os.environ.get("RANK", 0))
    )
    return root / f"iter_{iteration:07d}" / f"rank_{rank:05d}.jsonl"


def test_disabled_trace_is_a_noop(tmp_path):
    with trace_iteration(None, 1) as trace:
        assert trace is None
        assert active_trace() is None
        record_event(EventKind.PHASE, "ignored")
        record_tensor("ignored", torch.ones(1))
        handle = begin_collective_trace("ignored", "all_reduce", torch.ones(1))
        assert handle is None
        record_collective_result(handle, torch.ones(1))

    assert not list(tmp_path.rglob("*.jsonl"))


def test_trace_writes_versioned_rank_local_events_and_tensor_hash(tmp_path):
    with trace_iteration(tmp_path, 7, hash_tensors=True) as trace:
        assert active_trace() is trace
        record_event(EventKind.PHASE, "forward.begin", {"microbatch": 0})
        record_tensor("activation/layer.2", torch.tensor([1.0, 2.0]))

    events = _read_events(_trace_path(tmp_path, 7))
    assert [event["sequence"] for event in events] == list(range(len(events)))
    assert all(event["schema_version"] == 1 for event in events)
    tensor_event = next(event for event in events if event["kind"] == "tensor")
    assert tensor_event["payload"]["shape"] == [2]
    assert len(tensor_event["payload"]["sha256"]) == 64
    assert events[-1]["name"] == "iteration.end"
    assert active_trace() is None


def test_runtime_trace_records_kernel_selection_inputs(tmp_path, monkeypatch):
    environment = {
        "MAMBA_DETERMINISTIC": "1",
        "NVTE_FLASH_ATTN": "1",
        "NVTE_FUSED_ATTN": "0",
        "NVTE_UNFUSED_ATTN": "0",
        "TRITON_CACHE_AUTOTUNING": "0",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    def package_version(name):
        if name == "flash-attn":
            raise trace_module.importlib_metadata.PackageNotFoundError(name)
        return f"{name}-version"

    monkeypatch.setattr(trace_module.importlib_metadata, "version", package_version)
    with trace_iteration(tmp_path, 23):
        pass

    events = _read_events(_trace_path(tmp_path, 23))
    runtime = next(event for event in events if event["name"] == "runtime")["payload"]
    assert {name: runtime["environment"][name] for name in environment} == environment
    assert runtime["package_versions"] == {
        "flash-attn": None,
        "mamba-ssm": "mamba-ssm-version",
        "transformer-engine": "transformer-engine-version",
        "triton": "triton-version",
    }


def test_trace_hashes_size_one_zero_stride_tensor(tmp_path):
    tensor = torch.as_strided(torch.tensor([1.0]), size=(1,), stride=(0,))
    assert tensor.stride() == (0,)

    with trace_iteration(tmp_path, 19, hash_tensors=True):
        record_tensor("zero_stride", tensor)

    events = _read_events(_trace_path(tmp_path, 19))
    tensor_event = next(event for event in events if event["name"] == "zero_stride")
    assert len(tensor_event["payload"]["sha256"]) == 64


def test_recompute_trace_reports_forward_identity(tmp_path):
    tensor = torch.tensor([1.0, 2.0])

    def checkpointed(value):
        return value.square()

    with trace_iteration(tmp_path, 3, hash_tensors=True):
        handle = begin_recompute_trace(checkpointed, (tensor,))
        record_recompute_phase(handle, "forward", checkpointed(tensor))
        record_recompute_phase(handle, "recompute", checkpointed(tensor))

    events = _read_events(_trace_path(tmp_path, 3))
    recompute = next(event for event in events if event["name"] == "checkpoint.recompute")
    assert recompute["payload"]["checkpoint_id"] == 0
    assert recompute["payload"]["matches_forward"] is True


def test_recompute_trace_detects_changed_output(tmp_path):
    tensor = torch.tensor([1.0, 2.0])
    with trace_iteration(tmp_path, 4, hash_tensors=True):
        handle = begin_recompute_trace(lambda value: value, (tensor,))
        record_recompute_phase(handle, "forward", tensor)
        record_recompute_phase(handle, "recompute", tensor + 1)

    events = _read_events(_trace_path(tmp_path, 4))
    recompute = next(event for event in events if event["name"] == "checkpoint.recompute")
    assert recompute["payload"]["matches_forward"] is False


def test_te_checkpoint_records_paired_recompute(tmp_path, monkeypatch):
    phases = []

    def checkpointed(value, scale):
        phases.append(collective_trace_phase())
        return value.square() * scale

    def fake_te_checkpoint(function, *args, **kwargs):
        for key in ("distribute_saved_activations", "get_rng_state_tracker", "tp_group"):
            kwargs.pop(key)
        function(*args, **kwargs)
        return function(*args, **kwargs)

    monkeypatch.setattr(te_extension, "HAVE_TE", True)
    monkeypatch.setattr(te_extension, "is_te_min_version", lambda _: True)
    monkeypatch.setattr("transformer_engine.pytorch.distributed.checkpoint", fake_te_checkpoint)

    tensor = torch.tensor([1.0, 2.0])
    with trace_iteration(tmp_path, 20, hash_tensors=True):
        output = te_extension.te_checkpoint(checkpointed, False, None, None, tensor, scale=2)

    torch.testing.assert_close(output, tensor.square() * 2)
    assert phases == ["forward", "recompute"]
    events = _read_events(_trace_path(tmp_path, 20))
    recompute = [event for event in events if event["kind"] == "recompute"]
    assert [event["name"] for event in recompute] == ["checkpoint.forward", "checkpoint.recompute"]
    assert recompute[1]["payload"]["matches_forward"] is True


@pytest.mark.skipif(
    not torch.cuda.is_available() or not te_extension.HAVE_TE,
    reason="Transformer Engine checkpoint integration requires CUDA and TE",
)
def test_te_checkpoint_autograd_records_paired_recompute(tmp_path):
    phases = []

    def checkpointed(value):
        phases.append(collective_trace_phase())
        return value.square()

    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    try:
        model_parallel_cuda_manual_seed(123, force_reset_rng=True)
        tensor = torch.tensor([1.0, 2.0], device="cuda", requires_grad=True)
        with trace_iteration(tmp_path, 22, hash_tensors=True):
            output = te_extension.te_checkpoint(
                checkpointed, False, get_cuda_rng_tracker, get_tensor_model_parallel_group(), tensor
            )
            output.sum().backward()

        assert phases == ["forward", "recompute"]
        events = _read_events(_trace_path(tmp_path, 22))
        recompute = [event for event in events if event["kind"] == "recompute"]
        assert [event["name"] for event in recompute] == [
            "checkpoint.forward",
            "checkpoint.recompute",
        ]
        assert recompute[1]["payload"]["matches_forward"] is True
    finally:
        Utils.destroy_model_parallel()


def test_checkpoint_without_output_records_paired_recompute(tmp_path):
    phases = []

    def checkpointed(value):
        phases.append(collective_trace_phase())
        return value.square()

    tensor = torch.tensor([1.0, 2.0], requires_grad=True)
    with trace_iteration(tmp_path, 21, hash_tensors=True):
        checkpoint_object = CheckpointWithoutOutput()
        activation = checkpoint_object.checkpoint(checkpointed, tensor)
        output = activation * tensor
        checkpoint_object.discard_output_and_register_recompute(output)
        output.sum().backward()

    assert phases == ["forward", "recompute"]
    events = _read_events(_trace_path(tmp_path, 21))
    recompute = [event for event in events if event["kind"] == "recompute"]
    assert [event["name"] for event in recompute] == ["checkpoint.forward", "checkpoint.recompute"]
    assert recompute[1]["payload"]["matches_forward"] is True


def test_nonfinite_payloads_remain_valid_json(tmp_path):
    with trace_iteration(tmp_path, 5):
        record_event(EventKind.OPTIMIZER, "optimizer.end", {"grad_norm": float("nan")})

    events = _read_events(_trace_path(tmp_path, 5))
    optimizer = next(event for event in events if event["name"] == "optimizer.end")
    assert optimizer["payload"]["grad_norm"] == "nan"


def test_optimizer_state_trace_records_exact_tensor_state(tmp_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0], device=device))
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    parameter.grad = torch.tensor([0.25, 0.5], device=device)
    optimizer.step()
    optimizer.state[parameter]["integer_step"] = 1
    chained_optimizer = SimpleNamespace(chained_optimizers=[optimizer])

    with trace_iteration(tmp_path, 15, hash_tensors=True):
        record_optimizer_state("optimizer.output.state", chained_optimizer)

    events = _read_events(_trace_path(tmp_path, 15))
    tensor_state_events = [
        event
        for event in events
        if event["kind"] == "tensor" and event["name"].startswith("optimizer.output.state/")
    ]
    assert {event["name"].rsplit("/", 1)[-1] for event in tensor_state_events} == {
        "main_param",
        "step",
        "exp_avg",
        "exp_avg_sq",
    }
    assert all(len(event["payload"]["sha256"]) == 64 for event in tensor_state_events)
    scalar_event = next(event for event in events if event["name"].endswith("/scalars"))
    assert scalar_event["payload"] == {"integer_step": 1}


def test_optimizer_state_trace_requires_tensor_hashes(tmp_path):
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    with trace_iteration(tmp_path, 16):
        record_optimizer_state("optimizer.output.state", optimizer)

    events = _read_events(_trace_path(tmp_path, 16))
    assert not any(event["name"].startswith("optimizer.output.state/") for event in events)


def test_collective_trace_records_semantic_input_and_output(tmp_path):
    input_tensor = torch.tensor([1.0, 2.0])
    output_tensor = input_tensor + 1
    with trace_iteration(tmp_path, 8, hash_tensors=True):
        handle = begin_collective_trace(
            "test.collective", "all_reduce", input_tensor, metadata={"async_op": False}
        )
        record_collective_result(handle, output_tensor)
        record_collective_result(handle, output_tensor)

    events = _read_events(_trace_path(tmp_path, 8))
    collective = [event for event in events if event["kind"] == "collective"]
    assert [event["name"] for event in collective] == [
        "test.collective.begin",
        "test.collective.end",
    ]
    assert collective[0]["payload"]["inputs"][0]["sha256"]
    assert collective[1]["payload"]["outputs"][0]["sha256"]
    assert events[-1]["payload"]["pending_collectives"] == 0


@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_finalize_model_grad_collectives_record_existing_completion_points(tmp_path, deterministic):
    Utils.initialize_model_parallel()
    deterministic_algorithms_enabled = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(deterministic)
        group = torch.distributed.group.WORLD
        rank = torch.distributed.get_rank()
        reduced = torch.tensor([rank + 1.0], device="cuda")
        broadcast = torch.tensor([rank + 10.0], device="cuda")

        with trace_iteration(tmp_path, 23, hash_tensors=True):
            _all_reduce_with_determinism_trace(
                reduced,
                group=group,
                trace_name="finalize_model_grads.tensor_parallel_sum.backward",
                metadata={"gradient_kind": "non_tensor_parallel_parameter"},
            )
            _broadcast_with_determinism_trace(
                broadcast,
                src=0,
                group=group,
                trace_name="finalize_model_grads.num_tokens.pipeline_broadcast",
                metadata={"value_kind": "token_count"},
            )

        world_size = torch.distributed.get_world_size()
        torch.testing.assert_close(
            reduced, torch.tensor([world_size * (world_size + 1) / 2], device="cuda")
        )
        torch.testing.assert_close(broadcast, torch.tensor([10.0], device="cuda"))
        events = _read_events(_trace_path(tmp_path, 23))
        collective = [event for event in events if event["kind"] == "collective"]
        assert [event["name"] for event in collective] == [
            "finalize_model_grads.tensor_parallel_sum.backward.begin",
            "finalize_model_grads.tensor_parallel_sum.backward.end",
            "finalize_model_grads.num_tokens.pipeline_broadcast.begin",
            "finalize_model_grads.num_tokens.pipeline_broadcast.end",
        ]
        assert collective[0]["payload"]["operation"] == "all_reduce"
        assert collective[0]["payload"]["reduce_op"] == "sum"
        assert collective[0]["payload"]["implementation"] == (
            "logical_rank_all_to_all_sum_all_gather" if deterministic else "native"
        )
        if deterministic:
            assert collective[0]["payload"]["accumulation_dtype"] == "torch.float32"
        assert collective[2]["payload"]["operation"] == "broadcast"
        assert collective[2]["payload"]["src"] == 0
        assert all(
            event["payload"].get("inputs", event["payload"].get("outputs"))[0]["sha256"]
            for event in collective
        )
        assert events[-1]["payload"]["pending_collectives"] == 0
    finally:
        torch.use_deterministic_algorithms(deterministic_algorithms_enabled)
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    ("op", "expected"),
    [(torch.distributed.ReduceOp.SUM, 3.0), (torch.distributed.ReduceOp.AVG, 1.5)],
)
def test_finalize_model_grad_two_rank_reduction_keeps_exact_native_collective(
    tmp_path, op, expected
):
    Utils.initialize_model_parallel(tensor_model_parallel_size=2)
    deterministic_algorithms_enabled = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        group = get_tensor_model_parallel_group()
        group_rank = torch.distributed.get_rank(group=group)
        reduced = torch.tensor([group_rank + 1.0], device="cuda")

        with trace_iteration(tmp_path, 25, hash_tensors=True):
            _all_reduce_with_determinism_trace(
                reduced,
                group=group,
                trace_name="finalize_model_grads.tensor_parallel_sum.backward",
                op=op,
            )

        torch.testing.assert_close(reduced, torch.tensor([expected], device="cuda"))
        events = _read_events(_trace_path(tmp_path, 25))
        collective = [event for event in events if event["kind"] == "collective"]
        assert collective[0]["payload"]["implementation"] == "native_two_rank_exact"
        assert collective[0]["payload"]["group_size"] == 2
        assert events[-1]["payload"]["pending_collectives"] == 0
    finally:
        torch.use_deterministic_algorithms(deterministic_algorithms_enabled)
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_router_expert_bias_trace_preserves_large_integer_counts(tmp_path):
    Utils.initialize_model_parallel()
    try:
        group = torch.distributed.group.WORLD
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        tokens_per_expert = torch.tensor(
            [2**24 + int(rank == 0), 2**24], dtype=torch.int64, device="cuda"
        )
        expert_bias = torch.zeros(2, dtype=torch.float32, device="cuda")

        with trace_iteration(tmp_path, 24, hash_tensors=True):
            updated_bias = get_updated_expert_bias(
                tokens_per_expert, expert_bias, expert_bias_update_rate=0.1, tp_dp_cp_group=group
            )

        expected_counts = torch.tensor(
            [world_size * 2**24 + 1, world_size * 2**24], dtype=torch.int64, device="cuda"
        )
        torch.testing.assert_close(tokens_per_expert, expected_counts)
        assert expected_counts.to(torch.float32).unique().numel() == 1
        expected_offset = expected_counts.sum() - expected_counts * expected_counts.numel()
        expected_bias = torch.sign(expected_offset).to(expert_bias.dtype) * 0.1
        torch.testing.assert_close(updated_bias, expected_bias)

        events = _read_events(_trace_path(tmp_path, 24))
        collective = [event for event in events if event["kind"] == "collective"]
        assert [event["name"] for event in collective] == [
            "moe.router_expert_bias.token_count.begin",
            "moe.router_expert_bias.token_count.end",
        ]
        assert collective[0]["payload"]["operation"] == "all_reduce"
        assert collective[0]["payload"]["reduce_op"] == "sum"
        assert collective[0]["payload"]["exact_integer_accumulation"] is True
        assert collective[0]["payload"]["inputs"][0]["dtype"] == "torch.int64"
        assert collective[1]["payload"]["outputs"][0]["sha256"]
        assert events[-1]["payload"]["pending_collectives"] == 0
    finally:
        Utils.destroy_model_parallel()


def test_process_trace_records_worker_threads_and_pending_collectives(tmp_path):
    with trace_iteration(tmp_path, 14, hash_tensors=True):
        worker = threading.Thread(
            target=lambda: record_event(EventKind.PHASE, "worker.thread.event")
        )
        worker.start()
        worker.join()
        pending = begin_collective_trace(
            "test.pending_collective", "all_reduce", torch.tensor([1.0])
        )

    events = _read_events(_trace_path(tmp_path, 14))
    assert any(event["name"] == "worker.thread.event" for event in events)
    assert events[-1]["payload"]["pending_collectives"] == 1
    record_collective_result(pending, torch.tensor([1.0]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_checkpoint_autograd_records_paired_recompute(tmp_path):
    x = torch.tensor([1.0, 2.0], device="cuda", requires_grad=True)
    with trace_iteration(tmp_path, 6, hash_tensors=True):
        output = checkpoint(lambda value: value.square(), False, x)
        output.sum().backward()

    events = _read_events(_trace_path(tmp_path, 6))
    forward = [event for event in events if event["name"] == "checkpoint.forward"]
    recompute = [event for event in events if event["name"] == "checkpoint.recompute"]
    assert len(forward) == len(recompute) == 1
    assert recompute[0]["payload"]["matches_forward"] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_all_to_all_autograd_records_forward_and_backward(tmp_path):
    Utils.initialize_model_parallel(tensor_model_parallel_size=Utils.world_size)
    try:
        group = torch.distributed.group.WORLD
        world_size = group.size()
        rank = group.rank()
        values = torch.arange(world_size, dtype=torch.float32, device="cuda") + 10 * rank
        values.requires_grad_()

        with trace_iteration(tmp_path, 9, hash_tensors=True):
            output = all_to_all(group, values, trace_name="test.ep_all_to_all")
            output.sum().backward()

        expected = torch.arange(world_size, dtype=torch.float32, device="cuda") * 10 + rank
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(values.grad, torch.ones_like(values))
        events = _read_events(_trace_path(tmp_path, 9))
        collective = [event for event in events if event["kind"] == "collective"]
        assert [event["name"] for event in collective] == [
            "test.ep_all_to_all.forward.begin",
            "test.ep_all_to_all.forward.end",
            "test.ep_all_to_all.backward.begin",
            "test.ep_all_to_all.backward.end",
        ]
        assert all(event["payload"]["group_size"] == world_size for event in collective)
        assert all(event["payload"]["group_rank"] == rank for event in collective)
        assert collective[1]["payload"]["outputs"][0]["sha256"]
        assert collective[3]["payload"]["outputs"][0]["sha256"]

        checkpointed_values = values.detach().clone().requires_grad_()
        with trace_iteration(tmp_path, 10, hash_tensors=True):
            checkpointed_output = checkpoint(
                lambda tensor: all_to_all(
                    group, tensor, trace_name="test.checkpointed_ep_all_to_all"
                ),
                False,
                checkpointed_values,
            )
            checkpointed_output.sum().backward()

        checkpointed_events = _read_events(_trace_path(tmp_path, 10))
        checkpointed_collective = [
            event for event in checkpointed_events if event["kind"] == "collective"
        ]
        assert [event["name"] for event in checkpointed_collective] == [
            "test.checkpointed_ep_all_to_all.forward.begin",
            "test.checkpointed_ep_all_to_all.forward.end",
            "test.checkpointed_ep_all_to_all.recompute.begin",
            "test.checkpointed_ep_all_to_all.recompute.end",
            "test.checkpointed_ep_all_to_all.backward.begin",
            "test.checkpointed_ep_all_to_all.backward.end",
        ]
        assert all(
            (event["payload"].get("inputs") or event["payload"].get("outputs"))[0]["sha256"]
            for event in checkpointed_collective
        )
        torch.testing.assert_close(checkpointed_output, expected)
        torch.testing.assert_close(checkpointed_values.grad, torch.ones_like(checkpointed_values))

        async_values = values.detach().clone().requires_grad_()
        with trace_iteration(tmp_path, 11, hash_tensors=True):
            async_output = all_to_all(
                group, async_values, use_nccl_stream=True, trace_name="test.async_ep_all_to_all"
            )
            async_output.sum().backward()

        async_events = _read_events(_trace_path(tmp_path, 11))
        async_collective = [event for event in async_events if event["kind"] == "collective"]
        assert [event["name"] for event in async_collective] == [
            "test.async_ep_all_to_all.forward.begin",
            "test.async_ep_all_to_all.forward.end",
            "test.async_ep_all_to_all.backward.begin",
            "test.async_ep_all_to_all.backward.end",
        ]
        assert all(event["payload"]["async_op"] is True for event in async_collective)
        torch.testing.assert_close(async_output, expected)
        torch.testing.assert_close(async_values.grad, torch.ones_like(async_values))
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tensor_parallel_collective_trace_records_forward_and_backward(tmp_path):
    Utils.initialize_model_parallel(tensor_model_parallel_size=Utils.world_size)
    try:
        group = torch.distributed.group.WORLD
        world_size = group.size()
        rank = group.rank()

        with trace_iteration(tmp_path, 17, hash_tensors=True):
            reduce_values = torch.tensor(
                [rank + 1], dtype=torch.float32, device="cuda", requires_grad=True
            )
            reduced = reduce_from_tensor_model_parallel_region(reduce_values, group=group)
            reduced.sum().backward()

            copy_values = torch.tensor(
                [rank + 1], dtype=torch.float32, device="cuda", requires_grad=True
            )
            copied = copy_to_tensor_model_parallel_region(copy_values, group=group)
            copied.sum().backward()

            gather_last_values = (
                torch.tensor([rank, rank + world_size], dtype=torch.float32, device="cuda")
                .reshape(2, 1)
                .requires_grad_()
            )
            gathered_last = all_gather_last_dim_from_tensor_parallel_region(
                gather_last_values, group=group
            )
            gathered_last.sum().backward()

            reduce_scatter_last_values = (
                torch.arange(2 * world_size, dtype=torch.float32, device="cuda")
                .reshape(2, world_size)
                .requires_grad_()
            )
            reduced_scatter_last = reduce_scatter_last_dim_to_tensor_parallel_region(
                reduce_scatter_last_values, group=group
            )
            reduced_scatter_last.sum().backward()

            gather_first_values = torch.tensor(
                [[rank, rank + world_size]], dtype=torch.float32, device="cuda", requires_grad=True
            )
            gathered_first = gather_from_sequence_parallel_region(gather_first_values, group=group)
            gathered_first.sum().backward()

            reduce_scatter_first_values = (
                torch.arange(2 * world_size, dtype=torch.float32, device="cuda")
                .reshape(world_size, 2)
                .requires_grad_()
            )
            reduced_scatter_first = reduce_scatter_to_sequence_parallel_region(
                reduce_scatter_first_values, group=group
            )
            reduced_scatter_first.sum().backward()

        expected_sum = world_size * (world_size + 1) / 2
        torch.testing.assert_close(reduced, torch.tensor([expected_sum], device="cuda"))
        torch.testing.assert_close(copy_values.grad, torch.full_like(copy_values, world_size))
        expected_gather_last = torch.stack(
            (
                torch.arange(world_size, dtype=torch.float32, device="cuda"),
                torch.arange(world_size, dtype=torch.float32, device="cuda") + world_size,
            )
        )
        torch.testing.assert_close(gathered_last, expected_gather_last)
        torch.testing.assert_close(
            gather_last_values.grad, torch.full_like(gather_last_values, world_size)
        )
        torch.testing.assert_close(
            reduce_scatter_last_values.grad, torch.ones_like(reduce_scatter_last_values)
        )
        expected_gather_first = torch.stack(
            (
                torch.arange(world_size, dtype=torch.float32, device="cuda"),
                torch.arange(world_size, dtype=torch.float32, device="cuda") + world_size,
            ),
            dim=1,
        )
        torch.testing.assert_close(gathered_first, expected_gather_first)
        torch.testing.assert_close(
            gather_first_values.grad, torch.full_like(gather_first_values, world_size)
        )
        torch.testing.assert_close(
            reduce_scatter_first_values.grad, torch.ones_like(reduce_scatter_first_values)
        )

        events = _read_events(_trace_path(tmp_path, 17))
        collective = [event for event in events if event["kind"] == "collective"]
        expected_names = [
            "tensor_parallel.all_reduce.forward",
            "tensor_parallel.all_reduce.backward",
            "tensor_parallel.all_gather_last_dim.forward",
            "tensor_parallel.reduce_scatter_last_dim.backward",
            "tensor_parallel.reduce_scatter_last_dim.forward",
            "tensor_parallel.all_gather_last_dim.backward",
            "tensor_parallel.all_gather_first_dim.forward",
            "tensor_parallel.reduce_scatter_first_dim.backward",
            "tensor_parallel.reduce_scatter_first_dim.forward",
            "tensor_parallel.all_gather_first_dim.backward",
        ]
        assert [event["name"] for event in collective] == [
            f"{name}.{edge}" for name in expected_names for edge in ("begin", "end")
        ]
        assert all(event["payload"]["group_size"] == world_size for event in collective)
        assert all(event["payload"]["group_rank"] == rank for event in collective)
        assert all(
            (event["payload"].get("inputs") or event["payload"].get("outputs"))[0]["sha256"]
            for event in collective
        )
        assert events[-1]["payload"]["pending_collectives"] == 0

        checkpointed_values = torch.tensor(
            [rank], dtype=torch.float32, device="cuda", requires_grad=True
        )
        with trace_iteration(tmp_path, 18, hash_tensors=True):
            checkpointed_output = checkpoint(
                lambda tensor: all_gather_last_dim_from_tensor_parallel_region(tensor, group=group),
                False,
                checkpointed_values,
            )
            checkpointed_output.sum().backward()

        checkpointed_events = _read_events(_trace_path(tmp_path, 18))
        checkpointed_collective = [
            event for event in checkpointed_events if event["kind"] == "collective"
        ]
        assert [event["name"] for event in checkpointed_collective] == [
            "tensor_parallel.all_gather_last_dim.forward.begin",
            "tensor_parallel.all_gather_last_dim.forward.end",
            "tensor_parallel.all_gather_last_dim.recompute.begin",
            "tensor_parallel.all_gather_last_dim.recompute.end",
            "tensor_parallel.reduce_scatter_last_dim.backward.begin",
            "tensor_parallel.reduce_scatter_last_dim.backward.end",
        ]
        torch.testing.assert_close(
            checkpointed_output, torch.arange(world_size, dtype=torch.float32, device="cuda")
        )
        torch.testing.assert_close(
            checkpointed_values.grad, torch.full_like(checkpointed_values, world_size)
        )
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tensor_parallel_linear_collective_trace_records_existing_waits(tmp_path):
    Utils.initialize_model_parallel(tensor_model_parallel_size=Utils.world_size)
    try:
        group = torch.distributed.group.WORLD
        world_size = group.size()

        with trace_iteration(tmp_path, 20, hash_tensors=True):
            frozen_input = torch.ones(2, 3, dtype=torch.float32, device="cuda", requires_grad=True)
            frozen_weight = torch.ones(4, 3, dtype=torch.float32, device="cuda")
            frozen_output = linear_with_frozen_weight(
                frozen_input,
                frozen_weight,
                None,
                gradient_accumulation_fusion=False,
                allreduce_dgrad=True,
                sequence_parallel=False,
                tp_group=group,
            )
            frozen_output.sum().backward()

            allreduce_input = torch.ones(
                2, 3, dtype=torch.float32, device="cuda", requires_grad=True
            )
            allreduce_weight = torch.ones(
                4, 3, dtype=torch.float32, device="cuda", requires_grad=True
            )
            allreduce_output = linear_with_grad_accumulation_and_async_allreduce(
                allreduce_input,
                allreduce_weight,
                None,
                gradient_accumulation_fusion=False,
                allreduce_dgrad=True,
                sequence_parallel=False,
                tp_group=group,
            )
            allreduce_output.sum().backward()

            sequence_input = torch.ones(
                1, 3, dtype=torch.float32, device="cuda", requires_grad=True
            )
            sequence_weight = torch.ones(
                4, 3, dtype=torch.float32, device="cuda", requires_grad=True
            )
            sequence_output = linear_with_grad_accumulation_and_async_allreduce(
                sequence_input,
                sequence_weight,
                None,
                gradient_accumulation_fusion=False,
                allreduce_dgrad=False,
                sequence_parallel=True,
                tp_group=group,
            )
            sequence_output.sum().backward()

        expected_grad = torch.full_like(frozen_input, 4 * world_size)
        torch.testing.assert_close(frozen_input.grad, expected_grad)
        torch.testing.assert_close(allreduce_input.grad, expected_grad)
        torch.testing.assert_close(
            sequence_input.grad, torch.full_like(sequence_input, 4 * world_size)
        )

        events = _read_events(_trace_path(tmp_path, 20))
        collective = [event for event in events if event["kind"] == "collective"]
        expected_names = [
            "tensor_parallel.linear.frozen_dgrad_all_reduce.backward",
            "tensor_parallel.linear.dgrad_all_reduce.backward",
            "tensor_parallel.linear.sequence_parallel_all_gather.forward",
            "tensor_parallel.linear.sequence_parallel_all_gather.backward",
            "tensor_parallel.linear.dgrad_reduce_scatter.backward",
        ]
        assert [event["name"] for event in collective] == [
            f"{name}.{edge}" for name in expected_names for edge in ("begin", "end")
        ]
        assert [event["payload"]["async_op"] for event in collective[::2]] == [
            False,
            True,
            False,
            True,
            True,
        ]
        assert all(
            (event["payload"].get("inputs") or event["payload"].get("outputs"))[0]["sha256"]
            for event in collective
        )
        assert events[-1]["payload"]["pending_collectives"] == 0

        checkpointed_input = torch.ones(
            1, 3, dtype=torch.float32, device="cuda", requires_grad=True
        )
        checkpointed_weight = torch.ones(
            4, 3, dtype=torch.float32, device="cuda", requires_grad=True
        )
        with trace_iteration(tmp_path, 21, hash_tensors=True):
            checkpointed_output = checkpoint(
                lambda value: linear_with_grad_accumulation_and_async_allreduce(
                    value,
                    checkpointed_weight,
                    None,
                    gradient_accumulation_fusion=False,
                    allreduce_dgrad=False,
                    sequence_parallel=True,
                    tp_group=group,
                ),
                False,
                checkpointed_input,
            )
            checkpointed_output.sum().backward()

        checkpointed_events = _read_events(_trace_path(tmp_path, 21))
        checkpointed_collective = [
            event for event in checkpointed_events if event["kind"] == "collective"
        ]
        assert [event["name"] for event in checkpointed_collective] == [
            "tensor_parallel.linear.sequence_parallel_all_gather.forward.begin",
            "tensor_parallel.linear.sequence_parallel_all_gather.forward.end",
            "tensor_parallel.linear.sequence_parallel_all_gather.recompute.begin",
            "tensor_parallel.linear.sequence_parallel_all_gather.recompute.end",
            "tensor_parallel.linear.sequence_parallel_all_gather.backward.begin",
            "tensor_parallel.linear.sequence_parallel_all_gather.backward.end",
            "tensor_parallel.linear.dgrad_reduce_scatter.backward.begin",
            "tensor_parallel.linear.dgrad_reduce_scatter.backward.end",
        ]
        torch.testing.assert_close(
            checkpointed_input.grad, torch.full_like(checkpointed_input, 4 * world_size)
        )
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_pipeline_p2p_trace_records_after_existing_wait(tmp_path):
    Utils.initialize_model_parallel(pipeline_model_parallel_size=Utils.world_size)
    try:
        pp_group = get_pipeline_model_parallel_group()
        rank = pp_group.rank()
        world_size = pp_group.size()
        config = ModelParallelConfig(
            pipeline_dtype=torch.float32,
            pipeline_model_parallel_size=world_size,
            batch_p2p_comm=False,
        )
        communicator = P2PCommunicator(pp_group, config)

        send = torch.tensor([rank], dtype=torch.float32, device="cuda")
        with trace_iteration(tmp_path, 12, hash_tensors=True):
            recv, _, requests = communicator._communicate(
                tensor_send_next=send,
                tensor_send_prev=None,
                recv_prev=True,
                recv_next=False,
                tensor_shape=(1,),
                wait_on_reqs=False,
            )
            before_wait = _read_events(_trace_path(tmp_path, 12))
            before_names = [event["name"] for event in before_wait if event["kind"] == "collective"]
            assert set(before_names) == {
                "pipeline.forward.send_next.begin",
                "pipeline.forward.recv_prev.begin",
            }
            for request in requests.values():
                request.wait()

        expected_prev = (rank - 1) % world_size
        torch.testing.assert_close(
            recv, torch.tensor([expected_prev], dtype=torch.float32, device="cuda")
        )
        events = _read_events(_trace_path(tmp_path, 12))
        collective = [event for event in events if event["kind"] == "collective"]
        assert {event["name"] for event in collective} == {
            "pipeline.forward.send_next.begin",
            "pipeline.forward.send_next.end",
            "pipeline.forward.recv_prev.begin",
            "pipeline.forward.recv_prev.end",
        }
        recv_end = next(
            event for event in collective if event["name"] == "pipeline.forward.recv_prev.end"
        )
        assert recv_end["payload"]["outputs"][0]["sha256"]

        config.batch_p2p_comm = True
        send = torch.tensor([rank + 1], dtype=torch.float32, device="cuda")
        with trace_iteration(tmp_path, 13, hash_tensors=True):
            recv, _, requests = communicator._communicate(
                tensor_send_next=send,
                tensor_send_prev=None,
                recv_prev=True,
                recv_next=False,
                tensor_shape=(1,),
                wait_on_reqs=True,
            )
        assert requests is None
        torch.testing.assert_close(
            recv, torch.tensor([expected_prev + 1], dtype=torch.float32, device="cuda")
        )
        events = _read_events(_trace_path(tmp_path, 13))
        collective = [event for event in events if event["kind"] == "collective"]
        assert len(collective) == 4
        assert all(event["payload"]["batched"] is True for event in collective)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("use_distributed_optimizer", [False, True])
@pytest.mark.parametrize("overlap_grad_reduce", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_data_parallel_grad_collective_trace(
    tmp_path, use_distributed_optimizer, overlap_grad_reduce
):
    Utils.initialize_model_parallel()
    try:
        model, buffer, _ = get_model_and_buffers(
            input_dim=16,
            output_dim=16,
            num_layers=2,
            bias=True,
            shared_embedding=False,
            bucket_size=None,
            use_distributed_optimizer=use_distributed_optimizer,
            overlap_grad_reduce=overlap_grad_reduce,
            average_in_collective=False,
        )
        buffer.grad_data.fill_(torch.distributed.get_rank() + 1)
        with trace_iteration(tmp_path, 15, hash_tensors=True):
            model.finish_grad_sync()

        events = _read_events(_trace_path(tmp_path, 15))
        collective = [
            event
            for event in events
            if event["kind"] == "collective"
            and event["name"].startswith("data_parallel.grad_reduce")
        ]
        assert len(collective) == 2
        expected_operation = "reduce_scatter_tensor" if use_distributed_optimizer else "all_reduce"
        assert all(event["payload"]["operation"] == expected_operation for event in collective)
        if use_distributed_optimizer:
            assert collective[0]["payload"]["fp32_accumulation"] is False
        assert collective[0]["payload"]["inputs"][0]["sha256"]
        assert collective[1]["payload"]["outputs"][0]["sha256"]
        assert events[-1]["payload"]["pending_collectives"] == 0
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("force_sync", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_distributed_optimizer_param_gather_trace(tmp_path, force_sync):
    Utils.initialize_model_parallel()
    try:
        _, _, bucket_groups = get_model_and_buffers(
            input_dim=16,
            output_dim=16,
            num_layers=2,
            bias=True,
            shared_embedding=False,
            bucket_size=None,
            use_distributed_optimizer=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=False,
        )
        with trace_iteration(tmp_path, 16, hash_tensors=True):
            for bucket_group in bucket_groups:
                bucket_group.start_param_sync(force_sync=force_sync)
            if not force_sync:
                for bucket_group in bucket_groups:
                    bucket_group.finish_param_sync(skip_next_bucket_dispatch=True)

        events = _read_events(_trace_path(tmp_path, 16))
        collective = [
            event
            for event in events
            if event["kind"] == "collective"
            and event["name"].startswith("data_parallel.param_gather")
        ]
        assert len(collective) == 2
        assert all(
            event["payload"]["operation"] == "all_gather_into_tensor" for event in collective
        )
        assert collective[0]["payload"]["inputs"][0]["sha256"]
        assert collective[1]["payload"]["outputs"][0]["sha256"]
        assert events[-1]["payload"]["pending_collectives"] == 0
    finally:
        Utils.destroy_model_parallel()
