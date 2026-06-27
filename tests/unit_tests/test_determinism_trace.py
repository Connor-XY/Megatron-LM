# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
import os
import threading

import pytest
import torch

from megatron.core.determinism_trace import (
    EventKind,
    active_trace,
    begin_collective_trace,
    begin_recompute_trace,
    record_collective_result,
    record_event,
    record_recompute_phase,
    record_tensor,
    trace_iteration,
)
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.parallel_state import get_pipeline_model_parallel_group
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.tensor_parallel.mappings import all_to_all
from megatron.core.tensor_parallel.random import checkpoint
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


def test_nonfinite_payloads_remain_valid_json(tmp_path):
    with trace_iteration(tmp_path, 5):
        record_event(EventKind.OPTIMIZER, "optimizer.end", {"grad_norm": float("nan")})

    events = _read_events(_trace_path(tmp_path, 5))
    optimizer = next(event for event in events if event["name"] == "optimizer.end")
    assert optimizer["payload"]["grad_norm"] == "nan"


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
