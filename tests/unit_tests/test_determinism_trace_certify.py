# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json

from tools.determinism.certify_traces import certify_trace_path, main


def _write_trace(
    root,
    *,
    rank=0,
    iteration=1,
    suffix="",
    matches_forward=True,
    pending=0,
    fp32_accumulation=True,
    hierarchical_fp32_accumulation=True,
    group_size=2,
    device_hash=False,
    event_names=(),
):
    path = root / f"iter_{iteration:07d}" / f"rank_{rank:05d}{suffix}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    events = [
        {
            "schema_version": 1,
            "sequence": 0,
            "iteration": iteration,
            "rank": rank,
            "kind": "runtime",
            "name": "runtime",
            "payload": {"deterministic_algorithms": True},
        },
        {
            "schema_version": 1,
            "sequence": 1,
            "iteration": iteration,
            "rank": rank,
            "kind": "phase",
            "name": "iteration.begin",
            "payload": {},
        },
        {
            "schema_version": 1,
            "sequence": 2,
            "iteration": iteration,
            "rank": rank,
            "kind": "recompute",
            "name": "checkpoint.recompute",
            "payload": {"checkpoint_id": 0, "matches_forward": matches_forward},
        },
        {
            "schema_version": 1,
            "sequence": 3,
            "iteration": iteration,
            "rank": rank,
            "kind": "collective",
            "name": "data_parallel.grad_reduce.bucket_0.begin",
            "payload": {
                "fp32_accumulation": fp32_accumulation,
                "hierarchical_fp32_accumulation": hierarchical_fp32_accumulation,
                "group_size": group_size,
            },
        },
        {
            "schema_version": 1,
            "sequence": 4,
            "iteration": iteration,
            "rank": rank,
            "kind": "collective",
            "name": "data_parallel.grad_reduce.bucket_0.end",
            "payload": {
                "outputs": [{"device_digest": "abc"} if device_hash else {"sha256": "abc"}]
            },
        },
        {
            "schema_version": 1,
            "sequence": 5,
            "iteration": iteration,
            "rank": rank,
            "kind": "phase",
            "name": "iteration.end",
            "payload": {"pending_collectives": pending},
        },
    ]
    for event_name in event_names:
        events.insert(
            -1,
            {
                "schema_version": 1,
                "iteration": iteration,
                "rank": rank,
                "kind": "runtime",
                "name": event_name,
                "payload": {},
            },
        )
    for sequence, event in enumerate(events):
        event["sequence"] = sequence
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
    return path


def _write_summary_trace(root, *, rank=0, iteration=1, grad_norm=1.0):
    path = root / f"iter_{iteration:07d}" / f"rank_{rank:05d}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    events = [
        {
            "schema_version": 1,
            "sequence": 0,
            "iteration": iteration,
            "rank": rank,
            "kind": "runtime",
            "name": "runtime",
            "payload": {"deterministic_algorithms": True, "trace_detail": "summary"},
        },
        {
            "schema_version": 1,
            "sequence": 1,
            "iteration": iteration,
            "rank": rank,
            "kind": "phase",
            "name": "iteration.begin",
            "payload": {},
        },
        {
            "schema_version": 1,
            "sequence": 2,
            "iteration": iteration,
            "rank": rank,
            "kind": "phase",
            "name": "forward_backward.end",
            "payload": {"losses_reduced": [{"lm loss": 1.25}]},
        },
        {
            "schema_version": 1,
            "sequence": 3,
            "iteration": iteration,
            "rank": rank,
            "kind": "optimizer",
            "name": "optimizer.end",
            "payload": {"grad_norm": grad_norm},
        },
        {
            "schema_version": 1,
            "sequence": 4,
            "iteration": iteration,
            "rank": rank,
            "kind": "phase",
            "name": "iteration.end",
            "payload": {"pending_collectives": 0},
        },
    ]
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
    return path


def test_certifies_complete_bit_exact_trace(tmp_path):
    _write_trace(tmp_path)

    report = certify_trace_path(
        tmp_path,
        expected_ranks=1,
        expected_iterations=1,
        required_collective_prefixes=("data_parallel.",),
        require_dp_fp32_accumulation=True,
        require_dp_hierarchical_fp32_accumulation=True,
    )

    assert report["equal"]
    assert report["trace_files"] == 1
    assert report["recomputes"] == 1
    assert report["pending_collectives"] == 0


def test_certifies_sampled_rank_set_with_device_digests(tmp_path):
    _write_trace(tmp_path, rank=0, device_hash=True)
    _write_trace(tmp_path, rank=3, device_hash=True)

    report = certify_trace_path(tmp_path, expected_rank_spec="0,3", expected_iterations=1)

    assert report["equal"]
    assert report["ranks"] == [0, 3]
    assert report["tensor_digest_counts"] == {"sha256": 0, "device_digest": 2}
    assert report["diagnostic_only"]


def test_accepts_complete_summary_trace_as_diagnostic_only(tmp_path):
    _write_summary_trace(tmp_path, rank=0)
    _write_summary_trace(tmp_path, rank=1)

    report = certify_trace_path(tmp_path, expected_ranks=2, expected_iterations=1)

    assert report["equal"]
    assert report["trace_detail_counts"] == {"summary": 2}
    assert report["forward_backward_ends"] == 2
    assert report["optimizer_ends"] == 2
    assert report["recomputes"] == 0
    assert report["collectives"] == 0
    assert report["diagnostic_only"]


def test_rejects_incomplete_summary_trace(tmp_path):
    path = _write_summary_trace(tmp_path)
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    events = [event for event in events if event["name"] != "optimizer.end"]
    for sequence, event in enumerate(events):
        event["sequence"] = sequence
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")

    report = certify_trace_path(tmp_path)

    assert not report["equal"]
    assert "found 0 optimizer.end events for 1 files" in report["failures"]


def test_rejects_truncated_trace(tmp_path):
    path = _write_trace(tmp_path)
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    events.insert(
        -1,
        {
            "schema_version": 1,
            "iteration": 1,
            "rank": 0,
            "kind": "runtime",
            "name": "trace.truncated",
            "payload": {"max_events": 5},
        },
    )
    for sequence, event in enumerate(events):
        event["sequence"] = sequence
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")

    report = certify_trace_path(tmp_path)

    assert not report["equal"]
    assert report["trace_truncations"] == 1


def test_reports_recompute_pending_and_reduction_failures(tmp_path):
    _write_trace(
        tmp_path,
        matches_forward=False,
        pending=1,
        fp32_accumulation=False,
        hierarchical_fp32_accumulation=False,
        group_size=4,
    )

    report = certify_trace_path(
        tmp_path, require_dp_fp32_accumulation=True, require_dp_hierarchical_fp32_accumulation=True
    )

    assert not report["equal"]
    assert report["recompute_mismatches"] == 1
    assert report["pending_collectives"] == 1
    assert report["dp_grad_reductions_without_fp32_accumulation"] == 1
    assert report["multi_rank_dp_grad_reductions_without_hierarchical_fp32_accumulation"] == 1


def test_accepts_two_rank_fp32_reduction_without_hierarchy(tmp_path):
    _write_trace(tmp_path, hierarchical_fp32_accumulation=False, group_size=2)

    report = certify_trace_path(
        tmp_path, require_dp_fp32_accumulation=True, require_dp_hierarchical_fp32_accumulation=True
    )

    assert report["equal"]
    assert report["multi_rank_dp_grad_reductions"] == 1
    assert report["multi_rank_dp_grad_reductions_without_hierarchical_fp32_accumulation"] == 0


def test_requires_general_event_prefix(tmp_path):
    _write_trace(tmp_path, event_names=("te.attention.backend.selected",))

    report = certify_trace_path(
        tmp_path, required_event_prefixes=("te.attention.backend.selected",)
    )

    assert report["equal"]
    assert report["event_prefix_counts"] == {"te.attention.backend.selected": 1}


def test_rejects_missing_event_prefix_and_unavailable_te_backend(tmp_path):
    _write_trace(tmp_path, event_names=("te.attention.backend.unavailable",))

    report = certify_trace_path(
        tmp_path, required_event_prefixes=("te.attention.backend.selected",)
    )

    assert not report["equal"]
    assert report["event_prefix_counts"] == {"te.attention.backend.selected": 0}
    assert report["te_attention_backend_unavailable"] == 1


def test_cli_compares_two_certified_trace_trees(tmp_path, capsys):
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_trace(left)
    _write_trace(right)

    assert main([str(left), str(right), "--expected-ranks", "1", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["equal"]
    assert report["comparison"]["events_compared"] == 6


def test_cli_does_not_call_one_trace_tree_a_determinism_certificate(tmp_path, capsys):
    _write_trace(tmp_path)

    assert main([str(tmp_path), "--expected-ranks", "1"]) == 0
    assert capsys.readouterr().out.startswith("TRACE INVARIANTS PASSED:")


def test_cli_calls_sampled_device_digest_comparison_a_diagnostic_match(tmp_path, capsys):
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_trace(left, rank=3, device_hash=True)
    _write_trace(right, rank=3, device_hash=True)

    assert (
        main([str(left), str(right), "--expected-rank-spec", "3", "--expected-iterations", "1"])
        == 0
    )
    assert capsys.readouterr().out.startswith("DIAGNOSTIC MATCH:")


def test_rejects_duplicate_pair_that_hides_missing_rank_iteration_coverage(tmp_path):
    _write_trace(tmp_path, rank=0, iteration=1)
    _write_trace(tmp_path, rank=1, iteration=1)
    _write_trace(tmp_path, rank=0, iteration=2)
    _write_trace(tmp_path, rank=0, iteration=2, suffix="_duplicate")

    report = certify_trace_path(tmp_path, expected_ranks=2, expected_iterations=2)

    assert not report["equal"]
    assert "missing rank/iteration pairs: [(1, 2)]" in report["failures"]


def test_rejects_unsupported_schema_for_single_trace_tree(tmp_path, capsys):
    path = _write_trace(tmp_path)
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    events[0]["schema_version"] = 99
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")

    assert main([str(tmp_path)]) == 2
    assert "Unsupported schema version" in capsys.readouterr().err


def test_rejects_trace_errors_and_unbalanced_collectives(tmp_path):
    path = _write_trace(tmp_path)
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    events = [
        event
        for event in events
        if not (event["kind"] == "collective" and event["name"].endswith(".end"))
    ]
    events.insert(
        -1,
        {
            "schema_version": 1,
            "sequence": 0,
            "iteration": 1,
            "rank": 0,
            "kind": "phase",
            "name": "iteration.error",
            "payload": {"exception_type": "RuntimeError"},
        },
    )
    for sequence, event in enumerate(events):
        event["sequence"] = sequence
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")

    report = certify_trace_path(tmp_path)

    assert not report["equal"]
    assert report["iteration_errors"] == 1
    assert report["collective_begins_without_end"] == 1


def test_rejects_mixed_file_identity_and_non_contiguous_sequence(tmp_path):
    path = _write_trace(tmp_path)
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    events[-1]["rank"] = 1
    events[-1]["sequence"] = 99
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")

    report = certify_trace_path(tmp_path)

    assert not report["equal"]
    assert report["file_identity_failures"] == 1
    assert report["sequence_failures"] == 1


def test_rejects_duplicate_runtime_and_missing_iteration_begin(tmp_path):
    path = _write_trace(tmp_path)
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    events = [event for event in events if event["name"] != "iteration.begin"]
    events.insert(1, dict(events[0]))
    for sequence, event in enumerate(events):
        event["sequence"] = sequence
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")

    report = certify_trace_path(tmp_path)

    assert not report["equal"]
    assert report["runtime_count_failures"] == 1
    assert report["iteration_begin_count_failures"] == 1


def test_rejects_collective_end_before_begin(tmp_path):
    path = _write_trace(tmp_path)
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    begin_index = next(i for i, event in enumerate(events) if event["name"].endswith(".begin"))
    end_index = next(i for i, event in enumerate(events) if event["name"].endswith(".end"))
    events[begin_index], events[end_index] = events[end_index], events[begin_index]
    for sequence, event in enumerate(events):
        event["sequence"] = sequence
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")

    report = certify_trace_path(tmp_path)

    assert not report["equal"]
    assert report["collective_begins_without_end"] == 1
    assert report["collective_ends_without_begin"] == 1
