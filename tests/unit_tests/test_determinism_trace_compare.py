# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json

from megatron.core.determinism_trace import TRACE_SCHEMA_VERSION
from tools.determinism.compare_traces import SUPPORTED_SCHEMA_VERSION, compare_trace_paths, main


def test_comparator_schema_matches_trace_writer():
    assert SUPPORTED_SCHEMA_VERSION == TRACE_SCHEMA_VERSION


def _event(sequence, name, payload=None):
    return {
        "schema_version": 1,
        "sequence": sequence,
        "rank": 0,
        "iteration": 1,
        "kind": "phase",
        "name": name,
        "payload": payload or {},
    }


def _write_trace(root, events, relative="iter_0000001/rank_00000.jsonl"):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(event) + "\n" for event in events))


def test_semantic_comparison_ignores_cross_kind_arrival_order(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_trace(left, [_event(0, "forward"), _event(1, "optimizer")])
    _write_trace(right, [_event(0, "optimizer"), _event(1, "forward")])

    report = compare_trace_paths(left, right)

    assert report["equal"]
    assert report["events_compared"] == 2


def test_recompute_events_align_by_checkpoint_identity(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    first = _event(0, "checkpoint.recompute", {"checkpoint_id": 0, "hash": "a"})
    first["kind"] = "recompute"
    second = _event(1, "checkpoint.recompute", {"checkpoint_id": 1, "hash": "b"})
    second["kind"] = "recompute"
    _write_trace(left, [first, second])
    _write_trace(right, [second | {"sequence": 0}, first | {"sequence": 1}])

    assert compare_trace_paths(left, right)["equal"]


def test_op_events_align_by_explicit_identity(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    first = _event(0, "aten.add.Tensor", {"checkpoint_id": "aten:add:0", "digest": "a"})
    first["kind"] = "op"
    second = _event(1, "aten.add.Tensor", {"checkpoint_id": "aten:add:1", "digest": "b"})
    second["kind"] = "op"
    _write_trace(left, [first, second])
    _write_trace(right, [second | {"sequence": 0}, first | {"sequence": 1}])

    assert compare_trace_paths(left, right)["equal"]


def test_reports_first_semantic_event_value_difference(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_trace(left, [_event(0, "optimizer.end", {"grad_norm": 1.0})])
    _write_trace(right, [_event(0, "optimizer.end", {"grad_norm": 2.0})])

    report = compare_trace_paths(left, right)

    assert not report["equal"]
    assert report["divergence_count"] == 1
    divergence = report["divergences"][0]
    assert divergence["reason"] == "event_values"
    assert "name=optimizer.end" in divergence["event"]
    assert report["first_divergence"] == divergence
    assert report["divergent_ranks"] == [0]
    assert report["divergent_iterations"] == [1]
    assert report["recommended_followup"] == {
        "rank_spec": "0",
        "start_iteration": 1,
        "end_iteration": 1,
        "detail": "semantic_device_digests",
    }


def test_reports_differences_in_local_trace_sequence_order(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    early_left = _event(0, "z-early", {"hash": "left-early"})
    early_right = _event(0, "z-early", {"hash": "right-early"})
    late_left = _event(1, "a-late", {"grad_norm": 1.0})
    late_right = _event(1, "a-late", {"grad_norm": 2.0})
    early_left["kind"] = early_right["kind"] = "tensor"
    late_left["kind"] = late_right["kind"] = "optimizer"
    _write_trace(left, [early_left, late_left])
    _write_trace(right, [early_right, late_right])

    report = compare_trace_paths(left, right)

    assert [divergence["left_event"]["name"] for divergence in report["divergences"]] == [
        "z-early",
        "a-late",
    ]
    assert [
        (divergence["left_sequence"], divergence["right_sequence"])
        for divergence in report["divergences"]
    ] == [(0, 0), (1, 1)]


def test_reports_missing_rank_trace(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_trace(left, [_event(0, "forward")])
    _write_trace(left, [_event(0, "forward")], "iter_0000001/rank_00001.jsonl")
    _write_trace(right, [_event(0, "forward")])

    report = compare_trace_paths(left, right)

    assert report["divergences"] == [
        {"file": "iter_0000001/rank_00001.jsonl", "reason": "file_missing_right"}
    ]


def test_cli_exit_codes_and_invalid_schema(tmp_path, capsys):
    left = tmp_path / "left"
    right = tmp_path / "right"
    invalid = _event(0, "forward")
    invalid["schema_version"] = 99
    _write_trace(left, [invalid])
    _write_trace(right, [_event(0, "forward")])

    assert main([str(left), str(right)]) == 2
    assert "Unsupported schema version" in capsys.readouterr().err

    _write_trace(left, [_event(0, "forward")])
    assert main([str(left), str(right), "--json"]) == 0
    assert '"equal": true' in capsys.readouterr().out
