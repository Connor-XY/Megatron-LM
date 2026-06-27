# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json

from tools.determinism.certify_traces import certify_trace_path, main


def _write_trace(
    root, *, rank=0, iteration=1, suffix="", matches_forward=True, pending=0, fp32_accumulation=True
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
            "kind": "recompute",
            "name": "checkpoint.recompute",
            "payload": {"checkpoint_id": 0, "matches_forward": matches_forward},
        },
        {
            "schema_version": 1,
            "sequence": 2,
            "iteration": iteration,
            "rank": rank,
            "kind": "collective",
            "name": "data_parallel.grad_reduce.bucket_0.begin",
            "payload": {"fp32_accumulation": fp32_accumulation},
        },
        {
            "schema_version": 1,
            "sequence": 3,
            "iteration": iteration,
            "rank": rank,
            "kind": "collective",
            "name": "data_parallel.grad_reduce.bucket_0.end",
            "payload": {"outputs": [{"sha256": "abc"}]},
        },
        {
            "schema_version": 1,
            "sequence": 4,
            "iteration": iteration,
            "rank": rank,
            "kind": "phase",
            "name": "iteration.end",
            "payload": {"pending_collectives": pending},
        },
    ]
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")


def test_certifies_complete_bit_exact_trace(tmp_path):
    _write_trace(tmp_path)

    report = certify_trace_path(
        tmp_path,
        expected_ranks=1,
        expected_iterations=1,
        required_collective_prefixes=("data_parallel.",),
        require_dp_fp32_accumulation=True,
    )

    assert report["equal"]
    assert report["trace_files"] == 1
    assert report["recomputes"] == 1
    assert report["pending_collectives"] == 0


def test_reports_recompute_pending_and_reduction_failures(tmp_path):
    _write_trace(tmp_path, matches_forward=False, pending=1, fp32_accumulation=False)

    report = certify_trace_path(tmp_path, require_dp_fp32_accumulation=True)

    assert not report["equal"]
    assert report["recompute_mismatches"] == 1
    assert report["pending_collectives"] == 1
    assert report["dp_grad_reductions_without_fp32_accumulation"] == 1


def test_cli_compares_two_certified_trace_trees(tmp_path, capsys):
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_trace(left)
    _write_trace(right)

    assert main([str(left), str(right), "--expected-ranks", "1", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["equal"]
    assert report["comparison"]["events_compared"] == 5


def test_cli_does_not_call_one_trace_tree_a_determinism_certificate(tmp_path, capsys):
    _write_trace(tmp_path)

    assert main([str(tmp_path), "--expected-ranks", "1"]) == 0
    assert capsys.readouterr().out.startswith("TRACE INVARIANTS PASSED:")


def test_rejects_duplicate_pair_that_hides_missing_rank_iteration_coverage(tmp_path):
    _write_trace(tmp_path, rank=0, iteration=1)
    _write_trace(tmp_path, rank=1, iteration=1)
    _write_trace(tmp_path, rank=0, iteration=2)
    _write_trace(tmp_path, rank=0, iteration=2, suffix="_duplicate")

    report = certify_trace_path(tmp_path, expected_ranks=2, expected_iterations=2)

    assert not report["equal"]
    assert "missing rank/iteration pairs: [(1, 2)]" in report["failures"]
