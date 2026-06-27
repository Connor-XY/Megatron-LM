# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json

from tools.determinism.compare_training_logs import compare_training_logs, main


def _iteration_line(iteration, *, loss="1.250000E+01", grad_norm="9.002", elapsed="100.0"):
    return (
        f"[default0]: [2026-06-27 12:00:00.000000] iteration {iteration:8d}/{2:8d} | "
        f"consumed samples: {iteration * 32:12d} | elapsed time per iteration (ms): {elapsed} | "
        "learning rate: 1.000000E-04 | global batch size:    32 | "
        f"lm loss: {loss} | seq_load_balancing_loss: 1.300000E+00 | "
        f"loss scale: 1.0 | grad norm: {grad_norm} | "
        "number of skipped iterations:   0 | number of nan iterations:   0 |\n"
    )


def test_compares_every_serialized_nonvolatile_metric(tmp_path):
    left = tmp_path / "left.log"
    right = tmp_path / "right.log"
    left.write_text(_iteration_line(1, elapsed="101.0") + _iteration_line(2, loss="3.500000E+00"))
    right.write_text(
        "\x1b[32mstartup\x1b[0m\n"
        + _iteration_line(1, elapsed="999.0")
        + _iteration_line(2, loss="3.500000E+00")
    )

    report = compare_training_logs(left, right, expected_iterations=2)

    assert report["equal"]
    assert report["iterations_compared"] == 2
    assert report["metric_values_compared"] == 18
    assert report["left"]["iteration_count"] == 2
    assert report["left"]["first_iteration"] == 1
    assert report["left"]["last_iteration"] == 2


def test_reports_exact_metric_value_divergence(tmp_path):
    left = tmp_path / "left.log"
    right = tmp_path / "right.log"
    left.write_text(_iteration_line(1) + _iteration_line(2, grad_norm="6.923"))
    right.write_text(_iteration_line(1) + _iteration_line(2, grad_norm="6.924"))

    report = compare_training_logs(left, right)

    assert not report["equal"]
    assert report["divergences"] == [
        {
            "iteration": 2,
            "metric": "grad norm",
            "reason": "metric_value",
            "left": "6.923",
            "right": "6.924",
        }
    ]


def test_reports_missing_iterations_and_expected_count(tmp_path):
    left = tmp_path / "left.log"
    right = tmp_path / "right.log"
    left.write_text(_iteration_line(1) + _iteration_line(2))
    right.write_text(_iteration_line(1))

    report = compare_training_logs(left, right, expected_iterations=2)

    assert not report["equal"]
    assert report["divergences"][0] == {
        "reason": "iteration_count",
        "side": "right",
        "actual": 1,
        "expected": 2,
    }
    assert report["divergences"][1] == {"iteration": 2, "reason": "iteration_missing_right"}


def test_required_metric_must_exist_even_when_both_logs_omit_it(tmp_path):
    left = tmp_path / "left.log"
    right = tmp_path / "right.log"
    line = _iteration_line(1).replace(" grad norm: 9.002 |", "")
    left.write_text(line)
    right.write_text(line)

    report = compare_training_logs(left, right)

    assert report["divergences"] == [
        {"iteration": 1, "metric": "grad norm", "reason": "required_metric_missing_both"}
    ]


def test_cli_additional_required_metric_keeps_defaults(tmp_path, capsys):
    left = tmp_path / "left.log"
    right = tmp_path / "right.log"
    line = _iteration_line(1).replace(" grad norm: 9.002 |", "")
    left.write_text(line)
    right.write_text(line)

    assert main([str(left), str(right), "--require-metric", "learning rate"]) == 1
    assert "metric=grad norm" in capsys.readouterr().out


def test_identical_duplicate_rank_zero_lines_are_accepted(tmp_path):
    left = tmp_path / "left.log"
    right = tmp_path / "right.log"
    line = _iteration_line(1)
    left.write_text(line + line)
    right.write_text(line)

    report = compare_training_logs(left, right)

    assert report["equal"]
    assert report["left"]["duplicate_records"] == 1


def test_ignores_interleaved_rank_output_after_terminal_pipe(tmp_path):
    left = tmp_path / "left.log"
    right = tmp_path / "right.log"
    left.write_text(
        _iteration_line(1).rstrip()
        + "[Rank 64] Step Time : 257.68s | mem-allocated-gigabytes: 154.02 |\n"
    )
    right.write_text(
        _iteration_line(1).rstrip()
        + "[Rank 32] Step Time : 54.35s | mem-allocated-gigabytes: 139.08 |\n"
    )

    report = compare_training_logs(left, right)

    assert report["equal"]
    assert report["metric_values_compared"] == 9


def test_conflicting_duplicates_are_invalid_input(tmp_path, capsys):
    left = tmp_path / "left.log"
    right = tmp_path / "right.log"
    left.write_text(_iteration_line(1) + _iteration_line(1, loss="1.200000E+01"))
    right.write_text(_iteration_line(1))

    assert main([str(left), str(right)]) == 2
    assert "Conflicting records for iteration 1" in capsys.readouterr().err


def test_cli_exit_codes_and_json_report(tmp_path, capsys):
    left = tmp_path / "left.log"
    right = tmp_path / "right.log"
    left.write_text(_iteration_line(1))
    right.write_text(_iteration_line(1, loss="1.200000E+01"))

    assert main([str(left), str(right), "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert not report["equal"]

    assert main([str(left), str(left), "--expected-iterations", "1"]) == 0
    assert capsys.readouterr().out.startswith("MATCH:")


def test_logs_without_training_iterations_are_invalid(tmp_path, capsys):
    left = tmp_path / "left.log"
    right = tmp_path / "right.log"
    left.write_text("startup only\n")
    right.write_text(_iteration_line(1))

    assert main([str(left), str(right)]) == 2
    assert "No training iteration records found" in capsys.readouterr().err
