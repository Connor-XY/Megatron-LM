# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compare every serialized training metric from two Megatron console logs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ITERATION_PATTERN = re.compile(r"\biteration\s+(\d+)\s*/\s*(\d+)\s*\|")
ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
DEFAULT_REQUIRED_METRICS = ("lm loss", "grad norm")
DEFAULT_IGNORED_METRICS = frozenset(
    {
        "elapsed time per iteration (ms)",
        "throughput per GPU (TFLOP/s/GPU)",
        "energy per GPU (J/iter/GPU)",
        "power per GPU (W/GPU)",
    }
)


def _normalize_name(value: str) -> str:
    return " ".join(value.split())


def _parse_metrics(line: str, start: int, ignored_metrics: frozenset[str]) -> dict[str, str]:
    metrics = {}
    # Megatron terminates every metric with ``|``. Aggregated Slurm output can
    # append another rank's text immediately after that terminal delimiter. A
    # metric field starts with whitespace after its delimiter; an appended rank
    # prefix does not. That prefix therefore terminates this record even when
    # the appended output contains colons and its own pipe-delimited metrics.
    for field in line[start:].split("|")[:-1]:
        if metrics and not field[:1].isspace():
            break
        if ":" not in field:
            if metrics:
                break
            continue
        name, value = field.split(":", 1)
        name = _normalize_name(name)
        value = value.strip()
        if not name or not value or name in ignored_metrics:
            continue
        metrics[name] = value
    return metrics


def parse_training_log(
    path: str | Path, *, ignored_metrics: Iterable[str] = DEFAULT_IGNORED_METRICS
) -> dict[str, Any]:
    """Parse iteration records from a Megatron console log.

    Repeated identical records are accepted because aggregated distributed logs
    can tee the same rank-zero line more than once. Conflicting repetitions are
    rejected because there is no safe way to choose one as authoritative.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Training log does not exist: {path}")
    ignored = frozenset(_normalize_name(name) for name in ignored_metrics)
    iterations = {}
    duplicate_records = 0
    with path.open(encoding="utf-8", errors="replace") as log_file:
        for line_number, raw_line in enumerate(log_file, start=1):
            line = ANSI_ESCAPE_PATTERN.sub("", raw_line)
            match = ITERATION_PATTERN.search(line)
            if match is None:
                continue
            iteration = int(match.group(1))
            record = {
                "train_iters": int(match.group(2)),
                "metrics": _parse_metrics(line, match.end(), ignored),
            }
            if iteration in iterations:
                previous = iterations[iteration]
                if (
                    previous["train_iters"] != record["train_iters"]
                    or previous["metrics"] != record["metrics"]
                ):
                    raise ValueError(
                        f"Conflicting records for iteration {iteration} in {path}: "
                        f"lines {previous['line_number']} and {line_number}"
                    )
                duplicate_records += 1
                continue
            iterations[iteration] = record | {"line_number": line_number}
    if not iterations:
        raise ValueError(f"No training iteration records found in: {path}")
    return {
        "path": str(path.resolve()),
        "iterations": iterations,
        "duplicate_records": duplicate_records,
    }


def compare_training_logs(
    left_path: str | Path,
    right_path: str | Path,
    *,
    expected_iterations: int | None = None,
    required_metrics: Iterable[str] = DEFAULT_REQUIRED_METRICS,
    ignored_metrics: Iterable[str] = DEFAULT_IGNORED_METRICS,
    max_details: int = 10,
) -> dict[str, Any]:
    """Return a JSON-safe exact comparison of two Megatron training logs."""
    if expected_iterations is not None and expected_iterations < 1:
        raise ValueError("expected_iterations must be at least 1")
    if max_details < 1:
        raise ValueError("max_details must be at least 1")

    required = tuple(dict.fromkeys(_normalize_name(name) for name in required_metrics))
    left = parse_training_log(left_path, ignored_metrics=ignored_metrics)
    right = parse_training_log(right_path, ignored_metrics=ignored_metrics)
    left_iterations = left.pop("iterations")
    right_iterations = right.pop("iterations")
    for summary, iterations in ((left, left_iterations), (right, right_iterations)):
        summary["iteration_count"] = len(iterations)
        summary["first_iteration"] = min(iterations)
        summary["last_iteration"] = max(iterations)
    report = {
        "equal": True,
        "left": left,
        "right": right,
        "iterations_compared": 0,
        "metric_values_compared": 0,
        "divergence_count": 0,
        "divergences": [],
    }

    def add_divergence(divergence: dict[str, Any]) -> None:
        report["equal"] = False
        report["divergence_count"] += 1
        if len(report["divergences"]) < max_details:
            report["divergences"].append(divergence)

    if expected_iterations is not None:
        for side, iterations in (("left", left_iterations), ("right", right_iterations)):
            if len(iterations) != expected_iterations:
                add_divergence(
                    {
                        "reason": "iteration_count",
                        "side": side,
                        "actual": len(iterations),
                        "expected": expected_iterations,
                    }
                )

    all_iterations = sorted(set(left_iterations) | set(right_iterations))
    for iteration in all_iterations:
        if iteration not in left_iterations:
            add_divergence({"iteration": iteration, "reason": "iteration_missing_left"})
            continue
        if iteration not in right_iterations:
            add_divergence({"iteration": iteration, "reason": "iteration_missing_right"})
            continue

        report["iterations_compared"] += 1
        left_record = left_iterations[iteration]
        right_record = right_iterations[iteration]
        if left_record["train_iters"] != right_record["train_iters"]:
            add_divergence(
                {
                    "iteration": iteration,
                    "reason": "train_iters",
                    "left": left_record["train_iters"],
                    "right": right_record["train_iters"],
                }
            )

        left_metrics = left_record["metrics"]
        right_metrics = right_record["metrics"]
        for metric in required:
            if metric not in left_metrics and metric not in right_metrics:
                add_divergence(
                    {
                        "iteration": iteration,
                        "metric": metric,
                        "reason": "required_metric_missing_both",
                    }
                )

        for metric in sorted(set(left_metrics) | set(right_metrics)):
            if metric not in left_metrics:
                add_divergence(
                    {"iteration": iteration, "metric": metric, "reason": "metric_missing_left"}
                )
                continue
            if metric not in right_metrics:
                add_divergence(
                    {"iteration": iteration, "metric": metric, "reason": "metric_missing_right"}
                )
                continue
            report["metric_values_compared"] += 1
            if left_metrics[metric] != right_metrics[metric]:
                add_divergence(
                    {
                        "iteration": iteration,
                        "metric": metric,
                        "reason": "metric_value",
                        "left": left_metrics[metric],
                        "right": right_metrics[metric],
                    }
                )
    return report


def _print_human_report(report: dict[str, Any]) -> None:
    if report["equal"]:
        print(
            f"MATCH: {report['iterations_compared']} iteration(s), "
            f"{report['metric_values_compared']} serialized metric value(s) match"
        )
        return
    print(
        f"DIVERGED: {report['divergence_count']} difference(s) across "
        f"{report['iterations_compared']} matched iteration(s)"
    )
    for index, divergence in enumerate(report["divergences"], start=1):
        location = []
        if "iteration" in divergence:
            location.append(f"iteration={divergence['iteration']}")
        if "metric" in divergence:
            location.append(f"metric={divergence['metric']}")
        print(f"{index}. {' '.join(location) or divergence.get('side', 'comparison')}")
        print(f"   reason: {divergence['reason']}")
        if "left" in divergence or "right" in divergence:
            print(f"   left={divergence.get('left')!r} right={divergence.get('right')!r}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", help="First Megatron console log")
    parser.add_argument("right", help="Second Megatron console log")
    parser.add_argument(
        "--expected-iterations",
        type=int,
        help="Require exactly this many logged iterations in each input",
    )
    parser.add_argument(
        "--require-metric",
        action="append",
        default=[],
        help="Metric that must appear in every iteration (repeatable)",
    )
    parser.add_argument(
        "--ignore-metric",
        action="append",
        default=[],
        help="Additional volatile metric to exclude from comparison (repeatable)",
    )
    parser.add_argument(
        "--max-details", type=int, default=10, help="Maximum differences to print (default: 10)"
    )
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable JSON report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    required_metrics = DEFAULT_REQUIRED_METRICS + tuple(args.require_metric)
    ignored_metrics = DEFAULT_IGNORED_METRICS | frozenset(args.ignore_metric)
    try:
        report = compare_training_logs(
            args.left,
            args.right,
            expected_iterations=args.expected_iterations,
            required_metrics=required_metrics,
            ignored_metrics=ignored_metrics,
            max_details=args.max_details,
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human_report(report)
    return 0 if report["equal"] else 1


if __name__ == "__main__":
    sys.exit(main())
