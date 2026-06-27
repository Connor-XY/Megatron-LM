# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Certify rank-sharded training traces against deterministic-training invariants."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

try:
    from tools.determinism.compare_traces import compare_trace_paths
except ModuleNotFoundError:  # Direct ``python tools/determinism/certify_traces.py`` invocation.
    from compare_traces import compare_trace_paths


def _trace_files(path: Path) -> list[Path]:
    if not path.is_dir():
        raise FileNotFoundError(f"Trace directory does not exist: {path}")
    files = sorted(path.rglob("*.jsonl"))
    if not files:
        raise ValueError(f"No .jsonl trace files found under: {path}")
    return files


def _events(files: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in files:
        with path.open(encoding="utf-8") as trace_file:
            for line_number, line in enumerate(trace_file, start=1):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in {path}:{line_number}: {error.msg}"
                    ) from error
                if not isinstance(event, Mapping):
                    raise TypeError(f"Expected an event object in {path}:{line_number}")
                yield dict(event)


def certify_trace_path(
    trace_path: str | Path,
    *,
    expected_ranks: int | None = None,
    expected_iterations: int | None = None,
    required_collective_prefixes: tuple[str, ...] = (),
    require_dp_fp32_accumulation: bool = False,
) -> dict[str, Any]:
    """Return a JSON-safe certification report for one trace directory."""
    trace_path = Path(trace_path)
    files = _trace_files(trace_path)
    events = list(_events(files))
    failures = []

    rank_iterations = {(event.get("rank"), event.get("iteration")) for event in events}
    ranks = sorted({rank for rank, _ in rank_iterations if isinstance(rank, int)})
    iterations = sorted(
        {iteration for _, iteration in rank_iterations if isinstance(iteration, int)}
    )
    if expected_ranks is not None:
        expected = list(range(expected_ranks))
        if ranks != expected:
            failures.append(f"rank set is {ranks}, expected {expected}")
    if expected_iterations is not None:
        if len(iterations) != expected_iterations:
            failures.append(
                f"found {len(iterations)} traced iterations, expected {expected_iterations}"
            )
    if expected_ranks is not None and expected_iterations is not None:
        expected_files = expected_ranks * expected_iterations
        if len(files) != expected_files:
            failures.append(f"found {len(files)} trace files, expected {expected_files}")
        expected_pairs = {
            (rank, iteration) for rank in range(expected_ranks) for iteration in iterations
        }
        observed_pairs = {
            (rank, iteration)
            for rank, iteration in rank_iterations
            if isinstance(rank, int) and isinstance(iteration, int)
        }
        missing_pairs = sorted(expected_pairs - observed_pairs)
        unexpected_pairs = sorted(observed_pairs - expected_pairs)
        if missing_pairs:
            failures.append(f"missing rank/iteration pairs: {missing_pairs}")
        if unexpected_pairs:
            failures.append(f"unexpected rank/iteration pairs: {unexpected_pairs}")

    runtime_events = [event for event in events if event.get("name") == "runtime"]
    non_deterministic_runtime = [
        event
        for event in runtime_events
        if not event.get("payload", {}).get("deterministic_algorithms")
    ]
    if len(runtime_events) != len(files):
        failures.append(f"found {len(runtime_events)} runtime events for {len(files)} files")
    if non_deterministic_runtime:
        failures.append(
            f"{len(non_deterministic_runtime)} runtime events did not enable deterministic algorithms"
        )

    recomputes = [event for event in events if event.get("name") == "checkpoint.recompute"]
    recompute_mismatches = [
        event for event in recomputes if event.get("payload", {}).get("matches_forward") is not True
    ]
    if not recomputes:
        failures.append("no checkpoint recompute events found")
    if recompute_mismatches:
        failures.append(
            f"{len(recompute_mismatches)} checkpoint recomputes did not match original forward"
        )

    iteration_ends = [event for event in events if event.get("name") == "iteration.end"]
    pending_collectives = sum(
        int(event.get("payload", {}).get("pending_collectives", 0)) for event in iteration_ends
    )
    if len(iteration_ends) != len(files):
        failures.append(f"found {len(iteration_ends)} iteration.end events for {len(files)} files")
    if pending_collectives:
        failures.append(f"{pending_collectives} collectives were pending at trace-window end")

    collectives = [event for event in events if event.get("kind") == "collective"]
    completed = [event for event in collectives if str(event.get("name", "")).endswith(".end")]
    completed_without_hashes = [
        event
        for event in completed
        if not event.get("payload", {}).get("outputs")
        or any(not output.get("sha256") for output in event["payload"]["outputs"])
    ]
    if not completed:
        failures.append("no completed collective events found")
    if completed_without_hashes:
        failures.append(
            f"{len(completed_without_hashes)} completed collectives are missing output hashes"
        )

    prefix_counts = Counter()
    for prefix in required_collective_prefixes:
        prefix_counts[prefix] = sum(
            str(event.get("name", "")).startswith(prefix) for event in collectives
        )
        if prefix_counts[prefix] == 0:
            failures.append(f"no collective events start with {prefix!r}")

    dp_grad_begins = [
        event
        for event in collectives
        if str(event.get("name", "")).startswith("data_parallel.grad_reduce")
        and str(event.get("name", "")).endswith(".begin")
    ]
    dp_without_fp32 = [
        event
        for event in dp_grad_begins
        if event.get("payload", {}).get("fp32_accumulation") is not True
    ]
    if require_dp_fp32_accumulation:
        if not dp_grad_begins:
            failures.append("no data-parallel gradient-reduction events found")
        elif dp_without_fp32:
            failures.append(
                f"{len(dp_without_fp32)} data-parallel reductions did not use fp32 accumulation"
            )

    return {
        "equal": not failures,
        "path": str(trace_path.resolve()),
        "trace_files": len(files),
        "events": len(events),
        "ranks": ranks,
        "iterations": iterations,
        "recomputes": len(recomputes),
        "recompute_mismatches": len(recompute_mismatches),
        "collectives": len(collectives),
        "completed_collectives": len(completed),
        "pending_collectives": pending_collectives,
        "collective_prefix_counts": dict(prefix_counts),
        "dp_grad_reductions": len(dp_grad_begins),
        "dp_grad_reductions_without_fp32_accumulation": len(dp_without_fp32),
        "failures": failures,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", help="First rank-sharded trace directory")
    parser.add_argument("right", nargs="?", help="Optional second trace directory to compare")
    parser.add_argument("--expected-ranks", type=int)
    parser.add_argument("--expected-iterations", type=int)
    parser.add_argument(
        "--require-collective-prefix",
        action="append",
        default=[],
        help="Require at least one collective whose semantic name starts with this prefix",
    )
    parser.add_argument("--require-dp-fp32-accumulation", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit the complete JSON report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        kwargs = {
            "expected_ranks": args.expected_ranks,
            "expected_iterations": args.expected_iterations,
            "required_collective_prefixes": tuple(args.require_collective_prefix),
            "require_dp_fp32_accumulation": args.require_dp_fp32_accumulation,
        }
        report = {"left": certify_trace_path(args.left, **kwargs)}
        if args.right:
            report["right"] = certify_trace_path(args.right, **kwargs)
            report["comparison"] = compare_trace_paths(args.left, args.right)
        report["equal"] = all(
            section.get("equal", False)
            for section in report.values()
            if isinstance(section, Mapping)
        )
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif report["equal"]:
        left = report["left"]
        verdict = "CERTIFIED" if args.right else "TRACE INVARIANTS PASSED"
        print(
            f"{verdict}: files={left['trace_files']} events={left['events']} "
            f"recomputes={left['recomputes']} collectives={left['collectives']} pending=0"
        )
    else:
        print("CERTIFICATION FAILED", file=sys.stderr)
        for side in ("left", "right"):
            for failure in report.get(side, {}).get("failures", []):
                print(f"  {side}: {failure}", file=sys.stderr)
        comparison = report.get("comparison")
        if comparison is not None and not comparison["equal"]:
            print(
                f"  comparison: {comparison['divergence_count']} semantic divergences",
                file=sys.stderr,
            )
    return 0 if report["equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
