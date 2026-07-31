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
    from tools.determinism.compare_traces import SUPPORTED_SCHEMA_VERSION, compare_trace_paths
except ModuleNotFoundError:  # Direct ``python tools/determinism/certify_traces.py`` invocation.
    from compare_traces import SUPPORTED_SCHEMA_VERSION, compare_trace_paths


_REQUIRED_EVENT_KEYS = (
    "schema_version",
    "sequence",
    "iteration",
    "rank",
    "kind",
    "name",
    "payload",
)


def _trace_files(path: Path) -> list[Path]:
    if not path.is_dir():
        raise FileNotFoundError(f"Trace directory does not exist: {path}")
    files = sorted(path.rglob("*.jsonl"))
    if not files:
        raise ValueError(f"No .jsonl trace files found under: {path}")
    return files


def _events(files: Iterable[Path]) -> dict[Path, list[dict[str, Any]]]:
    events_by_file = {}
    for path in files:
        file_events = []
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
                missing = [key for key in _REQUIRED_EVENT_KEYS if key not in event]
                if missing:
                    raise ValueError(f"Missing keys in {path}:{line_number}: {', '.join(missing)}")
                if event["schema_version"] != SUPPORTED_SCHEMA_VERSION:
                    raise ValueError(
                        f"Unsupported schema version in {path}:{line_number}: "
                        f"{event['schema_version']!r}"
                    )
                for key in ("sequence", "iteration", "rank"):
                    if type(event[key]) is not int:
                        raise TypeError(
                            f"Expected integer {key} in {path}:{line_number}, "
                            f"got {event[key]!r}"
                        )
                for key in ("kind", "name"):
                    if not isinstance(event[key], str):
                        raise TypeError(
                            f"Expected string {key} in {path}:{line_number}, " f"got {event[key]!r}"
                        )
                if not isinstance(event["payload"], Mapping):
                    raise TypeError(f"Expected payload object in {path}:{line_number}")
                file_events.append(dict(event))
        events_by_file[path] = file_events
    return events_by_file


def certify_trace_path(
    trace_path: str | Path,
    *,
    expected_ranks: int | None = None,
    expected_iterations: int | None = None,
    required_event_prefixes: tuple[str, ...] = (),
    required_collective_prefixes: tuple[str, ...] = (),
    require_dp_fp32_accumulation: bool = False,
    require_dp_hierarchical_fp32_accumulation: bool = False,
) -> dict[str, Any]:
    """Return a JSON-safe certification report for one trace directory."""
    trace_path = Path(trace_path)
    files = _trace_files(trace_path)
    events_by_file = _events(files)
    events = [event for file_events in events_by_file.values() for event in file_events]
    failures = []

    file_identity_failures = 0
    sequence_failures = 0
    runtime_count_failures = 0
    iteration_begin_count_failures = 0
    iteration_end_count_failures = 0
    collective_begins_without_end = 0
    collective_ends_without_begin = 0
    for path, file_events in events_by_file.items():
        identities = {(event["rank"], event["iteration"]) for event in file_events}
        if len(identities) != 1:
            file_identity_failures += 1
            failures.append(f"{path} contains rank/iteration identities {sorted(identities)}")
        sequences = [event["sequence"] for event in file_events]
        if sequences != list(range(len(file_events))):
            sequence_failures += 1
            failures.append(f"{path} has non-contiguous event sequence {sequences}")
        runtime_count = sum(event["name"] == "runtime" for event in file_events)
        if runtime_count != 1:
            runtime_count_failures += 1
            failures.append(f"{path} has {runtime_count} runtime events, expected 1")
        iteration_begin_count = sum(event["name"] == "iteration.begin" for event in file_events)
        if iteration_begin_count != 1:
            iteration_begin_count_failures += 1
            failures.append(
                f"{path} has {iteration_begin_count} iteration.begin events, expected 1"
            )
        iteration_end_count = sum(event["name"] == "iteration.end" for event in file_events)
        if iteration_end_count != 1:
            iteration_end_count_failures += 1
            failures.append(f"{path} has {iteration_end_count} iteration.end events, expected 1")

        outstanding_collectives = Counter()
        missing_begins = Counter()
        for event in file_events:
            if event["kind"] != "collective":
                continue
            if event["name"].endswith(".begin"):
                outstanding_collectives[event["name"][: -len(".begin")]] += 1
            elif event["name"].endswith(".end"):
                collective_name = event["name"][: -len(".end")]
                if outstanding_collectives[collective_name]:
                    outstanding_collectives[collective_name] -= 1
                    if not outstanding_collectives[collective_name]:
                        del outstanding_collectives[collective_name]
                else:
                    missing_begins[collective_name] += 1
        missing_ends = outstanding_collectives
        collective_begins_without_end += sum(missing_ends.values())
        collective_ends_without_begin += sum(missing_begins.values())
        if missing_ends:
            failures.append(f"{path} has collective begins without ends: {dict(missing_ends)}")
        if missing_begins:
            failures.append(f"{path} has collective ends without begins: {dict(missing_begins)}")

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
    iteration_errors = [event for event in events if event.get("name") == "iteration.error"]
    if iteration_errors:
        failures.append(f"found {len(iteration_errors)} iteration.error events")
    pending_collectives = sum(
        int(event.get("payload", {}).get("pending_collectives", 0)) for event in iteration_ends
    )
    if pending_collectives:
        failures.append(f"{pending_collectives} collectives were pending at trace-window end")

    event_prefix_counts = Counter()
    for prefix in required_event_prefixes:
        event_prefix_counts[prefix] = sum(
            str(event.get("name", "")).startswith(prefix) for event in events
        )
        if event_prefix_counts[prefix] == 0:
            failures.append(f"no events start with {prefix!r}")

    te_backend_unavailable = [
        event for event in events if event.get("name") == "te.attention.backend.unavailable"
    ]
    if te_backend_unavailable:
        failures.append(
            f"{len(te_backend_unavailable)} TE attention backend selections were unavailable"
        )

    collectives = [event for event in events if event.get("kind") == "collective"]
    completed = [event for event in collectives if str(event.get("name", "")).endswith(".end")]
    # Hashed runs populate ``digest`` (the device-side hash). ``sha256`` is only
    # present in traces from before the host-copy path became opt-in, so accept
    # either as evidence that an output was hashed.
    completed_without_hashes = [
        event
        for event in completed
        if not event.get("payload", {}).get("outputs")
        or any(
            not (output.get("digest") or output.get("sha256"))
            for output in event["payload"]["outputs"]
        )
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
    multi_rank_dp_grad_begins = [
        event for event in dp_grad_begins if event.get("payload", {}).get("group_size", 0) > 1
    ]
    # A two-rank reduction has one floating-point addition per output element, so physical
    # topology cannot alter its accumulation order. Require the fixed hierarchy only where a
    # reduction tree can actually contain more than one addition.
    hierarchy_required_dp_grad_begins = [
        event for event in dp_grad_begins if event.get("payload", {}).get("group_size", 0) > 2
    ]
    multi_rank_dp_without_hierarchy = [
        event
        for event in hierarchy_required_dp_grad_begins
        if event.get("payload", {}).get("hierarchical_fp32_accumulation") is not True
    ]
    if require_dp_hierarchical_fp32_accumulation:
        if not dp_grad_begins:
            failures.append("no data-parallel gradient-reduction events found")
        elif not multi_rank_dp_grad_begins:
            failures.append("no multi-rank data-parallel gradient-reduction events found")
        elif multi_rank_dp_without_hierarchy:
            failures.append(
                f"{len(multi_rank_dp_without_hierarchy)} multi-rank data-parallel reductions "
                "did not use the hierarchical fp32 path"
            )

    return {
        "equal": not failures,
        "path": str(trace_path.resolve()),
        "trace_files": len(files),
        "events": len(events),
        "ranks": ranks,
        "iterations": iterations,
        "file_identity_failures": file_identity_failures,
        "sequence_failures": sequence_failures,
        "runtime_count_failures": runtime_count_failures,
        "iteration_begin_count_failures": iteration_begin_count_failures,
        "iteration_end_count_failures": iteration_end_count_failures,
        "iteration_errors": len(iteration_errors),
        "recomputes": len(recomputes),
        "recompute_mismatches": len(recompute_mismatches),
        "collectives": len(collectives),
        "completed_collectives": len(completed),
        "collective_begins_without_end": collective_begins_without_end,
        "collective_ends_without_begin": collective_ends_without_begin,
        "pending_collectives": pending_collectives,
        "event_prefix_counts": dict(event_prefix_counts),
        "te_attention_backend_unavailable": len(te_backend_unavailable),
        "collective_prefix_counts": dict(prefix_counts),
        "dp_grad_reductions": len(dp_grad_begins),
        "dp_grad_reductions_without_fp32_accumulation": len(dp_without_fp32),
        "multi_rank_dp_grad_reductions": len(multi_rank_dp_grad_begins),
        "multi_rank_dp_grad_reductions_without_hierarchical_fp32_accumulation": len(
            multi_rank_dp_without_hierarchy
        ),
        "failures": failures,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", help="First rank-sharded trace directory")
    parser.add_argument("right", nargs="?", help="Optional second trace directory to compare")
    parser.add_argument("--expected-ranks", type=int)
    parser.add_argument("--expected-iterations", type=int)
    parser.add_argument(
        "--require-event-prefix",
        action="append",
        default=[],
        help="Require at least one event whose semantic name starts with this prefix",
    )
    parser.add_argument(
        "--require-collective-prefix",
        action="append",
        default=[],
        help="Require at least one collective whose semantic name starts with this prefix",
    )
    parser.add_argument("--require-dp-fp32-accumulation", action="store_true")
    parser.add_argument("--require-dp-hierarchical-fp32-accumulation", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit the complete JSON report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        kwargs = {
            "expected_ranks": args.expected_ranks,
            "expected_iterations": args.expected_iterations,
            "required_event_prefixes": tuple(args.require_event_prefix),
            "required_collective_prefixes": tuple(args.require_collective_prefix),
            "require_dp_fp32_accumulation": args.require_dp_fp32_accumulation,
            "require_dp_hierarchical_fp32_accumulation": (
                args.require_dp_hierarchical_fp32_accumulation
            ),
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
