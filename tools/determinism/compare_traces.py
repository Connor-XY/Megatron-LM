# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compare two rank-sharded determinism trace trees by semantic event identity."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SUPPORTED_SCHEMA_VERSION = 1


def _natural_key(value: str) -> tuple:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in re.split(r"(\d+)", value)
        if part
    )


def _discover_files(path: Path) -> dict[str, Path]:
    if path.is_file():
        return {"<single-file>": path}
    if not path.is_dir():
        raise FileNotFoundError(f"Trace path does not exist: {path}")
    files = {
        file.relative_to(path).as_posix(): file
        for file in sorted(path.rglob("*.jsonl"), key=lambda item: _natural_key(str(item)))
    }
    if not files:
        raise ValueError(f"No .jsonl trace files found under: {path}")
    return files


def _load_events(path: Path) -> dict[str, dict[str, Any]]:
    events = {}
    occurrences: Counter[tuple] = Counter()
    with path.open(encoding="utf-8") as trace_file:
        for line_number, line in enumerate(trace_file, start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {error.msg}") from error
            if not isinstance(event, Mapping):
                raise TypeError(f"Expected an event object in {path}:{line_number}")
            if event.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported schema version in {path}:{line_number}: "
                    f"{event.get('schema_version')!r}"
                )
            required = ("sequence", "iteration", "rank", "kind", "name", "payload")
            missing = [key for key in required if key not in event]
            if missing:
                raise ValueError(f"Missing keys in {path}:{line_number}: {', '.join(missing)}")
            semantic_identity = None
            if isinstance(event["payload"], Mapping):
                semantic_identity = event["payload"].get("checkpoint_id")
            base = (
                event["iteration"],
                event["rank"],
                event["kind"],
                event["name"],
                semantic_identity,
            )
            occurrence = occurrences[base]
            occurrences[base] += 1
            semantic_key = "/".join(
                (
                    f"iter={event['iteration']}",
                    f"rank={event['rank']}",
                    f"kind={event['kind']}",
                    f"name={event['name']}",
                    f"identity={semantic_identity}",
                    f"occurrence={occurrence}",
                )
            )
            events[semantic_key] = {
                "sequence": event["sequence"],
                "event": {
                    "schema_version": event["schema_version"],
                    "iteration": event["iteration"],
                    "rank": event["rank"],
                    "kind": event["kind"],
                    "name": event["name"],
                    "payload": event["payload"],
                },
            }
    return events


def compare_trace_paths(
    left_path: str | Path, right_path: str | Path, max_details: int = 10
) -> dict:
    """Return a JSON-safe semantic comparison of two trace files or trees."""
    if max_details < 1:
        raise ValueError("max_details must be at least 1")
    left_path = Path(left_path)
    right_path = Path(right_path)
    left_files = _discover_files(left_path)
    right_files = _discover_files(right_path)
    all_files = sorted(set(left_files) | set(right_files), key=_natural_key)
    report = {
        "equal": True,
        "left": str(left_path.resolve()),
        "right": str(right_path.resolve()),
        "matched_files": 0,
        "events_compared": 0,
        "divergence_count": 0,
        "divergences": [],
        "first_divergence": None,
        "divergent_ranks": [],
        "divergent_iterations": [],
        "recommended_followup": None,
    }
    divergent_ranks = set()
    divergent_iterations = set()

    def add_divergence(
        divergence: dict, *, rank: int | None = None, iteration: int | None = None
    ) -> None:
        report["equal"] = False
        report["divergence_count"] += 1
        if report["first_divergence"] is None:
            report["first_divergence"] = divergence
        if len(report["divergences"]) < max_details:
            report["divergences"].append(divergence)
        if rank is not None:
            divergent_ranks.add(rank)
        if iteration is not None:
            divergent_iterations.add(iteration)

    def file_identity(relative_path: str) -> tuple[int | None, int | None]:
        match = re.search(r"iter_(\d+)/rank_(\d+)", relative_path)
        if match is None:
            return None, None
        return int(match.group(2)), int(match.group(1))

    for relative_path in all_files:
        if relative_path not in left_files:
            rank, iteration = file_identity(relative_path)
            add_divergence(
                {"file": relative_path, "reason": "file_missing_left"},
                rank=rank,
                iteration=iteration,
            )
            continue
        if relative_path not in right_files:
            rank, iteration = file_identity(relative_path)
            add_divergence(
                {"file": relative_path, "reason": "file_missing_right"},
                rank=rank,
                iteration=iteration,
            )
            continue
        report["matched_files"] += 1
        left_events = _load_events(left_files[relative_path])
        right_events = _load_events(right_files[relative_path])

        def event_order(event_key: str) -> tuple:
            sequences = [
                events[event_key]["sequence"]
                for events in (left_events, right_events)
                if event_key in events
            ]
            return (min(sequences), _natural_key(event_key))

        all_events = sorted(set(left_events) | set(right_events), key=event_order)
        for event_key in all_events:
            if event_key not in left_events:
                event = right_events[event_key]["event"]
                add_divergence(
                    {
                        "file": relative_path,
                        "event": event_key,
                        "reason": "event_missing_left",
                        "right_sequence": right_events[event_key]["sequence"],
                    },
                    rank=event["rank"],
                    iteration=event["iteration"],
                )
                continue
            if event_key not in right_events:
                event = left_events[event_key]["event"]
                add_divergence(
                    {
                        "file": relative_path,
                        "event": event_key,
                        "reason": "event_missing_right",
                        "left_sequence": left_events[event_key]["sequence"],
                    },
                    rank=event["rank"],
                    iteration=event["iteration"],
                )
                continue
            report["events_compared"] += 1
            if left_events[event_key]["event"] != right_events[event_key]["event"]:
                event = left_events[event_key]["event"]
                add_divergence(
                    {
                        "file": relative_path,
                        "event": event_key,
                        "reason": "event_values",
                        "left_sequence": left_events[event_key]["sequence"],
                        "right_sequence": right_events[event_key]["sequence"],
                        "left_event": left_events[event_key]["event"],
                        "right_event": right_events[event_key]["event"],
                    },
                    rank=event["rank"],
                    iteration=event["iteration"],
                )
    report["divergent_ranks"] = sorted(divergent_ranks)
    report["divergent_iterations"] = sorted(divergent_iterations)
    if divergent_ranks and divergent_iterations:
        first_iteration = min(divergent_iterations)
        report["recommended_followup"] = {
            "rank_spec": _format_rank_spec(divergent_ranks),
            "start_iteration": max(1, first_iteration - 1),
            "end_iteration": first_iteration,
            "detail": "semantic_device_digests",
        }
    return report


def _format_rank_spec(ranks: set[int]) -> str:
    """Format ranks as the tracer's compact comma/range syntax."""
    ordered = sorted(ranks)
    ranges = []
    start = previous = ordered[0]
    for rank in ordered[1:]:
        if rank == previous + 1:
            previous = rank
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = rank
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _print_human_report(report: dict) -> None:
    if report["equal"]:
        print(
            f"MATCH: {report['matched_files']} rank trace(s), "
            f"{report['events_compared']} semantic event(s) match"
        )
        return
    print(
        f"DIVERGED: {report['divergence_count']} difference(s) across "
        f"{report['matched_files']} matched rank trace(s)"
    )
    for index, divergence in enumerate(report["divergences"], start=1):
        location = divergence["file"]
        if "event" in divergence:
            location += f" :: {divergence['event']}"
        print(f"{index}. {location}")
        print(f"   reason: {divergence['reason']}")
        if "left_sequence" in divergence or "right_sequence" in divergence:
            print(
                "   local sequence: "
                f"left={divergence.get('left_sequence', 'missing')} "
                f"right={divergence.get('right_sequence', 'missing')}"
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", help="First trace file or directory")
    parser.add_argument("right", help="Second trace file or directory")
    parser.add_argument(
        "--max-details", type=int, default=10, help="Maximum differences to print (default: 10)"
    )
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable JSON report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = compare_trace_paths(args.left, args.right, max_details=args.max_details)
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human_report(report)
    return 0 if report["equal"] else 1


if __name__ == "__main__":
    sys.exit(main())
