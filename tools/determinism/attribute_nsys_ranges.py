# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Attribute matching Nsight Systems NVTX ranges to their enclosing ranges."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

_DYNAMIC_RANGE_SUFFIX = re.compile(r"(?:, seq = \d+)?(?:, op_id = \d+)$")
_UNATTRIBUTED = "<unattributed>"


def attribute_ranges(sqlite_path: Path, pattern: str, *, max_parents: int = 8) -> dict[str, Any]:
    """Return matching NVTX ranges and their nearest enclosing ranges.

    Args:
        sqlite_path: SQLite file exported from an Nsight Systems report.
        pattern: Substring matched against the resolved NVTX range name.
        max_parents: Maximum number of enclosing ranges retained per match.

    Returns:
        A JSON-serializable attribution report.
    """
    if not pattern:
        raise ValueError("pattern must be non-empty")
    if max_parents < 1:
        raise ValueError("max_parents must be positive")
    if not sqlite_path.is_file():
        raise FileNotFoundError(f"Nsight SQLite file does not exist: {sqlite_path}")

    query = """
        SELECT
            event.start,
            event.end,
            event.globalTid,
            COALESCE(event.text, string.value) AS name
        FROM NVTX_EVENTS AS event
        LEFT JOIN StringIds AS string ON event.textId = string.id
        WHERE event.end IS NOT NULL AND event.globalTid IS NOT NULL
        ORDER BY event.start, event.globalTid
    """
    with sqlite3.connect(sqlite_path) as connection:
        ranges = [
            {"start": start, "end": end, "global_tid": global_tid, "name": name or ""}
            for start, end, global_tid, name in connection.execute(query)
        ]

    events = []
    for target in ranges:
        if pattern not in target["name"]:
            continue
        parents = [
            parent
            for parent in ranges
            if parent["global_tid"] == target["global_tid"]
            and parent["start"] <= target["start"]
            and parent["end"] >= target["end"]
            and (parent["start"] < target["start"] or parent["end"] > target["end"])
        ]
        parents.sort(key=lambda parent: (parent["end"] - parent["start"], parent["name"]))
        events.append(
            {
                "name": target["name"],
                "duration_ms": (target["end"] - target["start"]) / 1e6,
                "global_tid": target["global_tid"],
                "parents": [
                    {"name": parent["name"], "duration_ms": (parent["end"] - parent["start"]) / 1e6}
                    for parent in parents[:max_parents]
                ],
            }
        )

    return {
        "sqlite": str(sqlite_path),
        "pattern": pattern,
        "match_count": len(events),
        "total_duration_ms": sum(event["duration_ms"] for event in events),
        "events": events,
    }


def _canonicalize_range_name(name: str) -> str:
    """Remove per-run sequence and operator IDs from an NVTX range name."""
    return _DYNAMIC_RANGE_SUFFIX.sub("", name)


def summarize_attribution(report: dict[str, Any], *, parent_depth: int = 1) -> dict[str, Any]:
    """Aggregate matched duration by one enclosing-parent level.

    ``parent_depth=1`` groups by the nearest enclosing range, ``2`` by the
    next-nearest range, and so on. Dynamic sequence and operator IDs are
    removed before grouping so repeated invocations share one entry.
    """
    if parent_depth < 1:
        raise ValueError("parent_depth must be positive")

    groups: dict[str, dict[str, Any]] = {}
    for event in report["events"]:
        if len(event["parents"]) >= parent_depth:
            name = _canonicalize_range_name(event["parents"][parent_depth - 1]["name"])
        else:
            name = _UNATTRIBUTED
        group = groups.setdefault(name, {"name": name, "match_count": 0, "duration_ms": 0.0})
        group["match_count"] += 1
        group["duration_ms"] += event["duration_ms"]

    ordered_groups = sorted(
        groups.values(), key=lambda group: (-group["duration_ms"], group["name"])
    )
    return {
        "sqlite": report["sqlite"],
        "pattern": report["pattern"],
        "parent_depth": parent_depth,
        "match_count": report["match_count"],
        "total_duration_ms": report["total_duration_ms"],
        "groups": ordered_groups,
    }


def _print_text(report: dict[str, Any], *, top: int | None = None) -> None:
    print(
        f"matched {report['match_count']} ranges for {report['pattern']!r}; "
        f"total {report['total_duration_ms']:.3f} ms"
    )
    events = sorted(report["events"], key=lambda event: -event["duration_ms"])
    if top is not None:
        events = events[:top]
    for index, event in enumerate(events, start=1):
        print(f"{index}. {event['name']} ({event['duration_ms']:.3f} ms)")
        for parent in event["parents"]:
            print(f"   -> {parent['name']} ({parent['duration_ms']:.3f} ms)")


def _print_summary(summary: dict[str, Any], *, top: int | None = None) -> None:
    print(
        f"matched {summary['match_count']} ranges for {summary['pattern']!r}; "
        f"total {summary['total_duration_ms']:.3f} ms; "
        f"grouped by parent depth {summary['parent_depth']}"
    )
    groups = summary["groups"] if top is None else summary["groups"][:top]
    total_duration_ms = summary["total_duration_ms"]
    for index, group in enumerate(groups, start=1):
        percentage = 100.0 * group["duration_ms"] / total_duration_ms if total_duration_ms else 0.0
        print(
            f"{index}. {group['name']}: {group['match_count']} ranges, "
            f"{group['duration_ms']:.3f} ms ({percentage:.1f}%)"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path, help="SQLite file exported by nsys stats")
    parser.add_argument("pattern", help="substring matched against NVTX range names")
    parser.add_argument("--max-parents", type=int, default=8)
    parser.add_argument(
        "--summary", action="store_true", help="aggregate matches by an enclosing parent"
    )
    parser.add_argument(
        "--parent-depth",
        type=int,
        default=1,
        help="parent level used by --summary: 1 is the nearest enclosing range",
    )
    parser.add_argument(
        "--top", type=int, help="limit text output to the highest-duration events or groups"
    )
    parser.add_argument("--json", action="store_true", help="emit the complete report as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        report = attribute_ranges(args.sqlite, args.pattern, max_parents=args.max_parents)
        if args.summary:
            report = summarize_attribution(report, parent_depth=args.parent_depth)
    except (FileNotFoundError, ValueError, sqlite3.Error) as error:
        parser.error(str(error))
    if args.top is not None and args.top < 1:
        parser.error("--top must be positive")
    if args.json:
        print(json.dumps(report, indent=2))
    elif args.summary:
        _print_summary(report, top=args.top)
    else:
        _print_text(report, top=args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
