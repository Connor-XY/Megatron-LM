# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Attribute matching Nsight Systems NVTX ranges to their enclosing ranges."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


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


def _print_text(report: dict[str, Any]) -> None:
    print(
        f"matched {report['match_count']} ranges for {report['pattern']!r}; "
        f"total {report['total_duration_ms']:.3f} ms"
    )
    for index, event in enumerate(report["events"], start=1):
        print(f"{index}. {event['name']} ({event['duration_ms']:.3f} ms)")
        for parent in event["parents"]:
            print(f"   -> {parent['name']} ({parent['duration_ms']:.3f} ms)")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path, help="SQLite file exported by nsys stats")
    parser.add_argument("pattern", help="substring matched against NVTX range names")
    parser.add_argument("--max-parents", type=int, default=8)
    parser.add_argument("--json", action="store_true", help="emit the complete report as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        report = attribute_ranges(args.sqlite, args.pattern, max_parents=args.max_parents)
    except (FileNotFoundError, ValueError, sqlite3.Error) as error:
        parser.error(str(error))
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
