# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compare two NT3 module-flight-recorder runs and locate the first birth."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_LINE = re.compile(r"rank=(\d+) seq=(\d+) mod=(\S+) call=(\d+) (.*)")


@dataclass(frozen=True)
class Record:
    rank: int
    sequence: int
    module_key: str
    call: int
    fields: dict[str, str]

    @property
    def identity(self) -> tuple[int, str, int]:
        return self.rank, self.module_key, self.call

    @property
    def direction(self) -> str:
        return self.module_key.rsplit("|", 1)[-1]


def _trace_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"trace path does not exist: {path}")
    files = sorted(path.glob("rank_*.txt"))
    if not files:
        raise ValueError(f"no rank_*.txt files found under {path}")
    return files


def load_records(path: str | Path) -> list[Record]:
    records = []
    identities = set()
    for trace_file in _trace_files(Path(path)):
        with trace_file.open(encoding="utf-8", errors="replace") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                match = _LINE.fullmatch(line.strip())
                if match is None:
                    continue
                fields = {}
                for token in match.group(5).split():
                    if "=" in token:
                        key, value = token.split("=", 1)
                        fields[key] = value
                record = Record(
                    rank=int(match.group(1)),
                    sequence=int(match.group(2)),
                    module_key=match.group(3),
                    call=int(match.group(4)),
                    fields=fields,
                )
                if record.identity in identities:
                    raise ValueError(
                        f"duplicate record identity in {trace_file}:{line_number}: "
                        f"{record.identity}"
                    )
                identities.add(record.identity)
                records.append(record)
    return sorted(records, key=lambda record: (record.sequence, record.rank))


def _content_fields(record: Record) -> dict[str, str]:
    return {
        name: value
        for name, value in record.fields.items()
        if not name.endswith(".status") or value != "ok"
    }


def _split_changes(record: Record, other: Record) -> tuple[list[str], list[str], list[str]]:
    left = _content_fields(record)
    right = _content_fields(other)
    changed = sorted(name for name in set(left) | set(right) if left.get(name) != right.get(name))
    if record.direction == "bwd":
        arrived_prefix, produced_prefix = "gout", "gin"
    else:
        arrived_prefix, produced_prefix = "in", "out"
    arrived = [name for name in changed if name.startswith(arrived_prefix)]
    produced = [name for name in changed if name.startswith(produced_prefix)]
    other_changes = [name for name in changed if name not in set(arrived + produced)]
    return arrived, produced, other_changes


def compare_runs(left_path: str | Path, right_path: str | Path) -> dict:
    left_records = load_records(left_path)
    right_records = load_records(right_path)
    right_by_identity = {record.identity: record for record in right_records}
    left_by_identity = {record.identity: record for record in left_records}

    differences = []
    births = []
    last_match_by_rank: dict[int, tuple[int, str, int]] = {}
    for left in left_records:
        right = right_by_identity.get(left.identity)
        if right is None:
            differences.append(
                {
                    "identity": left.identity,
                    "left_sequence": left.sequence,
                    "right_sequence": None,
                    "reason": "missing_right",
                }
            )
            continue
        arrived, produced, other = _split_changes(left, right)
        if not arrived and not produced and not other:
            last_match_by_rank[left.rank] = left.identity
            continue
        difference = {
            "identity": left.identity,
            "left_sequence": left.sequence,
            "right_sequence": right.sequence,
            "arrived_different": arrived,
            "produced_different": produced,
            "other_different": other,
            "previous_matching_identity": last_match_by_rank.get(left.rank),
            "reason": "values",
        }
        differences.append(difference)
        if not arrived and produced and not other:
            births.append(difference)

    for right in right_records:
        if right.identity not in left_by_identity:
            differences.append(
                {
                    "identity": right.identity,
                    "left_sequence": None,
                    "right_sequence": right.sequence,
                    "reason": "missing_left",
                }
            )

    def order(item: dict) -> tuple[int, int]:
        sequences = [
            sequence
            for sequence in (item.get("left_sequence"), item.get("right_sequence"))
            if sequence is not None
        ]
        return (min(sequences), item["identity"][0])

    differences.sort(key=order)
    births.sort(key=order)
    return {
        "equal": not differences,
        "left_records": len(left_records),
        "right_records": len(right_records),
        "difference_count": len(differences),
        "birth_count": len(births),
        "first_difference": differences[0] if differences else None,
        "first_birth": births[0] if births else None,
        "differences": differences,
    }


def _describe(item: dict | None) -> str:
    if item is None:
        return "none"
    rank, module_key, call = item["identity"]
    return (
        f"rank={rank} seq={item.get('left_sequence')}/{item.get('right_sequence')} "
        f"module={module_key} call={call} "
        f"arrived={item.get('arrived_different', [])} "
        f"produced={item.get('produced_different', [])}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", help="First recorder file or directory")
    parser.add_argument("right", help="Second recorder file or directory")
    parser.add_argument("--json", action="store_true", help="Print the full JSON report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = compare_runs(args.left, args.right)
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif report["equal"]:
        print(f"MATCH: {report['left_records']} chronological records")
    else:
        print(
            f"DIVERGED: differences={report['difference_count']} " f"births={report['birth_count']}"
        )
        print(f"first difference: {_describe(report['first_difference'])}")
        print(f"first birth:      {_describe(report['first_birth'])}")
        if report["first_birth"] is None:
            print("No captured birth: the first differing value arrived from an untraced boundary.")
    return 0 if report["equal"] else 1


if __name__ == "__main__":
    sys.exit(main())
