#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Find the module where one rank's activation first diverges from the rest.

Reads the per-rank JSONL written by all_rank_digest_trace and reports, for each
repeat, the earliest non-expert module whose per-rank digests partition into a
majority and a minority. That module is the producer of the divergence.

Usage:
    analyze_split.py <run-root> [<run-root> ...]

Each run root is expected to contain repeat-N/ directories, each holding a
digest-trace/ directory of rank*.jsonl files.

Two cautions, both learned the hard way on this bug:

- The trace covers a bounded layer and call range, so a repeat with no recorded
  split is not necessarily a clean repeat. Split counts are a lower bound.
- Judge a rate against a matched control run, never against a remembered
  number. On the case this tool was written for, the per-repeat rate moved
  between 0.22 and 0.75 with nothing but CUDA JIT cache state, so a five-repeat
  cell reading all-same is anywhere from 0.1% to 29% likely by luck.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

# Routed-expert activations legitimately differ on every rank under expert
# parallelism, which is a different signature from a minority split.
EXPERT = ("experts", "local_experts", "expert_mlp")


def load_repeat(trace_dir: Path) -> dict:
    """Collect digests per (module, call) across ranks."""
    events = defaultdict(lambda: {"input": {}, "output": {}})
    meta, order = {}, {}
    for path in sorted(trace_dir.glob("rank*.jsonl")):
        for line in path.read_text(errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A killed job can leave a partial final line.
                continue
            key = (record["module"], record["call"])
            for side in ("input", "output"):
                sig = record.get(side)
                if sig:
                    events[key][side][record["rank"]] = sig["digest"]
            meta.setdefault(
                key,
                {
                    "module_type": record.get("module_type"),
                    "shape": (record.get("output") or {}).get("shape"),
                },
            )
            order.setdefault(key, record.get("sequence", 0))
    return {"events": events, "meta": meta, "order": order}


def classify(per_rank: dict) -> tuple[str, Counter]:
    """One group means agreement; a small group against a large one is the defect."""
    counts = Counter(per_rank.values())
    if len(counts) <= 1:
        return "uniform", counts
    if len(counts) == len(per_rank):
        return "all_differ", counts
    return "minority_split", counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_roots", type=Path, nargs="+")
    parser.add_argument("--max-report", type=int, default=4)
    parser.add_argument("--json", action="store_true", help="emit raw JSON")
    args = parser.parse_args()

    rows = []
    for root in args.run_roots:
        if not root.is_dir():
            print(f"# missing run root: {root}")
            continue
        for repeat_dir in sorted(root.glob("repeat-*"), key=lambda p: int(p.name.split("-")[1])):
            trace_dir = repeat_dir / "digest-trace"
            if not trace_dir.is_dir():
                continue
            data = load_repeat(trace_dir)
            ranks = set()
            hits = []
            for key, sides in data["events"].items():
                module, call = key
                if any(tag in module for tag in EXPERT):
                    continue
                for side in ("input", "output"):
                    if not sides[side]:
                        continue
                    ranks.update(sides[side])
                    kind, counts = classify(sides[side])
                    if kind != "minority_split":
                        continue
                    minority = min(counts, key=counts.get)
                    hits.append(
                        {
                            "sequence": data["order"].get(key, 1 << 60),
                            "module": module,
                            "module_type": data["meta"][key]["module_type"],
                            "shape": data["meta"][key]["shape"],
                            "call": call,
                            "side": side,
                            "minority_ranks": sorted(
                                r for r, d in sides[side].items() if d == minority
                            ),
                        }
                    )
            hits.sort(key=lambda h: (h["sequence"], h["module"], h["side"]))
            rows.append(
                {
                    "run": root.name,
                    "repeat": repeat_dir.name,
                    "ranks": len(ranks),
                    "total_splits": len(hits),
                    "findings": hits[: args.max_report],
                }
            )

    if not rows:
        print("no digest traces found")
        return 1

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    print(f"{'run':>28} {'repeat':>10} {'ranks':>6} {'splits':>7}  first diverging module")
    for row in rows:
        first = row["findings"][0] if row["findings"] else None
        if first:
            mod = first["module"]
            if len(mod) > 52:
                mod = "..." + mod[-49:]
            tail = f"{mod} [{first['side']}] minority={first['minority_ranks']}"
        else:
            tail = "-"
        print(
            f"{row['run'][-28:]:>28} {row['repeat']:>10} {row['ranks']:>6} "
            f"{row['total_splits']:>7}  {tail}"
        )

    split = [r for r in rows if r["total_splits"]]
    print()
    print(f"repeats with a minority split: {len(split)} of {len(rows)}")
    if split:
        modules = Counter(r["findings"][0]["module"] for r in split)
        for module, n in modules.most_common():
            print(f"  first diverging module {module}: {n} of {len(split)}")
        minorities = [tuple(r["findings"][0]["minority_ranks"]) for r in split]
        involved = sorted({rank for m in minorities for rank in m})
        print(f"  minority sets: {minorities}")
        print(f"  ranks ever in the minority: {involved}")
        if len(involved) > 1:
            print("  more than one rank appears, so this is not a single faulty GPU")
    else:
        print("  no split recorded; note this is a lower bound, see the module docstring")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
