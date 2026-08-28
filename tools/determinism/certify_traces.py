# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Certify rank-sharded training traces against deterministic-training invariants."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterator, Mapping
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


def _parse_rank_spec(rank_spec: str) -> set[int] | None:
    """Parse the tracer's ``all`` or comma/range rank-selection syntax."""
    if rank_spec.strip().lower() in ("all", "*"):
        return None
    if not rank_spec.strip():
        raise ValueError("rank specification must not be empty")
    selected = set()
    for token in rank_spec.split(","):
        token = token.strip()
        if not token:
            raise ValueError(f"invalid empty item in rank specification {rank_spec!r}")
        try:
            if "-" not in token:
                rank = int(token)
                if rank < 0:
                    raise ValueError
                selected.add(rank)
                continue
            bounds = token.split("-")
            if len(bounds) != 2:
                raise ValueError
            start, end = (int(bound) for bound in bounds)
            if start < 0 or end < start:
                raise ValueError
            selected.update(range(start, end + 1))
        except ValueError as error:
            raise ValueError(f"invalid rank or range {token!r} in {rank_spec!r}") from error
    return selected


def _trace_files(path: Path) -> list[Path]:
    if not path.is_dir():
        raise FileNotFoundError(f"Trace directory does not exist: {path}")
    files = sorted(path.rglob("*.jsonl"))
    if not files:
        raise ValueError(f"No .jsonl trace files found under: {path}")
    return files


def _iter_events(path: Path) -> Iterator[dict[str, Any]]:
    """Validate and yield one event at a time without retaining the trace file."""
    with path.open(encoding="utf-8") as trace_file:
        for line_number, line in enumerate(trace_file, start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {error.msg}") from error
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
                        f"Expected integer {key} in {path}:{line_number}, got {event[key]!r}"
                    )
            for key in ("kind", "name"):
                if not isinstance(event[key], str):
                    raise TypeError(
                        f"Expected string {key} in {path}:{line_number}, got {event[key]!r}"
                    )
            if not isinstance(event["payload"], Mapping):
                raise TypeError(f"Expected payload object in {path}:{line_number}")
            yield dict(event)


def _count_tensor_digests(value: Any) -> tuple[int, int]:
    """Count exact and compact digest fields in one JSON-ready payload."""
    sha256_count = 0
    device_digest_count = 0
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, Mapping):
            sha256_count += bool(item.get("sha256"))
            device_digest_count += bool(item.get("device_digest"))
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return sha256_count, device_digest_count


def certify_trace_path(
    trace_path: str | Path,
    *,
    expected_ranks: int | None = None,
    expected_rank_spec: str | None = None,
    expected_iterations: int | None = None,
    required_event_prefixes: tuple[str, ...] = (),
    required_collective_prefixes: tuple[str, ...] = (),
    require_dp_fp32_accumulation: bool = False,
    require_dp_hierarchical_fp32_accumulation: bool = False,
) -> dict[str, Any]:
    """Return a JSON-safe certification report using bounded streaming state."""
    if expected_ranks is not None and expected_rank_spec is not None:
        raise ValueError("expected_ranks and expected_rank_spec are mutually exclusive")
    expected_rank_set = None
    if expected_ranks is not None:
        expected_rank_set = set(range(expected_ranks))
    elif expected_rank_spec is not None:
        expected_rank_set = _parse_rank_spec(expected_rank_spec)
        if expected_rank_set is None:
            raise ValueError("expected_rank_spec must enumerate ranks rather than select all")

    trace_path = Path(trace_path)
    files = _trace_files(trace_path)
    failures = []

    event_count = 0
    rank_iterations = set()
    file_identity_failures = 0
    sequence_failures = 0
    runtime_count_failures = 0
    iteration_begin_count_failures = 0
    iteration_end_count_failures = 0
    collective_begins_without_end = 0
    collective_ends_without_begin = 0
    runtime_events = 0
    non_deterministic_runtime = 0
    trace_details = Counter()
    forward_backward_ends = 0
    optimizer_ends = 0
    recomputes = 0
    recompute_mismatches = 0
    iteration_errors = 0
    pending_collectives = 0
    event_prefix_counts = Counter({prefix: 0 for prefix in required_event_prefixes})
    te_backend_unavailable = 0
    collectives = 0
    completed = 0
    completed_without_hashes = 0
    prefix_counts = Counter({prefix: 0 for prefix in required_collective_prefixes})
    dp_grad_begins = 0
    dp_without_fp32 = 0
    multi_rank_dp_grad_begins = 0
    multi_rank_dp_without_hierarchy = 0
    trace_truncations = 0
    sha256_digests = 0
    device_digests = 0

    for path in files:
        identities = set()
        expected_sequence = 0
        first_sequence_mismatch = None
        runtime_count = 0
        iteration_begin_count = 0
        iteration_end_count = 0
        outstanding_collectives = Counter()
        missing_begins = Counter()

        for event in _iter_events(path):
            event_count += 1
            identity = (event["rank"], event["iteration"])
            identities.add(identity)
            rank_iterations.add(identity)
            if event["sequence"] != expected_sequence and first_sequence_mismatch is None:
                first_sequence_mismatch = (expected_sequence, event["sequence"])
            expected_sequence += 1

            name = event["name"]
            kind = event["kind"]
            payload = event["payload"]
            payload_sha256, payload_device = _count_tensor_digests(payload)
            sha256_digests += payload_sha256
            device_digests += payload_device
            if name == "runtime":
                runtime_count += 1
                runtime_events += 1
                trace_details[payload.get("trace_detail", "semantic")] += 1
                if not payload.get("deterministic_algorithms"):
                    non_deterministic_runtime += 1
            elif name == "iteration.begin":
                iteration_begin_count += 1
            elif name == "iteration.end":
                iteration_end_count += 1
                pending_collectives += int(payload.get("pending_collectives", 0))
            elif name == "iteration.error":
                iteration_errors += 1
            elif name == "checkpoint.recompute":
                recomputes += 1
                if payload.get("matches_forward") is not True:
                    recompute_mismatches += 1
            elif name == "forward_backward.end":
                forward_backward_ends += 1
            elif name == "optimizer.end":
                optimizer_ends += 1
            elif name == "te.attention.backend.unavailable":
                te_backend_unavailable += 1
            elif name == "trace.truncated":
                trace_truncations += 1

            for prefix in required_event_prefixes:
                if name.startswith(prefix):
                    event_prefix_counts[prefix] += 1

            if kind != "collective":
                continue
            collectives += 1
            for prefix in required_collective_prefixes:
                if name.startswith(prefix):
                    prefix_counts[prefix] += 1

            if name.endswith(".begin"):
                collective_name = name[: -len(".begin")]
                outstanding_collectives[collective_name] += 1
                if name.startswith("data_parallel.grad_reduce"):
                    dp_grad_begins += 1
                    if payload.get("fp32_accumulation") is not True:
                        dp_without_fp32 += 1
                    group_size = payload.get("group_size", 0)
                    if isinstance(group_size, int) and group_size > 1:
                        multi_rank_dp_grad_begins += 1
                    if isinstance(group_size, int) and group_size > 2:
                        if payload.get("hierarchical_fp32_accumulation") is not True:
                            multi_rank_dp_without_hierarchy += 1
            elif name.endswith(".end"):
                completed += 1
                outputs = payload.get("outputs")
                if (
                    not isinstance(outputs, list)
                    or not outputs
                    or any(
                        not isinstance(output, Mapping)
                        or not (output.get("sha256") or output.get("device_digest"))
                        for output in outputs
                    )
                ):
                    completed_without_hashes += 1
                collective_name = name[: -len(".end")]
                if outstanding_collectives[collective_name]:
                    outstanding_collectives[collective_name] -= 1
                    if not outstanding_collectives[collective_name]:
                        del outstanding_collectives[collective_name]
                else:
                    missing_begins[collective_name] += 1

        if len(identities) != 1:
            file_identity_failures += 1
            failures.append(f"{path} contains rank/iteration identities {sorted(identities)}")
        if first_sequence_mismatch is not None:
            sequence_failures += 1
            expected, observed = first_sequence_mismatch
            failures.append(
                f"{path} has non-contiguous event sequence: expected {expected}, observed {observed}"
            )
        if runtime_count != 1:
            runtime_count_failures += 1
            failures.append(f"{path} has {runtime_count} runtime events, expected 1")
        if iteration_begin_count != 1:
            iteration_begin_count_failures += 1
            failures.append(
                f"{path} has {iteration_begin_count} iteration.begin events, expected 1"
            )
        if iteration_end_count != 1:
            iteration_end_count_failures += 1
            failures.append(f"{path} has {iteration_end_count} iteration.end events, expected 1")
        missing_ends = outstanding_collectives
        collective_begins_without_end += sum(missing_ends.values())
        collective_ends_without_begin += sum(missing_begins.values())
        if missing_ends:
            failures.append(f"{path} has collective begins without ends: {dict(missing_ends)}")
        if missing_begins:
            failures.append(f"{path} has collective ends without begins: {dict(missing_begins)}")

    ranks = sorted({rank for rank, _ in rank_iterations})
    iterations = sorted({iteration for _, iteration in rank_iterations})
    if expected_rank_set is not None:
        expected = sorted(expected_rank_set)
        if ranks != expected:
            failures.append(f"rank set is {ranks}, expected {expected}")
    if expected_iterations is not None:
        if len(iterations) != expected_iterations:
            failures.append(
                f"found {len(iterations)} traced iterations, expected {expected_iterations}"
            )
    if expected_rank_set is not None and expected_iterations is not None:
        expected_files = len(expected_rank_set) * expected_iterations
        if len(files) != expected_files:
            failures.append(f"found {len(files)} trace files, expected {expected_files}")
        expected_pairs = {
            (rank, iteration) for rank in expected_rank_set for iteration in iterations
        }
        observed_pairs = set(rank_iterations)
        missing_pairs = sorted(expected_pairs - observed_pairs)
        unexpected_pairs = sorted(observed_pairs - expected_pairs)
        if missing_pairs:
            failures.append(f"missing rank/iteration pairs: {missing_pairs}")
        if unexpected_pairs:
            failures.append(f"unexpected rank/iteration pairs: {unexpected_pairs}")

    if runtime_events != len(files):
        failures.append(f"found {runtime_events} runtime events for {len(files)} files")
    if non_deterministic_runtime:
        failures.append(
            f"{non_deterministic_runtime} runtime events did not enable deterministic algorithms"
        )

    unknown_trace_details = sorted(set(trace_details) - {"semantic", "summary"})
    if unknown_trace_details:
        failures.append(f"unknown trace detail levels: {unknown_trace_details}")
    if len(trace_details) != 1:
        failures.append(f"mixed trace detail levels: {dict(trace_details)}")
    summary_only = set(trace_details) == {"summary"}

    if summary_only:
        if forward_backward_ends != len(files):
            failures.append(
                f"found {forward_backward_ends} forward_backward.end events for {len(files)} files"
            )
        if optimizer_ends != len(files):
            failures.append(f"found {optimizer_ends} optimizer.end events for {len(files)} files")
    else:
        if not recomputes:
            failures.append("no checkpoint recompute events found")
        if recompute_mismatches:
            failures.append(
                f"{recompute_mismatches} checkpoint recomputes did not match original forward"
            )

    if iteration_errors:
        failures.append(f"found {iteration_errors} iteration.error events")
    if pending_collectives:
        failures.append(f"{pending_collectives} collectives were pending at trace-window end")

    for prefix in required_event_prefixes:
        if event_prefix_counts[prefix] == 0:
            failures.append(f"no events start with {prefix!r}")

    if te_backend_unavailable:
        failures.append(
            f"{te_backend_unavailable} TE attention backend selections were unavailable"
        )

    if not summary_only:
        if not completed:
            failures.append("no completed collective events found")
        if completed_without_hashes:
            failures.append(
                f"{completed_without_hashes} completed collectives are missing output hashes"
            )

    for prefix in required_collective_prefixes:
        if prefix_counts[prefix] == 0:
            failures.append(f"no collective events start with {prefix!r}")

    if require_dp_fp32_accumulation:
        if not dp_grad_begins:
            failures.append("no data-parallel gradient-reduction events found")
        elif dp_without_fp32:
            failures.append(
                f"{dp_without_fp32} data-parallel reductions did not use fp32 accumulation"
            )
    # A two-rank reduction has one floating-point addition per output element, so physical
    # topology cannot alter its accumulation order. Require the fixed hierarchy only where a
    # reduction tree can actually contain more than one addition.
    if require_dp_hierarchical_fp32_accumulation:
        if not dp_grad_begins:
            failures.append("no data-parallel gradient-reduction events found")
        elif not multi_rank_dp_grad_begins:
            failures.append("no multi-rank data-parallel gradient-reduction events found")
        elif multi_rank_dp_without_hierarchy:
            failures.append(
                f"{multi_rank_dp_without_hierarchy} multi-rank data-parallel reductions "
                "did not use the hierarchical fp32 path"
            )
    if trace_truncations:
        failures.append(f"found {trace_truncations} trace.truncated events")

    return {
        "equal": not failures,
        "path": str(trace_path.resolve()),
        "trace_files": len(files),
        "events": event_count,
        "ranks": ranks,
        "iterations": iterations,
        "file_identity_failures": file_identity_failures,
        "sequence_failures": sequence_failures,
        "runtime_count_failures": runtime_count_failures,
        "iteration_begin_count_failures": iteration_begin_count_failures,
        "iteration_end_count_failures": iteration_end_count_failures,
        "iteration_errors": iteration_errors,
        "recomputes": recomputes,
        "recompute_mismatches": recompute_mismatches,
        "collectives": collectives,
        "completed_collectives": completed,
        "collective_begins_without_end": collective_begins_without_end,
        "collective_ends_without_begin": collective_ends_without_begin,
        "pending_collectives": pending_collectives,
        "event_prefix_counts": dict(event_prefix_counts),
        "te_attention_backend_unavailable": te_backend_unavailable,
        "collective_prefix_counts": dict(prefix_counts),
        "dp_grad_reductions": dp_grad_begins,
        "dp_grad_reductions_without_fp32_accumulation": dp_without_fp32,
        "multi_rank_dp_grad_reductions": multi_rank_dp_grad_begins,
        "multi_rank_dp_grad_reductions_without_hierarchical_fp32_accumulation": (
            multi_rank_dp_without_hierarchy
        ),
        "trace_truncations": trace_truncations,
        "tensor_digest_counts": {"sha256": sha256_digests, "device_digest": device_digests},
        "trace_detail_counts": dict(trace_details),
        "forward_backward_ends": forward_backward_ends,
        "optimizer_ends": optimizer_ends,
        "diagnostic_only": summary_only or expected_rank_spec is not None or device_digests > 0,
        "failures": failures,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", help="First rank-sharded trace directory")
    parser.add_argument("right", nargs="?", help="Optional second trace directory to compare")
    rank_expectation = parser.add_mutually_exclusive_group()
    rank_expectation.add_argument("--expected-ranks", type=int)
    rank_expectation.add_argument(
        "--expected-rank-spec", help="Expected sampled ranks, for example 0,3,8-15"
    )
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
            "expected_rank_spec": args.expected_rank_spec,
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
        diagnostic_only = any(
            section.get("diagnostic_only", False)
            for section in (report.get("left", {}), report.get("right", {}))
        )
        if not args.right:
            verdict = "TRACE INVARIANTS PASSED"
        elif diagnostic_only:
            verdict = "DIAGNOSTIC MATCH"
        else:
            verdict = "CERTIFIED"
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
