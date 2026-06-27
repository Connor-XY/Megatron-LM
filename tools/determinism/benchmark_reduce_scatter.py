# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Benchmark native and deterministic reduce-scatter variants on an NCCL world."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import statistics
import time
from collections.abc import Callable

import torch

from megatron.core.distributed.reduce_scatter_with_fp32_accumulation import (
    reduce_scatter_with_fp32_accumulation,
)


def _measure(
    operation: Callable[[], None], *, warmup: int, iterations: int, device: torch.device
) -> list[float]:
    measurements = []
    for index in range(warmup + iterations):
        torch.distributed.barrier()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        operation()
        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        elapsed = torch.tensor(elapsed_ms, dtype=torch.float64, device=device)
        torch.distributed.all_reduce(elapsed, op=torch.distributed.ReduceOp.MAX)
        if index >= warmup:
            measurements.append(elapsed.item())
    return measurements


def _summary(measurements: list[float]) -> dict[str, float | list[float]]:
    return {
        "min_ms": min(measurements),
        "median_ms": statistics.median(measurements),
        "max_ms": max(measurements),
        "measurements_ms": measurements,
    }


def _build_hierarchical_groups(
    *, world_size: int, rank: int, local_group_size: int
) -> tuple[torch.distributed.ProcessGroup, torch.distributed.ProcessGroup]:
    if local_group_size <= 1 or world_size % local_group_size:
        raise ValueError(
            "hierarchical group size must be greater than one and divide the world size"
        )

    local_group = None
    inter_group = None
    num_local_groups = world_size // local_group_size
    for group_index in range(num_local_groups):
        ranks = list(range(group_index * local_group_size, (group_index + 1) * local_group_size))
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            local_group = group
    for local_index in range(local_group_size):
        ranks = [
            group_index * local_group_size + local_index for group_index in range(num_local_groups)
        ]
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            inter_group = group

    assert local_group is not None and inter_group is not None
    return local_group, inter_group


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numel", type=int, default=41_943_040)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--hierarchical-group-size",
        type=int,
        default=0,
        help=(
            "Also benchmark the production two-level ordered reduction using fixed contiguous "
            "logical-rank groups of this size (0 disables it)"
        ),
    )
    parser.add_argument(
        "--order",
        choices=("native-first", "ordered-first"),
        default="native-first",
        help="Run the methods forward or in reverse to expose ordering/cache effects",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.numel <= 0 or args.warmup < 0 or args.iterations <= 0:
        raise ValueError("numel and iterations must be positive; warmup must be non-negative")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    try:
        group = torch.distributed.group.WORLD
        world_size = torch.distributed.get_world_size(group)
        rank = torch.distributed.get_rank(group)
        if args.numel % world_size:
            raise ValueError(f"numel={args.numel} is not divisible by world_size={world_size}")

        hierarchical_groups = None
        if args.hierarchical_group_size:
            hierarchical_groups = _build_hierarchical_groups(
                world_size=world_size, rank=rank, local_group_size=args.hierarchical_group_size
            )

        torch.manual_seed(1234 + rank)
        input_tensor = torch.randn(args.numel, dtype=torch.float32, device=device)
        output_tensor = torch.empty(args.numel // world_size, dtype=torch.float32, device=device)

        operations = {
            "native": lambda: torch.distributed.reduce_scatter_tensor(
                output_tensor, input_tensor, op=torch.distributed.ReduceOp.SUM, group=group
            ),
            "ordered_fp32": lambda: reduce_scatter_with_fp32_accumulation(
                output_tensor,
                input_tensor,
                op=torch.distributed.ReduceOp.SUM,
                group=group,
                async_op=False,
            ),
        }
        if hierarchical_groups is not None:
            operations["hierarchical_fp32"] = lambda: reduce_scatter_with_fp32_accumulation(
                output_tensor,
                input_tensor,
                op=torch.distributed.ReduceOp.SUM,
                group=group,
                async_op=False,
                hierarchical_groups=hierarchical_groups,
            )
        operation_names = tuple(operations)
        order = (
            operation_names if args.order == "native-first" else tuple(reversed(operation_names))
        )
        results = {}
        outputs = {}
        for name in order:
            results[name] = _summary(
                _measure(
                    operations[name], warmup=args.warmup, iterations=args.iterations, device=device
                )
            )
            outputs[name] = output_tensor.clone()

        comparisons = {}
        output_sha256_by_rank = {}
        for name in operation_names:
            local_digest = hashlib.sha256(outputs[name].cpu().numpy().tobytes()).hexdigest()
            rank_digests = [None] * world_size
            torch.distributed.all_gather_object(rank_digests, local_digest)
            output_sha256_by_rank[name] = rank_digests
            if name == "native":
                continue
            absolute_difference = (outputs[name] - outputs["native"]).abs()
            max_abs_difference = absolute_difference.max()
            max_output_magnitude = torch.maximum(
                outputs[name].abs().max(), outputs["native"].abs().max()
            )
            torch.distributed.all_reduce(max_abs_difference, op=torch.distributed.ReduceOp.MAX)
            torch.distributed.all_reduce(max_output_magnitude, op=torch.distributed.ReduceOp.MAX)
            comparisons[name] = {
                "max_abs_difference": max_abs_difference.item(),
                "max_abs_over_max_magnitude": (max_abs_difference / max_output_magnitude).item(),
            }

        rank_to_host = [None] * world_size
        torch.distributed.all_gather_object(rank_to_host, socket.gethostname())

        if rank == 0:
            report = {
                "world_size": world_size,
                "numel": args.numel,
                "dtype": str(input_tensor.dtype),
                "order": list(order),
                "hierarchical_group_size": args.hierarchical_group_size or None,
                "results": results,
                "comparisons_to_native": comparisons,
                "output_sha256_by_rank": output_sha256_by_rank,
                "rank_to_host": rank_to_host,
                "ordered_over_native": (
                    results["ordered_fp32"]["median_ms"] / results["native"]["median_ms"]
                ),
            }
            if "hierarchical_fp32" in results:
                report["hierarchical_over_native"] = (
                    results["hierarchical_fp32"]["median_ms"] / results["native"]["median_ms"]
                )
            print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
