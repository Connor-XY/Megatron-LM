# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Run one distributed rank under Nsight Systems and exec the command normally elsewhere."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def build_command(command: list[str], *, rank: int, profile_rank: int, output: str) -> list[str]:
    """Build the process command for one torchrun rank.

    Args:
        command: Python script and arguments to execute.
        rank: Global rank of the current process.
        profile_rank: Global rank to wrap with Nsight Systems.
        output: Nsight Systems report path without an extension.

    Returns:
        The direct Python command or the Nsight-wrapped command.
    """
    python_command = [sys.executable, *command]
    if rank != profile_rank:
        return python_command
    return [
        "nsys",
        "profile",
        "-t",
        "cuda,nvtx",
        "-f",
        "true",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "-o",
        output,
        *python_command,
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-rank", type=int, default=0)
    parser.add_argument("--output", required=True, help="Nsight report path without extension")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Python script and arguments")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("a Python command is required after --")

    try:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except (KeyError, ValueError) as error:
        raise RuntimeError(
            "profile_rank.py must run under torchrun with RANK and WORLD_SIZE set"
        ) from error
    if not 0 <= args.profile_rank < world_size:
        raise ValueError(f"--profile-rank must be in [0, {world_size})")

    if rank == args.profile_rank:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    process_command = build_command(
        command, rank=rank, profile_rank=args.profile_rank, output=args.output
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os.execvp(process_command[0], process_command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
