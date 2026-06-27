# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import sys

import pytest

from tools.determinism.profile_rank import build_command, main


def test_non_profiled_rank_executes_python_command_directly():
    command = build_command(
        ["pretrain_gpt.py", "--mock-data"], rank=1, profile_rank=0, output="/tmp/report"
    )

    assert command == [sys.executable, "pretrain_gpt.py", "--mock-data"]


def test_profiled_rank_wraps_python_command_with_nsys():
    command = build_command(
        ["pretrain_gpt.py", "--mock-data"], rank=0, profile_rank=0, output="/tmp/report"
    )

    assert command[:2] == ["nsys", "profile"]
    assert "--capture-range=cudaProfilerApi" in command
    assert "--capture-range-end=stop" in command
    assert command[-3:] == [sys.executable, "pretrain_gpt.py", "--mock-data"]


def test_main_rejects_profile_rank_outside_world(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")

    with pytest.raises(ValueError, match=r"must be in \[0, 2\)"):
        main(["--profile-rank", "2", "--output", "/tmp/report", "--", "pretrain_gpt.py"])
