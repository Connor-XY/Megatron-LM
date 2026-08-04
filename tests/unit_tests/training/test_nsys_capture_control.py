# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for mutually exclusive Nsight capture-range backends."""

from unittest import mock

from megatron.training import training


class _Context:
    def __init__(self, events):
        self.events = events

    def __exit__(self, *args):
        self.events.append("context_exit")


def test_named_nvtx_capture_skips_cuda_profiler_api(monkeypatch):
    events = []
    cudart = mock.MagicMock()
    monkeypatch.setattr(training.torch.cuda, "cudart", cudart)
    monkeypatch.setattr(
        training.torch.cuda.nvtx,
        "range_push",
        lambda name: events.append(("range_push", name)),
    )
    monkeypatch.setattr(
        training.torch.cuda.nvtx,
        "range_pop",
        lambda: events.append("range_pop"),
    )

    training._start_nsys_capture("gtp_profile_window")
    training._stop_nsys_capture("gtp_profile_window", _Context(events))

    assert events == [
        ("range_push", "gtp_profile_window"),
        "context_exit",
        "range_pop",
    ]
    cudart.assert_not_called()


def test_default_capture_retains_cuda_profiler_api_order(monkeypatch):
    events = []
    cudart = mock.MagicMock()
    cudart.cudaProfilerStart.side_effect = lambda: events.append("cuda_start") or 0
    cudart.cudaProfilerStop.side_effect = lambda: events.append("cuda_stop") or 0
    monkeypatch.setattr(training.torch.cuda, "cudart", lambda: cudart)
    monkeypatch.setattr(training.torch.cuda, "check_error", lambda error: None)
    range_push = mock.MagicMock()
    range_pop = mock.MagicMock()
    monkeypatch.setattr(training.torch.cuda.nvtx, "range_push", range_push)
    monkeypatch.setattr(training.torch.cuda.nvtx, "range_pop", range_pop)

    training._start_nsys_capture(None)
    training._stop_nsys_capture(None, _Context(events))

    assert events == ["cuda_start", "cuda_stop", "context_exit"]
    range_push.assert_not_called()
    range_pop.assert_not_called()
