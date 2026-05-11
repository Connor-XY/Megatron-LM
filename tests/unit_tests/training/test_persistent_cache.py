# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Tests for megatron.training.persistent_cache."""

import argparse
import os
import subprocess
from unittest import mock

import pytest

from megatron.training import persistent_cache


def _make_args(**overrides):
    base = dict(
        persistent_cache_read_dir=None,
        persistent_cache_write_dir=None,
        persistent_cache_scopes=['triton', 'inductor'],
        persistent_cache_writeback_every_n_saves=8,
        persistent_cache_skip_validation=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture(autouse=True)
def _reset_singleton():
    persistent_cache._singleton = None
    yield
    persistent_cache._singleton = None


# ----------------------------------------------------------------------------
# enabled / disabled gating
# ----------------------------------------------------------------------------

def test_disabled_when_no_dirs_set():
    ctrl = persistent_cache.PersistentCacheController(_make_args())
    assert ctrl.enabled is False
    # All methods safe no-ops when disabled.
    ctrl.validate()
    assert ctrl.maybe_kick_writeback(42) is None


def test_enabled_when_read_dir_set():
    ctrl = persistent_cache.PersistentCacheController(
        _make_args(persistent_cache_read_dir='/tmp/r'))
    assert ctrl.enabled is True


def test_enabled_when_write_dir_set():
    ctrl = persistent_cache.PersistentCacheController(
        _make_args(persistent_cache_write_dir='/tmp/w'))
    assert ctrl.enabled is True


# ----------------------------------------------------------------------------
# Singleton
# ----------------------------------------------------------------------------

def test_get_returns_none_when_uninitialized():
    assert persistent_cache.get() is None


def test_init_is_idempotent():
    args = _make_args(persistent_cache_read_dir='/tmp/x')
    a = persistent_cache.init(args)
    b = persistent_cache.init(args)
    assert a is b
    assert persistent_cache.get() is a


# ----------------------------------------------------------------------------
# Throttle: maybe_kick_writeback fires every Nth save
# ----------------------------------------------------------------------------

@mock.patch.dict(os.environ, {'SLURM_LOCALID': '0'})
@mock.patch('subprocess.Popen')
def test_throttles_every_n_saves(mock_popen):
    mock_popen.return_value.poll.return_value = 0  # always reports completed
    ctrl = persistent_cache.PersistentCacheController(
        _make_args(persistent_cache_write_dir='/tmp/w',
                   persistent_cache_writeback_every_n_saves=3))
    for i in range(1, 7):
        ctrl.maybe_kick_writeback(i)
    # Save count divisible by 3 fires: count=3 and count=6 → 2 Popens.
    assert mock_popen.call_count == 2


@mock.patch.dict(os.environ, {'SLURM_LOCALID': '0'})
@mock.patch('subprocess.Popen')
def test_skips_when_previous_writeback_in_flight(mock_popen):
    running = mock.MagicMock()
    running.poll.return_value = None  # still running
    mock_popen.return_value = running
    ctrl = persistent_cache.PersistentCacheController(
        _make_args(persistent_cache_write_dir='/tmp/w',
                   persistent_cache_writeback_every_n_saves=1))
    ctrl.maybe_kick_writeback(1)
    ctrl.maybe_kick_writeback(2)
    assert mock_popen.call_count == 1


@mock.patch.dict(os.environ, {'SLURM_LOCALID': '1'})
@mock.patch('subprocess.Popen')
def test_writeback_gated_to_local_rank_zero(mock_popen):
    ctrl = persistent_cache.PersistentCacheController(
        _make_args(persistent_cache_write_dir='/tmp/w',
                   persistent_cache_writeback_every_n_saves=1))
    ctrl.maybe_kick_writeback(1)
    mock_popen.assert_not_called()


@mock.patch.dict(os.environ, {'SLURM_LOCALID': '0'})
@mock.patch('subprocess.Popen')
def test_writeback_disabled_when_no_write_dir(mock_popen):
    ctrl = persistent_cache.PersistentCacheController(
        _make_args(persistent_cache_read_dir='/tmp/r'))  # read only
    ctrl.maybe_kick_writeback(1)
    mock_popen.assert_not_called()


# ----------------------------------------------------------------------------
# validate(): env-var presence, empty-seed warning
# ----------------------------------------------------------------------------

def test_validate_skipped_when_disabled(caplog):
    ctrl = persistent_cache.PersistentCacheController(_make_args())
    ctrl.validate()  # no exception, no log spam
    assert "scope" not in caplog.text


def test_validate_skipped_when_skip_validation_set(caplog):
    ctrl = persistent_cache.PersistentCacheController(
        _make_args(persistent_cache_read_dir='/tmp/r',
                   persistent_cache_skip_validation=True))
    ctrl.validate()
    assert "scope" not in caplog.text


@mock.patch.dict(os.environ, {}, clear=True)
def test_validate_warns_on_missing_env_var(caplog):
    ctrl = persistent_cache.PersistentCacheController(
        _make_args(persistent_cache_read_dir='/tmp/r'))
    with caplog.at_level('WARNING', logger='megatron.training.persistent_cache'):
        ctrl.validate()
    assert "TRITON_CACHE_DIR is unset" in caplog.text
    assert "TORCHINDUCTOR_CACHE_DIR is unset" in caplog.text


def test_validate_warns_on_empty_seeded_dir(tmp_path, caplog):
    triton_dir = tmp_path / "triton"
    triton_dir.mkdir()  # exists but empty
    with mock.patch.dict(os.environ,
                         {'TRITON_CACHE_DIR': str(triton_dir),
                          'TORCHINDUCTOR_CACHE_DIR': str(triton_dir)},
                         clear=True):
        ctrl = persistent_cache.PersistentCacheController(
            _make_args(persistent_cache_read_dir='/tmp/r'))
        with caplog.at_level('WARNING', logger='megatron.training.persistent_cache'):
            ctrl.validate()
    assert "exists but is empty" in caplog.text


def test_validate_silent_on_populated_seed(tmp_path, caplog):
    triton_dir = tmp_path / "triton"
    triton_dir.mkdir()
    (triton_dir / "kernel.cubin").write_bytes(b"fake")
    with mock.patch.dict(os.environ,
                         {'TRITON_CACHE_DIR': str(triton_dir)},
                         clear=True):
        ctrl = persistent_cache.PersistentCacheController(
            _make_args(
                persistent_cache_read_dir='/tmp/r',
                persistent_cache_scopes=['triton']))
        with caplog.at_level('WARNING', logger='megatron.training.persistent_cache'):
            ctrl.validate()
    assert "empty" not in caplog.text
    assert "unset" not in caplog.text


# ----------------------------------------------------------------------------
# Final writeback: bounded wait, terminate on hang
# ----------------------------------------------------------------------------

@mock.patch.dict(os.environ, {'SLURM_LOCALID': '0'})
@mock.patch('subprocess.run')
@mock.patch('subprocess.Popen')
def test_final_writeback_terminates_hung_in_flight(mock_popen, mock_run):
    hung = mock.MagicMock()
    hung.poll.return_value = None
    hung.wait.side_effect = subprocess.TimeoutExpired(cmd='bash', timeout=60)
    mock_popen.return_value = hung
    ctrl = persistent_cache.PersistentCacheController(
        _make_args(persistent_cache_write_dir='/tmp/w',
                   persistent_cache_writeback_every_n_saves=1))
    ctrl.maybe_kick_writeback(1)
    ctrl._final_writeback()
    hung.terminate.assert_called_once()
    mock_run.assert_called()  # final writeback still attempted


@mock.patch.dict(os.environ, {'SLURM_LOCALID': '0'})
@mock.patch('subprocess.run')
@mock.patch('subprocess.Popen')
def test_final_writeback_completes_when_in_flight_finishes(mock_popen, mock_run):
    completed = mock.MagicMock()
    completed.poll.return_value = 0
    completed.wait.return_value = 0  # returns promptly
    mock_popen.return_value = completed
    ctrl = persistent_cache.PersistentCacheController(
        _make_args(persistent_cache_write_dir='/tmp/w',
                   persistent_cache_writeback_every_n_saves=1))
    ctrl.maybe_kick_writeback(1)
    ctrl._final_writeback()
    completed.terminate.assert_not_called()
    mock_run.assert_called()


# ----------------------------------------------------------------------------
# atexit registration: only on local rank 0
# ----------------------------------------------------------------------------

@mock.patch.dict(os.environ, {'SLURM_LOCALID': '0'})
@mock.patch('atexit.register')
def test_register_atexit_fires_on_local_rank_zero(mock_register):
    ctrl = persistent_cache.PersistentCacheController(
        _make_args(persistent_cache_write_dir='/tmp/w'))
    ctrl.register_atexit()
    mock_register.assert_called_once_with(ctrl._final_writeback)


@mock.patch.dict(os.environ, {'SLURM_LOCALID': '2'})
@mock.patch('atexit.register')
def test_register_atexit_skipped_on_non_zero_local_rank(mock_register):
    ctrl = persistent_cache.PersistentCacheController(
        _make_args(persistent_cache_write_dir='/tmp/w'))
    ctrl.register_atexit()
    mock_register.assert_not_called()


@mock.patch.dict(os.environ, {'SLURM_LOCALID': '0'})
@mock.patch('atexit.register')
def test_register_atexit_skipped_when_disabled(mock_register):
    ctrl = persistent_cache.PersistentCacheController(_make_args())
    ctrl.register_atexit()
    mock_register.assert_not_called()
