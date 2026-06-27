# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Lightweight, import-safe environment setup for deterministic execution."""

import os


def set_determinism_env_vars() -> None:
    """Set determinism controls that must precede CUDA and kernel-module imports.

    The cuBLAS, Transformer Engine, and NCCL defaults preserve an explicit
    launcher value. Mamba determinism is fail-closed: a cold Triton autotune
    cache can select different reduction tilings, so deterministic execution
    always disables autotuning and enables the package's deterministic kernels.
    """
    os.environ.setdefault("NCCL_ALGO", "Ring")
    os.environ.setdefault("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["MAMBA_DETERMINISTIC"] = "1"
    os.environ["TRITON_CACHE_AUTOTUNING"] = "0"
