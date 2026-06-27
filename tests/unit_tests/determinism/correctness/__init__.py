# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Bit-exact correctness tests for ``--deterministic-mode``.

Two-run comparison + parametrize over preset × parallelism. The package
import sets the determinism env vars eagerly so cuBLAS / TE / NCCL capture
them at their respective first-use sites. cuBLAS / TE / NCCL defaults preserve
launcher values; the Mamba controls force deterministic kernels and disable
cold-cache Triton autotuning. If pytest collection has already touched CUDA,
the launcher's shell-side exports remain the defense in depth used by CI.
"""

from megatron.determinism_env import set_determinism_env_vars

set_determinism_env_vars()
