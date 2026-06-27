# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Utilities for diagnosing training determinism.

Tensor dump files use PyTorch pickle serialization and must only be loaded from
trusted sources.
"""
