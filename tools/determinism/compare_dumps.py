# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Find the first semantic difference between two Megatron tensor dumps.

The training loop can already save activations, parameters, wgrads, and dgrads
as rank-sharded ``.pth`` files. This tool compares two such files or directory
trees by relative shard path and state-dict key. It deliberately ignores event
arrival order, which is not meaningful under PP/VPP interleaving.

Only load dump files produced by trusted jobs: ``torch.load`` uses pickle
serialization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


def _natural_key(value: str) -> tuple:
    """Sort embedded integers numerically (``layer.2`` before ``layer.10``)."""
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in re.split(r"(\d+)", value)
        if part
    )


def _discover_files(path: Path) -> dict[str, Path]:
    if path.is_file():
        return {"<single-file>": path}
    if not path.is_dir():
        raise FileNotFoundError(f"Dump path does not exist: {path}")
    files = {
        file.relative_to(path).as_posix(): file
        for file in sorted(path.rglob("*.pth"), key=lambda item: _natural_key(str(item)))
    }
    if not files:
        raise ValueError(f"No .pth dump files found under: {path}")
    return files


def _flatten_mapping(value: Any, prefix: tuple[str, ...] = ()) -> dict[str, Any]:
    if isinstance(value, Mapping):
        flattened = {}
        for key in sorted(value, key=lambda item: _natural_key(str(item))):
            flattened.update(_flatten_mapping(value[key], (*prefix, str(key))))
        return flattened
    return {"/".join(prefix): value}


def _load_dump(path: Path) -> dict[str, Any]:
    dump = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(dump, Mapping):
        raise TypeError(f"Expected a mapping in {path}, got {type(dump).__name__}")
    return _flatten_mapping(dump)


def _tensor_bytes(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.layout != torch.strided:
        tensor = tensor.to_dense()
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def _tensor_hash(raw_bytes: torch.Tensor) -> str:
    return hashlib.sha256(raw_bytes.numpy().tobytes()).hexdigest()


def _unravel_index(flat_index: int, shape: torch.Size) -> list[int]:
    if not shape:
        return []
    coordinates = []
    for dimension in reversed(shape):
        coordinates.append(flat_index % dimension)
        flat_index //= dimension
    return list(reversed(coordinates))


def _json_number(value: float) -> float | str:
    return value if math.isfinite(value) else repr(value)


def _numeric_error(left: torch.Tensor, right: torch.Tensor) -> tuple[float | str, float | str]:
    if left.is_complex():
        left_numeric = left.to(torch.complex128)
        right_numeric = right.to(torch.complex128)
    else:
        left_numeric = left.to(torch.float64)
        right_numeric = right.to(torch.float64)
    absolute = (left_numeric - right_numeric).abs()
    denominator = torch.maximum(left_numeric.abs(), right_numeric.abs())
    relative = absolute / torch.clamp(denominator, min=torch.finfo(torch.float64).tiny)
    absolute = torch.nan_to_num(absolute, nan=float("inf"), posinf=float("inf"))
    relative = torch.nan_to_num(relative, nan=float("inf"), posinf=float("inf"))
    return _json_number(float(absolute.max().item())), _json_number(float(relative.max().item()))


def _compare_tensors(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any] | None:
    metadata = {
        "left_shape": list(left.shape),
        "right_shape": list(right.shape),
        "left_dtype": str(left.dtype),
        "right_dtype": str(right.dtype),
    }
    if left.shape != right.shape or left.dtype != right.dtype:
        return {"reason": "tensor_metadata", **metadata}

    left_bytes = _tensor_bytes(left)
    right_bytes = _tensor_bytes(right)
    if torch.equal(left_bytes, right_bytes):
        return None

    bytes_per_element = left.element_size()
    element_mismatch = (
        left_bytes.reshape(left.numel(), bytes_per_element)
        != right_bytes.reshape(right.numel(), bytes_per_element)
    ).any(dim=1)
    mismatch_indices = element_mismatch.nonzero().reshape(-1)
    first_flat_index = int(mismatch_indices[0].item())
    left_flat = left.detach().cpu().contiguous().reshape(-1)
    right_flat = right.detach().cpu().contiguous().reshape(-1)
    max_absolute_error, max_relative_error = _numeric_error(left_flat, right_flat)

    return {
        "reason": "tensor_values",
        **metadata,
        "left_sha256": _tensor_hash(left_bytes),
        "right_sha256": _tensor_hash(right_bytes),
        "mismatch_count": int(element_mismatch.sum().item()),
        "numel": left.numel(),
        "mismatch_fraction": float(element_mismatch.float().mean().item()),
        "first_mismatch_index": _unravel_index(first_flat_index, left.shape),
        "left_first_value": repr(left_flat[first_flat_index].item()),
        "right_first_value": repr(right_flat[first_flat_index].item()),
        "max_absolute_error": max_absolute_error,
        "max_relative_error": max_relative_error,
    }


def _compare_values(left: Any, right: Any) -> dict[str, Any] | None:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return _compare_tensors(left, right)
    if type(left) is not type(right):
        return {
            "reason": "value_type",
            "left_type": type(left).__name__,
            "right_type": type(right).__name__,
        }
    try:
        equal = bool(left == right)
    except (TypeError, RuntimeError, ValueError):
        equal = False
    if equal:
        return None
    return {"reason": "non_tensor_value", "left_value": repr(left), "right_value": repr(right)}


def compare_paths(left_path: str | Path, right_path: str | Path, max_details: int = 10) -> dict:
    """Compare two dump files or directory trees and return a JSON-safe report."""
    if max_details < 1:
        raise ValueError("max_details must be at least 1")

    left_path = Path(left_path)
    right_path = Path(right_path)
    left_files = _discover_files(left_path)
    right_files = _discover_files(right_path)
    all_files = sorted(set(left_files) | set(right_files), key=_natural_key)

    report = {
        "equal": True,
        "left": str(left_path.resolve()),
        "right": str(right_path.resolve()),
        "matched_files": 0,
        "values_compared": 0,
        "divergence_count": 0,
        "divergences": [],
    }

    def add_divergence(divergence: dict) -> None:
        report["equal"] = False
        report["divergence_count"] += 1
        if len(report["divergences"]) < max_details:
            report["divergences"].append(divergence)

    for relative_path in all_files:
        if relative_path not in left_files:
            add_divergence({"file": relative_path, "reason": "file_missing_left"})
            continue
        if relative_path not in right_files:
            add_divergence({"file": relative_path, "reason": "file_missing_right"})
            continue

        report["matched_files"] += 1
        left_values = _load_dump(left_files[relative_path])
        right_values = _load_dump(right_files[relative_path])
        all_keys = sorted(set(left_values) | set(right_values), key=_natural_key)
        for key in all_keys:
            if key not in left_values:
                add_divergence({"file": relative_path, "key": key, "reason": "key_missing_left"})
                continue
            if key not in right_values:
                add_divergence({"file": relative_path, "key": key, "reason": "key_missing_right"})
                continue

            report["values_compared"] += 1
            difference = _compare_values(left_values[key], right_values[key])
            if difference is not None:
                add_divergence({"file": relative_path, "key": key, **difference})

    return report


def _print_human_report(report: dict) -> None:
    if report["equal"]:
        print(
            f"MATCH: {report['matched_files']} shard(s), "
            f"{report['values_compared']} value(s) are bit-exact"
        )
        return

    print(
        f"DIVERGED: {report['divergence_count']} difference(s) across "
        f"{report['matched_files']} matched shard(s)"
    )
    for index, divergence in enumerate(report["divergences"], start=1):
        location = divergence["file"]
        if "key" in divergence:
            location += f" :: {divergence['key']}"
        print(f"{index}. {location}")
        for key, value in divergence.items():
            if key not in ("file", "key"):
                print(f"   {key}: {value}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", help="First .pth dump or dump directory")
    parser.add_argument("right", help="Second .pth dump or dump directory")
    parser.add_argument(
        "--max-details", type=int, default=10, help="Maximum differences to print (default: 10)"
    )
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable JSON report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = compare_paths(args.left, args.right, max_details=args.max_details)
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human_report(report)
    return 0 if report["equal"] else 1


if __name__ == "__main__":
    sys.exit(main())
