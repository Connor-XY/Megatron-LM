# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
import subprocess

from tools.determinism.audit_sensitive_ops import (
    main,
    operation_fingerprint,
    scan_sensitive_operations,
)


def _write_fixture(tmp_path):
    source_root = tmp_path / "megatron"
    source_root.mkdir()
    (source_root / "training.py").write_text(
        """
import torch
import torch.distributed as dist
from torch.distributed import reduce_scatter_tensor as reduce_scatter

def train_step(tensor, output, group):
    dist.all_reduce(tensor, group=group)
    reduce_scatter(output, tensor, group=group)
    tensor.scatter_add_(0, output, tensor)
    torch.nn.functional.embedding(output, tensor)
    torch.topk(tensor, 2, sorted=False)
    torch.use_deterministic_algorithms(True)
""".lstrip(),
        encoding="utf-8",
    )
    excluded = source_root / "inference"
    excluded.mkdir()
    (excluded / "sampling.py").write_text("values.scatter_(0, indices, 1)\n", encoding="utf-8")
    return source_root


def test_scan_resolves_aliases_symbols_categories_and_exclusions(tmp_path):
    operations = scan_sensitive_operations(
        _write_fixture(tmp_path), exclude_globs=("inference/**",)
    )

    assert [(operation.category, operation.call) for operation in operations] == [
        ("floating_collective_reduction", "torch.distributed.all_reduce"),
        ("floating_collective_reduction", "torch.distributed.reduce_scatter_tensor"),
        ("indexed_reduction", "tensor.scatter_add_"),
        ("indexed_reduction", "torch.nn.functional.embedding"),
        ("order_selection", "torch.topk"),
        ("determinism_control", "torch.use_deterministic_algorithms"),
    ]
    assert all(operation.path == "megatron/training.py" for operation in operations)
    assert all(operation.symbol == "train_step" for operation in operations)


def test_category_filter_and_json_report(tmp_path, capsys):
    source_root = _write_fixture(tmp_path)

    assert main([str(source_root), "--category", "indexed_reduction", "--json"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["schema_version"] == 1
    assert report["operation_count"] == 2
    assert report["source_file_count"] == 1
    assert report["operation_fingerprint"] == operation_fingerprint(
        scan_sensitive_operations(source_root, categories=("indexed_reduction",))
    )
    assert report["category_counts"] == {"indexed_reduction": 2}
    assert [operation["call"] for operation in report["operations"]] == [
        "tensor.scatter_add_",
        "torch.nn.functional.embedding",
    ]


def test_scan_filters_common_non_tensor_name_collisions(tmp_path):
    source_root = tmp_path / "megatron"
    source_root.mkdir()
    (source_root / "control_plane.py").write_text(
        """
import asyncio

def schedule(module, queue, coroutine):
    asyncio.gather(coroutine)
    queue.put(coroutine)
    module.embedding(coroutine)
""".lstrip(),
        encoding="utf-8",
    )

    assert scan_sensitive_operations(source_root) == []


def test_git_tracked_only_ignores_untracked_sources(tmp_path):
    source_root = tmp_path / "megatron"
    source_root.mkdir()
    tracked = source_root / "tracked.py"
    tracked.write_text("tensor.index_add_(0, index, source)\n", encoding="utf-8")
    (source_root / "untracked.py").write_text(
        "torch.distributed.all_reduce(tensor)\n", encoding="utf-8"
    )
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", str(tracked)], check=True)

    operations = scan_sensitive_operations(source_root, git_tracked_only=True)

    assert [(operation.path, operation.call) for operation in operations] == [
        ("megatron/tracked.py", "tensor.index_add_")
    ]


def test_missing_or_empty_source_root_has_distinct_exit_codes(tmp_path, capsys):
    assert main([str(tmp_path / "missing")]) == 2
    assert "Source root does not exist" in capsys.readouterr().err

    source_root = tmp_path / "megatron"
    source_root.mkdir()
    (source_root / "safe.py").write_text("value = 1\n", encoding="utf-8")
    assert main([str(source_root), "--fail-empty"]) == 1
    assert capsys.readouterr().out == "TOTAL 0\n"


def test_verify_catalog_requires_exact_snapshot_and_every_source_path(tmp_path, capsys):
    source_root = _write_fixture(tmp_path)
    operations = scan_sensitive_operations(source_root)
    fingerprint = operation_fingerprint(operations)
    catalog = tmp_path / "catalog.md"
    catalog.write_text(
        f"<!-- sensitive-op-audit count={len(operations)} files=2 "
        f"fingerprint={fingerprint} -->\n"
        "megatron/training.py\nmegatron/inference/sampling.py\n",
        encoding="utf-8",
    )

    assert main([str(source_root), "--verify-catalog", str(catalog)]) == 0
    assert "CATALOG VERIFIED" in capsys.readouterr().out

    (source_root / "new_path.py").write_text(
        "torch.distributed.all_reduce(tensor)\n", encoding="utf-8"
    )
    assert main([str(source_root), "--verify-catalog", str(catalog)]) == 1
    error = capsys.readouterr().err
    assert "operation count changed" in error
    assert "operation fingerprint changed" in error
    assert "megatron/new_path.py" in error
