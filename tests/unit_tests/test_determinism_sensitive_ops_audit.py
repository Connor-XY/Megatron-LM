# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json

from tools.determinism.audit_sensitive_ops import main, scan_sensitive_operations


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
    assert report["category_counts"] == {"indexed_reduction": 2}
    assert [operation["call"] for operation in report["operations"]] == [
        "tensor.scatter_add_",
        "torch.nn.functional.embedding",
    ]


def test_missing_or_empty_source_root_has_distinct_exit_codes(tmp_path, capsys):
    assert main([str(tmp_path / "missing")]) == 2
    assert "Source root does not exist" in capsys.readouterr().err

    source_root = tmp_path / "megatron"
    source_root.mkdir()
    (source_root / "safe.py").write_text("value = 1\n", encoding="utf-8")
    assert main([str(source_root), "--fail-empty"]) == 1
    assert capsys.readouterr().out == "TOTAL 0\n"
