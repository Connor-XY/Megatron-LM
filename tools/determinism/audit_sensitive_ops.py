# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Inventory source operations that deserve a determinism classification.

This is a high-signal static audit, not a proof that an operation is deterministic
or non-deterministic. It finds floating-point collective reductions, rank-indexed
collectives, indexed reductions/writes, order-selection primitives, and explicit
determinism controls. Review the resulting candidates against the runtime path and
record the conclusion in ``docs/developer/determinism/op-catalog.md``.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

_DISTRIBUTED_OPERATIONS = {
    "_reduce_scatter_base": "floating_collective_reduction",
    "all_reduce": "floating_collective_reduction",
    "reduce": "floating_collective_reduction",
    "reduce_scatter": "floating_collective_reduction",
    "reduce_scatter_tensor": "floating_collective_reduction",
    "_all_gather_base": "rank_indexed_collective",
    "all_gather": "rank_indexed_collective",
    "all_gather_into_tensor": "rank_indexed_collective",
    "all_to_all": "rank_indexed_collective",
    "all_to_all_single": "rank_indexed_collective",
    "broadcast": "rank_indexed_collective",
    "gather": "rank_indexed_collective",
    "scatter": "rank_indexed_collective",
}

_TENSOR_OPERATIONS = {
    "bincount": "indexed_reduction",
    "embedding": "indexed_reduction",
    "embedding_bag": "indexed_reduction",
    "index_add": "indexed_reduction",
    "index_add_": "indexed_reduction",
    "index_reduce": "indexed_reduction",
    "index_reduce_": "indexed_reduction",
    "scatter_add": "indexed_reduction",
    "scatter_add_": "indexed_reduction",
    "scatter_reduce": "indexed_reduction",
    "scatter_reduce_": "indexed_reduction",
    "index_copy": "indexed_write_or_gather",
    "index_copy_": "indexed_write_or_gather",
    "index_put": "indexed_write_or_gather",
    "index_put_": "indexed_write_or_gather",
    "put_": "indexed_write_or_gather",
    "scatter": "indexed_write_or_gather",
    "scatter_": "indexed_write_or_gather",
    "gather": "indexed_write_or_gather",
    "index_select": "indexed_write_or_gather",
    "argsort": "order_selection",
    "sort": "order_selection",
    "topk": "order_selection",
}

_DETERMINISM_CONTROLS = {
    "are_deterministic_algorithms_enabled",
    "is_deterministic_algorithms_warn_only_enabled",
    "set_determinism_debug_mode",
    "use_deterministic_algorithms",
}


def _tensor_operation_category(call_name: str, leaf: str) -> str | None:
    """Classify a tensor operation while rejecting common non-tensor name collisions."""
    category = _TENSOR_OPERATIONS.get(leaf)
    if category is None:
        return None
    if leaf in {"embedding", "embedding_bag"} and not call_name.startswith("torch."):
        return None
    if leaf == "gather" and "asyncio" in call_name.split("."):
        return None
    return category


@dataclass(frozen=True, order=True)
class SensitiveOperation:
    """One determinism-sensitive source call."""

    path: str
    line: int
    column: int
    symbol: str
    category: str
    call: str


def _attribute_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _attribute_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


class _SensitiveOperationVisitor(ast.NodeVisitor):
    def __init__(self, path: str) -> None:
        self.path = path
        self.aliases: dict[str, str] = {}
        self.symbol_stack: list[str] = []
        self.operations: list[SensitiveOperation] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.asname:
                self.aliases[alias.asname] = alias.name
            elif "." not in alias.name:
                self.aliases[alias.name] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            for alias in node.names:
                if alias.name != "*":
                    self.aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_symbol(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_symbol(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_symbol(node)

    def _visit_symbol(self, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.symbol_stack.append(node.name)
        self.generic_visit(node)
        self.symbol_stack.pop()

    def _resolved_call_name(self, node: ast.Call) -> str | None:
        call_name = _attribute_name(node.func)
        if not call_name:
            return None
        root, separator, remainder = call_name.partition(".")
        resolved_root = self.aliases.get(root, root)
        return f"{resolved_root}.{remainder}" if separator else resolved_root

    def visit_Call(self, node: ast.Call) -> None:
        call_name = self._resolved_call_name(node)
        if call_name:
            leaf = call_name.rsplit(".", 1)[-1]
            category = None
            if call_name.startswith("torch.distributed."):
                category = _DISTRIBUTED_OPERATIONS.get(leaf)
            if category is None and leaf in _DETERMINISM_CONTROLS:
                category = "determinism_control"
            if category is None:
                category = _tensor_operation_category(call_name, leaf)
            if category is not None:
                self.operations.append(
                    SensitiveOperation(
                        path=self.path,
                        line=node.lineno,
                        column=node.col_offset,
                        symbol=".".join(self.symbol_stack) or "<module>",
                        category=category,
                        call=call_name,
                    )
                )
        self.generic_visit(node)


def scan_sensitive_operations(
    source_root: str | Path, *, categories: Iterable[str] = (), exclude_globs: Iterable[str] = ()
) -> list[SensitiveOperation]:
    """Return determinism-sensitive calls under ``source_root`` in stable order."""
    source_root = Path(source_root)
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")
    selected_categories = set(categories)
    excludes = tuple(exclude_globs)
    display_root = source_root.parent
    operations = []
    for path in sorted(source_root.rglob("*.py")):
        relative_source_path = path.relative_to(source_root).as_posix()
        if any(fnmatch.fnmatch(relative_source_path, pattern) for pattern in excludes):
            continue
        display_path = path.relative_to(display_root).as_posix()
        visitor = _SensitiveOperationVisitor(display_path)
        visitor.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        operations.extend(visitor.operations)
    if selected_categories:
        operations = [item for item in operations if item.category in selected_categories]
    return sorted(operations)


def _report(source_root: str | Path, operations: Sequence[SensitiveOperation]) -> dict:
    category_counts = Counter(operation.category for operation in operations)
    return {
        "schema_version": 1,
        "source_root": str(source_root),
        "operation_count": len(operations),
        "category_counts": dict(sorted(category_counts.items())),
        "operations": [asdict(operation) for operation in operations],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", nargs="?", default="megatron")
    parser.add_argument(
        "--category",
        action="append",
        default=[],
        choices=sorted(
            set(_DISTRIBUTED_OPERATIONS.values())
            | set(_TENSOR_OPERATIONS.values())
            | {"determinism_control"}
        ),
        help="Only emit this category; repeat to select several",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Glob relative to source_root to exclude; repeat as needed",
    )
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable report")
    parser.add_argument("--fail-empty", action="store_true", help="Return 1 when no calls match")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        operations = scan_sensitive_operations(
            args.source_root, categories=args.category, exclude_globs=args.exclude
        )
    except (FileNotFoundError, OSError, SyntaxError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    report = _report(args.source_root, operations)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for operation in operations:
            print(
                f"{operation.category}\t{operation.path}:{operation.line}"
                f"\t{operation.symbol}\t{operation.call}"
            )
        counts = ", ".join(
            f"{category}={count}" for category, count in report["category_counts"].items()
        )
        print(f"TOTAL {len(operations)}" + (f" ({counts})" if counts else ""))
    return 1 if args.fail_empty and not operations else 0


if __name__ == "__main__":
    raise SystemExit(main())
