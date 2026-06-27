# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json

import torch

from tools.determinism.compare_dumps import compare_paths, main


def _save_dump(root, values, filename="mp_rank_00.pth"):
    root.mkdir(parents=True, exist_ok=True)
    torch.save({"model_chunk0": values}, root / filename)


def test_equal_dump_directories(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    values = {"decoder.layers.2.mlp/output0": torch.tensor([1.0, float("nan")]), "metadata": "same"}
    _save_dump(left, values)
    _save_dump(right, values)

    report = compare_paths(left, right)

    assert report["equal"]
    assert report["matched_files"] == 1
    assert report["values_compared"] == 2


def test_reports_first_tensor_divergence_in_numeric_layer_order(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    _save_dump(
        left,
        {
            "decoder.layers.10.mlp/output0": torch.tensor([10.0]),
            "decoder.layers.2.mlp/output0": torch.tensor([2.0, 3.0], dtype=torch.bfloat16),
        },
    )
    _save_dump(
        right,
        {
            "decoder.layers.10.mlp/output0": torch.tensor([11.0]),
            "decoder.layers.2.mlp/output0": torch.tensor([2.0, 4.0], dtype=torch.bfloat16),
        },
    )

    report = compare_paths(left, right)

    assert not report["equal"]
    assert report["divergence_count"] == 2
    first = report["divergences"][0]
    assert first["key"] == "model_chunk0/decoder.layers.2.mlp/output0"
    assert first["reason"] == "tensor_values"
    assert first["mismatch_count"] == 1
    assert first["first_mismatch_index"] == [1]
    assert first["left_sha256"] != first["right_sha256"]


def test_reports_missing_shards_and_tensor_metadata(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    _save_dump(left, {"output": torch.ones(2)}, filename="mp_rank_00.pth")
    _save_dump(left, {"output": torch.ones(2)}, filename="mp_rank_01.pth")
    _save_dump(right, {"output": torch.ones(2, 1)}, filename="mp_rank_00.pth")

    report = compare_paths(left, right)

    assert report["divergence_count"] == 2
    assert report["divergences"][0]["reason"] == "tensor_metadata"
    assert report["divergences"][1] == {"file": "mp_rank_01.pth", "reason": "file_missing_right"}


def test_cli_exit_codes_and_json_output(tmp_path, capsys):
    left = tmp_path / "left"
    right = tmp_path / "right"
    _save_dump(left, {"output": torch.tensor([1])})
    _save_dump(right, {"output": torch.tensor([2])})

    assert main([str(left), str(right), "--json"]) == 1
    output = capsys.readouterr().out
    assert '"equal": false' in output

    assert main([str(left), str(left)]) == 0
    assert capsys.readouterr().out.startswith("MATCH:")


def test_single_files_match_independent_of_filename(tmp_path):
    left = tmp_path / "run_a.pth"
    right = tmp_path / "run_b.pth"
    torch.save({"output": torch.tensor([1, 2])}, left)
    torch.save({"output": torch.tensor([1, 2])}, right)

    assert compare_paths(left, right)["equal"]


def test_empty_directories_are_invalid(tmp_path, capsys):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()

    assert main([str(left), str(right)]) == 2
    assert "No .pth dump files found" in capsys.readouterr().err


def test_json_report_handles_non_finite_error_metrics(tmp_path):
    left = tmp_path / "left.pth"
    right = tmp_path / "right.pth"
    torch.save({"output": torch.tensor([float("nan")])}, left)
    torch.save({"output": torch.tensor([1.0])}, right)

    report = compare_paths(left, right)

    assert report["divergences"][0]["max_absolute_error"] == "inf"
    json.dumps(report, allow_nan=False)
