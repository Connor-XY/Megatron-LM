# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from tools.determinism.compare_modprobe import compare_runs, main


def _write(root, name, lines):
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_reports_first_chronological_birth(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    common = "in0=01 in0.meta=f32:1:1:1 in0.status=ok"
    _write(
        left,
        "rank_00000.txt",
        [
            f"rank=0 seq=0 mod=layer.0|fwd call=0 {common} out0=10 out0.status=ok",
            f"rank=0 seq=1 mod=layer.1|fwd call=0 {common} out0=20 out0.status=ok",
            "rank=0 seq=2 mod=layer.2|fwd call=0 in0=20 in0.status=ok out0=30 out0.status=ok",
        ],
    )
    _write(
        right,
        "rank_00000.txt",
        [
            f"rank=0 seq=0 mod=layer.0|fwd call=0 {common} out0=10 out0.status=ok",
            f"rank=0 seq=1 mod=layer.1|fwd call=0 {common} out0=21 out0.status=ok",
            "rank=0 seq=2 mod=layer.2|fwd call=0 in0=21 in0.status=ok out0=31 out0.status=ok",
        ],
    )

    report = compare_runs(left, right)

    assert report["first_difference"]["identity"] == (0, "layer.1|fwd", 0)
    assert report["first_birth"] == report["first_difference"]
    assert report["first_birth"]["previous_matching_identity"] == (0, "layer.0|fwd", 0)
    assert report["birth_count"] == 1


def test_backward_arrival_is_not_misreported_as_birth(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write(
        left,
        "rank_00000.txt",
        ["rank=0 seq=0 mod=layer|bwd call=0 gout0=10 gout0.status=ok gin0=20 gin0.status=ok"],
    )
    _write(
        right,
        "rank_00000.txt",
        ["rank=0 seq=0 mod=layer|bwd call=0 gout0=11 gout0.status=ok gin0=21 gin0.status=ok"],
    )

    report = compare_runs(left, right)

    assert report["first_difference"]["arrived_different"] == ["gout0"]
    assert report["first_birth"] is None


def test_cli_match_and_missing_identity(tmp_path, capsys):
    left = tmp_path / "left"
    right = tmp_path / "right"
    line = "rank=0 seq=0 mod=layer|fwd call=0 in0=10 out0=20"
    _write(left, "rank_00000.txt", [line])
    _write(right, "rank_00000.txt", [line])

    assert main([str(left), str(right)]) == 0
    assert "MATCH" in capsys.readouterr().out

    _write(right, "rank_00000.txt", [])
    assert main([str(left), str(right), "--json"]) == 1
    assert '"reason": "missing_right"' in capsys.readouterr().out
