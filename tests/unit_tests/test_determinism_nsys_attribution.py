# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import sqlite3

import pytest

from tools.determinism.attribute_nsys_ranges import (
    attribute_kernels,
    attribute_ranges,
    summarize_attribution,
)


@pytest.fixture
def nsys_sqlite(tmp_path):
    path = tmp_path / "report.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute(
            """
            CREATE TABLE NVTX_EVENTS (
                start INTEGER NOT NULL,
                end INTEGER,
                globalTid INTEGER,
                text TEXT,
                textId INTEGER
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
                start INTEGER NOT NULL,
                end INTEGER NOT NULL,
                globalTid INTEGER,
                correlationId INTEGER
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                start INTEGER NOT NULL,
                end INTEGER NOT NULL,
                streamId INTEGER NOT NULL,
                correlationId INTEGER,
                globalPid INTEGER,
                demangledName INTEGER,
                shortName INTEGER
            )
            """
        )
        connection.executemany(
            "INSERT INTO StringIds (id, value) VALUES (?, ?)",
            [
                (1, "megatron.forward"),
                (2, "aten::scatter_"),
                (3, "aten::index_put_"),
                (4, "selected_update_kernel"),
                (5, "scatter_setup_kernel"),
            ],
        )
        connection.executemany(
            "INSERT INTO NVTX_EVENTS (start, end, globalTid, text, textId) VALUES (?, ?, ?, ?, ?)",
            [
                (0, 1_000_000, 7, None, 1),
                (100_000, 900_000, 7, None, 2),
                (200_000, 500_000, 7, None, 3),
                (250_000, 450_000, 8, "aten::index_put_, other thread", None),
            ],
        )
        connection.executemany(
            """
            INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME
                (start, end, globalTid, correlationId)
            VALUES (?, ?, ?, ?)
            """,
            [(120_000, 130_000, 7, 10), (250_000, 260_000, 7, 11), (300_000, 310_000, 8, 12)],
        )
        connection.executemany(
            """
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL
                (start, end, streamId, correlationId, globalPid, demangledName, shortName)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (1_000_000, 1_050_000, 3, 10, 0, 5, 5),
                (1_100_000, 1_200_000, 7, 11, 0, 4, 4),
                (1_300_000, 1_500_000, 9, 12, 0, 4, 4),
            ],
        )
    return path


def test_attribute_ranges_returns_nearest_parent_chain(nsys_sqlite):
    report = attribute_ranges(nsys_sqlite, "aten::index_put_", max_parents=4)

    assert report["match_count"] == 2
    assert report["total_duration_ms"] == pytest.approx(0.5)
    assert report["events"][0]["parents"] == [
        {"name": "aten::scatter_", "duration_ms": 0.8},
        {"name": "megatron.forward", "duration_ms": 1.0},
    ]
    assert report["events"][1]["parents"] == []


def test_summarize_attribution_groups_canonical_parent_names():
    report = {
        "sqlite": "report.sqlite",
        "pattern": "aten::fill_",
        "match_count": 3,
        "total_duration_ms": 0.6,
        "events": [
            {
                "duration_ms": 0.1,
                "parents": [{"name": "aten::zeros, op_id = 12", "duration_ms": 1.0}],
            },
            {
                "duration_ms": 0.3,
                "parents": [{"name": "aten::zeros, seq = 7, op_id = 14", "duration_ms": 1.0}],
            },
            {"duration_ms": 0.2, "parents": []},
        ],
    }

    summary = summarize_attribution(report)

    assert summary["groups"] == [
        {"name": "aten::zeros", "match_count": 2, "duration_ms": pytest.approx(0.4)},
        {"name": "<unattributed>", "match_count": 1, "duration_ms": pytest.approx(0.2)},
    ]


def test_summarize_attribution_selects_parent_depth(nsys_sqlite):
    report = attribute_ranges(nsys_sqlite, "aten::index_put_", max_parents=4)

    summary = summarize_attribution(report, parent_depth=2)

    assert summary["groups"] == [
        {"name": "megatron.forward", "match_count": 1, "duration_ms": pytest.approx(0.3)},
        {"name": "<unattributed>", "match_count": 1, "duration_ms": pytest.approx(0.2)},
    ]


def test_attribute_kernels_deduplicates_nested_matching_ranges(nsys_sqlite):
    report = attribute_kernels(nsys_sqlite, "aten::")

    assert report["match_count"] == 3
    assert report["kernel_launch_count"] == 3
    assert report["total_kernel_duration_ms"] == pytest.approx(0.35)
    assert report["kernels"] == [
        {
            "name": "selected_update_kernel",
            "launch_count": 2,
            "duration_ms": pytest.approx(0.3),
            "stream_ids": [7, 9],
            "target_ranges": [
                "aten::index_put_",
                "aten::index_put_, other thread",
                "aten::scatter_",
            ],
        },
        {
            "name": "scatter_setup_kernel",
            "launch_count": 1,
            "duration_ms": pytest.approx(0.05),
            "stream_ids": [3],
            "target_ranges": ["aten::scatter_"],
        },
    ]


def test_attribute_kernels_rejects_missing_activity_tables(tmp_path):
    path = tmp_path / "incomplete.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")

    with pytest.raises(ValueError, match="missing kernel-attribution tables"):
        attribute_kernels(path, "aten::")


@pytest.mark.parametrize("pattern,max_parents", [("", 1), ("index_put", 0)])
def test_attribute_ranges_rejects_invalid_arguments(nsys_sqlite, pattern, max_parents):
    with pytest.raises(ValueError):
        attribute_ranges(nsys_sqlite, pattern, max_parents=max_parents)


def test_summarize_attribution_rejects_invalid_parent_depth(nsys_sqlite):
    report = attribute_ranges(nsys_sqlite, "aten::index_put_")
    with pytest.raises(ValueError, match="parent_depth must be positive"):
        summarize_attribution(report, parent_depth=0)
