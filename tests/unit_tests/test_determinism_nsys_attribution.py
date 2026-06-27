# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import sqlite3

import pytest

from tools.determinism.attribute_nsys_ranges import attribute_ranges


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
        connection.executemany(
            "INSERT INTO StringIds (id, value) VALUES (?, ?)",
            [(1, "megatron.forward"), (2, "aten::scatter_"), (3, "aten::index_put_")],
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


@pytest.mark.parametrize("pattern,max_parents", [("", 1), ("index_put", 0)])
def test_attribute_ranges_rejects_invalid_arguments(nsys_sqlite, pattern, max_parents):
    with pytest.raises(ValueError):
        attribute_ranges(nsys_sqlite, pattern, max_parents=max_parents)
