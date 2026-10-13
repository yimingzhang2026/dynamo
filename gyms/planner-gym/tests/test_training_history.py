# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guard chronological separation and preservation of request/cache identities."""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas", reason="Requires the history extra")
pa = pytest.importorskip("pyarrow", reason="Requires the history extra")
pq = pytest.importorskip("pyarrow.parquet", reason="Requires the history extra")

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.fixture
def history_module():
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_training_history.py"
    spec = importlib.util.spec_from_file_location("training_history_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_history_rejects_test_boundary_unsorted_rows_and_bad_blocks(history_module):
    writer = history_module.HistoryWriter(io.StringIO(), "sample", 16, 10)
    writer.emit(100, 16, 8, [1], 0)
    with pytest.raises(ValueError, match="outside training"):
        writer.emit(10000, 16, 8, [2], 1)
    with pytest.raises(ValueError, match="Unsorted"):
        writer.emit(99, 16, 8, [2], 1)
    with pytest.raises(ValueError, match="Block count"):
        writer.emit(101, 17, 8, [2], 1)
    assert writer.count == 1


def test_azure_history_uses_ns_stable_order_and_excludes_test_rows(
    tmp_path, history_module
):
    source = tmp_path / "azure.csv"
    pd.DataFrame(
        {
            "TIMESTAMP": [
                "2024-01-01T00:00:00Z",
                "2024-01-01T00:00:10Z",
                "2024-01-01T00:00:09.999999Z",
                "2024-01-01T00:00:00Z",
            ],
            "ContextTokens": [16, 32, 17, 16],
            "GeneratedTokens": [8, 9, 10, 11],
        }
    ).to_csv(source, index=False)
    spec = {
        "source": {
            "source_files": [
                {"file": source.name, "sha256": history_module.digest(source)}
            ],
            "full_trace_requests": 4,
        }
    }
    stream = io.StringIO()
    writer = history_module.HistoryWriter(stream, "azure-unit", 16, 10)
    history_module.csv_history(writer, spec, tmp_path)
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert [r["source_row"] for r in rows] == [0, 3, 2]
    assert [r["timestamp"] for r in rows] == [0, 0, 9999.999]
    assert [r["input_length"] for r in rows] == [16, 16, 17]
    assert [r["hash_ids"] for r in rows] == [[1], [2], [3, 4]]
    assert writer.statistics()["requests"] == 3
    source.write_text("corrupted")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        history_module.csv_history(writer, spec, tmp_path)


def test_art_includes_earlier_boundary_row_and_preserves_prefix_reuse(
    tmp_path, history_module
):
    assets = []
    for index, timestamps in enumerate(([0, 500], [999, 1000])):
        path = tmp_path / f"day-{index}.parquet"
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "request_id": f"request-{index}-{i}",
                        "timestamp_ms": timestamp,
                        "input_length": 16,
                        "output_length": 8,
                        "hash_ids": ["shared-prefix"],
                    }
                    for i, timestamp in enumerate(timestamps)
                ]
            ),
            path,
        )
        asset = {
            "url": f"https://example.invalid/day-{index}",
            "projection_file": path.name,
            "projection_sha256": history_module.digest(path),
            "rows": 2,
        }
        path.with_suffix(".source.json").write_text(json.dumps(asset))
        assets.append(asset)
    stream = io.StringIO()
    writer = history_module.HistoryWriter(stream, "art-chat", 16, 1)
    history_module.art_history(
        writer, {"assets": assets[:1], "boundary_projection": assets[1]}, tmp_path
    )
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert [r["timestamp"] for r in rows] == [0, 500, 999]
    assert [r["hash_ids"] for r in rows] == [[1], [1], [1]]
    assert len({r["source_row"] for r in rows}) == 3
    assert rows[-1]["source_request_id"] == "request-1-0"
