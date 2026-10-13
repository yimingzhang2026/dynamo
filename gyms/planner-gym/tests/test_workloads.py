# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the workload generator.

Pure-Python — no Dynamo runtime needed. Verifies: registry shape, record schema,
determinism, and that each axis (arrival / shape / prefix) actually does what it
claims.
"""

from __future__ import annotations

import json
import random

import pytest
from autoscaling_arena.workloads import (
    WORKLOADS,
    Workload,
    get_workload,
    validate_mooncake_trace,
)


def _records(name, seed=0):
    return get_workload(name).generate_records(seed)


# --- arrival axis ---------------------------------------------------------


# --- shape axis -----------------------------------------------------------


# --- prefix axis ----------------------------------------------------------


# --- materialization ------------------------------------------------------


def test_materialize_synthetic_roundtrips(tmp_path):
    path = get_workload("unit-a").materialize(tmp_path, seed=0)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines == _records("unit-a", seed=0)


def test_materialize_static_returns_anchor(tmp_path):
    workload = _static_workload(tmp_path)
    assert workload.materialize(tmp_path / "out") == workload.static_trace


def test_materialize_unsorted_static_trace_creates_sorted_copy(tmp_path):
    trace = tmp_path / "unsorted.jsonl"
    trace.write_text(
        "\n".join(
            json.dumps(
                {
                    "timestamp": timestamp,
                    "input_length": 8,
                    "output_length": 2,
                    "hash_ids": [timestamp],
                }
            )
            for timestamp in (20, 10)
        )
        + "\n"
    )
    workload = Workload(
        name="recorded-trace",
        description="test trace",
        static_trace=trace,
        presorted=False,
    )

    materialized = workload.materialize(tmp_path / "out")

    assert materialized != trace
    assert [
        json.loads(line)["timestamp"] for line in materialized.read_text().splitlines()
    ] == [10, 20]


def test_force_copy_static_stages_presorted_trace(tmp_path):
    workload = _static_workload(tmp_path)

    materialized = workload.materialize(tmp_path / "staged", force_copy_static=True)

    assert materialized != workload.static_trace
    assert materialized.read_bytes() == workload.static_trace.read_bytes()


def test_validate_recorded_trace_contract(tmp_path):
    trace = tmp_path / "recorded.jsonl"
    trace.write_text(
        '{"timestamp": 0, "input_length": 65, "output_length": 2, '
        '"hash_ids": [1, 2]}\n'
    )

    assert validate_mooncake_trace(trace, block_size=64, presorted=True) == 1


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ({"timestamp": 0}, "input_length"),
        (
            {
                "timestamp": 0,
                "input_length": 65,
                "output_length": 2,
                "hash_ids": [1],
            },
            "requires 2 hashes",
        ),
    ],
)
def test_validate_recorded_trace_rejects_malformed_rows(tmp_path, record, message):
    trace = tmp_path / "invalid.jsonl"
    trace.write_text(json.dumps(record) + "\n")

    with pytest.raises(ValueError, match=message):
        validate_mooncake_trace(trace, block_size=64)


def test_validate_recorded_trace_honors_presorted_declaration(tmp_path):
    trace = tmp_path / "unsorted.jsonl"
    trace.write_text(
        "\n".join(
            json.dumps(
                {
                    "timestamp": timestamp,
                    "input_length": 8,
                    "output_length": 2,
                    "hash_ids": [timestamp],
                }
            )
            for timestamp in (10, 5)
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="timestamps are not sorted"):
        validate_mooncake_trace(trace, block_size=512, presorted=True)


# --- max_requests cap + arrival_speedup ----------------------------------


def test_max_requests_caps_synthetic_count():
    full = _records("unit-a")
    capped = get_workload("unit-a").generate_records(0, max_requests=20)
    assert len(capped) == 20
    assert capped == full[:20]  # keeps the earliest-arriving N


def test_max_requests_field_on_workload():
    wl = WORKLOADS["unit-a"]
    import dataclasses

    capped = dataclasses.replace(wl, max_requests=15)
    assert len(capped.generate_records(0)) == 15


def test_materialize_max_requests(tmp_path):
    path = get_workload("unit-a").materialize(tmp_path, seed=0, max_requests=12)
    assert len(path.read_text().splitlines()) == 12


def test_materialize_static_truncates_to_copy(tmp_path):
    workload = _static_workload(tmp_path)
    path = workload.materialize(tmp_path / "out", max_requests=2)
    assert path != workload.static_trace
    assert len(path.read_text().splitlines()) == 2


def test_arrival_speedup_compresses_timestamps(tmp_path):
    import json

    base = [
        json.loads(line)
        for line in get_workload("unit-b")
        .materialize(tmp_path / "a", seed=0)
        .read_text()
        .splitlines()
    ]
    fast = [
        json.loads(line)
        for line in get_workload("unit-b")
        .materialize(tmp_path / "b", seed=0, arrival_speedup=2.0)
        .read_text()
        .splitlines()
    ]
    assert len(base) == len(fast)  # same requests, just compressed in time
    assert (
        max(r["timestamp"] for r in fast) <= max(r["timestamp"] for r in base) / 2 + 1
    )


def _static_workload(tmp_path):
    trace = tmp_path / "input.jsonl"
    trace.write_text(
        "\n".join(
            json.dumps(
                {
                    "timestamp": timestamp,
                    "input_length": 8,
                    "output_length": 2,
                    "hash_ids": [timestamp],
                }
            )
            for timestamp in (0, 100, 200)
        )
        + "\n"
    )
    return Workload(
        name="recorded",
        description="test trace",
        static_trace=trace,
        presorted=True,
    )


@pytest.mark.parametrize("block_size", [16, 512])
def test_prefix_policy_partial_terminal_blocks_have_unique_hashes(block_size):
    from autoscaling_arena.workloads.axes import SharedPrefix

    policy = SharedPrefix(8)
    rng = random.Random(0)
    first = policy.assign(rng, block_size - 4, block_size)
    second = policy.assign(rng, block_size + 7, block_size)
    full = policy.assign(rng, block_size * 2, block_size)
    another_partial = policy.assign(rng, block_size + 7, block_size)
    assert second[0] == full[0] == another_partial[0] == 0
    assert full == [0, 1]
    assert len({first[-1], second[-1], another_partial[-1]}) == 3
    assert not ({first[-1], second[-1], another_partial[-1]} & set(range(8)))


def test_prefix_policy_hashes_keep_one_token_length_across_generated_trace():
    workload = get_workload("unit-g")
    lengths = {}
    for record in workload.generate_records(seed=0):
        for index, hash_id in enumerate(record["hash_ids"]):
            size = min(
                workload.block_size,
                record["input_length"] - index * workload.block_size,
            )
            assert size > 0
            assert lengths.setdefault(hash_id, size) == size


@pytest.mark.parametrize(
    "missing", ["arrival", "shape", "prefix_factory", "duration_s"]
)
def test_synthetic_workload_missing_axis_names_the_configuration_error(missing):
    from dataclasses import replace

    workload = replace(get_workload("unit-a"), **{missing: None})
    with pytest.raises(ValueError, match=f"missing required axes: {missing}"):
        workload.generate_records(seed=0)


def test_synthetic_workload_lists_all_missing_axes():
    with pytest.raises(ValueError, match="arrival, shape, prefix_factory, duration_s"):
        Workload(name="incomplete", description="test").generate_records(seed=0)
