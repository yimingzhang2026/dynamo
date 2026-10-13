# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The versioned benchmark must work without an implicit workload registry."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import yaml
from autoscaling_arena.suite_config import load_suite_config

ROOT = Path(__file__).resolve().parents[1]


def test_production_registry_is_empty_in_a_fresh_process():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from autoscaling_arena.workloads import WORKLOADS; assert not WORKLOADS",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_all_cases_use_explicit_test_inputs_and_consistent_groups(tmp_path):
    source = ROOT / "configs/60traces"
    fixture = tmp_path / "input.jsonl"
    fixture.write_text(
        '{"timestamp":0,"input_length":16,"output_length":8,"hash_ids":[1]}\n'
    )
    catalog = json.loads((source / "catalog.json").read_text())
    expected = {case["case_id"] for case in catalog["cases"]}
    for case in catalog["cases"]:
        config = yaml.safe_load((source / f"{case['case_id']}.yaml").read_text())
        evaluations = config["evaluations"]
        assert "workloads" not in evaluations and "suites" not in evaluations
        assert evaluations["defaults"]["arrival_speedup"] == 1.0
        assert len(evaluations["traces"]) == 1
        trace = evaluations["traces"][0]
        assert "warmup" not in trace and "prepare" not in trace
        paths = trace.get("paths", [trace.get("path")])
        assert paths == [
            f"../../data/60traces/reproduction/generated/{case['case_id']}.jsonl"
        ]
        # Supply a tiny local file; no public download is needed to test config wiring.
        if "path" in trace:
            trace["path"] = str(fixture)
        else:
            trace["paths"] = [str(fixture)]
        (tmp_path / f"{case['case_id']}.yaml").write_text(yaml.safe_dump(config))
    for manifest in source.glob("suite*.yaml"):
        (tmp_path / manifest.name).write_text(manifest.read_text())
    suite = load_suite_config(tmp_path / "suite.yaml")
    assert {case.case_id for case in suite.cases} == expected
    assert sum(case.expected_runs for case in suite.cases) == 60
    groups = Counter("-".join(case.case_id.split("-")[:2]) for case in suite.cases)
    assert groups == {"A-HT": 15, "A-LC": 15, "D-HT": 15, "D-LC": 15}
    for group in groups:
        selected = load_suite_config(tmp_path / f"suite.{group}.yaml")
        assert len(selected.cases) == 15
        assert all(case.case_id.startswith(group + "-") for case in selected.cases)


def test_inventory_matches_case_recipes_and_separate_history():
    catalog = json.loads((ROOT / "configs/60traces/catalog.json").read_text())
    histories = json.loads((ROOT / "configs/60traces/history.json").read_text())[
        "histories"
    ]
    with (ROOT / "docs/traces-60.csv").open() as stream:
        rows = {row["case_id"]: row for row in csv.DictReader(stream)}
    assert len(rows) == 60
    for case in catalog["cases"]:
        row = rows[case["case_id"]]
        assert int(row["test_bytes"]) == case["preparation"]["expected_output_bytes"]
        assert int(row["test_requests"]) == case["preparation"]["expected_requests"]
        assert row["sha256"] == case["preparation"]["expected_output_sha256"]
        history = histories.get(case["source_id"])
        assert row["training_materialized"] == str(history is not None)
        if history:
            assert history["window_end_seconds"] <= case["test_start_s"]
            assert history["last_arrival_ms"] < case["test_start_s"] * 1000
            assert row["history_sha256"] == history["output_sha256"]
            assert row["history_input_file"] != row["input_file"]
            assert row["history_scale_factor"] == "1"
            assert "rollouts/validation not defined" in row["rl_training_split"]
        else:
            assert row["rl_training_split"] == "Held-out evaluation"
        assert int(row["static_warmup_s"]) == 0
        assert (
            float(row["test_duration_s"]) == case["test_end_s"] - case["test_start_s"]
        )
