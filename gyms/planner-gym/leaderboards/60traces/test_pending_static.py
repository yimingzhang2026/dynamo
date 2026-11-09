# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retired Static scores stay blank; retained controllers keep equivalent bounds."""
import json
from pathlib import Path

import pytest
import yaml
from build_leaderboards import load_static_latency
from import_results import experiment_setup

ROOT = Path(__file__).parent
GYM = ROOT.parents[1]


@pytest.fixture
def pending(tmp_path):
    cases = {
        c["case_id"]: c
        for c in json.loads((ROOT / "inputs/catalog.json").read_text())["cases"]
    }
    records = [
        {
            "result": {"case_id": c, "planner": "static", "status": "pending"},
            "config": yaml.safe_load(
                (GYM / "configs/60traces" / (c + ".yaml")).read_text()
            ),
            "source_sha256": None,
            "configuration": {"path": c + ".yaml", "sha256": None},
        }
        for c in cases
    ]
    evidence = {
        "source": "Static latency",
        "campaign": "test",
        "imported_at": "test",
        "records": records,
    }
    (tmp_path / "static-latency.json").write_text(json.dumps(evidence))
    return tmp_path, cases, evidence


def test_all_static_replacements_have_no_scores(pending):
    path, cases, _ = pending
    result = load_static_latency(path, cases)
    assert len(result["rows"]) == 60
    assert all(
        row["status"] == "pending"
        and row["metrics"] == {}
        and row["source_sha256"] is None
        for row in result["rows"].values()
    )


def test_pending_static_cannot_retain_old_metrics(pending):
    path, cases, evidence = pending
    evidence["records"][0]["result"]["metrics"] = {"good_rate": 0.9}
    (path / "static-latency.json").write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="must not retain previous"):
        load_static_latency(path, cases)


def test_historical_bounds_are_normalized_only_when_feasible_fleets_match():
    config = {
        "backend": {
            "topology": "agg",
            "gpu_budget": 16,
            "worker_limits": {"min_total": 2, "max_total": 16},
            "engines": {"aggregate": {"tp_size": 8, "attention_dp_size": 1}},
            "autoscalers": [
                {"start": {"decode": 2}, "config": {"min_decode": 2, "max_decode": 16}}
            ],
        }
    }
    resources = {"min_workers": {"aggregate": 2}, "max_workers": {"aggregate": 2}}
    result = experiment_setup(config, {}, resources)
    assert result["max_workers"] == {"aggregate": 2}
    config["backend"]["autoscalers"][0]["config"]["min_decode"] = 3
    with pytest.raises(ValueError, match="feasible fleets"):
        experiment_setup(config, {}, resources)
