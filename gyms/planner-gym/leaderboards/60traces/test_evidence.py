# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic controller evidence with real, independently validated workload fields."""

import copy
import json
from pathlib import Path

import yaml
from autoscaling_arena.resource_policy import case_resources
from build_leaderboards import KNOWLINE_MODEL, KNOWLINE_REVISION, PLANNERS

ROOT = Path(__file__).parent


def static_record(case):
    """Synthetic completed baseline, independent of campaign progress."""
    name = case["case_id"]
    config = yaml.safe_load(
        (ROOT.parents[1] / "configs/60traces" / (name + ".yaml")).read_text()
    )
    n, duration, gpus = (
        case["scaled_test_requests"],
        case["test_duration_s"],
        case["simulated_gpus"],
    )
    scores = {
        "good_count": float(n),
        "good_rate": 1.0,
        "goodput_rps": n / duration,
        "goodput_per_gpu": n / duration / gpus,
        "goodput_source": "request_records",
    }
    latency = {
        "mean_ttft_ms": 1.0,
        "p95_ttft_ms": 1.0,
        "p99_ttft_ms": 1.0,
        "mean_itl_ms": 1.0,
    }
    metrics = {**scores, **latency, "gpu_hours": gpus * duration / 3600}
    raw = {
        **latency,
        "scorecard_schema_version": 3,
        "short_output_itl": "skip",
        "attempted_requests": n,
        "completed_requests": n,
        "scale_events": 0,
        "oscillation_count": 0,
        "gpu_hours": metrics["gpu_hours"],
        "profiles": {config["slo_profiles"][0]["name"]: scores},
        "measurement": {
            "basis": "arrival_window",
            "completion_policy": "include_drain",
            "gpu_hours_basis": "lifecycle_window",
            "start_s": 0,
            "end_s": duration,
        },
    }
    return {
        "config": config,
        "source_path": "synthetic-static.json",
        "source_bytes": 1,
        "source_sha256": "synthetic-static",
        "configuration": {"path": "synthetic.yaml", "sha256": "synthetic"},
        "receipt": {
            "case_id": name,
            "result_sha256": "synthetic-static",
            "completed_at": "test",
        },
        "result": {
            "case_id": name,
            "planner": "static",
            "status": "completed",
            "metrics": metrics,
            "raw_metrics": raw,
            "latency": latency,
            "request_capture": {"records": n},
            "execution": {"wall_time_s": 1, "peak_rss_gib": 1},
            "runtime": {"packages": {"aisimulate": "synthetic"}},
            "inputs": {
                "id": name,
                "trace_sha256": case["trace_sha256"],
                "trace_bytes": case["expanded_test_bytes"],
                "requests": n,
                "duration_s": duration,
                "fixed_gpus": gpus,
            },
        },
    }


def completed_record(planner="mpc", case_id="A-HT-S01", *, policy=None):
    case = next(
        c
        for c in json.loads((ROOT / "inputs/catalog.json").read_text())["cases"]
        if c["case_id"] == case_id
    )
    static = static_record(case)
    if policy is None:
        policy = yaml.safe_load(
            (ROOT.parents[1] / "configs/60traces/comparison.yaml").read_text()
        )
    resources = case_resources(static["config"], policy)
    job = {
        "id": case_id + "-" + planner,
        "case_id": case_id,
        "planner": planner,
        "trace_sha256": case["trace_sha256"],
        "resources": resources,
    }
    decision = {
        "calls": 1,
        "errors": 0,
        "models": [KNOWLINE_MODEL],
        "replayed_decisions": 0,
    }
    runtime = {
        "topology": case["topology"],
        "gpu_budget": resources["gpu_budget"],
        "worker_limits": copy.deepcopy(resources["worker_limits"]),
        "autoscaler_config": {"model": KNOWLINE_MODEL, "provider": "self_hosted"},
        "jev": decision,
    }
    report = {
        "match": {"name": job["id"], "labels": {"model_revision": KNOWLINE_REVISION}},
        "summary": {"status": "ok", "succeeded_runs": 1, "failed_runs": 0},
        "results": [
            {
                "status": "ok",
                "autoscaler": PLANNERS[planner],
                "autoscaler_type": "jev" if planner == "knowline31b" else planner,
                "metrics": copy.deepcopy(static["result"]["metrics"]),
                "raw_metrics": copy.deepcopy(static["result"]["raw_metrics"]),
                "runtime": runtime,
                "evaluation": {
                    "trace": {"sha256": case["trace_sha256"]},
                    "max_requests": None,
                    "arrival_speedup": 1,
                    "sla": {k: case[k] for k in ("ttft_ms", "itl_ms")},
                },
            }
        ],
    }
    record = {
        "job_id": job["id"],
        "plan_job": job,
        "saved_status": "completed",
        "source": "Synthetic test fixture",
        "source_path": "fixture.json",
        "source_sha256": "test-fixture",
        "source_bytes": 1,
        "configuration": {"path": "fixture.yaml", "sha256": "test-config"},
        "report": report,
        "decision_model": {
            "model": KNOWLINE_MODEL,
            "revision": KNOWLINE_REVISION,
            "provider": "self_hosted",
        },
        "decision_summary": copy.deepcopy(decision),
    }
    return record, case, job
