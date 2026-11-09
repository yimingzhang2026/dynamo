# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the complete 60-trace HTML, CSV and analysis tables from final results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from autoscaling_arena.resource_policy import (
    case_resources,
    resource_policy_description,
)

PLANNERS = {
    "mpc": "cloudAI-MPC",
    "planner": "Dynamo",
    "keda": "KEDA",
    "reactive": "Reactive",
    "static": "Static",
    "knowline31b": "KnowLine-31B",
}
CORE = ("mpc", "planner", "keda", "reactive", "static", "knowline31b")
KNOWLINE_MODEL = "PelaAI/KnowLine-31B-Gen1"
KNOWLINE_REVISION = "8f052a8130e599b53e5324c34f0f5ff90696e1c6"


def finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def close(a, b):
    return finite(a) and finite(b) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-6)


def validate_record(record, case, expected_job):
    """Reject partial, mismatched or incompatible reports before ranking."""
    job_id = expected_job["id"]
    resources = expected_job["resources"]
    report = record.get("report", {})
    records = report.get("results", [])
    row = {
        "job_id": job_id,
        "case_id": case["case_id"],
        "planner": expected_job["planner"],
        "name": PLANNERS[expected_job["planner"]],
        "status": "invalid",
        "metrics": {},
        "source": record["source"],
        "source_path": record["source_path"],
        "source_bytes": record["source_bytes"],
        "source_sha256": record["source_sha256"],
        "provenance": report.get("provenance", {}),
        "execution": {},
        "error": None,
        "validation_issues": [],
        "trace_sha256": case["trace_sha256"],
    }
    if record.get("saved_status") == "pending":
        if records:
            raise ValueError("A pending result must not retain previous scores")
        row.update(
            status="pending",
            configuration=record["configuration"],
            runtime={
                "topology": case["topology"],
                "gpu_budget": resources["gpu_budget"],
                "worker_limits": resources["worker_limits"],
            },
        )
        return row
    if len(records) != 1:
        row.update(
            error="Expected exactly one full-case result",
            validation_issues=["result_count"],
        )
        return row
    saved = records[0]
    row.update(execution=saved.get("execution", {}), error=saved.get("error"))
    if saved.get("status") != "ok" or record.get("saved_status") != "completed":
        row["status"] = (
            "failed"
            if saved.get("status") == "failed" or record.get("saved_status") == "failed"
            else "incomplete"
        )
        if not row["error"]:
            row["error"] = "Saved run is not a completed, successful result"
        return row
    issues = []

    def require(condition, message):
        if not condition:
            issues.append(message)

    summary = report.get("summary", {})
    require(
        summary.get("status") == "ok"
        and summary.get("succeeded_runs") == 1
        and summary.get("failed_runs") == 0,
        "Result summary is not one successful run",
    )
    expected_type = (
        "jev" if expected_job["planner"] == "knowline31b" else expected_job["planner"]
    )
    require(
        saved.get("autoscaler_type") == expected_type, "Controller identity mismatch"
    )
    require(report.get("match", {}).get("name") == job_id, "Case identity mismatch")
    require(
        record.get("plan_job", {}).get("trace_sha256") == case["trace_sha256"],
        "Plan trace hash mismatch",
    )
    evaluation = saved.get("evaluation", {})
    trace = evaluation.get("trace", {})
    hashes = (
        [p.get("sha256") for p in trace["shards"]]
        if "shards" in trace
        else [trace.get("sha256")]
    )
    require(hashes == [case["trace_sha256"]], "Executed trace hash mismatch")
    require(
        evaluation.get("max_requests") is None
        and evaluation.get("arrival_speedup") == 1,
        "Truncated or accelerated trace",
    )
    slo = evaluation.get("sla", {})
    require(
        close(slo.get("ttft_ms"), case["ttft_ms"])
        and close(slo.get("itl_ms"), case["itl_ms"]),
        "Latency objectives mismatch",
    )
    raw = saved.get("raw_metrics", {})
    require(
        raw.get("scorecard_schema_version") == 3
        and raw.get("short_output_itl") == "skip",
        "Scoring contract mismatch",
    )
    require(
        raw.get("attempted_requests") == case["scaled_test_requests"],
        "Full request count mismatch",
    )
    measurement = raw.get("measurement", {})
    require(
        measurement.get("basis") == "arrival_window"
        and measurement.get("completion_policy") == "include_drain"
        and measurement.get("gpu_hours_basis") == "lifecycle_window",
        "Measurement basis mismatch",
    )
    require(
        close(measurement.get("start_s"), 0)
        and close(measurement.get("end_s"), case["test_duration_s"]),
        "Full measurement window mismatch",
    )
    runtime = saved.get("runtime", {})
    require(
        runtime.get("gpu_budget") == resources["gpu_budget"],
        "Campaign GPU budget mismatch",
    )
    require(runtime.get("topology") == case["topology"], "Serving topology mismatch")
    require(
        runtime.get("worker_limits") == resources["worker_limits"],
        "Worker limits mismatch",
    )
    if expected_job["planner"] == "knowline31b":
        adapter = runtime.get("autoscaler_config", {})
        model = record.get("decision_model", {})
        decisions = record.get("decision_summary", {})
        last_attempt = runtime.get("jev", {})
        require(
            saved.get("autoscaler") == "KnowLine-31B",
            "KnowLine controller label mismatch",
        )
        require(
            adapter.get("model") == KNOWLINE_MODEL
            and adapter.get("provider") == "self_hosted"
            and model.get("model") == KNOWLINE_MODEL
            and model.get("provider") == "self_hosted",
            "KnowLine decision model/provider mismatch",
        )
        require(
            model.get("revision") == KNOWLINE_REVISION
            and report.get("match", {}).get("labels", {}).get("model_revision")
            == KNOWLINE_REVISION,
            "KnowLine model revision mismatch",
        )
        require(
            decisions.get("errors") == 0
            and decisions.get("models") == [KNOWLINE_MODEL]
            and (
                decisions.get("calls", 0) > 0
                or (
                    resources["feasible_fleet_count"] == 1
                    and decisions.get("forced_decisions", 0) > 0
                )
            )
            and last_attempt.get("errors") == 0,
            "KnowLine decision evidence missing or contains errors",
        )
        require(
            sum(
                last_attempt.get(key, 0)
                for key in ("calls", "replayed_decisions", "forced_decisions")
            )
            == sum(
                decisions.get(key, 0)
                for key in ("calls", "replayed_decisions", "forced_decisions")
            ),
            "KnowLine cumulative decision count mismatch",
        )
    metrics = saved.get("metrics", {})
    for field in ["goodput_per_gpu", "goodput_rps", "good_rate", "gpu_hours"]:
        require(
            finite(metrics.get(field)) and metrics[field] >= 0,
            "Invalid metric: " + field,
        )
    if finite(metrics.get("good_rate")):
        require(metrics["good_rate"] <= 1, "SLO success fraction exceeds one")
    require(
        finite(metrics.get("gpu_hours")) and metrics["gpu_hours"] > 0,
        "GPU-hour denominator must be positive",
    )
    if (
        all(
            finite(metrics.get(k))
            for k in ["goodput_per_gpu", "goodput_rps", "gpu_hours"]
        )
        and metrics["gpu_hours"] > 0
    ):
        expected = (
            metrics["goodput_rps"]
            * case["test_duration_s"]
            / 3600
            / metrics["gpu_hours"]
        )
        require(
            close(metrics["goodput_per_gpu"], expected),
            "Efficiency disagrees with GPU-hour normalization",
        )
    if issues:
        row.update(error="; ".join(issues), validation_issues=issues)
        return row
    row.update(
        status="completed",
        metrics=metrics,
        error=None,
        measurement=measurement,
        attempted_requests=raw["attempted_requests"],
        completed_requests=raw.get("completed_requests"),
        runtime=runtime,
        evaluation=evaluation,
    )
    if expected_job["planner"] == "knowline31b":
        row.update(
            decision_model=record["decision_model"],
            decision_summary=record["decision_summary"],
            configuration=record["configuration"],
        )
    return row


def validate_static_latency(entry, case):
    """Validate the current fixed-fleet baseline for every score and latency view."""
    saved, config = entry["result"], entry["config"]
    job, raw, metrics = saved["inputs"], saved["raw_metrics"], saved["metrics"]
    backend, defaults = config["backend"], config["evaluations"]["defaults"]
    controllers, profiles = backend["autoscalers"], config["slo_profiles"]
    issues = []

    def require(condition, message):
        if not condition:
            issues.append(message)

    require(
        saved.get("case_id")
        == job.get("id")
        == entry["receipt"].get("case_id")
        == case["case_id"]
        and saved.get("planner") == "static"
        and saved.get("status") == "completed",
        "identity/status",
    )
    require(
        entry["source_sha256"] == entry["receipt"]["result_sha256"],
        "completion receipt",
    )
    require(
        job.get("trace_sha256") == case["trace_sha256"]
        and job.get("trace_bytes") == case["expanded_test_bytes"],
        "trace identity",
    )
    require(
        job.get("requests")
        == raw.get("attempted_requests")
        == raw.get("completed_requests")
        == saved["request_capture"].get("records")
        == case["scaled_test_requests"],
        "request accounting",
    )
    require(
        close(job.get("duration_s"), case["test_duration_s"])
        and close(job.get("fixed_gpus"), case["simulated_gpus"]),
        "full window/allocation",
    )
    require(
        backend.get("topology") == case["topology"]
        and len(controllers) == 1
        and controllers[0].get("type") == "static",
        "fixed controller/topology",
    )
    fixed = controllers[0].get("config", {}) if len(controllers) == 1 else {}
    require(
        sum(fixed.get(k, 0) or 0 for k in ("num_prefill", "num_decode"))
        == (2 if case["deployment_size"] == "small" else 16),
        "fixed replica count",
    )
    require(
        len(profiles) == 1
        and close(profiles[0].get("ttft_ms"), case["ttft_ms"])
        and close(profiles[0].get("itl_ms"), case["itl_ms"])
        and profiles[0].get("e2e_ms") is None,
        "latency objectives",
    )
    require(
        defaults.get("arrival_speedup") == 1 and defaults.get("max_requests") is None,
        "truncated/accelerated replay",
    )
    measurement = raw.get("measurement", {})
    require(
        raw.get("scorecard_schema_version") == 3
        and raw.get("short_output_itl") == "skip"
        and measurement.get("basis") == "arrival_window"
        and measurement.get("completion_policy") == "include_drain"
        and measurement.get("gpu_hours_basis") == "lifecycle_window"
        and close(measurement.get("start_s"), 0)
        and close(measurement.get("end_s"), case["test_duration_s"]),
        "scoring contract",
    )
    require(
        raw.get("scale_events") == raw.get("oscillation_count") == 0,
        "unexpected scaling",
    )
    require(
        close(
            metrics.get("gpu_hours"),
            case["simulated_gpus"] * case["test_duration_s"] / 3600,
        ),
        "fixed GPU accounting",
    )
    for key in (
        "good_rate",
        "goodput_rps",
        "goodput_per_gpu",
        "mean_ttft_ms",
        "p99_ttft_ms",
        "mean_itl_ms",
    ):
        require(finite(metrics.get(key)) and metrics[key] >= 0, "invalid " + key)
    require(
        finite(metrics.get("good_rate")) and metrics["good_rate"] <= 1,
        "invalid good rate",
    )
    for key in ("mean_ttft_ms", "p99_ttft_ms", "mean_itl_ms"):
        require(
            close(metrics.get(key), raw.get(key))
            and close(metrics.get(key), saved["latency"].get(key)),
            "latency summary mismatch: " + key,
        )
    scored = (
        raw.get("profiles", {}).get(profiles[0]["name"], {})
        if len(profiles) == 1
        else {}
    )
    for key in ("good_rate", "goodput_rps", "goodput_per_gpu"):
        require(close(metrics.get(key), scored.get(key)), "score mismatch: " + key)
    if finite(metrics.get("good_rate")):
        expected = (
            metrics["good_rate"]
            * case["scaled_test_requests"]
            / case["test_duration_s"]
        )
        require(
            close(metrics.get("goodput_rps"), expected)
            and close(
                metrics.get("goodput_per_gpu"), expected / case["simulated_gpus"]
            ),
            "score normalization",
        )
    if issues:
        raise ValueError(case["case_id"] + ": Static rerun " + ", ".join(issues))
    return {
        "case_id": case["case_id"],
        "job_id": case["case_id"] + "-static",
        "planner": "static",
        "name": "Static",
        "status": "completed",
        "latency_rerun": True,
        "metrics": metrics,
        "source": "Static replay",
        "source_path": entry["source_path"],
        "source_sha256": entry["source_sha256"],
        "source_bytes": entry["source_bytes"],
        "configuration": entry["configuration"],
        "trace_sha256": case["trace_sha256"],
        "measurement": measurement,
        "attempted_requests": raw["attempted_requests"],
        "completed_requests": raw["completed_requests"],
        "execution": {
            "wall_time_s": saved["execution"]["wall_time_s"],
            "peak_rss_mb": saved["execution"]["peak_rss_gib"] * 1024,
        },
        "runtime": {
            "topology": backend["topology"],
            "fixed_replicas": (
                {"aggregate": fixed["num_decode"]}
                if backend["topology"] == "agg"
                else {"prefill": fixed["num_prefill"], "decode": fixed["num_decode"]}
            ),
            "simulated_gpus": job["fixed_gpus"],
        },
        "provenance": {
            **saved["runtime"],
            "completed_at": entry["receipt"]["completed_at"],
            "request_capture": saved["request_capture"],
            "aisimulate_version": saved["runtime"]["packages"]["aisimulate"],
            "origin": "Static full-trace latency evaluation",
        },
        "error": None,
    }


def load_static_latency(inputs, cases):
    path = inputs / "static-latency.json"
    if not path.exists():
        return None
    data = path.read_bytes()
    evidence = json.loads(data)
    records = evidence["records"]
    if (
        len(records) != len(cases)
        or {r["result"]["case_id"] for r in records} != set(cases)
        or evidence.get("source") != "Static latency"
    ):
        raise ValueError("Expected exactly one Static latency rerun for all 60 cases")
    rows = {}
    for entry in records:
        case_id = entry["result"]["case_id"]
        if entry["result"].get("status") != "pending":
            rows[case_id] = validate_static_latency(entry, cases[case_id])
            continue
        if (
            set(entry["result"]) != {"case_id", "planner", "status"}
            or entry["result"]["planner"] != "static"
            or entry.get("source_sha256") is not None
            or entry.get("receipt") is not None
        ):
            raise ValueError(
                "Pending Static result must not retain previous scores or receipts"
            )
        backend = entry["config"]["backend"]
        (controller,) = backend["autoscalers"]
        fixed = controller["config"]
        pools = (
            ("aggregate",) if backend["topology"] == "agg" else ("prefill", "decode")
        )
        fleet = {
            p: fixed["num_decode" if p == "aggregate" else "num_" + p] for p in pools
        }
        gpus = sum(
            fleet[p]
            * backend["engines"][p]["tp_size"]
            * backend["engines"][p].get("attention_dp_size", 1)
            for p in pools
        )
        if (
            controller["type"] != "static"
            or backend["topology"] != cases[case_id]["topology"]
            or gpus != cases[case_id]["simulated_gpus"]
        ):
            raise ValueError(
                "Pending Static configuration differs from the original fleet"
            )
        rows[case_id] = {
            "case_id": case_id,
            "job_id": case_id + "-static",
            "planner": "static",
            "name": "Static",
            "status": "pending",
            "metrics": {},
            "source": "Static replay",
            "source_path": "",
            "source_sha256": None,
            "source_bytes": 0,
            "configuration": entry["configuration"],
            "trace_sha256": cases[case_id]["trace_sha256"],
            "runtime": {
                "topology": backend["topology"],
                "fixed_replicas": fleet,
                "simulated_gpus": gpus,
            },
            "execution": {},
            "provenance": {},
            "validation_issues": [],
            "error": None,
        }
    return {
        "source": evidence["source"],
        "campaign": evidence["campaign"],
        "imported_at": evidence["imported_at"],
        "input_manifest": {
            "path": "inputs/static-latency.json",
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        },
        "rows": rows,
    }


def descending_ranks(values):
    ordered = sorted(values, key=lambda name: (-values[name], name))
    ranks = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        for name in ordered[start:end]:
            ranks[name] = (start + 1 + end) / 2
        start = end
    return ranks


def load_diagnostics(inputs, report):
    """Bind reviewed explanations to exactly the selected red results."""
    path = inputs / "red-diagnoses.json"
    if not path.exists():
        return None
    evidence_path = inputs / "diagnostic-evidence.json"
    data, evidence_data = path.read_bytes(), evidence_path.read_bytes()
    diagnoses, evidence = json.loads(data), json.loads(evidence_data)["runs"]
    policy = {"mean_severe_ratio": 5, "good_rate_severe_below": 0.2}
    if diagnoses.get("policy") != policy:
        raise ValueError(
            "Diagnosis color policy does not match the current red thresholds"
        )
    expected = {}
    for page in report["pages"].values():
        for case in page["cases"]:
            for planner in report["core"]:
                row = (
                    report.get("static_latency", {})
                    .get("rows", {})
                    .get(case["case_id"])
                    if planner == "static"
                    else None
                ) or case["rows"][planner]
                m = row["metrics"]
                if row["status"] == "completed" and (
                    m.get("good_rate", 1) < 0.2
                    or any(
                        m.get(metric, 0) / case[slo] >= 5
                        for metric, slo in [
                            ("mean_ttft_ms", "ttft_ms"),
                            ("mean_itl_ms", "itl_ms"),
                        ]
                    )
                ):
                    expected[case["case_id"] + "-" + planner] = (case, row)
    if set(diagnoses["rows"]) != set(expected) or set(evidence) != set(expected):
        raise ValueError("Diagnoses and evidence must cover every red result exactly")
    for key, (case, row) in expected.items():
        diagnosis, saved = diagnoses["rows"][key], evidence[key]
        basis = "timeline" if saved["timeline_available"] else "summary_inference"
        if (
            not diagnosis.get("summary", "").strip()
            or not diagnosis.get("note", "").strip()
            or diagnosis.get("basis") != basis
            or diagnosis.get("evidence_key") != key
            or any(
                item.get("source_sha256") != row["source_sha256"]
                or item.get("trace_sha256") != case["trace_sha256"]
                for item in (diagnosis, saved)
            )
            or saved["sla"] != {k: case[k] for k in ("ttft_ms", "itl_ms")}
            or any(
                saved["metrics"].get(k) != row["metrics"].get(k)
                for k in ("good_rate", "mean_ttft_ms", "mean_itl_ms")
            )
        ):
            raise ValueError("Diagnosis does not match the selected result: " + key)
    return {
        **diagnoses,
        "input_manifest": [
            {
                "path": "inputs/" + p.name,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            for p, raw in [(path, data), (evidence_path, evidence_data)]
        ],
    }


def load_experiment_setup(inputs, report):
    path = inputs / "experiment-setup.json"
    if not path.exists():
        return None
    data = path.read_bytes()
    rows = json.loads(data)["rows"]
    cases = {
        c["case_id"]: c for page in report["pages"].values() for c in page["cases"]
    }
    if set(rows) != set(cases):
        raise ValueError("Experiment settings must cover every trace exactly")
    jobs = {
        j["case_id"]: j for j in json.loads((inputs / "plan.json").read_text())["jobs"]
    }
    for case_id, row in rows.items():
        c = cases[case_id]
        expected = jobs[case_id]["resources"]
        static = c["rows"]["static"]["runtime"]
        if (
            row["topology"] != c["topology"]
            or row["initial_workers"] != static["fixed_replicas"]
            or any(
                row[key] != expected[key]
                for key in (
                    "gpus_per_worker",
                    "initial_workers",
                    "min_workers",
                    "max_workers",
                    "gpu_budget",
                    "worker_limits",
                )
            )
            or any(
                c["rows"][p]["runtime"]["gpu_budget"] != expected["gpu_budget"]
                or c["rows"][p]["runtime"]["worker_limits"] != expected["worker_limits"]
                for p in report["core"]
                if p != "static"
            )
        ):
            raise ValueError(
                "Experiment settings disagree with saved evidence: " + case_id
            )
        row.update(
            baseline_gpus=expected["baseline_gpus"],
            initial_gpus=expected["baseline_gpus"],
            **{
                key: expected[key]
                for key in (
                    "gpu_budget_multiplier",
                    "gpu_budget_extra_workers",
                    "max_worker_gpus",
                    "headroom_gpus",
                )
                if key in expected
            },
            feasible_fleet_count=expected["feasible_fleet_count"],
            min_gpus=sum(
                cost * expected["min_workers"][pool]
                for pool, cost in expected["gpus_per_worker"].items()
            ),
        )
    return {
        "rows": rows,
        "budget_mismatches": {},
        "input_manifest": {
            "path": "inputs/" + path.name,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        },
    }


def ranking(rows, case_ids, planners):
    shared = [
        case
        for case in case_ids
        if all(
            rows.get(case + "-" + p, {}).get("status") == "completed" for p in planners
        )
    ]
    rank_by_case = {}
    win_credit = {p: 0.0 for p in planners}
    for case in shared:
        values = {
            p: rows[case + "-" + p]["metrics"]["goodput_per_gpu"] for p in planners
        }
        rank_by_case[case] = descending_ranks(values)
        winners = [p for p in planners if values[p] == max(values.values())]
        for p in winners:
            win_credit[p] += 1 / len(winners)
    leaders = []
    for planner in planners:
        selected = [rows[case + "-" + planner] for case in shared]
        completed = sum(
            rows.get(case + "-" + planner, {}).get("status") == "completed"
            for case in case_ids
        )
        leaders.append(
            {
                "planner": planner,
                "name": PLANNERS[planner],
                "completed": completed,
                "expected": len(case_ids),
                "compared": len(shared),
                "mean_rank": (
                    sum(rank_by_case[c][planner] for c in shared) / len(shared)
                    if shared
                    else None
                ),
                "wins": win_credit[planner],
                "mean_good_rate": (
                    sum(r["metrics"]["good_rate"] for r in selected) / len(shared)
                    if shared
                    else None
                ),
                "gpu_hours": (
                    sum(r["metrics"]["gpu_hours"] for r in selected) if shared else None
                ),
            }
        )
    leaders.sort(
        key=lambda r: (
            r["mean_rank"] if r["mean_rank"] is not None else math.inf,
            r["name"],
        )
    )
    overall = (
        descending_ranks({r["planner"]: -r["mean_rank"] for r in leaders})
        if shared
        else {}
    )
    for row in leaders:
        row["rank"] = overall.get(row["planner"])
    return {
        "shared_cases": shared,
        "excluded_cases": [c for c in case_ids if c not in shared],
        "case_count": len(case_ids),
        "rows": leaders,
        "complete": len(shared) == len(case_ids),
    }


def error_text(row):
    value = row.get("error")
    if isinstance(value, dict):
        return f"{value.get('type', 'Error')}: {value.get('message', '')}"
    return str(value or "")


def make_report(inputs, *, include_diagnostics=True):
    catalog = json.loads((inputs / "catalog.json").read_text())
    cases = {c["case_id"]: c for c in catalog["cases"]}
    plan = json.loads((inputs / "plan.json").read_text())
    jobs = {j["id"]: j for j in plan["jobs"] if j["planner"] in PLANNERS}
    expected = {
        case_id + "-" + planner
        for case_id in cases
        for planner in CORE
        if planner != "static"
    }
    if len(cases) != 60 or len(plan["jobs"]) != 300 or set(jobs) != expected:
        raise ValueError("Expected 60 traces and 300 dynamic runs")
    baseline_configs = {
        r["result"]["case_id"]: r["config"]
        for r in json.loads((inputs / "static-latency.json").read_text())["records"]
    }
    for job in jobs.values():
        if job["resources"] != case_resources(
            baseline_configs[job["case_id"]], plan["policy"]
        ):
            raise ValueError(
                "Planned resource envelope differs from the campaign policy: "
                + job["id"]
            )
    static_latency = load_static_latency(inputs, cases)
    if static_latency is None:
        raise ValueError("Current Static replay results are required for every case")
    input_manifest = [
        {
            "path": "inputs/" + filename,
            "bytes": (inputs / filename).stat().st_size,
            "sha256": hashlib.sha256((inputs / filename).read_bytes()).hexdigest(),
        }
        for filename in [
            "catalog.json",
            "plan.json",
            "static-latency.json",
            "results.json",
        ]
    ]
    records = json.loads((inputs / "results.json").read_text())["records"]
    if len(records) != len(jobs) or {r["job_id"] for r in records} != set(jobs):
        raise ValueError(
            "Expected exactly one final result per planned case/controller"
        )
    rows = {}
    for record in records:
        job = jobs[record["job_id"]]
        row = validate_record(record, cases[job["case_id"]], job)
        if row["status"] not in {"completed", "pending"}:
            raise ValueError(job["id"] + ": " + str(row["error"]))
        rows[job["id"]] = row
    for case_id in cases:
        rows[case_id + "-static"] = static_latency["rows"][case_id]
    pages = {}
    for topology, prefix, title in [
        ("agg", "A", "Aggregated serving"),
        ("disagg", "D", "Disaggregated serving"),
    ]:
        page_cases = sorted(
            (c for c in cases.values() if c["topology"] == topology),
            key=lambda c: (
                c["regime"] != "HT",
                c["deployment_size"] != "small",
                c["case_id"],
            ),
        )
        details = []
        for case in page_cases:
            case_id = case["case_id"]
            details.append(
                {
                    "case_id": case_id,
                    "dataset": case["dataset"],
                    "source_id": case["source_id"],
                    "topology": topology,
                    "regime": case["regime"],
                    "size": case["deployment_size"],
                    "duration_s": case["test_duration_s"],
                    "requests": case["scaled_test_requests"],
                    "ttft_ms": case["ttft_ms"],
                    "itl_ms": case["itl_ms"],
                    "scale_factor": case["scale_factor"],
                    "trace_sha256": case["trace_sha256"],
                    "rows": {p: rows[case_id + "-" + p] for p in CORE},
                }
            )
        views = {}
        for regime in ["all", "HT", "LC"]:
            ids = [
                c["case_id"]
                for c in page_cases
                if regime == "all" or c["regime"] == regime
            ]
            selected = [
                r for r in rows.values() if r["case_id"] in ids and r["planner"] in CORE
            ]
            views[regime] = {
                "case_ids": ids,
                "counts": dict(Counter(r["status"] for r in selected)),
                "expected": len(selected),
                "core": ranking(rows, ids, CORE),
            }
        pages[topology] = {
            "title": title,
            "prefix": prefix,
            "cases": details,
            "views": views,
        }
    main_rows = [r for r in rows.values() if r["planner"] in CORE]
    report = {
        **({"static_latency": static_latency} if static_latency else {}),
        "schema_version": 5,
        "resource_policy": {
            **plan["policy"]["resources"],
            **(
                {"worker_limits": plan["policy"]["worker_limits"]}
                if "worker_limits" in plan["policy"]
                else {}
            ),
        },
        "generated_at": datetime.now(UTC).isoformat(),
        "counts": dict(Counter(r["status"] for r in main_rows)),
        "expected": len(main_rows),
        "evidence_counts": dict(Counter(r["status"] for r in rows.values())),
        "evidence_expected": len(rows),
        "scheduled_runs": sum(j["planner"] in CORE for j in jobs.values()),
        "evidence_scheduled_runs": len(jobs),
        "static_baselines": sum(
            r["status"] == "completed" for r in static_latency["rows"].values()
        ),
        "knowline_runs": sum(j["planner"] == "knowline31b" for j in jobs.values()),
        "knowline_completed": sum(
            r["planner"] == "knowline31b" and r["status"] == "completed"
            for r in main_rows
        ),
        "case_count": len(cases),
        "planners": {p: PLANNERS[p] for p in CORE},
        "core": list(CORE),
        "pages": pages,
        "input_manifest": input_manifest,
        "methodology": [
            "Within each trace, higher goodput per average simulated GPU ranks first. The leaderboard averages these ranks; lower is better.",
            "Every shared case has equal weight. Exact ties receive average ranks and share win credit. Raw scores are available per case.",
            "Only traces completed and validated for every compared controller enter a ranking. Failure and missing metrics stay null, never zero.",
            f"Every case allows up to {resource_policy_description(plan['policy']['resources'])}. All planners start with that original fleet; Static holds it fixed. The per-case setup lists the exact worker bounds. Active, starting and draining workers share the GPU ceiling.",
            "The main comparison covers cloudAI-MPC, Dynamo, KEDA, Reactive, Static and KnowLine-31B on 60 traces. Pending results have no score and do not enter ranks.",
            "KnowLine-31B is the open decision model PelaAI/KnowLine-31B-Gen1, served through the self-hosted decision adapter. Each completed cell represents one independent full-trace result.",
            "KnowLine decides worker scaling; throughput, request latency and simulated GPU hours describe the serving workload, not KnowLine inference performance. Decision-call latency is recorded separately and is excluded from simulated time and GPU-hour rankings.",
            "Static uses the current validated fixed-allocation replay in every tab, including ranks, winner highlighting and latency.",
            "Every Static replay is checked against the trace fingerprint, full request count, SLOs, measurement window and fixed GPU allocation. The same saved result supplies all Static metrics.",
            "Autoscaling scores preserve the planned trace fingerprint, request count, arrival window, SLOs, worker limits and scorecard contract. Static holds the original 2- or 16-worker starting deployment fixed.",
            "These are saved CPU simulations, not physical GPU-serving measurements. GPU hours are simulated; execution wall time and RAM are diagnostics, not ranking inputs.",
        ],
        "issues": [r for r in main_rows if r["status"] != "completed"],
    }
    diagnostics = load_diagnostics(inputs, report) if include_diagnostics else None
    if diagnostics:
        report["diagnostics"] = diagnostics
    setup = load_experiment_setup(inputs, report)
    if setup:
        report["experiment_setup"] = setup
    return report


def write_csv(path, rows, fields):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export(report, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "leaderboards.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    summary_rows = []
    run_fields = [
        "case_id",
        "regime",
        "dataset",
        "planner",
        "status",
        "source",
        "goodput_per_gpu",
        "goodput_rps",
        "good_rate",
        "gpu_hours",
        "mean_ttft_ms",
        "p99_ttft_ms",
        "mean_itl_ms",
        "wall_time_s",
        "peak_rss_gib",
        "error",
        "source_path",
        "source_sha256",
        "dynamo_sha",
        "aisimulate_version",
        "score_is_derived",
        "derived_metrics",
        "decision_model",
        "decision_model_revision",
        "decision_calls",
        "decision_errors",
        "decision_latency_total_s",
        "decision_latency_p50_s",
        "decision_latency_p95_s",
    ]
    for topology, page in report["pages"].items():
        csv_rows = []
        for case in page["cases"]:
            for planner, row in case["rows"].items():
                execution = row.get("execution", {})
                peak = execution.get("peak_rss_mb")
                data = {
                    field: row.get("metrics", {}).get(field) for field in run_fields
                }
                data.update(
                    case_id=case["case_id"],
                    regime=case["regime"],
                    dataset=case["dataset"],
                    planner=PLANNERS[planner],
                    status=row["status"],
                    source=row.get("source"),
                    wall_time_s=execution.get("wall_time_s"),
                    peak_rss_gib=peak / 1024 if finite(peak) else None,
                    error=error_text(row),
                    source_path=row.get("source_path"),
                    source_sha256=row.get("source_sha256"),
                    dynamo_sha=row.get("provenance", {}).get("dynamo_sha"),
                    aisimulate_version=row.get("provenance", {}).get(
                        "aisimulate_version"
                    ),
                    score_is_derived=row.get("provenance", {}).get("score_is_derived"),
                    derived_metrics=";".join(row.get("derived_metrics", [])),
                    decision_model=row.get("decision_model", {}).get("model"),
                    decision_model_revision=row.get("decision_model", {}).get(
                        "revision"
                    ),
                    decision_calls=row.get("decision_summary", {}).get("calls"),
                    decision_errors=row.get("decision_summary", {}).get("errors"),
                    decision_latency_total_s=row.get("decision_summary", {}).get(
                        "total_wall_latency_s"
                    ),
                    decision_latency_p50_s=row.get("decision_summary", {}).get(
                        "p50_wall_latency_s"
                    ),
                    decision_latency_p95_s=row.get("decision_summary", {}).get(
                        "p95_wall_latency_s"
                    ),
                )
                csv_rows.append(data)
        write_csv(
            output
            / (
                "aggregated-runs.csv" if topology == "agg" else "disaggregated-runs.csv"
            ),
            csv_rows,
            run_fields,
        )
        for regime, view in page["views"].items():
            for row in view["core"]["rows"]:
                summary_rows.append(
                    {"topology": topology, "regime": regime, "cohort": "core", **row}
                )
    write_csv(
        output / "leaderboard-summary.csv",
        summary_rows,
        [
            "topology",
            "regime",
            "cohort",
            "planner",
            "name",
            "completed",
            "expected",
            "compared",
            "mean_rank",
            "wins",
            "mean_good_rate",
            "gpu_hours",
            "rank",
        ],
    )
    validation = {
        "generated_at": report["generated_at"],
        "counts": report["counts"],
        "evidence_counts": report["evidence_counts"],
        "selected_completed": report["counts"].get("completed", 0),
        "static_replays_validated": report["static_baselines"],
        "knowline_results_validated": report["knowline_completed"],
        "shared_core_cases": {
            key: page["views"]["all"]["core"]["shared_cases"]
            for key, page in report["pages"].items()
        },
        "input_manifest": report["input_manifest"],
    }
    if report.get("static_latency"):
        supplement = report["static_latency"]
        validation["static_latency"] = {
            "validated_runs": sum(
                r["status"] == "completed" for r in supplement["rows"].values()
            ),
            "input_manifest": supplement["input_manifest"],
        }
        write_csv(
            output / "static-latency-runs.csv",
            [
                {
                    "case_id": case_id,
                    "status": row["status"],
                    **row["metrics"],
                    "source_sha256": row["source_sha256"],
                }
                for case_id, row in supplement["rows"].items()
            ],
            [
                "case_id",
                "status",
                "good_count",
                "good_rate",
                "goodput_rps",
                "goodput_per_gpu",
                "goodput_source",
                "gpu_hours",
                "mean_ttft_ms",
                "p95_ttft_ms",
                "p99_ttft_ms",
                "mean_itl_ms",
                "source_sha256",
            ],
        )
    if report.get("diagnostics"):
        supplement = report["diagnostics"]
        validation["diagnostics"] = {
            "validated_red_results": len(supplement["rows"]),
            "bases": dict(Counter(r["basis"] for r in supplement["rows"].values())),
            "input_manifest": supplement["input_manifest"],
        }
    if report.get("experiment_setup"):
        supplement = report["experiment_setup"]
        validation["experiment_setup"] = {
            "validated_cases": len(supplement["rows"]),
            "case_budget_mismatches": supplement["budget_mismatches"],
            "input_manifest": supplement["input_manifest"],
        }
    (output / "validation.json").write_text(
        json.dumps(validation, indent=2, allow_nan=False) + "\n"
    )


def render(report, template, selected):
    payload = (
        json.dumps({"report": report, "selected": selected}, allow_nan=False)
        .replace("<", "\\u003c")
        .replace("&", "\\u0026")
    )
    return template.replace("__REPORT_DATA__", payload)


def write_report(report, output, *, data_only=False):
    export(report, output)
    if not data_only:
        template = Path(__file__).with_name("leaderboard.html").read_text()
        for filename, selected in [
            ("index.html", None),
            ("aggregated.html", "agg"),
            ("disaggregated.html", "disagg"),
            ("latency.html", "latency"),
            ("slo-rankings.html", "slo-ranking"),
        ]:
            (output / filename).write_text(render(report, template, selected))


def check_report(report, output, *, data_only=False):
    """Rebuild with the saved timestamp and compare every generated byte."""
    saved = json.loads((output / "leaderboards.json").read_text())
    report = {**report, "generated_at": saved["generated_at"]}
    with tempfile.TemporaryDirectory(prefix="planner-report-check-") as directory:
        stage = Path(directory)
        write_report(report, stage, data_only=data_only)
        mismatches = [
            path.name
            for path in sorted(stage.iterdir())
            if not (output / path.name).is_file()
            or path.read_bytes() != (output / path.name).read_bytes()
        ]
        if mismatches:
            raise ValueError(
                "Saved report differs from its inputs: " + ", ".join(mismatches)
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path(__file__).parent / "inputs")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent)
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify saved artifacts without changing them",
    )
    args = parser.parse_args()
    report = make_report(args.inputs)
    if args.check:
        check_report(report, args.out, data_only=args.data_only)
    else:
        write_report(report, args.out, data_only=args.data_only)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "counts": report["counts"],
                "core_shared": {
                    key: len(page["views"]["all"]["core"]["shared_cases"])
                    for key, page in report["pages"].items()
                },
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
