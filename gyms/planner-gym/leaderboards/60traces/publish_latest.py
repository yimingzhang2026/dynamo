# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publish only current validated results; unfinished replacements have no scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_leaderboards import export, make_report, render, validate_record
from import_results import experiment_setup, portable, static_entry

ROOT = Path(__file__).parent


def validate_static_export(value, case, resources):
    """Static shares the campaign ceiling but keeps its original resident fleet."""
    proof = value["budget_audit"]
    backend = value["config"]["backend"]
    runtime = value["report"]["results"][0]["runtime"]
    fixed = backend["autoscalers"][0]["config"]
    initial = {
        role: fixed["num_decode" if role == "aggregate" else "num_" + role]
        for role in resources["initial_workers"]
    }
    if (
        proof["gpu_budget"] != resources["gpu_budget"]
        or backend["gpu_budget"] != resources["gpu_budget"]
        or runtime["gpu_budget"] != resources["gpu_budget"]
        or proof["peak_allocated_gpus"] > case["simulated_gpus"]
        or proof["telemetry_samples"] <= 0
        or initial != resources["initial_workers"]
    ):
        raise ValueError("Static GPU audit failed: " + value["job_id"])


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def refresh(inputs, exports, *, diagnostics=()):
    jobs = {j["id"]: j for j in json.loads((inputs / "plan.json").read_text())["jobs"]}
    cases = {
        c["case_id"]: c
        for c in json.loads((inputs / "catalog.json").read_text())["cases"]
    }
    records = {
        r["job_id"]: r
        for r in json.loads((inputs / "results.json").read_text())["records"]
    }
    updates = {}
    static_path = inputs / "static-latency.json"
    static_evidence = json.loads(static_path.read_text())
    static_records = {r["result"]["case_id"]: r for r in static_evidence["records"]}
    static_updates = set()
    for directory in exports:
        for path in sorted(directory.rglob("*.json")):
            value = json.loads(path.read_text())
            if "job_id" not in value:
                continue
            job_id = value["job_id"]
            if value.get("plan_job", {}).get("planner") == "static":
                case_id = value["plan_job"]["case_id"]
                if (
                    case_id not in cases
                    or job_id != case_id + "-static"
                    or case_id in static_updates
                ):
                    raise ValueError("Unknown or duplicate Static export: " + job_id)
                validate_static_export(
                    value, cases[case_id], jobs[case_id + "-mpc"]["resources"]
                )
                static_records[case_id] = static_entry(
                    value, cases[case_id], value["config"]
                )
                static_updates.add(case_id)
                continue
            if job_id not in jobs or job_id in updates:
                raise ValueError("Unknown or duplicate exported job: " + job_id)
            job = jobs[job_id]
            budget = job["resources"]["gpu_budget"]
            proof = value["budget_audit"]
            if (
                proof["gpu_budget"] != budget
                or proof["peak_allocated_gpus"] > budget
                or proof["telemetry_samples"] <= 0
            ):
                raise ValueError("GPU audit failed: " + job_id)
            if value["config"]["backend"]["gpu_budget"] != budget:
                raise ValueError("Configured GPU budget mismatch: " + job_id)
            if (
                validate_record(value, cases[job["case_id"]], job)["status"]
                != "completed"
            ):
                raise ValueError("Result validation failed: " + job_id)
            updates[job_id] = value
    setups = json.loads((inputs / "experiment-setup.json").read_text())["rows"]
    for key, record in updates.items():
        config = record.pop("config")
        if jobs[key]["planner"] == "knowline31b":
            record["source"] = "KnowLine"
            setups[jobs[key]["case_id"]] = experiment_setup(
                config, record["configuration"], jobs[key]["resources"]
            )
        records[key] = portable(record)
    for key, job in jobs.items():
        if key not in records:
            records[key] = {
                "job_id": key,
                "plan_job": job,
                "saved_status": "pending",
                "source": "Replay",
                "source_path": "",
                "source_sha256": None,
                "source_bytes": 0,
                "configuration": {"path": f"configs/{key}.yaml", "sha256": None},
            }
        row = validate_record(records[key], cases[job["case_id"]], job)
        if row["status"] not in {"completed", "pending"}:
            raise ValueError(key + ": " + str(row["validation_issues"]))
    for case, setup in setups.items():
        setup["source_config"] = records[case + "-knowline31b"]["configuration"]
    static_evidence["records"] = [static_records[key] for key in sorted(static_records)]
    write(static_path, static_evidence)
    write(
        inputs / "results.json", {"records": [records[key] for key in sorted(records)]}
    )
    write(inputs / "experiment-setup.json", {"rows": setups})
    report = make_report(inputs, include_diagnostics=False)
    refresh_diagnostics(inputs, report, diagnostics)
    return make_report(inputs)


def timeline_explanation(saved, case, setup):
    timeline = saved["timeline"]
    stats = timeline["stats"]
    peaks = {
        role: stats[f"queued_{role}_requests"]["max"] for role in ("prefill", "decode")
    }
    if case["topology"] == "agg":
        queue = f"The queue peaked at {peaks['decode']:,.0f} requests."
    else:
        queue = f"Queues peaked at {peaks['prefill']:,.0f} prefill and {peaks['decode']:,.0f} decode requests."
    fleet = timeline["fleet_residence"][0]
    share = fleet["seconds"] / case["duration_s"]
    allocation = (
        f"{fleet['decode']} shared workers"
        if case["topology"] == "agg"
        else f"{fleet['prefill']} prefill and {fleet['decode']} decode workers"
    )
    observation = f"{queue} The most frequent ready fleet was {allocation}, present for {share:.1%} of the arrival window; the ready fleet changed {timeline['fleet_changes']:,} times."
    if setup["min_gpus"] == setup["gpu_budget"]:
        note = f"The minimum fleet uses all {setup['gpu_budget']} available GPUs. No larger fleet is feasible under this case's budget, so scaling cannot add capacity to relieve the observed queues."
    else:
        note = f"The case allows {setup['gpu_budget']} GPUs. These histories show the applied fleet and observed queues; they do not by themselves establish which alternative scaling decisions would improve SLOs."
    return observation, note


def refresh_diagnostics(inputs, report, directories=()):
    """Retain matching evidence; describe new SLO failures from measured summaries."""
    previous = (
        json.loads((inputs / "red-diagnoses.json").read_text())
        if (inputs / "red-diagnoses.json").exists()
        else {"rows": {}}
    )
    old_evidence = (
        json.loads((inputs / "diagnostic-evidence.json").read_text())["runs"]
        if (inputs / "diagnostic-evidence.json").exists()
        else {}
    )
    supplied = {}
    for directory in directories:
        for path in sorted(directory.rglob("*.json")):
            item = json.loads(path.read_text())
            key = item["job_id"]
            if key in supplied:
                raise ValueError("Duplicate timeline evidence: " + key)
            supplied[key] = item
    diagnoses, evidence = {}, {}
    for page in report["pages"].values():
        for case in page["cases"]:
            for planner in report["core"]:
                row = (
                    report["static_latency"]["rows"][case["case_id"]]
                    if planner == "static"
                    else case["rows"][planner]
                )
                if row["status"] != "completed":
                    continue
                metrics = row["metrics"]
                ttft = metrics["mean_ttft_ms"] / case["ttft_ms"]
                itl = metrics["mean_itl_ms"] / case["itl_ms"]
                if metrics["good_rate"] >= 0.2 and max(ttft, itl) < 5:
                    continue
                key = case["case_id"] + "-" + planner
                if key in supplied:
                    saved = supplied[key]
                    if (
                        saved["source_sha256"] != row["source_sha256"]
                        or saved["trace_sha256"] != case["trace_sha256"]
                        or saved["sla"] != {k: case[k] for k in ("ttft_ms", "itl_ms")}
                        or saved["metrics"] != metrics
                        or not saved.get("timeline_available")
                        or saved["timeline"]["samples"] <= 0
                    ):
                        raise ValueError(
                            "Timeline does not match current result: " + key
                        )
                    observation, note = timeline_explanation(
                        saved, case, report["experiment_setup"]["rows"][case["case_id"]]
                    )
                    evidence[key] = saved
                    diagnoses[key] = {
                        "source_sha256": row["source_sha256"],
                        "trace_sha256": case["trace_sha256"],
                        "basis": "timeline",
                        "evidence_key": key,
                        "summary": f"{metrics['good_rate']:.1%} of requests met both objectives. {observation}",
                        "note": note,
                    }
                    continue
                if (
                    previous["rows"].get(key, {}).get("source_sha256")
                    == row["source_sha256"]
                    and key in old_evidence
                ):
                    diagnoses[key], evidence[key] = (
                        previous["rows"][key],
                        old_evidence[key],
                    )
                    continue
                evidence[key] = {
                    "source_path": row["source_path"],
                    "source_sha256": row["source_sha256"],
                    "trace_sha256": case["trace_sha256"],
                    "timeline_available": False,
                    "duration_s": case["duration_s"],
                    "metrics": metrics,
                    "sla": {"ttft_ms": case["ttft_ms"], "itl_ms": case["itl_ms"]},
                }
                diagnoses[key] = {
                    "source_sha256": row["source_sha256"],
                    "trace_sha256": case["trace_sha256"],
                    "basis": "summary_inference",
                    "evidence_key": key,
                    "summary": f"{metrics['good_rate']:.1%} of requests met both objectives. Mean TTFT was {ttft:.2f}× its SLO; mean ITL was {itl:.2f}× its SLO.",
                    "note": "These measured summaries identify the SLO violation. They do not establish a queueing or scaling root cause without timeline analysis.",
                }
    write(
        inputs / "red-diagnoses.json",
        {
            "policy": {"mean_severe_ratio": 5, "good_rate_severe_below": 0.2},
            "rows": diagnoses,
        },
    )
    write(inputs / "diagnostic-evidence.json", {"runs": evidence})


def publish(report, output):
    for obsolete in ("knowline-vs-jev.html", "knowline-vs-jev.csv"):
        (output / obsolete).unlink(missing_ok=True)
    export(report, output)
    template = (ROOT / "leaderboard.html").read_text()
    for filename, selected in [
        ("index.html", None),
        ("aggregated.html", "agg"),
        ("disaggregated.html", "disagg"),
        ("latency.html", "latency"),
        ("slo-rankings.html", "slo-ranking"),
    ]:
        (output / filename).write_text(render(report, template, selected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exports", type=Path, nargs="+")
    parser.add_argument("--inputs", type=Path, default=ROOT / "inputs")
    parser.add_argument("--mirror", type=Path)
    parser.add_argument("--diagnostics", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    report = refresh(
        args.inputs,
        args.exports,
        diagnostics=args.diagnostics,
    )
    publish(report, ROOT)
    if args.mirror:
        publish(report, args.mirror)
    print(json.dumps(report["evidence_counts"]))


if __name__ == "__main__":
    main()
