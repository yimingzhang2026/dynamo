# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Import one completed portable campaign; validate all evidence before publishing."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import yaml
from autoscaling_arena.worker_limits import WorkerLimits
from build_leaderboards import (
    KNOWLINE_MODEL,
    KNOWLINE_REVISION,
    validate_record,
    validate_static_latency,
)
from compact_results import compact

ROOT = Path(__file__).parent


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def portable(value):
    """Keep scientific evidence while dropping process locations and identifiers."""
    if isinstance(value, dict):
        return {
            k: portable(v)
            for k, v in value.items()
            if k
            not in {
                "hostname",
                "host",
                "pid",
                "cwd",
                "argv",
                "command",
                "resolved_config",
                "session_dir",
                "artifact_root",
                "output_dir",
                "executable",
            }
        }
    if isinstance(value, list):
        return [portable(v) for v in value]
    if isinstance(value, str) and Path(value).is_absolute():
        return Path(value).name
    return value


def static_entry(record, case, config):
    saved = record["report"]["results"][0]
    trace = saved["evaluation"]["trace"]
    hashes = (
        [s["sha256"] for s in trace["shards"]]
        if "shards" in trace
        else [trace.get("sha256")]
    )
    if (
        hashes != [case["trace_sha256"]]
        or saved["autoscaler_type"] != "static"
        or saved["evaluation"]["sla"]["ttft_ms"] != case["ttft_ms"]
        or saved["evaluation"]["sla"]["itl_ms"] != case["itl_ms"]
    ):
        raise ValueError("Static executed workload does not match " + case["case_id"])
    raw = saved["raw_metrics"]
    profile = config["slo_profiles"][0]["name"]
    metrics = {
        **raw["profiles"][profile],
        **{
            k: raw[k]
            for k in (
                "gpu_hours",
                "mean_ttft_ms",
                "p95_ttft_ms",
                "p99_ttft_ms",
                "mean_itl_ms",
            )
        },
    }
    versions = record["report"]["provenance"]
    result = {
        "case_id": case["case_id"],
        "planner": "static",
        "status": "completed",
        "metrics": metrics,
        "latency": metrics,
        "raw_metrics": raw,
        "inputs": {
            "id": case["case_id"],
            "trace_sha256": case["trace_sha256"],
            "trace_bytes": case["expanded_test_bytes"],
            "requests": case["scaled_test_requests"],
            "duration_s": case["test_duration_s"],
            "fixed_gpus": case["simulated_gpus"],
        },
        "execution": {
            "wall_time_s": saved["execution"]["wall_time_s"],
            "peak_rss_gib": saved["execution"]["peak_rss_mb"] / 1024,
        },
        "runtime": {"packages": versions.get("packages", {})},
        "request_capture": {"records": raw["attempted_requests"]},
    }
    if "aisimulate" not in result["runtime"]["packages"]:
        result["runtime"]["packages"]["aisimulate"] = versions["aisimulate_version"]
    entry = {
        k: record[k]
        for k in ("source_path", "source_sha256", "source_bytes", "configuration")
    }
    entry.update(
        result=result,
        config=config,
        receipt={
            "case_id": case["case_id"],
            "result_sha256": record["source_sha256"],
            "completed_at": datetime.now(UTC).isoformat(),
        },
    )
    if "budget_audit" in record:
        entry["budget_audit"] = copy.deepcopy(record["budget_audit"])
    validate_static_latency(entry, case)
    return entry


def experiment_setup(config, configuration, resources=None):
    backend = config["backend"]
    (controller,) = backend["autoscalers"]
    pools = ["aggregate"] if backend["topology"] == "agg" else ["prefill", "decode"]
    result = {
        "topology": backend["topology"],
        "gpus_per_worker": {
            p: backend["engines"][p]["tp_size"]
            * backend["engines"][p]["attention_dp_size"]
            for p in pools
        },
        "initial_workers": {
            p: controller["start"]["decode" if p == "aggregate" else p] for p in pools
        },
        "worker_limits": backend["worker_limits"],
        "min_workers": {
            p: controller["config"]["min_" + ("decode" if p == "aggregate" else p)]
            for p in pools
        },
        "max_workers": {
            p: controller["config"]["max_" + ("decode" if p == "aggregate" else p)]
            for p in pools
        },
        "gpu_budget": backend["gpu_budget"],
        "source_config": configuration,
    }
    if resources is not None:
        costs = result["gpus_per_worker"]
        disagg = backend["topology"] == "disagg"
        limits = WorkerLimits(**result["worker_limits"])
        common = dict(
            topology=backend["topology"],
            gpu_budget=result["gpu_budget"],
            prefill_gpus=costs.get("prefill", 0),
            decode_gpus=costs["decode" if disagg else "aggregate"],
        )

        def targets(minimum, maximum):
            return limits.targets(
                **common,
                min_prefill=minimum.get("prefill", 1),
                max_prefill=maximum.get("prefill"),
                min_decode=minimum["decode" if disagg else "aggregate"],
                max_decode=maximum["decode" if disagg else "aggregate"],
            )

        if targets(result["min_workers"], result["max_workers"]) != targets(
            resources["min_workers"], resources["max_workers"]
        ):
            raise ValueError(
                "Saved controller bounds differ from the campaign's feasible fleets"
            )
        result.update(
            min_workers=resources["min_workers"], max_workers=resources["max_workers"]
        )
    return result


def import_campaign(campaign, inputs, knowline_campaign=None):
    plan = json.loads((campaign / "plan.json").read_text())
    statuses = json.loads((campaign / "status.json").read_text())["jobs"]
    cases = {
        c["case_id"]: c
        for c in json.loads((inputs / "catalog.json").read_text())["cases"]
    }
    expected = {
        c + "-" + p
        for c in cases
        for p in ("mpc", "planner", "keda", "reactive", "static", "knowline31b")
    }
    if len(plan["jobs"]) != 360 or {j["id"] for j in plan["jobs"]} != expected:
        raise ValueError("Expected exactly 360 jobs for the complete 60-trace campaign")
    records, static, setups = [], [], {}
    for job in plan["jobs"]:
        case = cases[job["case_id"]]
        decision_summary = None
        if knowline_campaign is not None and job["planner"] == "knowline31b":
            directory = knowline_campaign / "cases" / job["case_id"]
            state = json.loads((directory / "state.json").read_text())
            if state["state"] != "completed":
                raise ValueError("Incomplete KnowLine case: " + job["case_id"])
            attempt = Path(state["result"]).parent.name
            path = directory / attempt / "results.json"
            config_path = directory / attempt / "config.yaml"
            decision_summary = state["decision_summary"]
            config_bytes = config_path.read_bytes()
            config_sha256 = hashlib.sha256(config_bytes).hexdigest()
        else:
            if statuses.get(job["id"]) != "completed":
                raise ValueError("Incomplete job: " + job["id"])
            config_path = campaign / job["config"]
            config_bytes = config_path.read_bytes()
            config_sha256 = hashlib.sha256(config_bytes).hexdigest()
            if config_sha256 != job["config_sha256"]:
                raise ValueError("Configuration changed: " + job["id"])
            path = (
                campaign / "results" / job["case_id"] / job["planner"] / "results.json"
            )
        config = yaml.safe_load(config_bytes)
        report, fingerprint = compact(path)
        if (
            report.get("summary", {}).get("status") != "ok"
            or len(report["results"]) != 1
            or report["results"][0]["status"] != "ok"
        ):
            raise ValueError("Unsuccessful result: " + job["id"])
        expected_budget = job["resources"]["gpu_budget"]
        runtime = report["results"][0].get("runtime", {})
        executed_budget = runtime.get(
            "gpu_budget", runtime.get("planner_config", {}).get("max_gpu_budget")
        )
        if (
            config["backend"]["gpu_budget"] != expected_budget
            or executed_budget != expected_budget
        ):
            raise ValueError(
                f"{job['id']}: configuration and executed GPU budget must both "
                f"match the campaign policy ({expected_budget}); got "
                f"{config['backend']['gpu_budget']} and {executed_budget}"
            )
        record = {
            "job_id": job["id"],
            "plan_job": {
                k: job[k]
                for k in ("id", "case_id", "planner", "trace_sha256", "resources")
            },
            "source": "KnowLine" if job["planner"] == "knowline31b" else "Replay",
            "source_path": "results/"
            + job["case_id"]
            + "/"
            + job["planner"]
            + "/results.json",
            "source_sha256": fingerprint,
            "source_bytes": path.stat().st_size,
            "saved_status": "completed",
            "report": report,
            "configuration": {
                "path": "configs/" + job["id"] + ".yaml",
                "sha256": config_sha256,
            },
        }
        if job["planner"] == "static":
            static.append(static_entry(record, case, config))
            continue
        if job["planner"] == "knowline31b":
            record["decision_model"] = {
                "model": KNOWLINE_MODEL,
                "revision": KNOWLINE_REVISION,
                "provider": "self_hosted",
            }
            record["decision_summary"] = copy.deepcopy(
                report["results"][0]["runtime"]["jev"]
            )
            if decision_summary is not None:
                record["decision_summary"] = decision_summary
            else:
                record["decision_summary"]["calls"] += record["decision_summary"].get(
                    "replayed_decisions", 0
                )
            setups[job["case_id"]] = experiment_setup(config, record["configuration"])
        row = validate_record(record, case, job)
        if row["status"] != "completed":
            raise ValueError(job["id"] + ": " + str(row["error"]))
        report["match"] = {
            k: v for k, v in report["match"].items() if k in ("name", "labels")
        }
        records.append(record)
    # Publication starts only after every report, config, trace and scoring contract validates.
    write(inputs / "results.json", portable({"records": records}))
    write(
        inputs / "plan.json",
        {"policy": plan["policy"], "jobs": [r["plan_job"] for r in records]},
    )
    write(
        inputs / "static-latency.json",
        portable(
            {
                "source": "Static latency",
                "campaign": "60traces-static",
                "imported_at": datetime.now(UTC).isoformat(),
                "records": static,
            }
        ),
    )
    write(inputs / "experiment-setup.json", portable({"rows": setups}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument(
        "--knowline-campaign", type=Path, help="Completed GPU campaign directory"
    )
    parser.add_argument("--inputs", type=Path, default=ROOT / "inputs")
    args = parser.parse_args()
    import_campaign(
        args.campaign.resolve(), args.inputs.resolve(), args.knowline_campaign
    )
    print(
        "Imported 328 dynamic results, 60 Static latency results and 60 experiment settings."
    )


if __name__ == "__main__":
    main()
