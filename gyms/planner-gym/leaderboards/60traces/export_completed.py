# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reduce only successful, GPU-audited results for latest-only publication."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml
from compact_results import compact

GYM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GYM / "scripts"))
from audit_gpu_usage import audit_budget  # noqa: E402


def export(campaign, output, knowline=False):
    plan = json.loads((campaign / "plan.json").read_text())
    jobs = plan["cases"] if knowline else plan["jobs"]
    statuses = (
        {} if knowline else json.loads((campaign / "status.json").read_text())["jobs"]
    )
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    for job in jobs:
        case = job["case_id"]
        planner = "knowline31b" if knowline else job["planner"]
        job_id = case + "-" + planner
        if knowline:
            state_path = campaign / "cases" / case / "state.json"
            if not state_path.exists():
                continue
            receipt = json.loads(state_path.read_text())
            if receipt["state"] != "completed":
                continue
            path = Path(receipt["result"])
            config_path = path.parent / "config.yaml"
        else:
            directory = campaign / "results" / case / planner
            receipt_path = directory / "COMPLETE.json"
            if statuses.get(job_id) != "completed":
                continue
            receipt = (
                json.loads(receipt_path.read_text()) if receipt_path.exists() else None
            )
            path = directory / "results.json"
            config_path = campaign / job["config"]
        destination = output / (job_id + ".json")
        if destination.exists():
            count += 1
            continue
        config_bytes = config_path.read_bytes()
        config = yaml.safe_load(config_bytes)
        budget = config["backend"]["gpu_budget"]
        if not knowline:
            if hashlib.sha256(config_bytes).hexdigest() != job["config_sha256"]:
                raise ValueError("Configuration changed: " + job_id)
            if budget != job["resources"]["gpu_budget"]:
                raise ValueError("Configured GPU budget differs from plan: " + job_id)
        # A normal CPU launcher preserves its session; compact it directly.
        # Archived sessions instead carry a completion receipt from their audit.
        if receipt is None:
            receipt = {
                "budget_audit": audit_budget(campaign / job["session"], budget),
            }
        audit = receipt["budget_audit"]
        assert (
            audit["gpu_budget"] == budget
            and audit["peak_allocated_gpus"] <= budget
            and audit["telemetry_samples"] > 0
        )
        report, fingerprint = compact(path)
        (row,) = report["results"]
        assert row["status"] == "ok" and row["runtime"]["gpu_budget"] == budget
        assert (
            row["raw_metrics"]["attempted_requests"]
            == row["raw_metrics"]["completed_requests"]
        )
        if not knowline and "results_sha256" in receipt:
            assert fingerprint == receipt["results_sha256"]
        value = {
            "job_id": job_id,
            "plan_job": {
                "id": job_id,
                "case_id": case,
                "planner": planner,
                "trace_sha256": job["trace_sha256"],
            },
            "source": "Replay",
            "source_path": f"results/{job_id}/results.json",
            "source_sha256": fingerprint,
            "source_bytes": path.stat().st_size,
            "saved_status": "completed",
            "report": report,
            "configuration": {
                "path": f"configs/{job_id}.yaml",
                "sha256": hashlib.sha256(config_bytes).hexdigest(),
            },
            "config": config,
            "budget_audit": audit,
        }
        if knowline:
            value.update(
                decision_model={
                    "model": "PelaAI/KnowLine-31B-Gen1",
                    "revision": "8f052a8130e599b53e5324c34f0f5ff90696e1c6",
                    "provider": "self_hosted",
                },
                decision_summary=receipt["decision_summary"],
            )
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temporary.replace(destination)
        count += 1
    print(
        json.dumps({"exported": count, "total": len(jobs), "directory": str(output)}),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("campaign", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--knowline", action="store_true")
    a = p.parse_args()
    export(a.campaign, a.output, a.knowline)
