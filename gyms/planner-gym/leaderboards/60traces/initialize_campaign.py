# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Start publication for a new policy with empty dynamic cells and verified Static scores."""

import argparse
import json
from pathlib import Path

import yaml
from autoscaling_arena.resource_policy import case_resources
from build_leaderboards import load_static_latency
from publish_latest import publish, refresh, write

ROOT = Path(__file__).parent
GYM = ROOT.parents[1]


def initialize(inputs, policy):
    cases = json.loads((inputs / "catalog.json").read_text())["cases"]
    load_static_latency(inputs, {c["case_id"]: c for c in cases})
    jobs, records, setups = [], [], {}
    planners = ("mpc", "planner", "keda", "reactive", "knowline31b")
    for case in cases:
        name = case["case_id"]
        original = yaml.safe_load((GYM / f"configs/60traces/{name}.yaml").read_text())
        resources = case_resources(original, policy)
        for planner in planners:
            job = {
                "id": name + "-" + planner,
                "case_id": name,
                "planner": planner,
                "trace_sha256": case["trace_sha256"],
                "resources": resources,
            }
            jobs.append(job)
            records.append(
                {
                    "job_id": job["id"],
                    "plan_job": job,
                    "saved_status": "pending",
                    "source": "Replay",
                    "source_path": "",
                    "source_sha256": None,
                    "source_bytes": 0,
                    "configuration": {
                        "path": f"configs/{job['id']}.yaml",
                        "sha256": None,
                    },
                }
            )
        setups[name] = {
            **resources,
            "topology": case["topology"],
            "source_config": records[-1]["configuration"],
        }
    write(inputs / "plan.json", {"schema_version": 2, "policy": policy, "jobs": jobs})
    write(inputs / "results.json", {"records": records})
    write(inputs / "experiment-setup.json", {"rows": setups})
    return refresh(inputs, [])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=ROOT / "inputs")
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--mirror", type=Path)
    args = parser.parse_args()
    policy = yaml.safe_load((GYM / "configs/60traces/comparison.yaml").read_text())
    report = initialize(args.inputs, policy)
    publish(report, args.output)
    if args.mirror:
        publish(report, args.mirror)
    print(json.dumps(report["counts"]))
