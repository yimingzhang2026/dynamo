#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the four engine layouts on a tiny synthetic trace, outside benchmark results."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml
from autoscaling_arena.match_config import load_match_config
from prepare_comparison import HEADER, ROOT, SOURCE, make_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/native-smoke")
    parser.add_argument(
        "--include-jev", action="store_true", help="include live JEV API smoke calls"
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    policy = yaml.safe_load((SOURCE / "comparison.mac.yaml").read_text())
    policy["jev"]["max_calls"] = 10
    planners = [p for p in policy["planners"] if p != "jev" or args.include_jev]
    trace = output / "synthetic.jsonl"
    with trace.open("x") as stream:
        for index in range(8):
            row = {
                "schema": "dynamo.request.trace.v1",
                "event_type": "request_end",
                "event_time_unix_ms": index * 2000,
                "request": {
                    "request_id": f"smoke-{index}",
                    "request_received_ms": index * 2000,
                    "output_tokens": 8,
                    "replay": {
                        "trace_block_size": 16,
                        "input_length": 256,
                        "input_sequence_hashes": list(
                            range(index * 16, (index + 1) * 16)
                        ),
                    },
                },
            }
            stream.write(json.dumps(row) + "\n")
    paths = []
    for group in ("A-HT", "A-LC", "D-HT", "D-LC"):
        case = f"{group}-S05"
        base = yaml.safe_load((SOURCE / f"{case}.yaml").read_text())
        for planner in planners:
            job = {
                "id": f"smoke-{group}-{planner}",
                "case_id": case,
                "planner": planner,
                "timeout_s": 600,
                "memory_gib": 5,
            }
            path = output / f"{job['id']}.yaml"
            config = make_config(base, job, policy, path, output)
            config[
                "description"
            ] = "Synthetic native runtime validation; never a benchmark result."
            config["labels"]["comparison"] = "native-smoke-only"
            config["evaluations"]["traces"] = [
                {
                    "name": "synthetic-smoke",
                    "format": "dynamo",
                    "block_size": 16,
                    "paths": [str(trace)],
                }
            ]
            config["evaluations"]["defaults"]["measurement_window"] = {
                "start_s": 0,
                "end_s": 14,
            }
            path.write_text(HEADER + yaml.safe_dump(config, sort_keys=False))
            load_match_config(path)
            paths.append(path)
    print(f"Prepared {len(paths)} synthetic smoke runs in {output}", flush=True)
    if not args.execute:
        return 0
    environment = {**os.environ, **policy["parallel"]["thread_environment"]}
    for path in paths:
        print(f"Running {path.stem}", flush=True)
        with path.with_suffix(".log").open("x") as log:
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/run_match_config.py"),
                    str(path),
                    "--session-dir",
                    str(output / "sessions" / path.stem),
                ],
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            print(f"Failed {path.stem}; inspect {path.with_suffix('.log')}", flush=True)
            return result.returncode
    print("All native smoke runs passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
