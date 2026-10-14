#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One authorized, sequential retry of non-JEV memory failures after a campaign."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import yaml
from autoscaling_arena.match_config import parse_match_config

ROOT = Path(__file__).resolve().parents[1]
MODULES = {}
for name in ("prepare_comparison", "run_comparison"):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    MODULES[name] = module
prepare = MODULES["prepare_comparison"]
run = MODULES["run_comparison"]
ARCHIVE = "recovery/memory-limit-v1"
RSS_GIB = 27


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".memory-retry.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def candidates(directory: Path, plan: dict, status: dict) -> dict:
    """Only structured memory failures qualify; never infer from log text."""
    selected = {}
    for job in plan["jobs"]:
        if job["planner"] == "jev" or status["jobs"].get(job["id"]) != "failed":
            continue
        result = (
            directory / "results" / job["case_id"] / job["planner"] / "results.json"
        )
        report = json.loads(result.read_text())
        rows = report["results"]
        if (
            len(rows) != 1
            or rows[0].get("status") != "failed"
            or rows[0].get("error", {}).get("type") != "RunMemoryLimitError"
        ):
            continue
        execution = rows[0]["execution"]
        if execution["max_memory_mb"] >= RSS_GIB * 1024:
            continue
        selected[job["id"]] = {
            "old_limit_gib": execution["max_memory_mb"] / 1024,
            "observed_peak_gib": execution["peak_rss_mb"] / 1024,
            "retry_limit_gib": RSS_GIB,
        }
    return selected


def prepare_retry(directory: Path) -> dict:
    """Caller holds launcher.lock; archive originals before publishing new work."""
    plan = json.loads((directory / "plan.json").read_text())
    status = json.loads((directory / "status.json").read_text())
    if any(state not in {"completed", "failed"} for state in status["jobs"].values()):
        raise ValueError("Original campaign must finish before memory retries")
    run.validate_plan(plan, directory)
    archive = directory / ARCHIVE
    if archive.exists():
        raise ValueError(
            "Memory retry archive already exists; refusing another attempt"
        )
    selected = candidates(directory, plan, status)
    if not selected:
        return selected

    # A 27 GiB worker guard leaves 3 GiB inside the existing 30 GiB container.
    # The job reserves 28 GiB; the dispatcher additionally requires 1 GiB free.
    plan["policy"]["parallel"].update(
        memory_gib=29, reserve_memory_gib=1, max_parallel_runs=1
    )
    configs = {}
    sources = []
    for job in plan["jobs"]:
        if job["id"] not in selected:
            continue
        job.update(memory_gib=RSS_GIB + 1, exclusive=True)
        path = directory / job["config"]
        base = yaml.safe_load((prepare.SOURCE / f"{job['case_id']}.yaml").read_text())
        config = prepare.make_config(base, job, plan["policy"], path, directory)
        expected = yaml.safe_load(path.read_text())
        expected["execution"]["max_memory_mb"] = RSS_GIB * 1024
        if config != expected:
            raise ValueError(f"{job['id']}: regeneration changes more than memory")
        parse_match_config(config, source_path=path)
        content = prepare.HEADER + yaml.safe_dump(config, sort_keys=False)
        configs[path] = content
        job["config_sha256"] = hashlib.sha256(content.encode()).hexdigest()
        sources.extend(
            [
                directory / "logs" / f"{job['id']}.log",
                directory / job["session"],
                directory / "results" / job["case_id"] / job["planner"],
            ]
        )
        status["jobs"][job["id"]] = "pending"
    for source in sources:
        if not source.exists():
            raise ValueError(f"Missing original attempt: {source}")

    # Preserve the original 220/48 partition and completed jobs. Only retry
    # reservations change; no formerly blocked job or JEV attempt is added.
    plan["batches"] = [[job_id] for job_id in status["jobs"]]
    plan["notes"].append(
        "Authorized memory retry: non-JEV memory failures only, one attempt each, "
        "27 GiB process-tree RSS, sequential, unchanged 30 GiB container. "
        "Exceeding 27 GiB is not proof that more than 30 GiB is required."
    )
    archive.mkdir(parents=True)
    for name in ("plan.json", "status.json"):
        shutil.copy2(directory / name, archive / f"{name}.before")
    save(archive / "retries.json", selected)
    for path in configs:
        target = archive / path.relative_to(directory)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for source in sources:
        target = archive / source.relative_to(directory)
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
    for path, content in configs.items():
        path.write_text(content)
    run.validate_plan(plan, directory)
    save(directory / "plan.json", plan)
    save(directory / "status.json", status)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--wait-for-lock", action="store_true")
    args = parser.parse_args()
    directory = args.plan.resolve().parent
    if not args.execute:
        print(
            json.dumps(
                candidates(
                    directory,
                    json.loads(args.plan.read_text()),
                    json.loads((directory / "status.json").read_text()),
                ),
                indent=2,
            )
        )
        return 0
    if sys.platform != "linux":
        raise RuntimeError("Memory recovery execution requires the Linux replay host")
    with (directory / "memory-retry.lock").open("a") as recovery_lock:
        fcntl.flock(recovery_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = directory / "memory-retry.json"
        save(
            state_path,
            {
                "phase": "waiting_for_original_campaign",
                "pid": os.getpid(),
                "rss_limit_gib": RSS_GIB,
                "container_limit_gib": 30,
                "automatic_jev_retries": False,
            },
        )
        try:
            with (directory / "launcher.lock").open("a") as lock:
                print(
                    "Waiting for the original campaign; no active jobs will be interrupted.",
                    flush=True,
                )
                fcntl.flock(
                    lock, fcntl.LOCK_EX | (0 if args.wait_for_lock else fcntl.LOCK_NB)
                )
                capacity = run.memory_capacity()[0]
                if not 29 <= capacity <= 30.01:
                    raise ValueError(
                        f"Expected the existing 30 GiB container, found {capacity:g} GiB"
                    )
                selected = prepare_retry(directory)
                save(
                    state_path,
                    {
                        "phase": "prepared" if selected else "nothing_to_retry",
                        "pid": os.getpid(),
                        "retries": selected,
                        "rss_limit_gib": RSS_GIB,
                        "container_limit_gib": 30,
                        "automatic_jev_retries": False,
                    },
                )
        except Exception as error:
            save(state_path, {"phase": "needs_inspection", "error": str(error)})
            raise
        if not selected:
            return 0
        print(
            f"Prepared {len(selected)} sequential memory retries at {RSS_GIB} GiB RSS.",
            flush=True,
        )
        (directory / "launcher.pid").write_text(str(os.getpid()) + "\n")
        os.chdir(ROOT)
        os.execv(
            sys.executable,
            [
                sys.executable,
                "-u",
                "scripts/run_comparison.py",
                str(directory / "plan.json"),
                "--execute",
                "--allow-blocked",
                "--resume-pending",
                "--keep-going",
                "--rolling",
                "--wait-for-lock",
                "--planners",
                "mpc",
                "planner",
                "keda",
                "reactive",
            ],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
