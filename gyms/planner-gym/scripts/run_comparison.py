#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect a comparison plan, or explicitly execute its resource-bounded batches on Linux."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]


def physical_cpu_ids() -> list[int]:
    """Choose one allowed logical CPU per physical core, avoiding SMT double-counting."""
    cores = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        key = (
            (topology / "physical_package_id").read_text().strip(),
            (topology / "core_id").read_text().strip(),
        )
        cores.setdefault(key, cpu)
    return list(cores.values())


def stop_process(process: subprocess.Popen) -> None:
    """Give the match runner time to clean up its isolated worker, then reap leftovers."""
    if process.poll() is not None:
        return
    try:
        parent = psutil.Process(process.pid)
        descendants = parent.children(recursive=True)
        parent.send_signal(signal.SIGINT)
    except psutil.NoSuchProcess:
        process.wait()
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass  # The explicit terminate/kill phase below owns remaining processes.
    remaining = [p for p in [parent, *descendants] if p.is_running()]
    for child in remaining:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            continue
    _, alive = psutil.wait_procs(remaining, timeout=3)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            continue
    process.wait()


def validate_plan(plan: dict, directory: Path) -> None:
    """Refuse config drift, duplicate execution or a batch that exceeds its resources."""
    jobs = {job["id"]: job for job in plan["jobs"]}
    if len(jobs) != len(plan["jobs"]):
        raise ValueError("duplicate job IDs")
    scheduled = [job_id for batch in plan["batches"] for job_id in batch]
    if len(scheduled) != len(set(scheduled)):
        raise ValueError("a job appears in more than one batch")
    if set(scheduled) & set(plan["blocked_jobs"]) or set(scheduled) | set(
        plan["blocked_jobs"]
    ) != set(jobs):
        raise ValueError("scheduled/blocked jobs do not partition the complete plan")
    policy = plan["policy"]["parallel"]
    for batch_ids in plan["batches"]:
        batch = [jobs[job_id] for job_id in batch_ids]
        if (
            not batch
            or len(batch) > policy["max_parallel_runs"]
            or sum(j["cores"] for j in batch)
            > policy["physical_cores"] - policy["reserve_cores"]
            or sum(j["memory_gib"] for j in batch)
            > policy["memory_gib"] - policy["reserve_memory_gib"]
            or sum(j["planner"] == "jev" for j in batch) > policy["max_parallel_jev"]
            or (len(batch) > 1 and any(j["exclusive"] for j in batch))
        ):
            raise ValueError(
                "batch exceeds the CPU, memory, JEV or exclusive-run limits"
            )
    for job in jobs.values():
        config = directory / job["config"]
        if hashlib.sha256(config.read_bytes()).hexdigest() != job["config_sha256"]:
            raise ValueError(f"{job['id']}: config changed; regenerate the plan")


def execute(plan: dict, directory: Path) -> int:
    if sys.platform != "linux":
        raise RuntimeError("this execution setup requires a Linux native-replay host")
    if plan["blocked_jobs"]:
        raise ValueError(
            "host resources cannot accommodate every job; regenerate for a larger host"
        )
    taskset = shutil.which("taskset")
    if taskset is None:
        raise RuntimeError("taskset is required to enforce CPU affinity")
    policy = plan["policy"]["parallel"]
    cpus = physical_cpu_ids()
    if (
        len(cpus) < policy["physical_cores"]
        or psutil.virtual_memory().total / 1024**3 < policy["memory_gib"] * 0.98
    ):
        raise RuntimeError(
            "actual host is smaller than the planned physical CPU/RAM capacity"
        )
    if (
        any(j["planner"] == "jev" for j in plan["jobs"])
        and not os.environ.get("TYPESAFE_API_KEY", "").strip()
    ):
        raise RuntimeError("TYPESAFE_API_KEY is required for the planned JEV jobs")
    jobs = {job["id"]: job for job in plan["jobs"]}
    environment = {**os.environ, **policy["thread_environment"]}
    logs = directory / "logs"
    logs.mkdir(exist_ok=True)
    for index, batch_ids in enumerate(plan["batches"], 1):
        batch = [jobs[job_id] for job_id in batch_ids]
        required = sum(j["memory_gib"] for j in batch) + policy["reserve_memory_gib"]
        if psutil.virtual_memory().available / 1024**3 < required:
            raise RuntimeError(
                f"batch {index} requires {required} GiB available including host reserve"
            )
        print(
            f"Batch {index}/{len(plan['batches'])}: {', '.join(batch_ids)}", flush=True
        )
        with ExitStack() as stack:
            processes = []
            offset = 0
            for job in batch:
                allocated = cpus[offset : offset + job["cores"]]
                offset += job["cores"]
                command = [
                    taskset,
                    "--cpu-list",
                    ",".join(map(str, allocated)),
                    sys.executable,
                    str(ROOT / "scripts/run_match_config.py"),
                    str(directory / job["config"]),
                    "--session-dir",
                    str(directory / job["session"]),
                    "--expect-trace-sha256",
                    job["trace_sha256"],
                ]
                log = stack.enter_context((logs / f"{job['id']}.log").open("x"))
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                stack.callback(stop_process, process)
                processes.append(process)
            while any(p.poll() is None for p in processes):
                time.sleep(0.25)
            failures = [
                job["id"]
                for job, process in zip(batch, processes)
                if process.returncode != 0
            ]
        if failures:
            print(
                "Failed: "
                + ", ".join(failures)
                + "; inspect logs before retrying. No automatic retries.",
                file=sys.stderr,
            )
            return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="launch the prepared benchmarks and JEV API calls",
    )
    args = parser.parse_args()
    path = args.plan.resolve()
    plan = json.loads(path.read_text())
    validate_plan(plan, path.parent)
    print(
        f"{plan['new_runs']} new runs; {len(plan['batches'])} batches; {len(plan['blocked_jobs'])} blocked; JEV cap {plan['max_jev_calls']} calls."
    )
    if not args.execute:
        print("Plan inspection only. Nothing executed.")
        return 0
    return execute(plan, path.parent)


if __name__ == "__main__":
    raise SystemExit(main())
