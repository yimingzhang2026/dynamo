#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect a comparison plan, or explicitly execute its resource-bounded batches on Linux."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
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
CGROUP = Path("/sys/fs/cgroup")


def memory_capacity(cgroup: Path = CGROUP) -> tuple[float, float]:
    """Return total/available GiB, respecting a Linux container's memory ceiling."""
    memory = psutil.virtual_memory()
    total, available = memory.total, memory.available
    limit_path = cgroup / "memory.max"
    if limit_path.exists():
        value = limit_path.read_text().strip()
        if value != "max":
            limit = int(value)
            current = int((cgroup / "memory.current").read_text())
            stats = dict(
                line.split()
                for line in (cgroup / "memory.stat").read_text().splitlines()
            )
            reclaimable = int(stats.get("inactive_file", 0))
            total = min(total, limit)
            available = min(available, max(0, limit - current + reclaimable))
    return total / 1024**3, available / 1024**3


def cpu_capacity(cpus: list[int], cgroup: Path = CGROUP) -> int:
    """Respect CPU quota as well as affinity/topology (Docker reports VM topology)."""
    quota_path = cgroup / "cpu.max"
    if not quota_path.exists():
        return len(cpus)
    quota, period = quota_path.read_text().split()
    return (
        len(cpus)
        if quota == "max"
        else min(len(cpus), math.floor(int(quota) / int(period)))
    )


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


def choose_rolling_job(
    pending: list[dict], active: list[dict], policy: dict, available_gib: float
) -> dict | None:
    """Prefer small, different cases while enforcing all concurrent resource limits."""
    active_cases = {job["case_id"] for job in active}
    candidates = sorted(
        pending,
        key=lambda job: (
            job["exclusive"],
            job["memory_gib"],
            job["case_id"] in active_cases,
            job["duration_s"],
            job["id"],
        ),
    )
    for job in candidates:
        combined = [*active, job]
        if (
            len(combined) > policy["max_parallel_runs"]
            or sum(j["cores"] for j in combined)
            > policy["physical_cores"] - policy["reserve_cores"]
            or sum(j["memory_gib"] for j in combined)
            > policy["memory_gib"] - policy["reserve_memory_gib"]
            or sum(j["planner"] == "jev" for j in combined) > policy["max_parallel_jev"]
            or (len(combined) > 1 and any(j["exclusive"] for j in combined))
            or job["memory_gib"] + policy["reserve_memory_gib"] > available_gib
        ):
            continue
        return job
    return None


def execute_rolling(
    jobs: dict,
    status: dict,
    policy: dict,
    directory: Path,
    cpus: list[int],
    taskset: str,
    environment: dict,
    save_status,
    planners: set[str] | None,
    keep_going: bool,
) -> int:
    """Refill individual CPU slots without waiting for an entire batch to finish."""
    pending = [
        jobs[job_id]
        for job_id, state in status["jobs"].items()
        if state == "pending"
        and (planners is None or jobs[job_id]["planner"] in planners)
    ]
    active = {}
    halt = False
    with ExitStack() as cleanup:
        while pending or active:
            for job_id, (job, process, allocated, resources) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                status["jobs"][job_id] = "completed" if code == 0 else "failed"
                save_status()
                resources.close()
                del active[job_id]
                print(f"Finished {job_id}: {status['jobs'][job_id]}", flush=True)
                if code != 0 and not keep_going:
                    halt = True
            if not halt:
                while pending:
                    job = choose_rolling_job(
                        pending,
                        [item[0] for item in active.values()],
                        policy,
                        memory_capacity()[1],
                    )
                    if job is None:
                        break
                    used = {cpu for item in active.values() for cpu in item[2]}
                    allocated = [cpu for cpu in cpus if cpu not in used][: job["cores"]]
                    resources = cleanup.enter_context(ExitStack())
                    log = resources.enter_context(
                        (directory / "logs" / f"{job['id']}.log").open("x")
                    )
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
                    process = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    resources.callback(stop_process, process)
                    active[job["id"]] = (job, process, allocated, resources)
                    pending = [item for item in pending if item["id"] != job["id"]]
                    status["jobs"][job["id"]] = "running"
                    save_status()
                    print(
                        f"Started {job['id']} on CPUs {allocated}; "
                        f"{len(active)}/{policy['max_parallel_runs']} active",
                        flush=True,
                    )
            if not active:
                if halt:
                    break
                if pending:
                    raise RuntimeError(
                        "no pending job fits the available memory including host reserve"
                    )
            if active:
                time.sleep(0.25)
    return int("failed" in status["jobs"].values())


def execute(
    plan: dict,
    directory: Path,
    *,
    allow_blocked: bool = False,
    resume_pending: bool = False,
    planners: set[str] | None = None,
    keep_going: bool = False,
    rolling: bool = False,
) -> int:
    if sys.platform != "linux":
        raise RuntimeError("this execution setup requires a Linux native-replay host")
    if plan["blocked_jobs"] and not allow_blocked:
        raise ValueError(
            "host resources cannot accommodate every job; use --allow-blocked to run only scheduled jobs"
        )
    taskset = shutil.which("taskset")
    if taskset is None:
        raise RuntimeError("taskset is required to enforce CPU affinity")
    policy = plan["policy"]["parallel"]
    cpus = physical_cpu_ids()
    if (
        cpu_capacity(cpus) < policy["physical_cores"]
        or memory_capacity()[0] < policy["memory_gib"] * 0.98
    ):
        raise RuntimeError(
            "actual host is smaller than the planned physical CPU/RAM capacity"
        )
    if (
        any(
            j["planner"] == "jev"
            and j["id"] not in plan["blocked_jobs"]
            and (planners is None or j["planner"] in planners)
            for j in plan["jobs"]
        )
        and not os.environ.get("TYPESAFE_API_KEY", "").strip()
    ):
        raise RuntimeError("TYPESAFE_API_KEY is required for the planned JEV jobs")
    jobs = {job["id"]: job for job in plan["jobs"]}
    environment = {**os.environ, **policy["thread_environment"]}
    logs = directory / "logs"
    logs.mkdir(exist_ok=True)
    status = {
        "pending_resources": plan["blocked_jobs"],
        "jobs": {job_id: "pending" for batch in plan["batches"] for job_id in batch},
    }
    if resume_pending:
        previous = json.loads((directory / "status.json").read_text())
        if (
            set(previous["jobs"]) != set(status["jobs"])
            or previous["pending_resources"] != status["pending_resources"]
        ):
            raise ValueError("saved status does not match the plan")
        if any(
            value not in {"pending", "completed", "failed"}
            for value in previous["jobs"].values()
        ):
            raise ValueError(
                "resolve running/interrupted jobs before resuming pending work"
            )
        status = previous
    else:
        with (directory / "status.json").open("x") as stream:
            stream.write(json.dumps(status, indent=2) + "\n")

    def save_status():
        temporary = directory / "status.json.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(directory / "status.json")

    save_status()
    if rolling:
        return execute_rolling(
            jobs,
            status,
            policy,
            directory,
            cpus[: policy["physical_cores"] - policy["reserve_cores"]],
            taskset,
            environment,
            save_status,
            planners,
            keep_going,
        )
    for index, batch_ids in enumerate(plan["batches"], 1):
        batch = [
            jobs[job_id]
            for job_id in batch_ids
            if status["jobs"][job_id] == "pending"
            and (planners is None or jobs[job_id]["planner"] in planners)
        ]
        if not batch:
            continue
        required = sum(j["memory_gib"] for j in batch) + policy["reserve_memory_gib"]
        if memory_capacity()[1] < required:
            raise RuntimeError(
                f"batch {index} requires {required} GiB available including host reserve"
            )
        print(
            f"Batch {index}/{len(plan['batches'])}: {', '.join(j['id'] for j in batch)}",
            flush=True,
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
                status["jobs"][job["id"]] = "running"
                save_status()
            while any(p.poll() is None for p in processes):
                time.sleep(0.25)
            failures = [
                job["id"]
                for job, process in zip(batch, processes)
                if process.returncode != 0
            ]
            for job, process in zip(batch, processes):
                status["jobs"][job["id"]] = (
                    "completed" if process.returncode == 0 else "failed"
                )
            save_status()
        if failures:
            print(
                "Failed: "
                + ", ".join(failures)
                + "; inspect logs before retrying. No automatic retries.",
                file=sys.stderr,
            )
            if not keep_going:
                return 1
    return int("failed" in status["jobs"].values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument(
        "--rolling",
        action="store_true",
        help="prioritize small cases and refill free slots as each run finishes",
    )
    parser.add_argument(
        "--wait-for-lock",
        action="store_true",
        help="wait for the active launcher to finish before resuming a selected queue",
    )
    parser.add_argument(
        "--resume-pending",
        action="store_true",
        help="run only pending jobs; never retry failed or completed jobs",
    )
    parser.add_argument(
        "--planners",
        nargs="+",
        choices=["mpc", "planner", "keda", "reactive", "jev"],
        help="execute only these planners; leave others pending",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="record failures and continue independent batches without retrying",
    )
    parser.add_argument(
        "--allow-blocked",
        action="store_true",
        help="leave oversized jobs pending and execute the batches that fit",
    )
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
    with (path.parent / "launcher.lock").open("a") as lock:
        if args.wait_for_lock:
            print(
                "Waiting for the active launcher to release the campaign lock.",
                flush=True,
            )
        fcntl.flock(lock, fcntl.LOCK_EX | (0 if args.wait_for_lock else fcntl.LOCK_NB))
        return execute(
            plan,
            path.parent,
            allow_blocked=args.allow_blocked,
            resume_pending=args.resume_pending,
            planners=set(args.planners) if args.planners else None,
            keep_going=args.keep_going,
            rolling=args.rolling,
        )


if __name__ == "__main__":
    raise SystemExit(main())
