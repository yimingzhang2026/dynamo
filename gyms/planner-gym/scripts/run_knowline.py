# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a frozen selection with one to four independent lanes per GPU node."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import queue
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import knowline_campaign as core
import psutil
from audit_gpu_usage import audit_budget
from autoscaling_arena.jev_report import summarize_decisions

PLAN = Path(os.environ.get("KNOWLINE_PARALLEL_PLAN", core.ROOT / "parallel-plan.json"))
MAX_SERVER_RETRIES = 2
BATCH = Path(
    os.environ.get("KNOWLINE_BATCH", Path(__file__).with_name("knowline.sbatch"))
)


def assign_shards(cases, lanes_per_node=4, nodes=2):
    """Balance longest cases first across a fixed number of node queues."""
    if type(lanes_per_node) is not int or not 1 <= lanes_per_node <= 4:
        raise ValueError("Expected between one and four lanes per node")
    if type(nodes) is not int or nodes < 1:
        raise ValueError("Expected a positive integer node count")
    if not 1 <= len(cases) <= 60 or len({job["case_id"] for job in cases}) != len(
        cases
    ):
        raise ValueError("Expected between one and 60 distinct traces")

    def work(job):
        return job["duration_s"] * (2.02 if job["group"].startswith("D") else 0.67)

    ordered = sorted(
        cases,
        key=lambda job: (
            -work(job),
            -job["bytes"],
            job["case_id"],
        ),
    )
    shards = [{"shard": i, "case_ids": []} for i in range(min(nodes, len(cases)))]
    loads = [0.0] * len(shards)
    for job in ordered:
        owner = min(range(len(shards)), key=lambda i: (loads[i], i))
        shards[owner]["case_ids"].append(job["case_id"])
        loads[owner] += work(job)
    return shards


def prepare(*, lanes_per_node=4, nodes=2):
    if PLAN.exists():
        raise RuntimeError("Parallel plan already exists; inspect before replacing")
    original_path = core.ROOT / "plan.json"
    original = json.loads(original_path.read_text())
    if not core.digest(core.__file__) == original["launcher_sha256"]:
        raise ValueError(
            "Campaign validation failed: core.digest(core.__file__) == original['launcher_sha256']"
        )
    for job in original["cases"]:
        if not core.digest(job["config"]) == job["config_sha256"]:
            raise ValueError(
                "Campaign validation failed: core.digest(job['config']) == job['config_sha256']"
            )
        trace = core.TRACE_DIR / f"{job['case_id']}.jsonl"
        if not trace.stat().st_size == job["bytes"]:
            raise ValueError(job["case_id"])
    shards = assign_shards(original["cases"], lanes_per_node, nodes)
    plan = {
        "created_at": core.now(),
        "model": core.MODEL,
        "model_revision": core.REVISION,
        "shards": shards,
        "maximum_parallel_traces": sum(
            min(lanes_per_node, len(shard["case_ids"])) for shard in shards
        ),
        "maximum_concurrent_nodes": len(shards),
        "lanes_per_node": lanes_per_node,
        "gpus_per_node": 8,
        "cpu_cores_per_node": 64,
        "simulation_memory_guard_gib": 384,
        "isolated_simulation_memory_guard_gib": 1400,
        "original_plan_sha256": core.digest(original_path),
        "core_sha256": core.digest(core.__file__),
        "wrapper_sha256": core.digest(__file__),
        "batch_sha256": core.digest(BATCH),
        "round_work_budget_s": 13200,
        "max_rounds_per_shard": 16,
    }
    core.save(PLAN, plan)
    print(
        json.dumps(
            {
                "prepared_shards": len(plan["shards"]),
                "maximum_parallel_traces": plan["maximum_parallel_traces"],
                "maximum_gpus": len(plan["shards"]) * 8,
                "plan": str(PLAN),
            },
            indent=2,
        ),
        flush=True,
    )


def check_plan():
    plan = json.loads(PLAN.read_text())
    if not core.digest(__file__) == plan["wrapper_sha256"]:
        raise ValueError(
            "Campaign validation failed: core.digest(__file__) == plan['wrapper_sha256']"
        )
    if not core.digest(core.__file__) == plan["core_sha256"]:
        raise ValueError(
            "Campaign validation failed: core.digest(core.__file__) == plan['core_sha256']"
        )
    if not core.digest(BATCH) == plan["batch_sha256"]:
        raise ValueError(
            "Campaign validation failed: core.digest(BATCH) == plan['batch_sha256']"
        )
    if not core.digest(core.ROOT / "plan.json") == plan["original_plan_sha256"]:
        raise ValueError(
            "Campaign validation failed: core.digest(core.ROOT / 'plan.json') == plan['original_plan_sha256']"
        )
    return plan


def recover_case(case):
    """Recover a stopped allocation without losing its successful result or prefix."""
    directory = core.ROOT / "cases" / case
    state = core.read_state(case)
    if state["state"] != "running":
        return
    rows = core.harvest(directory, state)
    attempt = state["attempts"][-1]
    result_path = Path(attempt["directory"]) / "results.json"
    report = json.loads(result_path.read_text()) if result_path.exists() else {}
    results = report.get("results", [])
    ok = (
        report.get("summary", {}).get("status") == "ok"
        and len(results) == 1
        and (results[0].get("status") == "ok")
        and (results[0].get("autoscaler") == "KnowLine-31B")
    )
    if ok:
        state["budget_audit"] = audit_budget(
            Path(attempt["directory"]) / "session", results[0]["runtime"]["gpu_budget"]
        )
        state.update(
            state="completed",
            result=str(result_path),
            completed_at=core.now(),
            decision_summary=summarize_decisions(directory / "saved-decisions.jsonl"),
        )
    else:
        attempt.update(interrupted=True, recovered_at=core.now())
        state.update(state="pending_continuation", saved_decisions=len(rows))
    state["updated_at"] = core.now()
    core.save(directory / "state.json", state)


def shard_status(shard, cases, **details):
    states = {case: core.read_state(case) for case in cases}
    counts = {}
    for row in states.values():
        counts[row["state"]] = counts.get(row["state"], 0) + 1
    value = {
        "updated_at": core.now(),
        "shard": shard,
        "counts": counts,
        "case_ids": cases,
        **details,
    }
    core.save(core.ROOT / "shards" / f"{shard:02d}" / "state.json", value)
    return value


def aggregate_status():
    """One writer publishes campaign status while shards update separate cases."""
    with (core.ROOT / "parallel-status.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        original = json.loads((core.ROOT / "plan.json").read_text())
        states = {
            job["case_id"]: core.read_state(job["case_id"]) for job in original["cases"]
        }
        counts = {}
        for row in states.values():
            counts[row["state"]] = counts.get(row["state"], 0) + 1
        result = {
            "updated_at": core.now(),
            "total": len(original["cases"]),
            "counts": counts,
            "cases": states,
        }
        core.save(core.ROOT / "parallel-status.json", result)
        if counts.get("completed") == len(original["cases"]):
            (core.ROOT / "CAMPAIGN_COMPLETE").write_text(core.now() + "\n")
        return result


def run_lane(lane, jobs, cpus, allocation, deadline, isolated):
    """Reuse healthy servers; resume failed calls only after replacing the server."""
    process = None
    generation = 0
    errors = []
    try:
        while deadline - time.monotonic() > 240:
            try:
                job = jobs.get_nowait()
            except queue.Empty:
                break
            case = job["case_id"]
            directory = core.ROOT / "cases" / case
            with (directory / "parallel.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                state = core.read_state(case)
                if state["state"] in core.TERMINAL:
                    continue
                trace = core.TRACE_DIR / f"{case}.jsonl"
                if not trace.is_file() or trace.stat().st_size != job["bytes"]:
                    state.update(state="waiting_for_data", updated_at=core.now())
                    core.save(directory / "state.json", state)
                    continue
                # A lane/server startup failure leaves unstarted cases pending
                # for the next allocation instead of labelling them failed runs.
                if psutil.virtual_memory().available < 200 * 1024**3:
                    raise RuntimeError(
                        "Insufficient host memory headroom before replay"
                    )
                while deadline - time.monotonic() > 240:
                    if process is not None and process.poll() is not None:
                        process = None
                    if process is None:
                        logs = allocation
                        if generation:
                            logs = (
                                allocation
                                / "server-restarts"
                                / f"lane-{lane}"
                                / f"start-{generation:03d}"
                            )
                            logs.mkdir(parents=True)
                        generation += 1
                        process, port = core.start_server(lane, cpus, logs)
                    try:
                        core.run_case(job, lane, cpus, port, deadline, isolated)
                    except Exception as exc:
                        state = core.read_state(case)
                        core.harvest(directory, state)
                        error = {
                            "case_id": case,
                            "type": type(exc).__name__,
                            "message": str(exc),
                        }
                        if state["state"] != "completed":
                            state.update(
                                state="failed", updated_at=core.now(), last_error=error
                            )
                            core.save(directory / "state.json", state)
                        errors.append(error)
                    state = core.read_state(case)
                    if core.is_server_failure(state):
                        failures = state.get("server_failures", 0) + 1
                        state.update(
                            state=(
                                "pending_server_retry"
                                if failures <= MAX_SERVER_RETRIES
                                else "failed"
                            ),
                            server_failures=failures,
                            updated_at=core.now(),
                        )
                        core.save(directory / "state.json", state)
                        print(
                            f"{core.now()} {case}: replacing server lane {lane}; transport failure {failures}",
                            flush=True,
                        )
                        if failures > MAX_SERVER_RETRIES:
                            errors.append(
                                {
                                    "case_id": case,
                                    "type": "ServerRetryLimit",
                                    "message": str(state["last_error"]),
                                }
                            )
                    # A timed-out replay can leave an inference in flight. Never
                    # hand that server to another case, even if /health is OK.
                    if state["state"] != "completed":
                        core.stop_server(process)
                        process = None
                    if state["state"] != "pending_server_retry":
                        break
    finally:
        core.stop_server(process)
    return errors


def run(shard, round_number):
    started = time.monotonic()
    plan = check_plan()
    if (
        not 0 <= shard < len(plan["shards"])
        or not 1 <= round_number <= plan["max_rounds_per_shard"]
    ):
        raise ValueError("Invalid shard or continuation round")
    shard_root = core.ROOT / "shards" / f"{shard:02d}"
    shard_root.mkdir(parents=True, exist_ok=True)
    with (shard_root / "allocation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        names = plan["shards"][shard]["case_ids"]
        for name in names:
            recover_case(name)
        status = shard_status(shard, names, round=round_number)
        original = json.loads((core.ROOT / "plan.json").read_text())
        jobs = {job["case_id"]: job for job in original["cases"]}
        active = [
            jobs[name]
            for name in names
            if core.read_state(name)["state"] not in core.TERMINAL
        ]
        if not active:
            aggregate_status()
            print("Shard is finished:", status["counts"], flush=True)
            return
        allocation = core.ROOT / "allocations" / os.environ["SLURM_JOB_ID"]
        allocation.mkdir(parents=True)
        if psutil.virtual_memory().total < 1900 * 1024**3:
            raise RuntimeError(
                "This allocation requires a GPU node with approximately 2 TiB RAM"
            )
        gpu_inventory = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        )
        if len(gpu_inventory.strip().splitlines()) != 8:
            raise RuntimeError("Expected eight visible GPUs on the exclusive GPU node")
        (allocation / "gpus.csv").write_text(gpu_inventory)
        core.save(
            allocation / "started.json",
            {
                "started_at": core.now(),
                "shard": shard,
                "round": round_number,
                "host": os.uname().nodename,
                "case_ids": names,
            },
        )
        if round_number < plan["max_rounds_per_shard"]:
            successor = subprocess.check_output(
                [
                    "sbatch",
                    "--parsable",
                    *(
                        ["--account=" + os.environ["SLURM_JOB_ACCOUNT"]]
                        if os.environ.get("SLURM_JOB_ACCOUNT")
                        else []
                    ),
                    *(
                        ["--partition=" + os.environ["SLURM_JOB_PARTITION"]]
                        if os.environ.get("SLURM_JOB_PARTITION")
                        else []
                    ),
                    f"--dependency=afterany:{os.environ['SLURM_JOB_ID']}",
                    str(BATCH),
                    str(shard),
                    str(round_number + 1),
                ],
                text=True,
            ).strip()
            core.save(
                allocation / "successor.json",
                {"job_id": successor, "shard": shard, "round": round_number + 1},
            )
        else:
            successor = None
        cpus = sorted(os.sched_getaffinity(0))[:64]
        if len(cpus) < 8:
            raise RuntimeError("Fewer than eight allocated CPU cores")
        lanes = plan["lanes_per_node"]
        deadline = started + plan["round_work_budget_s"]
        regular = [
            job
            for job in active
            if core.read_state(job["case_id"])["state"] != "pending_isolated"
        ]
        isolated = lanes == 1 or not regular
        chosen = regular if regular else active[:1]
        active_lanes = 1 if isolated else min(lanes, len(chosen))
        width = len(cpus) // active_lanes
        pending = queue.Queue()
        for job in chosen:
            pending.put(job)
        errors = []
        with ThreadPoolExecutor(max_workers=active_lanes) as pool:
            futures = [
                pool.submit(
                    run_lane,
                    lane,
                    pending,
                    cpus[lane * width : (lane + 1) * width],
                    allocation,
                    deadline,
                    isolated,
                )
                for lane in range(active_lanes)
            ]
            for future in futures:
                try:
                    errors.extend(future.result())
                except (
                    Exception
                ) as exc:  # noqa: BLE001 - fail the allocation after all lanes finish.
                    errors.append({"type": type(exc).__name__, "message": str(exc)})
                    print(errors[-1], flush=True)
        status = shard_status(
            shard, names, round=round_number, job_id=os.environ["SLURM_JOB_ID"]
        )
        aggregate_status()
        unfinished = any(
            core.read_state(name)["state"] not in core.TERMINAL for name in names
        )
        core.save(
            allocation / "finished.json",
            {"finished_at": core.now(), "counts": status["counts"], "errors": errors},
        )
        if not unfinished and successor is not None:
            subprocess.run(["scancel", successor], check=True)
        if unfinished and successor is None:
            core.save(
                shard_root / "ALLOCATION_LIMIT_REACHED.json",
                {"at": core.now(), "counts": status["counts"]},
            )
        print(json.dumps(status), flush=True)
        if errors:
            raise RuntimeError("Shard encountered errors; inspect its case states")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "run", "status"])
    parser.add_argument("--shard", type=int)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--lanes-per-node", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--nodes", type=int, default=2)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(lanes_per_node=args.lanes_per_node, nodes=args.nodes)
    elif args.command == "status":
        print(json.dumps(aggregate_status()["counts"]))
    else:
        run(args.shard, args.round)
