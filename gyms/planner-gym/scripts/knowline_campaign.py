# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Selected full traces, four GPU pairs, durable exact-decision continuation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import yaml
from audit_gpu_usage import audit_budget
from autoscaling_arena.decision_replay import merge_prefix
from autoscaling_arena.decision_replay import valid_prefix as validated_decision_prefix
from autoscaling_arena.jev_report import summarize_decisions
from autoscaling_arena.match_config import load_match_config
from autoscaling_arena.resource_policy import case_resources
from prepare_comparison import make_config

GYM = Path(os.environ.get("PLANNER_GYM", Path(__file__).resolve().parents[1])).resolve()
BASE = Path(os.environ.get("KNOWLINE_WORKSPACE", GYM / "runs/knowline")).resolve()
ROOT = BASE / "campaign"
TRACE_DIR = Path(
    os.environ.get("PLANNER_TRACE_DIR", GYM / "data/60traces/reproduction/generated")
).resolve()
MODEL = "PelaAI/KnowLine-31B-Gen1"
REVISION = "8f052a8130e599b53e5324c34f0f5ff90696e1c6"
GROUPS = ("A-HT", "A-LC", "D-HT", "D-LC")
MAX_ROUNDS = 16
TERMINAL = {"completed", "failed", "needs_longer_allocation"}


def now():
    return datetime.now(UTC).isoformat()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_state(case):
    path = ROOT / "cases" / case / "state.json"
    return (
        json.loads(path.read_text())
        if path.exists()
        else {"state": "pending", "attempts": []}
    )


def valid_prefix(path):
    return validated_decision_prefix(path, model=MODEL)


def harvest(directory, state):
    cache = directory / "saved-decisions.jsonl"
    rows = valid_prefix(cache)
    for attempt in state["attempts"]:
        logs = list(
            Path(attempt["directory"]).glob("session/runs/*/jev-decisions.jsonl")
        )
        if len(logs) > 1:
            raise ValueError("Expected one run per attempt")
        if logs:
            rows = merge_prefix(rows, valid_prefix(logs[0]))
    temporary = cache.with_suffix(".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(cache)
    state["saved_decisions"] = len(rows)
    return rows


def prepare(case_ids=None, *, policy=None):
    if (ROOT / "plan.json").exists():
        raise RuntimeError("Campaign is already prepared")
    catalog = json.loads((GYM / "configs/60traces/catalog.json").read_text())["cases"]
    if not len(catalog) == len({row["case_id"] for row in catalog}) == 60:
        raise ValueError(
            "Campaign validation failed: len(catalog) == len({row['case_id'] for row in catalog}) == 60"
        )
    if case_ids is not None:
        if not case_ids or len(case_ids) != len(set(case_ids)):
            raise ValueError("Selection must contain distinct case IDs")
        unknown = set(case_ids) - {row["case_id"] for row in catalog}
        if unknown:
            raise ValueError(f"Unknown selected cases: {sorted(unknown)}")
        catalog = [row for row in catalog if row["case_id"] in set(case_ids)]
    policy = (
        yaml.safe_load((GYM / "configs/60traces/comparison.yaml").read_text())
        if policy is None
        else copy.deepcopy(policy)
    )
    jobs = []
    for case in catalog:
        name = case["case_id"]
        directory = ROOT / "cases" / name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "base.yaml"
        original = yaml.safe_load(
            (GYM / "configs/60traces" / f"{name}.yaml").read_text()
        )
        job = {
            "resources": case_resources(original, policy),
            "id": f"{name}-knowline31b",
            "case_id": name,
            "planner": "knowline31b",
            "timeout_s": 13200,
            "memory_gib": 385,
        }
        config = make_config(original, job, policy, path, ROOT)
        config["description"] += " Independent open decision model."
        config["labels"].update(
            planner="knowline-31b",
            comparison="60traces-knowline",
            model_revision=REVISION,
            serving_backend="transformers-cu126",
        )
        adapter = config["backend"]["autoscalers"][0]
        adapter["name"] = "KnowLine-31B"
        adapter["config"].update(
            provider="self_hosted",
            model=MODEL,
            base_url="http://127.0.0.1:18080",
            timeout_s=180,
            failure_mode="raise",
        )
        trace = config["evaluations"]["traces"][0]
        trace_path = str(TRACE_DIR / f"{name}.jsonl")
        if trace["format"] == "dynamo":
            trace.pop("path", None)
            trace["paths"] = [trace_path]
        else:
            trace.pop("paths", None)
            trace["path"] = trace_path
        config["publish"] = {
            "artifact_root": str(directory / "unused-artifacts"),
            "destinations": [{"type": "json", "path": str(directory / "unused.json")}],
        }
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        input_present = Path(trace_path).exists()
        if input_present:
            load_match_config(path)
        if not config["slo_profiles"] == original["slo_profiles"]:
            raise ValueError(
                "Campaign validation failed: config['slo_profiles'] == original['slo_profiles']"
            )
        if (
            not config["evaluations"]["defaults"]["seed"]
            == original["evaluations"]["defaults"]["seed"]
        ):
            raise ValueError(
                "Campaign validation failed: config['evaluations']['defaults']['seed'] == original['evaluations']['defaults']['seed']"
            )
        for key in ("start_s", "end_s"):
            if (
                not config["evaluations"]["defaults"]["measurement_window"][key]
                == original["evaluations"]["defaults"]["measurement_window"][key]
            ):
                raise ValueError(
                    "Campaign validation failed: config['evaluations']['defaults']['measurement_window'][key] == original['evaluations']['defaults']['measurement_window'][key]"
                )
        jobs.append(
            {
                "case_id": name,
                "group": name[:4],
                "resources": job["resources"],
                "duration_s": case["test_duration_s"],
                "bytes": case["expanded_test_bytes"],
                "trace_sha256": case["trace_sha256"],
                "config_sha256": digest(path),
                "config": str(path),
                "validation": "passed" if input_present else "pending_trace_staging",
            }
        )
    plan = {
        "created_at": now(),
        "model": MODEL,
        "revision": REVISION,
        "cases": jobs,
        "total_cases": len(jobs),
        "simultaneous_runs": 4,
        "gpu_pairs": [[0, 1], [2, 3], [4, 5], [6, 7]],
        "memory_guard_gib_per_run": 384,
        "isolated_memory_guard_gib": 1400,
        "round_wall_budget_s": 13200,
        "max_rounds": MAX_ROUNDS,
        "continuation": "Replay exact saved decisions from full trace start; reject changed requests.",
        "launcher_sha256": digest(__file__),
    }
    save(ROOT / "plan.json", plan)
    print(json.dumps({"prepared": len(jobs), "path": str(ROOT)}, indent=2), flush=True)


def collect_status(plan):
    states = {job["case_id"]: read_state(job["case_id"]) for job in plan["cases"]}
    counts = {}
    for state in states.values():
        counts[state["state"]] = counts.get(state["state"], 0) + 1
    result = {
        "updated_at": now(),
        "total": len(plan["cases"]),
        "counts": counts,
        "cases": states,
    }
    save(ROOT / "status.json", result)
    if counts.get("completed") == len(plan["cases"]):
        (ROOT / "CAMPAIGN_COMPLETE").write_text(now() + "\n")
    return result


def stop_server(process):
    """Release the GPU pair before starting another server on it."""
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def is_server_failure(state):
    """Retry transport failures, never malformed model decisions or audit errors."""
    error = state.get("last_error") or {}
    transport_errors = (
        "TimeoutError",
        "ReadTimeout",
        "WriteTimeout",
        "ConnectTimeout",
        "PoolTimeout",
        "ConnectError",
        "ReadError",
        "WriteError",
        "RemoteProtocolError",
    )
    return (
        state.get("state") == "failed"
        and error.get("type") == "RuntimeError"
        and error.get("message")
        in {
            f"Jev decision failed ({kind}); see decision log"
            for kind in transport_errors
        }
    )


def start_server(lane, cpus, allocation):
    model = BASE / "models/KnowLine-31B-Gen1-8f052a8"
    provenance = json.loads((model / "model-manifest.json").read_text())
    if not (provenance["model"] == MODEL and provenance["revision"] == REVISION):
        raise ValueError(
            "Campaign validation failed: provenance['model'] == MODEL and provenance['revision'] == REVISION"
        )
    for name, expected in provenance["serving_source_sha256"].items():
        if not digest(model / name) == expected:
            raise ValueError(
                "Campaign validation failed: digest(model / name) == expected"
            )
    port = 18080 + lane
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": f"{lane * 2},{lane * 2 + 1}"}
    log = (allocation / f"server-{lane}.log").open("x")
    command = [
        "taskset",
        "-c",
        ",".join(map(str, cpus)),
        os.environ.get("KNOWLINE_SERVE_PYTHON", str(BASE / "serve-env/bin/python")),
        str(model / "knowline_server.py"),
        "--model",
        str(model),
        "--backend",
        "hf",
        "--temperature",
        "1",
        "--workers",
        "1",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--temperatures",
        str(model / "temperatures.json"),
        "--box-prompt",
        "gemma",
        "--box-coords",
        "yxyx",
    ]
    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    log.close()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        until = time.monotonic() + 1200
        while time.monotonic() < until:
            if process.poll() is not None:
                raise RuntimeError(
                    f"Server lane {lane} exited with {process.returncode}"
                )
            try:
                with opener.open(
                    f"http://127.0.0.1:{port}/health", timeout=5
                ) as response:
                    if response.status == 200:
                        return (process, port)
            except (OSError, TimeoutError):
                pass
            time.sleep(3)
        raise TimeoutError("Model startup timed out")
    except BaseException:
        stop_server(process)
        raise


def run_case(job, lane, cpus, port, deadline, isolated):
    name = job["case_id"]
    directory = ROOT / "cases" / name
    state = read_state(name)
    rows = harvest(directory, state)
    before = len(rows)
    if not digest(job["config"]) == job["config_sha256"]:
        raise ValueError(
            "Campaign validation failed: digest(job['config']) == job['config_sha256']"
        )
    trace = TRACE_DIR / f"{name}.jsonl"
    if not trace.exists() or trace.stat().st_size != job["bytes"]:
        state.update(state="waiting_for_data", updated_at=now())
        save(directory / "state.json", state)
        return
    if not (directory / "trace-verified.json").exists():
        if digest(trace) != job["trace_sha256"]:
            raise ValueError(f"{name}: trace checksum mismatch")
        save(
            directory / "trace-verified.json",
            {"sha256": job["trace_sha256"], "bytes": job["bytes"]},
        )
    remaining = int(deadline - time.monotonic()) - 90
    if remaining < 120:
        return
    attempt_dir = directory / f"attempt-{len(state['attempts']) + 1:03d}"
    attempt_dir.mkdir()
    config = yaml.safe_load(Path(job["config"]).read_text())
    config["execution"].update(
        timeout_s=remaining, max_memory_mb=(1400 if isolated else 384) * 1024
    )
    adapter = config["backend"]["autoscalers"][0]["config"]
    adapter["base_url"] = f"http://127.0.0.1:{port}"
    if rows:
        saved = attempt_dir / "prior-decisions.jsonl"
        saved.write_bytes((directory / "saved-decisions.jsonl").read_bytes())
        adapter.update(
            prior_calls=max(len(rows), max(row["api_calls_used"] for row in rows)),
            replay_decisions=str(saved),
            replay_decisions_sha256=digest(saved),
        )
    config["publish"] = {
        "artifact_root": str(attempt_dir / "artifacts"),
        "destinations": [{"type": "json", "path": str(attempt_dir / "results.json")}],
    }
    config_path = attempt_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    attempt = {
        "directory": str(attempt_dir),
        "job_id": os.environ["SLURM_JOB_ID"],
        "started_at": now(),
        "prior_decisions": before,
        "isolated": isolated,
        "timeout_s": remaining,
        "cpu_ids": cpus,
        "gpu_pair": [lane * 2, lane * 2 + 1],
    }
    state["attempts"].append(attempt)
    state.update(state="running", updated_at=now())
    save(directory / "state.json", state)
    print(
        f"{now()} Starting {name}, {before} saved decisions, {remaining}s budget",
        flush=True,
    )
    command = [
        "taskset",
        "-c",
        ",".join(map(str, cpus)),
        sys.executable,
        str(GYM / "scripts/run_match_config.py"),
        str(config_path),
        "--session-dir",
        str(attempt_dir / "session"),
        "--expect-trace-sha256",
        job["trace_sha256"],
    ]
    with (attempt_dir / "launcher.log").open("x") as log:
        process = subprocess.run(
            command,
            check=False,
            cwd=GYM,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    attempt.update(exit_code=process.returncode, finished_at=now())
    rows = harvest(directory, state)
    report_path = attempt_dir / "results.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    result = (report.get("results") or [{}])[0]
    error = result.get("error") or {}
    if (
        process.returncode == 0
        and report.get("summary", {}).get("status") == "ok"
        and (len(report.get("results", [])) == 1)
        and (result.get("status") == "ok")
        and (result.get("autoscaler") == "KnowLine-31B")
    ):
        state["budget_audit"] = audit_budget(
            attempt_dir / "session", config["backend"]["gpu_budget"]
        )
        state.update(state="completed", result=str(report_path), completed_at=now())
        state.pop("last_error", None)
        state["decision_summary"] = summarize_decisions(
            directory / "saved-decisions.jsonl"
        )
        if state["decision_summary"]["errors"]:
            raise ValueError("Successful result contains failed decisions")
    elif error.get("type") == "RunTimeoutError":
        stalled = state.get("stalled_allocations", 0) + 1 if len(rows) <= before else 0
        state.update(
            state="needs_longer_allocation" if stalled >= 2 else "pending_continuation",
            stalled_allocations=stalled,
            last_error=error,
        )
    elif error.get("type") == "RunMemoryLimitError" and (not isolated):
        state.update(state="pending_isolated", last_error=error)
    else:
        state.update(
            state="failed",
            last_error=error or {"type": "MissingReportOrLauncherFailure"},
        )
    state["updated_at"] = now()
    save(directory / "state.json", state)
    print(f"{now()} {name}: {state['state']}; {len(rows)} saved decisions", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "status"])
    parser.add_argument(
        "--selection", type=Path, help="JSON list of exact case IDs to prepare"
    )
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(json.loads(args.selection.read_text()) if args.selection else None)
    else:
        print(
            json.dumps(
                collect_status(json.loads((ROOT / "plan.json").read_text()))["counts"]
            )
        )
