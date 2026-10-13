# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from autoscaling_arena.match_config import feasible_worker_targets, parse_match_config

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_comparison", ROOT / "scripts/prepare_comparison.py"
)
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)
RUN_SPEC = importlib.util.spec_from_file_location(
    "run_comparison", ROOT / "scripts/run_comparison.py"
)
run = importlib.util.module_from_spec(RUN_SPEC)
RUN_SPEC.loader.exec_module(run)


@pytest.fixture
def campaign():
    catalog = json.loads((ROOT / "configs/60traces/catalog.json").read_text())
    policy = yaml.safe_load((ROOT / "configs/60traces/comparison.yaml").read_text())
    references = [
        {
            "benchmark_id": c["case_id"],
            "trace_sha256": c["trace_sha256"],
            "peak_rss_gib": 3,
            "simulation_wall_s": 60,
        }
        for c in catalog["cases"]
    ]
    return prepare.build_jobs(catalog, references, policy), policy


def test_selection_excludes_long_jev_cases_and_has_no_duplicate_static_runs(campaign):
    jobs, policy = campaign
    assert Counter(j["planner"] for j in jobs) == {
        "mpc": 60,
        "planner": 60,
        "keda": 60,
        "reactive": 60,
        "jev": 28,
    }
    assert Counter(j["group"] for j in jobs) == {
        "A-HT": 67,
        "A-LC": 67,
        "D-HT": 67,
        "D-LC": 67,
    }
    assert all(
        j["duration_s"] < 86400 and j["max_calls"] == 600
        for j in jobs
        if j["planner"] == "jev"
    )
    assert sum(j["max_calls"] for j in jobs) == 16800
    batches, blocked = prepare.schedule_batches(jobs, policy["parallel"])
    assert not blocked
    assert sorted(j["id"] for b in batches for j in b) == sorted(j["id"] for j in jobs)
    for batch in batches:
        assert len(batch) <= 4
        assert sum(j["planner"] == "jev" for j in batch) <= 1


def test_all_generated_configs_have_the_same_feasible_worker_range(campaign, tmp_path):
    jobs, policy = campaign
    fixture = tmp_path / "tiny.jsonl"
    fixture.write_text(
        '{"timestamp":0,"input_length":16,"output_length":8,"hash_ids":[1]}\n'
    )
    for job in jobs:
        base = yaml.safe_load(
            (ROOT / f"configs/60traces/{job['case_id']}.yaml").read_text()
        )
        path = tmp_path / "configs" / f"{job['id']}.yaml"
        config = prepare.make_config(base, job, policy, path, tmp_path)
        trace = config["evaluations"]["traces"][0]
        if "path" in trace:
            trace["path"] = str(fixture)
        else:
            trace["paths"] = [str(fixture)]
        resolved = parse_match_config(config, source_path=path)
        scaler = resolved.backend.autoscalers[0]
        assert resolved.expected_runs == 1
        assert resolved.execution.max_parallel_runs == 1
        targets = feasible_worker_targets(resolved.backend, scaler)
        assert len(targets) == (120 if job["group"].startswith("D") else 15)
        if job["planner"] == "mpc":
            assert scaler.config["cold_start_s"] == 0
            assert scaler.config["slo_ttft_ms"] == base["slo_profiles"][0]["ttft_ms"]
            model = scaler.config["forward_model"]
            if job["group"] == "D-LC":
                assert model["config"]["worker_type"] == "prefill"
                assert model["config"]["tp"] == 4
                assert model["decode_config"]["worker_type"] == "decode"
                assert model["decode_config"]["tp"] == 8
            elif job["group"].startswith("A"):
                assert model["config"] == model["decode_config"]
        if job["planner"] == "jev":
            assert scaler.config["max_calls"] == 600


def test_batches_respect_memory_cpu_exclusivity_and_report_oversized_jobs(campaign):
    jobs, policy = campaign
    selected = [dict(j) for j in jobs[:5]]
    selected[0].update(memory_gib=174, exclusive=True)
    selected[1].update(memory_gib=300, exclusive=True)
    selected[2].update(memory_gib=30)
    selected[3].update(memory_gib=30)
    parallel = {**policy["parallel"], "physical_cores": 6}
    batches, blocked = prepare.schedule_batches(selected, parallel)
    assert [j["id"] for j in blocked] == [selected[1]["id"]]
    for batch in batches:
        assert sum(j["memory_gib"] for j in batch) <= 224
        assert sum(j["cores"] for j in batch) <= 4
        if any(j["exclusive"] for j in batch):
            assert len(batch) == 1


def test_launcher_rejects_duplicated_jobs_and_config_drift(campaign, tmp_path):
    jobs, policy = campaign
    jobs = [dict(jobs[0])]
    path = tmp_path / "match.yaml"
    path.write_text("first config")
    jobs[0].update(
        config=path.name, config_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )
    plan = {
        "jobs": jobs,
        "policy": policy,
        "batches": [[jobs[0]["id"]]],
        "blocked_jobs": [],
    }
    run.validate_plan(plan, tmp_path)
    plan["batches"].append([jobs[0]["id"]])
    with pytest.raises(ValueError, match="more than one batch"):
        run.validate_plan(plan, tmp_path)
    plan["batches"].pop()
    path.write_text("changed config")
    with pytest.raises(ValueError, match="config changed"):
        run.validate_plan(plan, tmp_path)


def test_launcher_rejects_a_memory_oversubscribed_batch(campaign, tmp_path):
    jobs, policy = campaign
    jobs = [dict(j) for j in jobs[:4]]
    for job in jobs:
        job.update(memory_gib=60)
    plan = {
        "jobs": jobs,
        "policy": policy,
        "batches": [[j["id"] for j in jobs]],
        "blocked_jobs": [],
    }
    with pytest.raises(ValueError, match="batch exceeds"):
        run.validate_plan(plan, tmp_path)


def test_linux_launcher_assigns_disjoint_cores_and_caps_threads(
    campaign, tmp_path, monkeypatch
):
    jobs, policy = campaign
    jobs = [dict(j) for j in jobs[:2]]
    for job in jobs:
        job.update(config=f"{job['id']}.yaml", session=f"sessions/{job['id']}")
    plan = {
        "jobs": jobs,
        "policy": policy,
        "batches": [[j["id"] for j in jobs]],
        "blocked_jobs": [],
    }
    monkeypatch.setattr(run.sys, "platform", "linux")
    monkeypatch.setattr(run, "physical_cpu_ids", lambda: list(range(0, 64, 2)))
    monkeypatch.setattr(run, "cpu_capacity", lambda cpus: len(cpus))
    monkeypatch.setattr(run, "memory_capacity", lambda: (256, 240))
    monkeypatch.setattr(run.shutil, "which", lambda name: "/usr/bin/taskset")
    monkeypatch.setattr(
        run.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=256 * 1024**3, available=240 * 1024**3),
    )
    calls = []

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, poll=lambda: 0)

    monkeypatch.setattr(run.subprocess, "Popen", fake_popen)
    assert run.execute(plan, tmp_path) == 0
    assert [command[2] for command, _ in calls] == ["0,2", "4,6"]
    assert all(kwargs["env"]["OPENBLAS_NUM_THREADS"] == "1" for _, kwargs in calls)
    assert (
        len({command[command.index("--session-dir") + 1] for command, _ in calls}) == 2
    )


def test_linux_launcher_refuses_insufficient_available_memory(
    campaign, tmp_path, monkeypatch
):
    jobs, policy = campaign
    job = dict(jobs[0], memory_gib=174)
    plan = {
        "jobs": [job],
        "policy": policy,
        "batches": [[job["id"]]],
        "blocked_jobs": [],
    }
    monkeypatch.setattr(run.sys, "platform", "linux")
    monkeypatch.setattr(run, "physical_cpu_ids", lambda: list(range(32)))
    monkeypatch.setattr(run, "cpu_capacity", lambda cpus: len(cpus))
    monkeypatch.setattr(run, "memory_capacity", lambda: (256, 200))
    monkeypatch.setattr(run.shutil, "which", lambda name: "/usr/bin/taskset")
    monkeypatch.setattr(
        run.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=256 * 1024**3, available=200 * 1024**3),
    )
    with pytest.raises(RuntimeError, match="requires 206 GiB available"):
        run.execute(plan, tmp_path)


def test_container_limits_override_vm_resources(tmp_path, monkeypatch):
    monkeypatch.setattr(
        run.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=32 * 1024**3, available=30 * 1024**3),
    )
    (tmp_path / "memory.max").write_text(str(28 * 1024**3))
    (tmp_path / "memory.current").write_text(str(6 * 1024**3))
    (tmp_path / "memory.stat").write_text(f"inactive_file {2 * 1024**3}\n")
    (tmp_path / "cpu.max").write_text("600000 100000")
    assert run.memory_capacity(tmp_path) == (28, 24)
    assert run.cpu_capacity(list(range(18)), tmp_path) == 6
    assert run.cpu_capacity([0, 1], tmp_path) == 2
    (tmp_path / "memory.max").write_text("max")
    (tmp_path / "cpu.max").write_text("max 100000")
    assert run.memory_capacity(tmp_path) == (32, 30)
    assert run.cpu_capacity(list(range(18)), tmp_path) == 18


def test_resume_runs_only_pending_selected_planners_and_keeps_failures(
    campaign, tmp_path, monkeypatch
):
    jobs, policy = campaign
    selected = [
        dict(job, config=f"{job['id']}.yaml", session=f"sessions/{job['id']}")
        for job in jobs
        if job["case_id"] == "A-HT-S05"
    ]
    states = {j["id"]: "pending" for j in selected}
    states[selected[0]["id"]] = "completed"
    states[selected[1]["id"]] = "failed"
    (tmp_path / "status.json").write_text(
        json.dumps({"jobs": states, "pending_resources": []})
    )
    plan = {
        "jobs": selected,
        "policy": policy,
        "batches": [[j["id"]] for j in selected],
        "blocked_jobs": [],
    }
    monkeypatch.setattr(run.sys, "platform", "linux")
    monkeypatch.setattr(run, "physical_cpu_ids", lambda: list(range(32)))
    monkeypatch.setattr(run, "cpu_capacity", lambda cpus: len(cpus))
    monkeypatch.setattr(run, "memory_capacity", lambda: (256, 240))
    monkeypatch.setattr(run.shutil, "which", lambda name: "/usr/bin/taskset")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    commands = []

    def launch(command, **kwargs):
        commands.append(command)
        code = 1 if len(commands) == 1 else 0
        return SimpleNamespace(returncode=code, poll=lambda: code)

    monkeypatch.setattr(run.subprocess, "Popen", launch)
    assert (
        run.execute(
            plan,
            tmp_path,
            resume_pending=True,
            planners={"mpc", "planner", "keda", "reactive"},
            keep_going=True,
        )
        == 1
    )
    after = json.loads((tmp_path / "status.json").read_text())["jobs"]
    assert len(commands) == 2
    assert after[selected[0]["id"]] == "completed"
    assert after[selected[1]["id"]] == "failed"
    assert after[selected[2]["id"]] == "failed"
    assert after[selected[3]["id"]] == "completed"
    assert after[selected[4]["id"]] == "pending"
    assert (
        run.execute(
            plan,
            tmp_path,
            resume_pending=True,
            planners={"mpc", "planner", "keda", "reactive"},
            keep_going=True,
        )
        == 1
    )
    assert len(commands) == 2  # Neither failures nor completed runs are retried.


def test_oversized_cases_remain_pending_when_partial_execution_is_explicit(
    campaign, tmp_path, monkeypatch
):
    _, policy = campaign
    plan = {"jobs": [], "policy": policy, "batches": [], "blocked_jobs": ["large-case"]}
    monkeypatch.setattr(run.sys, "platform", "linux")
    monkeypatch.setattr(run, "physical_cpu_ids", lambda: list(range(32)))
    monkeypatch.setattr(run, "cpu_capacity", lambda cpus: len(cpus))
    monkeypatch.setattr(run, "memory_capacity", lambda: (256, 240))
    monkeypatch.setattr(run.shutil, "which", lambda name: "/usr/bin/taskset")
    with pytest.raises(ValueError, match="allow-blocked"):
        run.execute(plan, tmp_path)
    assert run.execute(plan, tmp_path, allow_blocked=True) == 0
    assert json.loads((tmp_path / "status.json").read_text())["pending_resources"] == [
        "large-case"
    ]
