# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "retry_memory_failures", ROOT / "scripts/retry_memory_failures.py"
)
retry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(retry)
pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    trace = source / "tiny.jsonl"
    trace.write_text(
        '{"timestamp":0,"input_length":16,"output_length":8,"hash_ids":[1]}\n'
    )
    base = yaml.safe_load((ROOT / "configs/60traces/A-HT-S05.yaml").read_text())
    base["evaluations"]["traces"][0]["path"] = str(trace)
    base["evaluations"]["traces"][0]["format"] = "mooncake"
    base["evaluations"]["traces"][0].pop("paths", None)
    (source / "A-HT-S05.yaml").write_text(yaml.safe_dump(base))
    monkeypatch.setattr(retry.prepare, "SOURCE", source)
    policy = yaml.safe_load((ROOT / "configs/60traces/comparison.mac.yaml").read_text())
    directory = tmp_path / "campaign"
    directory.mkdir()
    jobs = []
    states = {
        "mpc": "failed",
        "keda": "failed",
        "jev": "failed",
        "reactive": "completed",
    }
    for planner in [*states, "planner"]:
        job_id = f"A-HT-S05-{planner}"
        config = directory / "configs" / "A-HT-S05" / f"{planner}.yaml"
        config.parent.mkdir(parents=True, exist_ok=True)
        job = {
            "id": job_id,
            "case_id": "A-HT-S05",
            "group": "A-HT",
            "planner": planner,
            "memory_gib": 174 if planner == "planner" else 8,
            "cores": 2,
            "exclusive": planner == "planner",
            "timeout_s": 3600,
            "config": str(config.relative_to(directory)),
            "session": f"sessions/A-HT-S05/{planner}",
            "trace_sha256": "unchanged",
        }
        config.write_text(
            retry.prepare.HEADER
            + yaml.safe_dump(
                retry.prepare.make_config(base, job, policy, config, directory),
                sort_keys=False,
            )
        )
        job["config_sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
        jobs.append(job)
        if planner not in states:
            continue
        (directory / job["session"]).mkdir(parents=True)
        (directory / job["session"] / "original.txt").write_text(
            "full original trace artifacts"
        )
        logs = directory / "logs"
        logs.mkdir(exist_ok=True)
        (logs / f"{job_id}.log").write_text("original log")
        results = directory / "results" / job["case_id"] / planner
        results.mkdir(parents=True)
        report = {
            "results": [
                {
                    "status": states[planner],
                    "error": {
                        "type": "RunTimeoutError"
                        if planner == "keda"
                        else "RunMemoryLimitError"
                    },
                    "execution": {"max_memory_mb": 7168, "peak_rss_mb": 7200},
                }
            ]
        }
        (results / "results.json").write_text(json.dumps(report))
    blocked = ["A-HT-S05-planner"]
    status = {
        "jobs": {f"A-HT-S05-{k}": v for k, v in states.items()},
        "pending_resources": blocked,
    }
    plan = {
        "jobs": jobs,
        "policy": policy,
        "blocked_jobs": blocked,
        "batches": [[k] for k in status["jobs"]],
        "notes": [],
    }
    retry.save(directory / "plan.json", plan)
    retry.save(directory / "status.json", status)
    return directory, plan, status


@pytest.mark.parametrize("state", ["pending", "running"])
def test_retry_waits_for_entire_original_campaign(campaign, state):
    directory, _, status = campaign
    status["jobs"]["A-HT-S05-reactive"] = state
    retry.save(directory / "status.json", status)
    with pytest.raises(ValueError, match="must finish"):
        retry.prepare_retry(directory)
    assert not (directory / retry.ARCHIVE).exists()


def test_retry_preserves_attempts_and_benchmark_semantics(campaign):
    directory, old_plan, old_status = campaign
    snapshots = {
        p.relative_to(directory): p.read_bytes()
        for p in directory.rglob("*")
        if p.is_file()
    }
    selected = retry.prepare_retry(directory)
    assert set(selected) == {"A-HT-S05-mpc"}
    plan = json.loads((directory / "plan.json").read_text())
    status = json.loads((directory / "status.json").read_text())
    assert status["pending_resources"] == old_status["pending_resources"]
    assert status["jobs"] == {**old_status["jobs"], "A-HT-S05-mpc": "pending"}
    assert plan["policy"]["jev"] == old_plan["policy"]["jev"]
    assert plan["policy"]["parallel"]["max_parallel_runs"] == 1
    for before, after in zip(old_plan["jobs"], plan["jobs"]):
        if before["id"] not in selected:
            assert before == after
            assert (directory / before["config"]).read_bytes() == snapshots[
                Path(before["config"])
            ]
            continue
        assert after["memory_gib"] == 28
        assert after["exclusive"] is True
        old_config = yaml.safe_load(snapshots[Path(before["config"])])
        new_config = yaml.safe_load((directory / after["config"]).read_text())
        old_config["execution"]["max_memory_mb"] = 27 * 1024
        assert old_config == new_config
        assert after["trace_sha256"] == before["trace_sha256"]
    archive = directory / retry.ARCHIVE
    for relative, content in snapshots.items():
        if "mpc" in str(relative):
            assert (archive / relative).read_bytes() == content
        elif relative.parts[0] not in {"plan.json", "status.json"}:
            assert (directory / relative).read_bytes() == content
    retry.run.validate_plan(plan, directory)
    assert not (directory / "sessions/A-HT-S05/mpc").exists()
    # Once the retry is terminal, another invocation still cannot spend a second attempt.
    status["jobs"]["A-HT-S05-mpc"] = "failed"
    retry.save(directory / "status.json", status)
    with pytest.raises(ValueError, match="already exists"):
        retry.prepare_retry(directory)


def test_no_repeat_when_failure_already_reached_safe_ceiling(campaign):
    directory, plan, status = campaign
    path = directory / "results/A-HT-S05/mpc/results.json"
    report = json.loads(path.read_text())
    report["results"][0]["execution"]["max_memory_mb"] = 27 * 1024
    path.write_text(json.dumps(report))
    assert retry.candidates(directory, plan, status) == {}
