# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise portable campaign import through full coverage and semantic validation."""

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest
import yaml
from build_leaderboards import make_report
from import_results import import_campaign
from test_evidence import completed_record, static_record

ROOT = Path(__file__).parent
GYM = ROOT.parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_benchmark", GYM / "scripts/prepare_comparison.py"
)
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


@pytest.fixture
def completed_campaign(tmp_path):
    inputs = tmp_path / "inputs"
    shutil.copytree(ROOT / "inputs", inputs)
    catalog = json.loads((inputs / "catalog.json").read_text())
    references = prepare.load_references(inputs / "static-latency.json")
    policy = yaml.safe_load((GYM / "configs/60traces/comparison.yaml").read_text())
    jobs = prepare.build_jobs(catalog, references, policy)
    cases = {c["case_id"]: c for c in catalog["cases"]}
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    for job in jobs:
        case = cases[job["case_id"]]
        config_path = campaign / "configs" / f"{job['id']}.yaml"
        base = yaml.safe_load(
            (GYM / "configs/60traces" / f"{job['case_id']}.yaml").read_text()
        )
        config = prepare.make_config(base, job, policy, config_path, campaign)
        config_path.parent.mkdir(exist_ok=True)
        config_path.write_text(yaml.safe_dump(config))
        job.update(
            config=str(config_path.relative_to(campaign)),
            config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        )
        if job["planner"] == "static":
            saved = static_record(case)["result"]
            report = {
                "match": {"name": job["id"]},
                "summary": {"status": "ok", "succeeded_runs": 1, "failed_runs": 0},
                "provenance": {"packages": saved["runtime"]["packages"]},
                "results": [
                    {
                        "status": "ok",
                        "autoscaler_type": "static",
                        "metrics": saved["metrics"],
                        "raw_metrics": saved["raw_metrics"],
                        "execution": {
                            "wall_time_s": saved["execution"]["wall_time_s"],
                            "peak_rss_mb": saved["execution"]["peak_rss_gib"] * 1024,
                        },
                        "evaluation": {
                            "trace": {"sha256": case["trace_sha256"]},
                            "sla": {
                                "ttft_ms": case["ttft_ms"],
                                "itl_ms": case["itl_ms"],
                            },
                        },
                    }
                ],
            }
        else:
            # Synthetic complete campaign: reuse this case's verified workload/score
            # fields, then set the controller metadata exercised by the importer.
            report = completed_record(job["planner"], job["case_id"])[0]["report"]
            controller = config["backend"]["autoscalers"][0]
            report["match"] = {"name": job["id"], "labels": config["labels"]}
            saved = report["results"][0]
            saved.update(
                autoscaler=controller["name"], autoscaler_type=controller["type"]
            )
            saved["runtime"]["autoscaler_config"] = controller["config"]
            if job["planner"] == "knowline31b":
                saved["runtime"]["jev"] = {
                    "calls": 1,
                    "errors": 0,
                    "models": [controller["config"]["model"]],
                    "replayed_decisions": 0,
                }
        # Import fixtures carry the executed budget required by publication.
        report["results"][0].setdefault("runtime", {})["gpu_budget"] = config[
            "backend"
        ]["gpu_budget"]
        path = campaign / "results" / job["case_id"] / job["planner"] / "results.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(report))
    (campaign / "plan.json").write_text(json.dumps({"policy": policy, "jobs": jobs}))
    (campaign / "status.json").write_text(
        json.dumps({"jobs": {j["id"]: "completed" for j in jobs}})
    )
    return campaign, inputs


def test_full_campaign_round_trip_keeps_scores_and_requires_new_diagnoses(
    completed_campaign,
):
    campaign, inputs = completed_campaign
    before = make_report(inputs)
    import_campaign(campaign, inputs)
    after = make_report(inputs, include_diagnostics=False)
    assert before["experiment_setup"]["budget_mismatches"] == {}
    assert after["experiment_setup"]["budget_mismatches"] == {}
    assert after["counts"] == {"completed": 360}
    assert after["resource_policy"] == {"gpu_budget_extra_workers": 3}
    assert any(
        "plus 3 × the largest per-worker GPU cost" in line
        for line in after["methodology"]
    )
    for setup in after["experiment_setup"]["rows"].values():
        assert setup["gpu_budget"] == setup["initial_gpus"] + 3 * max(
            setup["gpus_per_worker"].values()
        )
        assert setup["headroom_gpus"] == 24
        assert "gpu_budget_multiplier" not in setup
    assert "comparison" not in after
    expected = {
        json.loads(path.read_text())["match"]["name"].removesuffix(
            "-static"
        ): json.loads(path.read_text())["results"][0]["metrics"]
        for path in (campaign / "results").glob("*/static/results.json")
    }
    assert {
        k: r["metrics"] for k, r in after["static_latency"]["rows"].items()
    } == expected
    with pytest.raises(ValueError, match="Diagnos"):
        make_report(inputs)
    assert str(campaign) not in (inputs / "results.json").read_text()


def test_bad_executed_static_trace_leaves_all_published_inputs_unchanged(
    completed_campaign,
):
    campaign, inputs = completed_campaign
    before = {p.name: p.read_bytes() for p in inputs.glob("*.json")}
    path = next((campaign / "results").glob("*/static/results.json"))
    report = json.loads(path.read_text())
    report["results"][0]["evaluation"]["trace"]["sha256"] = "wrong-trace"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Static executed workload"):
        import_campaign(campaign, inputs)
    assert {p.name: p.read_bytes() for p in inputs.glob("*.json")} == before


@pytest.mark.parametrize("change_config", [False, True])
def test_inflated_gpu_budget_is_rejected_before_publication(
    completed_campaign, change_config
):
    campaign, inputs = completed_campaign
    before = {p.name: p.read_bytes() for p in inputs.glob("*.json")}
    path = campaign / "results/D-LC-S01/mpc/results.json"
    report = json.loads(path.read_text())
    report["results"][0]["runtime"]["gpu_budget"] = 124
    path.write_text(json.dumps(report))
    if change_config:
        config_path = campaign / "configs/D-LC-S01-mpc.yaml"
        config = yaml.safe_load(config_path.read_text())
        config["backend"]["gpu_budget"] = 124
        config_path.write_text(yaml.safe_dump(config))
        plan = json.loads((campaign / "plan.json").read_text())
        next(j for j in plan["jobs"] if j["id"] == "D-LC-S01-mpc")[
            "config_sha256"
        ] = hashlib.sha256(config_path.read_bytes()).hexdigest()
        (campaign / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="executed GPU budget must both match"):
        import_campaign(campaign, inputs)
    assert {p.name: p.read_bytes() for p in inputs.glob("*.json")} == before
