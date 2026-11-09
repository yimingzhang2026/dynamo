# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise complete publication locally with synthetic, GPU-free evidence."""

import copy
import sys
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parent
sys.path[:0] = [str(CODE), str(CODE.parents[1] / "src")]
import publish_collection as publisher  # noqa: E402
from export_completed import export  # noqa: E402

pytest_plugins = ["test_import_results"]


def seal(folder, jobs):
    publisher.save(
        folder / "manifest.json",
        {
            "campaign": publisher.CAMPAIGN,
            "completed": len(jobs),
            "jobs": jobs,
            "source_archive_sha256": publisher.SOURCE_SHA256,
            "files": {
                str(p.relative_to(folder)): {
                    "sha256": publisher.digest(p),
                    "bytes": p.stat().st_size,
                }
                for p in folder.rglob("*")
                if p.is_file() and p.name != "manifest.json"
            },
        },
    )


@pytest.fixture
def bundles(completed_campaign, tmp_path):
    campaign, inputs = completed_campaign
    plan = publisher.read(campaign / "plan.json")
    cpu_plan = copy.deepcopy(plan)
    cpu_plan["jobs"] = [job for job in plan["jobs"] if job["planner"] != "knowline31b"]
    publisher.save(campaign / "plan.json", cpu_plan)
    model_campaign = tmp_path / "model-campaign"
    model_campaign.mkdir()
    model_jobs = []
    for job in plan["jobs"]:
        path = campaign / "results" / job["case_id"] / job["planner"] / "results.json"
        budget = job["resources"]["gpu_budget"]
        audit = {
            "gpu_budget": budget,
            "peak_allocated_gpus": job["resources"]["baseline_gpus"],
            "telemetry_samples": 1,
        }
        if job["planner"] == "knowline31b":
            config = campaign / job["config"]
            # The model exporter reads the executed attempt config beside its result.
            (path.parent / "config.yaml").write_bytes(config.read_bytes())
            directory = model_campaign / "cases" / job["case_id"]
            directory.mkdir(parents=True)
            publisher.save(
                directory / "state.json",
                {
                    "state": "completed",
                    "result": str(path),
                    "budget_audit": audit,
                    "decision_summary": publisher.read(path)["results"][0]["runtime"][
                        "jev"
                    ],
                },
            )
            model_jobs.append({**job, "config": str(config)})
        else:
            publisher.save(
                path.parent / "COMPLETE.json",
                {"results_sha256": publisher.digest(path), "budget_audit": audit},
            )
    publisher.save(model_campaign / "plan.json", {"cases": model_jobs})
    collection = tmp_path / "collection"
    for host, source, knowline in (
        ("cpu", campaign, False),
        ("knowline", model_campaign, True),
    ):
        folder = collection / host
        export(source, folder / "exports", knowline=knowline)
        publisher.save(folder / "plan.json", publisher.read(source / "plan.json"))
        (folder / "catalog.json").write_bytes((inputs / "catalog.json").read_bytes())
        (folder / "diagnostics").mkdir()
        seal(folder, sorted(p.stem for p in (folder / "exports").glob("*.json")))
    return collection


def test_complete_bundle_publishes_360_current_results(bundles, tmp_path):
    output = tmp_path / "leaderboards" / publisher.CAMPAIGN
    report = publisher.build(bundles, output)
    assert report["counts"] == {"completed": 360}
    assert report["resource_policy"] == {"gpu_budget_extra_workers": 3}
    assert report["static_baselines"] == report["knowline_completed"] == 60
    assert set(report["core"]) == {
        "mpc",
        "planner",
        "keda",
        "reactive",
        "static",
        "knowline31b",
    }
    assert len(list(output.glob("*.html"))) == 5
    assert publisher.read(output / "missing-results.json") == []
    assert len(publisher.read(output / "inputs/static-latency.json")["records"]) == 60
    assert all(
        "budget_audit" in row
        for row in publisher.read(output / "inputs/static-latency.json")["records"]
    )
    assert "gpu_budget_extra_workers" in (output / "index.html").read_text()


def test_corrupt_download_is_rejected_before_publication(bundles, tmp_path):
    path = next((bundles / "cpu/exports").glob("*.json"))
    path.write_text(path.read_text() + " ")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="Transfer checksum mismatch"):
        publisher.build(bundles, output)
    assert not output.exists()


def test_incomplete_campaign_cannot_publish(bundles, tmp_path):
    folder = bundles / "knowline"
    next((folder / "exports").glob("*.json")).unlink()
    seal(folder, sorted(p.stem for p in (folder / "exports").glob("*.json")))
    with pytest.raises(ValueError, match="Unexpected collected campaign"):
        publisher.build(bundles, tmp_path / "output")
