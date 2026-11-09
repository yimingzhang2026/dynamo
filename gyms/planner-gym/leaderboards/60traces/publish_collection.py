# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate the downloaded current campaign and publish its separate leaderboard."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

CODE = Path(__file__).resolve().parent
GYM = CODE.parents[1]
CAMPAIGN = "efficiency-plus3-20261106"
sys.path[:0] = [str(CODE), str(GYM / "src")]
from build_leaderboards import make_report  # noqa: E402
from import_results import experiment_setup, portable  # noqa: E402
from publish_latest import publish, refresh  # noqa: E402

SOURCE_SHA256 = "440a7f925d4cd9f1772140a8fd43434c1df8f7d66bf209c5b09b0e285f557bf7"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_bundle(path, expected, campaign=CAMPAIGN, source_sha256=None):
    manifest = read(path / "manifest.json")
    require(
        manifest["campaign"] == campaign and manifest["completed"] == expected,
        "Unexpected collected campaign",
    )
    if source_sha256 is not None:
        require(
            manifest["source_archive_sha256"] == source_sha256, "Wrong campaign source"
        )
    for name, record in manifest["files"].items():
        relative = Path(name)
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            "Unsafe manifest path",
        )
        file = path / relative
        require(
            file.is_file()
            and file.stat().st_size == record["bytes"]
            and digest(file) == record["sha256"],
            "Transfer checksum mismatch: " + name,
        )
    exports = {p.stem for p in (path / "exports").glob("*.json")}
    require(
        exports == set(manifest["jobs"]) and len(exports) == expected,
        "Missing or unexpected collected result",
    )
    return manifest


def build(collection, output, *, campaign=CAMPAIGN, source_sha256=SOURCE_SHA256):
    cpu = collection / "cpu"
    knowline = collection / "knowline"
    manifests = [
        verify_bundle(cpu, 300, campaign, source_sha256),
        verify_bundle(knowline, 60, campaign, source_sha256),
    ]
    require(
        read(cpu / "catalog.json") == read(knowline / "catalog.json"),
        "Campaign catalogs differ",
    )
    cpu_plan, model_plan = read(cpu / "plan.json"), read(knowline / "plan.json")
    policy = copy.deepcopy(cpu_plan["policy"])
    require(
        policy["resources"] == {"gpu_budget_extra_workers": 3}, "Wrong resource policy"
    )
    policy["planners"] = ["mpc", "planner", "keda", "reactive", "static", "knowline31b"]
    jobs = [job for job in cpu_plan["jobs"] if job["planner"] != "static"]
    jobs += [
        {**job, "id": job["case_id"] + "-knowline31b", "planner": "knowline31b"}
        for job in model_plan["cases"]
    ]
    cases = {case["case_id"] for case in read(cpu / "catalog.json")["cases"]}
    expected = {
        case + "-" + planner for case in cases for planner in policy["planners"]
    }
    require(
        set(manifests[0]["jobs"]) | set(manifests[1]["jobs"]) == expected,
        "Expected all 360 current results",
    )
    require(
        len(jobs) == 300 and len({job["id"] for job in jobs}) == 300,
        "Expected 300 distinct dynamic outcomes",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".plus3-publication-", dir=output.parent
    ) as temp:
        stage = Path(temp)
        inputs = stage / "inputs"
        inputs.mkdir()
        shutil.copyfile(cpu / "catalog.json", inputs / "catalog.json")
        save(
            inputs / "plan.json",
            portable({"schema_version": 2, "policy": policy, "jobs": jobs}),
        )
        save(inputs / "results.json", {"records": []})
        save(
            inputs / "static-latency.json",
            {
                "source": "Static latency",
                "campaign": campaign,
                "imported_at": datetime.now(UTC).isoformat(),
                "records": [],
            },
        )
        setups = {}
        for job in model_plan["cases"]:
            value = read(knowline / "exports" / (job["case_id"] + "-knowline31b.json"))
            setups[job["case_id"]] = experiment_setup(
                value["config"], value["configuration"], job["resources"]
            )
        save(inputs / "experiment-setup.json", {"rows": setups})
        # Keep summaries for all cases; only usable timelines enter red-cell analysis.
        timelines = stage / ".timelines"
        timelines.mkdir()
        for directory in (cpu / "diagnostics", knowline / "diagnostics"):
            for path in directory.glob("*.json"):
                if read(path).get("timeline_available"):
                    shutil.copyfile(path, timelines / path.name)
        refresh(
            inputs, [cpu / "exports", knowline / "exports"], diagnostics=[timelines]
        )
        for path in inputs.glob("*.json"):
            save(path, portable(read(path)))
        report = make_report(inputs)
        require(
            report["counts"] == {"completed": 360}
            and report["static_baselines"] == report["knowline_completed"] == 60,
            "Not every result passed publication validation",
        )
        require(
            not report["experiment_setup"]["budget_mismatches"],
            "GPU budgets differ between planners",
        )
        require(
            all(
                len(page["views"]["all"]["core"]["shared_cases"]) == 30
                for page in report["pages"].values()
            ),
            "Incomplete ranking coverage",
        )
        publish(report, stage)
        shutil.rmtree(timelines)
        shutil.copyfile(CODE / "check_browser.mjs", stage / "check_browser.mjs")
        receipt = {
            "published_at": datetime.now(UTC).isoformat(),
            "campaign": campaign,
            "counts": report["counts"],
            "policy": report["resource_policy"],
            "static_completed": 60,
            "knowline_completed": 60,
            "source_archive_sha256": source_sha256,
            "collection_manifests": {
                host: digest(collection / host / "manifest.json")
                for host in ("cpu", "knowline")
            },
            "gpu_accounting": ["active", "starting", "draining"],
            "checks": [
                "transfer hashes",
                "complete trace/request counts",
                "GPU ceilings and resident GPU audits",
                "original fixed Static fleet",
                "SLO and drain measurement contract",
                "KnowLine identity and decision replay",
                "all 30 shared cases per topology",
            ],
            "leaderboard": str(output / "index.html"),
        }
        save(stage / "publication-validation.json", receipt)
        save(stage / "missing-results.json", [])
        output.mkdir(exist_ok=True)
        for path in stage.rglob("*"):
            if path.is_file():
                destination = output / path.relative_to(stage)
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(path, destination)
        print(json.dumps(receipt, indent=2), flush=True)
        print("Open: " + str(output / "index.html"), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "collection", type=Path, help="Directory containing cpu/ and knowline/ bundles"
    )
    parser.add_argument("--output", type=Path, default=CODE)
    parser.add_argument("--campaign", default=CAMPAIGN)
    parser.add_argument("--source-sha256", default=SOURCE_SHA256)
    args = parser.parse_args()
    with (args.collection.parent / "collection.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        build(
            args.collection,
            args.output,
            campaign=args.campaign,
            source_sha256=args.source_sha256,
        )


if __name__ == "__main__":
    main()
