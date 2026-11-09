#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single entry point for CPU replay, two-node KnowLine execution and publication."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

GYM = Path(__file__).resolve().parents[1]
CPU_PLANNERS = ("mpc", "planner", "keda", "reactive", "static")


def invoke(script, *arguments):
    subprocess.run(
        [sys.executable, str(GYM / script), *map(str, arguments)],
        cwd=GYM,
        check=True,
    )


def prepare_runtime(workspace):
    """Serialize native setup across the two compute-node workers."""
    workspace.mkdir(parents=True, exist_ok=True)
    with (workspace / "native.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _prepare_runtime(workspace)


def _prepare_runtime(workspace):
    """Reuse a build only when both its source and installed binary match."""
    root = GYM.parents[1]
    sources = {
        root / name for name in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml")
    }
    for name in ("lib", "vendor/aisimulate-core", ".cargo"):
        sources.update(
            path
            for path in (root / name).rglob("*")
            if path.is_file()
            and "target" not in path.parts
            and path.suffix in {".rs", ".toml", ".lock"}
        )
    sources.add(GYM / "scripts/build_queue_runtime.sh")
    digest = hashlib.sha256()
    for path in sorted(sources):
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(path.read_bytes())
    source_hash = digest.hexdigest()
    build = workspace / "native"
    receipt = build / "source-identity.json"
    import install_queue_runtime as runtime

    if receipt.exists():
        saved = json.loads(receipt.read_text())
        try:
            current = runtime.check()
            if (
                saved["source_sha256"] == source_hash
                and runtime.digest(Path(current["library"])) == saved["binary_sha256"]
            ):
                return
        except (subprocess.CalledProcessError, OSError, ValueError):
            pass
    subprocess.run(
        [
            "bash",
            str(GYM / "scripts/build_queue_runtime.sh"),
            str(root),
            sys.executable,
            str(build),
        ],
        cwd=GYM,
        check=True,
    )
    record = runtime.install(build / "overlay/dynamo/_core.abi3.so")
    receipt.write_text(
        json.dumps(
            {"source_sha256": source_hash, "binary_sha256": record["binary_sha256"]},
            indent=2,
        )
        + "\n"
    )


def run_cpu(workspace, policy_path):
    directory = workspace / "cpu"
    plan = directory / "plan.json"
    if not plan.exists():
        policy = yaml.safe_load(policy_path.read_text())
        policy["planners"] = list(CPU_PLANNERS)
        selected = workspace / "cpu-policy.yaml"
        selected.write_text(yaml.safe_dump(policy, sort_keys=False))
        invoke(
            "scripts/prepare_comparison.py", "--policy", selected, "--output", directory
        )
    saved = json.loads(plan.read_text())
    if len(saved["jobs"]) != 300 or {j["planner"] for j in saved["jobs"]} != set(
        CPU_PLANNERS
    ):
        raise ValueError("Expected the complete 300-job CPU campaign")
    invoke(
        "scripts/run_comparison.py",
        plan,
        "--rolling",
        "--execute",
        "--keep-going",
        *(["--resume-pending"] if (directory / "status.json").exists() else []),
    )
    invoke(
        "leaderboards/60traces/export_completed.py",
        directory,
        workspace / "exports/cpu",
    )


def knowline_modules(workspace):
    # The normal implementation includes server replacement, memory isolation,
    # exact-decision continuation and skipping completed cases.
    os.environ["PLANNER_GYM"] = str(GYM)
    os.environ["PLANNER_PYTHON"] = sys.executable
    os.environ["PLANNER_CAMPAIGN_WORKSPACE"] = str(workspace)
    os.environ["KNOWLINE_WORKSPACE"] = str(workspace / "knowline")
    os.environ["KNOWLINE_PARALLEL_PLAN"] = str(
        workspace / "knowline/campaign/parallel-plan.json"
    )
    os.environ["KNOWLINE_BATCH"] = str(GYM / "scripts/knowline.sbatch")
    import knowline_campaign
    import run_knowline

    return knowline_campaign, run_knowline


def submit_knowline(workspace, policy_path, account, partition):
    core, parallel = knowline_modules(workspace)
    model = core.BASE / "models/KnowLine-31B-Gen1-8f052a8/model-manifest.json"
    if not model.exists():
        raise FileNotFoundError(
            "Prepare the pinned KnowLine model first; see docs/comparison-60.md"
        )
    if not (core.ROOT / "plan.json").exists():
        core.prepare(policy=yaml.safe_load(policy_path.read_text()))
    if not parallel.PLAN.exists():
        parallel.prepare(nodes=2, lanes_per_node=4)
    plan = parallel.check_plan()
    if plan["maximum_concurrent_nodes"] != 2:
        raise ValueError("The benchmark uses exactly two GPU node queues")
    receipt = core.ROOT / "submission.json"
    if receipt.exists():
        print(receipt.read_text())
        print(json.dumps(parallel.aggregate_status()["counts"]))
        return
    job = subprocess.check_output(
        [
            "sbatch",
            "--parsable",
            "--array=0-1%2",
            *(["--account=" + account] if account else []),
            *(["--partition=" + partition] if partition else []),
            str(parallel.BATCH),
        ],
        cwd=GYM,
        text=True,
    ).strip()
    core.save(receipt, {"job_id": job, "submitted_at": core.now(), "nodes": 2})
    print(receipt.read_text())


def publish_results(collection, output):
    if (collection / "cpu/manifest.json").is_file():
        invoke(
            "leaderboards/60traces/publish_collection.py",
            collection,
            "--output",
            output,
        )
        return
    # Plain exports from this entry point can also be published after transfer.
    code = GYM / "leaderboards/60traces"
    sys.path.insert(0, str(code))
    from publish_latest import publish, refresh

    cases = {
        c["case_id"]
        for c in json.loads((code / "inputs/catalog.json").read_text())["cases"]
    }
    for kind, planners in (("cpu", CPU_PLANNERS), ("knowline", ("knowline31b",))):
        expected = {f"{case}-{planner}" for case in cases for planner in planners}
        actual = {p.stem for p in (collection / kind).glob("*.json")}
        if actual != expected:
            raise ValueError(
                f"Expected all completed {kind} exports; missing: {sorted(expected - actual)}"
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".publication-", dir=output.parent
    ) as temporary:
        stage = Path(temporary)
        shutil.copytree(code / "inputs", stage / "inputs")
        report = refresh(
            stage / "inputs", [collection / "cpu", collection / "knowline"]
        )
        if report["counts"] != {"completed": 360}:
            raise ValueError("Publication requires all 360 validated results")
        publish(report, stage)
        for path in stage.rglob("*"):
            if path.is_file():
                target = output / path.relative_to(stage)
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(path, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("cpu", "knowline", "worker", "collect", "publish", "status")
    )
    parser.add_argument("--workspace", type=Path, default=GYM / "runs/benchmark")
    parser.add_argument(
        "--policy", type=Path, default=GYM / "configs/60traces/comparison.yaml"
    )
    parser.add_argument("--account")
    parser.add_argument("--partition")
    parser.add_argument("--shard", type=int)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--kind", choices=("cpu", "knowline"))
    parser.add_argument(
        "--collection",
        type=Path,
        default=GYM / "runs/efficiency-plus3-20261106/collection",
    )
    parser.add_argument("--output", type=Path, default=GYM / "leaderboards/60traces")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    sys.path.insert(0, str(GYM / "src"))
    os.environ["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(GYM / "src"), os.environ.get("PYTHONPATH")))
    )
    if args.command == "publish":
        publish_results(args.collection.resolve(), args.output.resolve())
        return
    if args.command == "worker":
        if args.shard is None:
            parser.error("worker requires --shard")
        if sys.platform != "linux":
            parser.error("Replay execution requires Linux")
        prepare_runtime(workspace)
        _, parallel = knowline_modules(workspace)
        parallel.run(args.shard, args.round)
        return
    if args.command == "collect":
        if args.kind is None:
            parser.error("collect requires --kind cpu or knowline")
        campaign = workspace / (
            "knowline/campaign" if args.kind == "knowline" else "cpu"
        )
        invoke(
            "leaderboards/60traces/export_completed.py",
            campaign,
            workspace / "exports" / args.kind,
            *(["--knowline"] if args.kind == "knowline" else []),
        )
        return
    if args.command == "status":
        for kind, file in (
            ("cpu", workspace / "cpu/status.json"),
            ("knowline", workspace / "knowline/campaign/parallel-status.json"),
        ):
            print(kind, file.read_text() if file.exists() else "Not started")
        return
    if sys.platform != "linux":
        parser.error(
            "Replay execution requires the configured Linux replay environment"
        )
    workspace.mkdir(parents=True, exist_ok=True)
    with (workspace / f"{args.command}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == "cpu":
            prepare_runtime(workspace)
            run_cpu(workspace, args.policy)
        else:
            submit_knowline(workspace, args.policy, args.account, args.partition)


if __name__ == "__main__":
    main()
