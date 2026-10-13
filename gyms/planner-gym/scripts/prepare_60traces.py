#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reconstruct selected historical inputs and verify their original SHA-256."""
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
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data/60traces/reproduction"


@lru_cache(maxsize=32)
def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(case: dict) -> Path:
    base = ROOT / case["base"]
    scaler = ROOT / case["scaler"]
    for path, expected in (
        (base, case["base_sha256"]),
        (scaler, case["scaler_sha256"]),
    ):
        if digest(path) != expected:
            raise ValueError(f"Checksum mismatch: {path.relative_to(ROOT)}")
    output = ROOT / case["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if output.exists():
            if digest(output) != case["expected_output_sha256"]:
                raise ValueError(
                    f"Existing output differs from reference; refusing to overwrite {output}"
                )
            print(f"{case['id']}: existing input verified", flush=True)
            return output
        required = case["expected_output_bytes"] + 1024**3
        if shutil.disk_usage(output.parent).free < required:
            raise OSError(
                f"{case['id']}: requires {required / 1024**3:.1f} GiB free for generation"
            )
        with tempfile.TemporaryDirectory(
            prefix=f".{case['id']}-", dir=output.parent
        ) as directory:
            temporary = Path(directory) / output.name
            command = [
                sys.executable,
                str(scaler),
                "--base",
                str(base),
                "--manifest",
                str(ROOT / case["base_manifest"]),
                "--output",
                str(temporary),
                "--alpha",
                repr(case["alpha"]),
                "--seed",
                str(case["seed"]),
                "--jitter-seconds",
                repr(case["jitter_seconds"]),
                "--window-start-seconds",
                repr(case["window_start_seconds"]),
                "--window-duration-seconds",
                repr(case["window_duration_seconds"]),
                "--output-format",
                case["format"],
                "--max-output-requests",
                str(case["expected_requests"] + 1),
                "--max-wall-seconds",
                "3600",
                # Frozen scalers compare Linux KiB; macOS reports bytes.
                # This keeps the actual RSS guard at 8 GiB on either platform.
                "--max-rss-gib",
                str(8 * (1024 if sys.platform == "darwin" else 1)),
            ]
            subprocess.run(command, check=True, timeout=3630)
            if digest(temporary) != case["expected_output_sha256"]:
                raise ValueError(
                    f"{case['id']}: generated trace does not match the measured reference"
                )
            if temporary.stat().st_size != case["expected_output_bytes"]:
                raise ValueError(f"{case['id']}: generated size differs from reference")
            sidecar = temporary.with_suffix(".jsonl.manifest.json")
            manifest = json.loads(sidecar.read_text())
            if manifest["requests"] != case["expected_requests"]:
                raise ValueError(
                    f"{case['id']}: generated request count differs from reference"
                )
            manifest["output_path"] = str(output)
            if sys.platform == "darwin":
                manifest["preparation_peak_rss_kib"] /= 1024
            manifest["preparation_platform"] = sys.platform
            sidecar.write_text(json.dumps(manifest, indent=2) + "\n")
            os.replace(sidecar, output.with_suffix(".jsonl.manifest.json"))
            os.replace(temporary, output)
    print(f"{case['id']}: SHA-256 verified; config {case['config']}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument(
        "--case", action="append", help="case ID; repeat to select several"
    )
    selector.add_argument(
        "--all", action="store_true", help="generate all 60 inputs (45.37 GB)"
    )
    selector.add_argument(
        "--list", action="store_true", help="list IDs, base, scale and expanded size"
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="number of independent cases to prepare concurrently",
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    cases = json.loads((ROOT / "cases.json").read_text())["cases"]
    if args.list:
        for case in cases:
            print(
                f"{case['id']:10} {case['dataset']:22} {case['alpha']:12.8g}x {case['expected_output_bytes']/1e6:10.1f} MB"
            )
        return
    selected = {case["id"] for case in cases} if args.all else set(args.case)
    unknown = selected - {case["id"] for case in cases}
    if unknown:
        parser.error("Unknown case IDs: " + ", ".join(sorted(unknown)))
    selected_cases = [case for case in cases if case["id"] in selected]
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        for _ in executor.map(prepare, selected_cases):
            pass


if __name__ == "__main__":
    main()
