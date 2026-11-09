#!/usr/bin/env python3
"""Download pinned upstream inputs, then reconstruct SHA-verified 1x test traces."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(asset, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha(destination) != asset["sha256"]:
            raise ValueError(f"Existing source checksum mismatch: {destination}")
        return
    temporary = destination.with_suffix(destination.suffix + ".partial")
    print(f"Downloading {asset['url']}", flush=True)
    try:
        request = urllib.request.Request(
            asset["url"], headers={"User-Agent": "planner-gym-reproduction/1.0"}
        )
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open(
            "wb"
        ) as output:
            while block := response.read(4 * 1024 * 1024):
                output.write(block)
        if sha(temporary) != asset["sha256"]:
            raise ValueError(f'Upstream source checksum mismatch: {asset["url"]}')
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--dataset", action="append")
    choice.add_argument("--all", action="store_true")
    choice.add_argument("--list", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT.parent / "data/60traces/reproduction/data"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT.parent / "data/60traces/reproduction/work/source-inputs",
    )
    args = parser.parse_args()
    recipes = json.loads((ROOT / "base-recipes.json").read_text())
    assets = json.loads((ROOT / "source-assets.json").read_text())
    if args.list:
        for ident, recipe in recipes.items():
            print(f"{ident:24} {recipe['expected_requests']:10,} test requests")
        return
    selected = list(recipes) if args.all else args.dataset
    unknown = set(selected) - set(recipes)
    if unknown:
        parser.error("Unknown datasets: " + ", ".join(sorted(unknown)))
    stage = args.source_dir.resolve()
    data = args.data_dir.resolve()
    (stage / "source").mkdir(parents=True, exist_ok=True)
    (stage / "catalog.json").write_bytes((ROOT / "source-catalog.json").read_bytes())
    for ident in selected:
        recipe = recipes[ident]
        output = data / ident / "test-1x.jsonl"
        manifest = output.with_suffix(".jsonl.manifest.json")
        if output.exists():
            if sha(output) != recipe["expected_base_sha256"] or not manifest.exists():
                raise ValueError(
                    f"Existing base differs from reference or lacks metadata: {output}"
                )
            print(f"{ident}: existing base verified", flush=True)
            continue
        if ident == "art-chat":
            if args.download_only:
                print(
                    "ART uses selective Parquet column reads during conversion; see source-assets.json"
                )
                continue
        else:
            for asset in assets[ident]["assets"]:
                prefix = "ragpulse/" if ident == "ragpulse" else ""
                fetch(asset, stage / "source" / (prefix + asset["file"]))
        if args.download_only:
            continue
        converter = ROOT / recipe["converter"]
        if sha(converter) != recipe["converter_sha256"]:
            raise ValueError(f"Converter checksum mismatch: {converter}")
        subprocess.run(
            [
                sys.executable,
                str(converter),
                "--dataset",
                ident,
                "--root",
                str(stage),
                "--output",
                str(data),
            ],
            check=True,
        )
        if sha(output) != recipe["expected_base_sha256"]:
            raise ValueError(
                f"{ident}: conversion output does not match historical base SHA-256"
            )
        print(f"{ident}: base SHA-256 verified", flush=True)


if __name__ == "__main__":
    main()
