#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download the pinned KnowLine weights and the model author's serving code."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL = "PelaAI/KnowLine-31B-Gen1"
REVISION = "8f052a8130e599b53e5324c34f0f5ff90696e1c6"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    snapshot_download(
        MODEL,
        revision=REVISION,
        local_dir=args.directory,
        max_workers=4,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.model",
            "*.py",
            "*.sh",
            "*.jinja",
            "*.txt",
            "LICENSE*",
        ],
    )
    record = {
        "model": MODEL,
        "revision": REVISION,
        "serving_source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in args.directory.glob("*.py")
        },
    }
    (args.directory / "model-manifest.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
