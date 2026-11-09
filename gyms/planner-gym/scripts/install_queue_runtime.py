#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Install/check the optimized native runtime without modifying running processes."""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REVISION = "incremental-queue-stats-v1"


def check(overlay=None):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    if overlay is not None:
        env["PYTHONPATH"] = str(overlay)
    code = (
        "import json; from dynamo import _core; "
        "from dynamo.replay import run_trace_replay; "
        f'assert getattr(_core, "REPLAY_QUEUE_STATISTICS_REVISION", None) == {REVISION!r}, '
        '"Rebuild the native replay runtime with the vendored queue optimizations"; '
        'print(json.dumps({"revision":_core.REPLAY_QUEUE_STATISTICS_REVISION,"library":_core.__file__}))'
    )
    return json.loads(
        subprocess.check_output([sys.executable, "-c", code], env=env, text=True)
    )


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def install(library):
    library = library.resolve(strict=True)
    spec = importlib.util.find_spec("dynamo._core")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "Install the matching ai-dynamo and ai-dynamo-runtime Python packages first"
        )
    target = Path(spec.origin).resolve()
    site = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    if site.resolve() not in target.parents:
        raise RuntimeError(
            "Run the installer with PYTHONPATH unset in the target environment"
        )
    # Probe ELF/Python ABI compatibility and the native revision in a subprocess.
    with tempfile.TemporaryDirectory(prefix="planner-native-probe-") as directory:
        overlay = Path(directory)
        (overlay / "dynamo").mkdir()
        (overlay / "dynamo" / target.name).symlink_to(library)
        check(overlay)
    binary_sha = digest(library)
    previous_sha = digest(target)
    if previous_sha != binary_sha:
        backup = target.with_name(target.name + ".before-" + previous_sha[:12])
        if not backup.exists():
            os.link(target, backup)
        descriptor, staged = tempfile.mkstemp(
            prefix=target.name + ".install-", dir=target.parent
        )
        os.close(descriptor)
        try:
            shutil.copy2(library, staged)
            # Never truncate an inode mapped by an active simulation.
            os.replace(staged, target)
        finally:
            if os.path.exists(staged):
                os.unlink(staged)
    record = check()
    record.update(
        installed_at=datetime.now(timezone.utc).isoformat(),
        binary_sha256=binary_sha,
        previous_binary_sha256=previous_sha,
        source_library=str(library),
    )
    manifest = target.with_name("replay-queue-optimization.json")
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--check", action="store_true")
    choice.add_argument("--library", type=Path)
    args = parser.parse_args()
    print(json.dumps(check() if args.check else install(args.library), indent=2))


if __name__ == "__main__":
    main()
