# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime promotion must preserve active mappings and reject incompatible builds."""
import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.fixture
def installer(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/install_queue_runtime.py"
    spec = importlib.util.spec_from_file_location("queue_runtime_install", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = (
        tmp_path
        / "env/lib"
        / f"python{module.sys.version_info.major}.{module.sys.version_info.minor}"
        / "site-packages/dynamo/_core.abi3.so"
    )
    target.parent.mkdir(parents=True)
    target.write_bytes(b"original mapped library")
    candidate = tmp_path / "candidate.so"
    candidate.write_bytes(b"optimized library")
    monkeypatch.setattr(module.sys, "prefix", str(tmp_path / "env"))
    monkeypatch.setattr(
        module.importlib.util,
        "find_spec",
        lambda _: SimpleNamespace(origin=str(target)),
    )
    monkeypatch.setattr(
        module, "check", lambda overlay=None: {"revision": module.REVISION}
    )
    return module, target, candidate


def test_promotion_preserves_old_open_file_and_records_revision(installer):
    module, target, candidate = installer
    with target.open("rb") as active_mapping:
        original_inode = target.stat().st_ino
        record = module.install(candidate)
        assert active_mapping.read() == b"original mapped library"
        assert target.stat().st_ino != original_inode
    assert target.read_bytes() == b"optimized library"
    assert record["revision"] == module.REVISION
    assert (
        next(target.parent.glob("*.before-*")).read_bytes()
        == b"original mapped library"
    )


def test_incompatible_candidate_does_not_replace_runtime(installer, monkeypatch):
    module, target, candidate = installer

    def reject(overlay=None):
        raise subprocess.CalledProcessError(1, ["native import probe"])

    monkeypatch.setattr(module, "check", reject)
    with pytest.raises(subprocess.CalledProcessError):
        module.install(candidate)
    assert target.read_bytes() == b"original mapped library"
    assert not list(target.parent.glob("*.before-*"))


def test_reinstall_is_idempotent(installer):
    module, target, candidate = installer
    module.install(candidate)
    inode = target.stat().st_ino
    module.install(candidate)
    assert target.stat().st_ino == inode
    assert len(list(target.parent.glob("*.before-*"))) == 1
