# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check dispatch without building a runtime, submitting jobs or running replay."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_campaign as campaign

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.fixture
def entrypoint(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("PYTHONPATH", "")
    runtime = Mock()
    monkeypatch.setattr(campaign, "prepare_runtime", runtime)

    def invoke(command, *extra):
        monkeypatch.setattr(
            sys,
            "argv",
            ["run_campaign.py", command, "--workspace", str(tmp_path), *extra],
        )
        campaign.main()

    return invoke, runtime, tmp_path


def test_gpu_submission_does_not_build_on_login_node(entrypoint, monkeypatch):
    invoke, runtime, workspace = entrypoint
    submit = Mock()
    monkeypatch.setattr(campaign, "submit_knowline", submit)
    invoke("knowline", "--account", "test", "--partition", "batch")
    runtime.assert_not_called()
    submit.assert_called_once_with(
        workspace, campaign.GYM / "configs/60traces/comparison.yaml", "test", "batch"
    )


def test_compute_worker_prepares_runtime_before_replay(entrypoint, monkeypatch):
    invoke, runtime, workspace = entrypoint
    run = Mock(side_effect=lambda *args: runtime.assert_called_once_with(workspace))
    monkeypatch.setattr(
        campaign, "knowline_modules", lambda path: (None, SimpleNamespace(run=run))
    )
    invoke("worker", "--shard", "1", "--round", "2")
    run.assert_called_once_with(1, 2)


def test_cpu_prepares_runtime_before_dispatch(entrypoint, monkeypatch):
    invoke, runtime, workspace = entrypoint
    run = Mock(side_effect=lambda *args: runtime.assert_called_once_with(workspace))
    monkeypatch.setattr(campaign, "run_cpu", run)
    invoke("cpu")
    run.assert_called_once()


def test_status_does_not_initialize_runtime(entrypoint):
    invoke, runtime, _ = entrypoint
    invoke("status")
    runtime.assert_not_called()
