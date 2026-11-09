# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Make ``src/`` importable without installing the package."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from autoscaling_arena.workloads import WORKLOADS, Workload, axes


@pytest.fixture(autouse=True)
def unit_workloads(monkeypatch):
    """Small generated records for parser/runner contracts, isolated per test."""
    for index in range(7):
        name = f"unit-{chr(97 + index)}"
        monkeypatch.setitem(
            WORKLOADS,
            name,
            Workload(
                name=name,
                description="Unit-test input",
                duration_s=2,
                arrival=lambda t: 30,
                shape=lambda rng: (1088, 8),
                prefix_factory=axes.NoSharing
                if index != 6
                else lambda: axes.SharedPrefix(2),
            ),
        )
