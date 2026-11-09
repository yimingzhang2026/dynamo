# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU admission includes workers still starting or draining in either pool."""

from types import SimpleNamespace

import pytest

pytest.importorskip("dynamo.planner.offline.replay_adapter")
from autoscaling_arena.runners.budgeted_replay import (  # noqa: E402
    BudgetedReplayPlannerAdapter,
)

from dynamo.planner.core.types import PlannerEffects, ScalingDecision  # noqa: E402
from dynamo.planner.offline.replay_adapter import ReplayPlannerAdapter  # noqa: E402


def admit(
    monkeypatch,
    *,
    budget,
    costs,
    current,
    draining=(0, 0),
    starting=(0, 0),
    desired,
    mode="disagg",
):
    adapter = object.__new__(BudgetedReplayPlannerAdapter)
    adapter._config = SimpleNamespace(
        max_gpu_budget=budget,
        prefill_engine_num_gpu=costs[0],
        decode_engine_num_gpu=costs[1],
    )
    adapter._is_disagg = mode == "disagg"
    snapshot = {}
    for index, role in enumerate(("prefill", "decode")):
        snapshot[f"non_draining_{role}_count"] = current[index]
        snapshot[f"active_{role}_ids"] = list(range(current[index] - starting[index]))
        snapshot[f"starting_{role}_ids"] = list(
            range(current[index] - starting[index], current[index])
        )
        snapshot[f"draining_{role}_ids"] = list(
            range(current[index], current[index] + draining[index])
        )
    monkeypatch.setattr(
        ReplayPlannerAdapter,
        "_compute_scale_decision",
        lambda self, effects, result, now_s: effects.scale_to,
    )
    decision = adapter._compute_scale_decision(
        PlannerEffects(
            scale_to=ScalingDecision(num_prefill=desired[0], num_decode=desired[1])
        ),
        snapshot,
        5.0,
    )
    return (decision.num_prefill, decision.num_decode)


@pytest.mark.parametrize(
    "budget,costs,start,target,intermediate",
    [
        (128, (8, 8), (12, 4), (15, 1), (12, 1)),
        (80, (4, 8), (8, 6), (12, 4), (8, 4)),
        (112, (4, 8), (12, 8), (8, 10), (8, 8)),
        (16, (8, 8), (1, 1), (1, 1), (1, 1)),
        (12, (4, 8), (1, 1), (1, 1), (1, 1)),
    ],
)
def test_rebalance_waits_for_drain(
    monkeypatch, budget, costs, start, target, intermediate
):
    assert (
        admit(monkeypatch, budget=budget, costs=costs, current=start, desired=target)
        == intermediate
    )
    draining = tuple(max(0, old - new) for old, new in zip(start, intermediate))
    assert (
        admit(
            monkeypatch,
            budget=budget,
            costs=costs,
            current=intermediate,
            draining=draining,
            desired=target,
        )
        == intermediate
    )
    assert (
        admit(
            monkeypatch,
            budget=budget,
            costs=costs,
            current=intermediate,
            desired=target,
        )
        == target
    )


def test_starting_workers_consume_budget(monkeypatch):
    assert admit(
        monkeypatch,
        budget=80,
        costs=(4, 8),
        current=(8, 6),
        starting=(2, 1),
        desired=(12, 4),
    ) == (8, 4)


def test_partial_headroom_allows_only_affordable_additions(monkeypatch):
    assert admit(
        monkeypatch, budget=128, costs=(8, 8), current=(10, 4), desired=(15, 1)
    ) == (12, 1)


def test_aggregate_rebound_waits_for_drain(monkeypatch):
    assert admit(
        monkeypatch,
        budget=16,
        costs=(0, 8),
        current=(0, 1),
        draining=(0, 1),
        desired=(None, 2),
        mode="agg",
    ) == (None, 1)
    assert admit(
        monkeypatch,
        budget=16,
        costs=(0, 8),
        current=(0, 1),
        desired=(None, 2),
        mode="agg",
    ) == (None, 2)
