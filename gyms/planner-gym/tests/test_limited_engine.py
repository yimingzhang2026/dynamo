# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib
import json

import pytest
from autoscaling_arena.worker_limits import WorkerLimits

types = pytest.importorskip("dynamo.planner.core.types")
LimitedEngine = importlib.import_module(
    "autoscaling_arena.adapters.limited"
).LimitedEngine


class ScriptedEngine:
    supports_ais_bootstrap = False

    def __init__(self, effects):
        self.effects = effects
        self.closed = False

    def initial_tick(self, start_s):
        return types.ScheduledTick(at_s=start_s)

    async def tick(self, scheduled_tick, tick_input):
        return self.effects

    async def shutdown(self):
        self.closed = True


def test_runtime_clamps_partial_targets_and_records_the_applied_action(tmp_path):
    scheduled = types.ScheduledTick(at_s=15)
    effects = types.PlannerEffects(
        scale_to=types.ScalingDecision(num_decode=30), next_tick=scheduled
    )
    inner = ScriptedEngine(effects)
    path = tmp_path / "decisions.jsonl"
    wrapper = LimitedEngine(
        inner,
        targets=WorkerLimits(2, 16).targets(
            topology="disagg",
            gpu_budget=124,
            prefill_gpus=4,
            decode_gpus=8,
        ),
        start=(1, 1),
        log_path=path,
    )
    counts = types.WorkerCounts(expected_num_prefill=1, expected_num_decode=1)
    applied = asyncio.run(
        wrapper.tick(scheduled, types.TickInput(now_s=0, worker_counts=counts))
    )
    assert applied.scale_to.num_decode == 15
    assert applied.scale_to.num_prefill is None
    assert applied.next_tick is scheduled and applied.diagnostics is effects.diagnostics
    row = json.loads(path.read_text())
    assert row["requested"] == {"prefill": 1, "decode": 30}
    assert row["applied"] == {"prefill": 1, "decode": 15}
    assert wrapper.supports_ais_bootstrap is False
    asyncio.run(wrapper.shutdown())
    assert inner.closed


def test_valid_aggregate_decision_keeps_inactive_pool_untouched(tmp_path):
    effects = types.PlannerEffects(scale_to=types.ScalingDecision(num_decode=8))
    wrapper = LimitedEngine(
        ScriptedEngine(effects),
        targets=WorkerLimits(2, 16).targets(
            topology="agg",
            gpu_budget=128,
            prefill_gpus=0,
            decode_gpus=8,
        ),
        start=(0, 2),
        log_path=tmp_path / "decisions.jsonl",
    )
    applied = asyncio.run(
        wrapper.tick(wrapper.initial_tick(0), types.TickInput(now_s=0))
    )
    assert applied is effects
    assert applied.scale_to.num_prefill is None
