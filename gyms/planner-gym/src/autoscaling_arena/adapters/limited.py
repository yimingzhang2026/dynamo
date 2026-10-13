# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply the same allocation constraints to every replay controller."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from autoscaling_arena.worker_limits import project_target

from dynamo.planner.core.engine_protocol import EngineProtocol
from dynamo.planner.core.types import (
    PlannerEffects,
    ScalingDecision,
    ScheduledTick,
    TickInput,
)


class LimitedEngine:
    def __init__(
        self,
        engine: EngineProtocol,
        *,
        targets: tuple[tuple[int, int], ...],
        start: tuple[int, int],
        log_path: Path,
    ) -> None:
        if start not in targets:
            raise ValueError("starting fleet is outside worker/GPU limits")
        self._engine = engine
        self._targets = targets
        self._current = start
        self._log_path = log_path

    def __getattr__(self, name: str) -> Any:
        # Forward optional replay capabilities (bootstrap and telemetry hooks).
        return getattr(self._engine, name)

    def initial_tick(self, start_s: float) -> ScheduledTick:
        return self._engine.initial_tick(start_s)

    async def tick(
        self, scheduled_tick: ScheduledTick, tick_input: TickInput
    ) -> PlannerEffects:
        effects = await self._engine.tick(scheduled_tick, tick_input)
        counts = tick_input.worker_counts
        p, d = self._current
        if counts is not None:
            if counts.expected_num_prefill is not None:
                p = counts.expected_num_prefill
            if counts.expected_num_decode is not None:
                d = counts.expected_num_decode
        self._current = (p, d)
        if effects.scale_to is None:
            return effects
        requested = (
            p if effects.scale_to.num_prefill is None else effects.scale_to.num_prefill,
            d if effects.scale_to.num_decode is None else effects.scale_to.num_decode,
        )
        applied = project_target(requested, self._current, self._targets)
        if applied != requested:
            with self._log_path.open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "tick_s": tick_input.now_s,
                            "requested": {
                                "prefill": requested[0],
                                "decode": requested[1],
                            },
                            "applied": {"prefill": applied[0], "decode": applied[1]},
                        }
                    )
                    + "\n"
                )
        self._current = applied
        if applied == requested:
            return effects
        return replace(
            effects,
            scale_to=ScalingDecision(
                num_prefill=applied[0]
                if effects.scale_to.num_prefill is not None or applied[0] != p
                else None,
                num_decode=applied[1]
                if effects.scale_to.num_decode is not None or applied[1] != d
                else None,
            ),
        )

    async def shutdown(self) -> None:
        await self._engine.shutdown()
