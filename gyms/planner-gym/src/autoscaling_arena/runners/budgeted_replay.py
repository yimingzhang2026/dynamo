# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Enforce the replay GPU budget during worker startup and graceful draining."""

import logging
from dataclasses import replace
from typing import Any

from dynamo.planner.core.types import PlannerEffects, ScalingDecision
from dynamo.planner.offline.replay_adapter import ReplayPlannerAdapter

logger = logging.getLogger(__name__)


class BudgetedReplayPlannerAdapter(ReplayPlannerAdapter):
    """Admit scale-ups only against GPUs already free in the native snapshot.

    A lower desired replica count does not release GPUs until those workers
    finish draining. Constrain effects before the parent records its targets
    and scaling events, so subsequent observations and GPU accounting describe
    the admitted transition.
    """

    def _compute_scale_decision(
        self, effects: PlannerEffects, result: dict[str, Any], now_s: float
    ) -> tuple[int | None, int | None]:
        budget = self._config.max_gpu_budget
        if budget >= 0 and effects.scale_to is not None:
            roles = ("prefill", "decode") if self._is_disagg else ("decode",)
            current = {role: result[f"non_draining_{role}_count"] for role in roles}
            costs = {
                role: getattr(self._config, f"{role}_engine_num_gpu") for role in roles
            }
            allocated = sum(
                costs[role]
                * len(
                    set().union(
                        *(
                            result[f"{state}_{role}_ids"]
                            for state in ("active", "starting", "draining")
                        )
                    )
                )
                for role in roles
            )
            if allocated > budget:
                raise ValueError(
                    f"native replay allocation {allocated} exceeds GPU budget {budget}"
                )
            requested = {
                role: getattr(effects.scale_to, f"num_{role}") for role in roles
            }
            desired = {
                role: current[role] if requested[role] is None else requested[role]
                for role in roles
            }
            additions = sum(
                max(0, desired[role] - current[role]) * costs[role] for role in roles
            )
            if allocated + additions > budget:
                # Keep every requested reduction. Growing either pool can use
                # only current headroom, never capacity promised by that reduction.
                headroom = budget - allocated
                p_current = current.get("prefill", 0)
                p_desired = desired.get("prefill", 0)
                candidates = (
                    (p, d)
                    for p in range(min(p_current, p_desired), p_desired + 1)
                    for d in range(
                        min(current["decode"], desired["decode"]), desired["decode"] + 1
                    )
                    if max(0, p - p_current) * costs.get("prefill", 0)
                    + max(0, d - current["decode"]) * costs["decode"]
                    <= headroom
                )
                p, d = min(
                    candidates,
                    key=lambda pair: (
                        (pair[0] - p_desired) ** 2 + (pair[1] - desired["decode"]) ** 2,
                        sum(pair),
                        pair,
                    ),
                )
                logger.info(
                    "GPU budget deferred scale-up at t=%s: requested=%s admitted=%s allocated=%s budget=%s",
                    now_s,
                    desired,
                    (p, d),
                    allocated,
                    budget,
                )
                effects = replace(
                    effects,
                    scale_to=ScalingDecision(
                        num_prefill=p if self._is_disagg else None, num_decode=d
                    ),
                )
        return super()._compute_scale_decision(effects, result, now_s)
