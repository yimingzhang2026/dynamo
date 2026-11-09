# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared feasible fleet targets, independent of any planner implementation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkerLimits:
    min_total: int
    max_total: int

    def __post_init__(self) -> None:
        if any(type(n) is not int or n < 1 for n in (self.min_total, self.max_total)):
            raise ValueError("worker limits must be positive integers")
        if self.min_total > self.max_total:
            raise ValueError("worker_limits.min_total must be <= max_total")

    def targets(
        self,
        *,
        topology: str,
        gpu_budget: int,
        prefill_gpus: int,
        decode_gpus: int,
        min_prefill: int = 1,
        max_prefill: int | None = None,
        min_decode: int = 1,
        max_decode: int | None = None,
    ) -> tuple[tuple[int, int], ...]:
        """Return (prefill, decode) pairs allowed by all worker/GPU limits."""
        if topology not in {"agg", "disagg"}:
            raise ValueError("worker limits require agg or disagg topology")
        if decode_gpus < 1 or (topology == "disagg" and prefill_gpus < 1):
            raise ValueError("active pools must have positive GPU costs")
        p_max = min(self.max_total, max_prefill or self.max_total)
        d_max = min(self.max_total, max_decode or self.max_total)
        prefills = range(min_prefill, p_max + 1) if topology == "disagg" else (0,)
        return tuple(
            (p, d)
            for p in prefills
            for d in range(min_decode, d_max + 1)
            if self.min_total <= p + d <= self.max_total
            and p * prefill_gpus + d * decode_gpus <= gpu_budget
        )


def project_target(
    requested: tuple[int, int],
    current: tuple[int, int],
    targets: tuple[tuple[int, int], ...],
) -> tuple[int, int]:
    """Closest feasible integer fleet; ties prefer less movement, then fewer workers.

    A final lexicographic tie-break makes this deterministic for every planner.
    The caller records changes so a constrained action is never hidden.
    """
    if not targets:
        raise ValueError("worker/GPU limits leave no feasible fleet")
    if requested in targets:
        return requested
    return min(
        targets,
        key=lambda pair: (
            sum((n - r) ** 2 for n, r in zip(pair, requested)),
            sum(abs(n - c) for n, c in zip(pair, current)),
            sum(pair),
            pair,
        ),
    )


def resident_workers(snapshot, pool):
    """Count every allocated worker, including starts and drains."""
    states = [
        set(snapshot[state + "_" + pool + "_ids"])
        for state in ("active", "starting", "draining")
    ]
    if any(a & b for i, a in enumerate(states) for b in states[i + 1 :]):
        raise ValueError("Overlapping worker lifecycle states")
    return len(set().union(*states))


def aggregate_action_mask(current, deltas, minimum, maximum, resident, *, exact=True):
    """Bounded relative actions that fit without freeing draining workers early."""
    if not minimum <= current <= resident <= maximum:
        raise ValueError(
            "Resident fleet exceeds the configured capacity or is inconsistent"
        )
    targets = [max(minimum, min(maximum, current + d)) for d in deltas]
    chosen = {}
    for i in sorted(range(len(deltas)), key=lambda i: (abs(deltas[i]), i)):
        chosen.setdefault(targets[i], i)
    return [
        (not exact or i in chosen.values())
        and max(0, target - current) <= maximum - resident
        for i, target in enumerate(targets)
    ]


def joint_delta_mask(requested, deltas, settings, resident=None):
    """Feasible pairs of relative actions under pool bounds and one GPU budget."""
    costs = [settings[p + "_gpus_per_worker"] for p in ("prefill", "decode")]
    check_resident = resident is not None
    resident = requested if resident is None else resident
    allocated = sum(n * c for n, c in zip(resident, costs, strict=True))
    if (
        any(r < n for r, n in zip(resident, requested, strict=True))
        or allocated > settings["gpu_budget"]
    ):
        raise ValueError(
            "Resident workers exceed the joint GPU budget or are inconsistent"
        )
    return [
        [
            settings["min_prefill"] <= requested[0] + p <= settings["max_prefill"]
            and settings["min_decode"] <= requested[1] + d <= settings["max_decode"]
            and (requested[0] + p) * settings["prefill_gpus_per_worker"]
            + (requested[1] + d) * settings["decode_gpus_per_worker"]
            <= settings["gpu_budget"]
            and (
                not check_resident
                or max(0, p) * costs[0] + max(0, d) * costs[1]
                <= settings["gpu_budget"] - allocated
            )
            for d in deltas
        ]
        for p in deltas
    ]


def admit_disagg_target(requested, current, snapshot, settings):
    """Keep reductions; admit additions only on GPUs already free, including drains."""
    costs = [settings[pool + "_gpus_per_worker"] for pool in ("prefill", "decode")]
    allocated = sum(
        cost
        * len(
            set().union(
                *(
                    snapshot[state + "_" + pool + "_ids"]
                    for state in ("active", "starting", "draining")
                )
            )
        )
        for pool, cost in zip(("prefill", "decode"), costs)
    )
    if allocated > settings["gpu_budget"]:
        raise ValueError("Resident workers exceed the joint GPU budget")
    headroom = settings["gpu_budget"] - allocated
    available = tuple(
        (p, d)
        for p in range(min(current[0], requested[0]), requested[0] + 1)
        for d in range(min(current[1], requested[1]), requested[1] + 1)
        if max(0, p - current[0]) * costs[0] + max(0, d - current[1]) * costs[1]
        <= headroom
    )
    return list(project_target(tuple(requested), tuple(current), available))
