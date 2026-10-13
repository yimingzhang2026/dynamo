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
