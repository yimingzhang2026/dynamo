# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify resident GPU usage from persisted native replay telemetry."""

import json
from pathlib import Path


def audit_budget(session: Path, budget: int) -> dict:
    (report,) = session.rglob("trace-report.json")
    native = json.loads(report.read_text())
    costs = {role: native[f"{role}_gpus_per_worker"] for role in ("prefill", "decode")}
    (telemetry,) = session.rglob("telemetry.jsonl")
    peak = samples = 0
    with telemetry.open() as stream:
        for line in stream:
            sample = json.loads(line)
            allocated = sum(
                cost
                * len(
                    set().union(
                        *(
                            sample[f"{state}_{role}_ids"]
                            for state in ("active", "starting", "draining")
                        )
                    )
                )
                for role, cost in costs.items()
            )
            if allocated > budget:
                raise ValueError(
                    f"GPU budget exceeded at {sample['sampled_at_ms']} ms: {allocated} > {budget}"
                )
            peak = max(peak, allocated)
            samples += 1
    if not samples:
        raise ValueError("Native GPU audit requires nonempty telemetry")
    return {
        "peak_allocated_gpus": peak,
        "gpu_budget": budget,
        "telemetry_samples": samples,
    }
