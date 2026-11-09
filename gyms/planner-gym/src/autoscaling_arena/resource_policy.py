# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared resource accounting for trace comparisons and result validation."""

from __future__ import annotations

from autoscaling_arena.worker_limits import WorkerLimits


def resource_policy_label(resources: dict) -> str:
    if "gpu_budget_extra_workers" in resources:
        return f"baseline-plus-{resources['gpu_budget_extra_workers']}-max-workers"
    return f"baseline-gpus-x{resources['gpu_budget_multiplier']}"


def resource_policy_description(resources: dict) -> str:
    if "gpu_budget_extra_workers" in resources:
        return (
            "the original Static GPU allocation plus "
            f"{resources['gpu_budget_extra_workers']} × the largest per-worker GPU cost"
        )
    return f"{resources['gpu_budget_multiplier']}× the original Static GPU allocation"


def case_resources(base: dict, policy: dict) -> dict:
    """Keep the fixed baseline/start; derive pool bounds from a shared GPU ceiling."""
    spec = policy["resources"]
    if set(spec) not in ({"gpu_budget_multiplier"}, {"gpu_budget_extra_workers"}):
        raise ValueError("Specify exactly one GPU policy: multiplier or extra workers")
    key = next(iter(spec))
    value = spec[key]
    minimum_value = 0 if key == "gpu_budget_extra_workers" else 1
    if type(value) is not int or value < minimum_value:
        raise ValueError(f"{key} must be an integer >= {minimum_value}")
    backend = base["backend"]
    disagg = backend["topology"] == "disagg"
    pools = ("prefill", "decode") if disagg else ("aggregate",)
    fixed = backend["autoscalers"][0]["config"]
    costs = {
        pool: backend["engines"][pool]["tp_size"]
        * backend["engines"][pool].get("attention_dp_size", 1)
        for pool in pools
    }
    initial = {
        pool: fixed["num_decode" if pool == "aggregate" else "num_" + pool]
        for pool in pools
    }
    baseline = sum(initial[pool] * costs[pool] for pool in pools)
    if key == "gpu_budget_extra_workers":
        headroom = value * max(costs.values())
        budget = baseline + headroom
        policy_evidence = {
            "schema_version": 2,
            key: value,
            "max_worker_gpus": max(costs.values()),
            "headroom_gpus": headroom,
        }
    else:
        budget = value * baseline
        policy_evidence = {"schema_version": 1, key: value}
    minimum = {pool: 1 for pool in pools}
    maximum = {
        pool: (budget - sum(costs[other] for other in pools if other != pool))
        // costs[pool]
        for pool in pools
    }
    # Derive count bounds from the shared GPU ceiling unless explicitly configured.
    total = max(maximum.values()) + (1 if disagg else 0)
    limits = (
        WorkerLimits(**policy["worker_limits"])
        if "worker_limits" in policy
        else WorkerLimits(2 if disagg else 1, total)
    )
    if not disagg:
        minimum["aggregate"] = limits.min_total
    for pool in pools:
        maximum[pool] = min(maximum[pool], limits.max_total - (len(pools) - 1))
    targets = limits.targets(
        topology=backend["topology"],
        gpu_budget=budget,
        prefill_gpus=costs.get("prefill", 0),
        decode_gpus=costs["decode" if disagg else "aggregate"],
    )
    start = (initial.get("prefill", 0), initial["decode" if disagg else "aggregate"])
    if start not in targets:
        raise ValueError(
            "Static allocation is outside the comparison resource envelope"
        )
    # A singleton is a valid resource-constrained case. Never increase its GPU
    # ceiling merely to manufacture scaling choices.
    return {
        **policy_evidence,
        "baseline_gpus": baseline,
        "gpu_budget": budget,
        "initial_workers": initial,
        "gpus_per_worker": costs,
        "min_workers": minimum,
        "max_workers": maximum,
        "worker_limits": {"min_total": limits.min_total, "max_total": limits.max_total},
        "feasible_fleet_count": len(targets),
    }


def apply_case_resources(base: dict, policy: dict) -> dict:
    """Apply the common ceiling without changing the original Static fleet."""
    resources = case_resources(base, policy)
    base["backend"]["gpu_budget"] = resources["gpu_budget"]
    base["backend"]["worker_limits"] = resources["worker_limits"]
    base.setdefault("labels", {})["resource_policy"] = resource_policy_label(resources)
    return resources
