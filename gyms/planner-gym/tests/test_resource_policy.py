# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml
from autoscaling_arena.resource_policy import (
    apply_case_resources,
    case_resources,
    resource_policy_description,
)
from autoscaling_arena.worker_limits import (
    WorkerLimits,
    admit_disagg_target,
    joint_delta_mask,
)

ROOT = Path(__file__).resolve().parents[1]


def test_all_cases_offer_scaling_and_every_gpu_affordable_fleet():
    policy = {"resources": {"gpu_budget_multiplier": 2}}
    paths = sorted((ROOT / "configs/60traces").glob("[AD]-*.yaml"))
    assert len(paths) == 60
    for path in paths:
        raw = yaml.safe_load(path.read_text())
        r = case_resources(raw, policy)
        costs = r["gpus_per_worker"]
        disagg = raw["backend"]["topology"] == "disagg"
        kwargs = {
            "topology": raw["backend"]["topology"],
            "gpu_budget": r["gpu_budget"],
            "prefill_gpus": costs.get("prefill", 0),
            "decode_gpus": costs["decode" if disagg else "aggregate"],
        }
        allowed = WorkerLimits(**r["worker_limits"]).targets(**kwargs)
        physical = WorkerLimits(2 if disagg else 1, r["gpu_budget"]).targets(**kwargs)
        assert allowed == physical
        assert len(allowed) == r["feasible_fleet_count"] > 1
        assert r["gpu_budget"] == 2 * raw["backend"]["gpu_budget"]
        start = (
            r["initial_workers"].get("prefill", 0),
            r["initial_workers"]["decode" if disagg else "aggregate"],
        )
        assert start in allowed
        assert any(sum(target) > sum(start) for target in allowed)


def test_singleton_keeps_original_budget_and_start():
    raw = yaml.safe_load((ROOT / "configs/60traces/D-HT-S01.yaml").read_text())
    result = case_resources(raw, {"resources": {"gpu_budget_multiplier": 1}})
    assert result["gpu_budget"] == raw["backend"]["gpu_budget"] == 16
    assert result["initial_workers"] == {"prefill": 1, "decode": 1}
    assert result["feasible_fleet_count"] == 1


@pytest.mark.parametrize("historical", [False, True])
def test_saved_original_policy_preserves_all_original_case_budgets(historical):
    policy = yaml.safe_load((ROOT / "configs/60traces/comparison.yaml").read_text())
    policy["resources"] = {"gpu_budget_multiplier": 1}
    if historical:
        policy["worker_limits"] = {"min_total": 2, "max_total": 16}
    singleton = []
    for path in sorted((ROOT / "configs/60traces").glob("[AD]-*.yaml")):
        raw = yaml.safe_load(path.read_text())
        result = case_resources(raw, policy)
        assert result["gpu_budget"] == raw["backend"]["gpu_budget"]
        assert result["baseline_gpus"] == result["gpu_budget"]
        assert result["gpu_budget_multiplier"] == 1
        if historical:
            assert result["worker_limits"] == {"min_total": 2, "max_total": 16}
        if result["feasible_fleet_count"] == 1:
            singleton.append(path.stem)
    assert len(singleton) == (40 if historical else 20)
    assert all(name.rsplit("-", 1)[-1].startswith("S") for name in singleton)


def test_current_policy_adds_three_max_cost_workers_for_all_sixty_cases():
    policy = yaml.safe_load((ROOT / "configs/60traces/comparison.yaml").read_text())
    assert policy["resources"] == {"gpu_budget_extra_workers": 3}
    budgets = set()
    for path in sorted((ROOT / "configs/60traces").glob("[AD]-*.yaml")):
        raw = yaml.safe_load(path.read_text())
        original_start = dict(raw["backend"]["autoscalers"][0]["config"])
        original_budget = raw["backend"]["gpu_budget"]
        r = apply_case_resources(raw, policy)
        costs = r["gpus_per_worker"]
        budgets.add(r["gpu_budget"])
        assert r["gpu_budget"] == original_budget + 3 * max(costs.values())
        assert r["headroom_gpus"] == 24
        assert r["baseline_gpus"] == original_budget
        assert r["schema_version"] == 2
        assert "gpu_budget_multiplier" not in r
        assert raw["backend"]["autoscalers"][0]["config"] == original_start
        assert raw["labels"]["resource_policy"] == "baseline-plus-3-max-workers"
        disagg = raw["backend"]["topology"] == "disagg"
        kwargs = dict(
            topology=raw["backend"]["topology"],
            gpu_budget=r["gpu_budget"],
            prefill_gpus=costs.get("prefill", 0),
            decode_gpus=costs["decode" if disagg else "aggregate"],
        )
        allowed = WorkerLimits(**r["worker_limits"]).targets(**kwargs)
        physical = WorkerLimits(2 if disagg else 1, r["gpu_budget"]).targets(**kwargs)
        assert allowed == physical  # No independent worker cap hides affordable fleets.
        assert len(allowed) == r["feasible_fleet_count"] > 1
        start = (
            r["initial_workers"].get("prefill", 0),
            r["initial_workers"]["decode" if disagg else "aggregate"],
        )
        assert start in allowed
    assert budgets == {36, 40, 104, 136, 152}


def test_headroom_uses_actual_worker_gpu_cost_including_attention_dp():
    raw = yaml.safe_load((ROOT / "configs/60traces/D-LC-S01.yaml").read_text())
    raw["backend"]["engines"]["decode"]["attention_dp_size"] = 2
    result = case_resources(raw, {"resources": {"gpu_budget_extra_workers": 3}})
    assert result["gpus_per_worker"] == {"prefill": 4, "decode": 16}
    assert result["baseline_gpus"] == 20
    assert result["headroom_gpus"] == 48
    assert result["gpu_budget"] == 68
    assert "3 × the largest per-worker GPU cost" in resource_policy_description(result)


@pytest.mark.parametrize(
    "spec",
    [
        {},
        {"gpu_budget_multiplier": 1, "gpu_budget_extra_workers": 3},
        {"gpu_budget_extra_workers": -1},
        {"gpu_budget_extra_workers": True},
        {"gpu_budget_extra_workers": 1.5},
        {"gpu_budget_multiplier": 0},
        {"gpu_budget_extra_workers": 3, "gpu_budget": 999},
    ],
)
def test_ambiguous_or_invalid_budget_policy_is_rejected(spec):
    raw = yaml.safe_load((ROOT / "configs/60traces/D-LC-S01.yaml").read_text())
    with pytest.raises(ValueError):
        case_resources(raw, {"resources": spec})


def test_joint_actions_cannot_spend_the_same_free_gpus_twice():
    settings = {
        "min_prefill": 1,
        "max_prefill": 3,
        "min_decode": 1,
        "max_decode": 3,
        "prefill_gpus_per_worker": 8,
        "decode_gpus_per_worker": 8,
        "gpu_budget": 24,
    }
    mask = joint_delta_mask([1, 1], [0, 1], settings)
    assert mask == [[True, True], [True, False]]


def test_training_scale_up_waits_for_draining_gpus():
    settings = {
        "prefill_gpus_per_worker": 4,
        "decode_gpus_per_worker": 8,
        "gpu_budget": 24,
    }
    snapshot = {
        "active_prefill_ids": [0, 1],
        "starting_prefill_ids": [],
        "draining_prefill_ids": [],
        "active_decode_ids": [0],
        "starting_decode_ids": [],
        "draining_decode_ids": [9],
    }
    assert admit_disagg_target([3, 1], [2, 1], snapshot, settings) == [2, 1]
    assert admit_disagg_target([1, 2], [2, 1], snapshot, settings) == [1, 1]
    snapshot["draining_decode_ids"] = []
    assert admit_disagg_target([3, 1], [2, 1], snapshot, settings) == [3, 1]
