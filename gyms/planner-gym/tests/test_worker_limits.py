# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml
from autoscaling_arena.match_config import (
    MatchConfigError,
    feasible_worker_targets,
    parse_match_config,
)
from autoscaling_arena.worker_limits import WorkerLimits, project_target


def test_all_asymmetric_disagg_splits_are_allowed_with_unequal_gpu_costs():
    targets = WorkerLimits(2, 16).targets(
        topology="disagg", gpu_budget=124, prefill_gpus=4, decode_gpus=8
    )
    assert len(targets) == 120
    assert (1, 15) in targets and (15, 1) in targets and (1, 1) in targets
    assert (8, 9) not in targets
    for p in range(32):
        for d in range(32):
            applied = project_target((p, d), (1, 1), targets)
            assert applied in targets
            if (p, d) in targets:
                assert applied == (p, d)
    # At capacity, a one-pool scale-up cannot silently force the other pool down.
    assert project_target((15, 2), (15, 1), targets) == (15, 1)


def test_gpu_budget_still_constrains_the_total_worker_range():
    targets = WorkerLimits(2, 16).targets(
        topology="disagg", gpu_budget=16, prefill_gpus=8, decode_gpus=8
    )
    assert targets == ((1, 1),)
    agg = WorkerLimits(2, 16).targets(
        topology="agg", gpu_budget=128, prefill_gpus=0, decode_gpus=8
    )
    assert agg == tuple((0, n) for n in range(2, 17))


def disagg_config():
    path = Path(__file__).parent / "fixtures/match.controlled-disagg.example.yaml"
    config = yaml.safe_load(path.read_text())
    config["backend"].update(
        gpu_budget=16, worker_limits={"min_total": 2, "max_total": 16}
    )
    config["backend"]["autoscalers"] = [
        {
            "name": "test",
            "type": "reactive",
            "start": {"prefill": 1, "decode": 1},
            "config": {
                "min_prefill": 1,
                "max_prefill": 15,
                "min_decode": 1,
                "max_decode": 15,
            },
        }
    ]
    return config


def test_match_requires_allocation_enforcement_before_relaxing_pool_maxima(tmp_path):
    config = disagg_config()
    resolved = parse_match_config(config, source_path=tmp_path / "match.yaml")
    assert (
        len(feasible_worker_targets(resolved.backend, resolved.backend.autoscalers[0]))
        == 120
    )
    del config["backend"]["worker_limits"]
    with pytest.raises(MatchConfigError, match="maximum fleet requires"):
        parse_match_config(config, source_path=tmp_path / "match.yaml")


@pytest.mark.parametrize(
    "bounds",
    [
        {"min_total": True, "max_total": 16},
        {"min_total": 17, "max_total": 16},
        {"min_total": 0, "max_total": 16},
        {"min_total": 3, "max_total": 16},
        {"min_total": 2, "max_total": 16, "typo": 1},
    ],
)
def test_invalid_limits_or_start_fail_before_replay(tmp_path, bounds):
    config = disagg_config()
    config["backend"]["worker_limits"] = bounds
    with pytest.raises(MatchConfigError):
        parse_match_config(config, source_path=tmp_path / "match.yaml")
