# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy

import pytest
from publish_latest import validate_static_export


def fixture():
    value = {
        "job_id": "D-HT-S01-static",
        "budget_audit": {
            "gpu_budget": 40,
            "peak_allocated_gpus": 16,
            "telemetry_samples": 10,
        },
        "config": {
            "backend": {
                "gpu_budget": 40,
                "autoscalers": [{"config": {"num_prefill": 1, "num_decode": 1}}],
            }
        },
        "report": {"results": [{"runtime": {"gpu_budget": 40}}]},
    }
    return (
        value,
        {"simulated_gpus": 16},
        {"gpu_budget": 40, "initial_workers": {"prefill": 1, "decode": 1}},
    )


def test_static_can_share_larger_ceiling_without_using_extra_gpus():
    validate_static_export(*fixture())


@pytest.mark.parametrize(
    "field", ["config", "runtime", "audit", "resident", "fleet", "samples"]
)
def test_static_rejects_budget_or_fleet_drift(field):
    value, case, resources = copy.deepcopy(fixture())
    if field == "config":
        value["config"]["backend"]["gpu_budget"] = 16
    elif field == "runtime":
        value["report"]["results"][0]["runtime"]["gpu_budget"] = 16
    elif field == "audit":
        value["budget_audit"]["gpu_budget"] = 16
    elif field == "resident":
        value["budget_audit"]["peak_allocated_gpus"] = 24
    elif field == "fleet":
        value["config"]["backend"]["autoscalers"][0]["config"]["num_decode"] = 2
    else:
        value["budget_audit"]["telemetry_samples"] = 0
    with pytest.raises(ValueError, match="Static GPU audit failed"):
        validate_static_export(value, case, resources)
