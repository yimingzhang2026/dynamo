# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU collection audits live session evidence without a campaign-specific wrapper."""

import hashlib
import json

import pytest
import yaml
from export_completed import export

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.mark.parametrize("draining", [False, True])
def test_unarchived_completed_cpu_job_requires_resident_budget_audit(
    tmp_path, draining
):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    job = {
        "id": "case-mpc",
        "case_id": "case",
        "planner": "mpc",
        "config": "config.yaml",
        "session": "session",
        "trace_sha256": "a" * 64,
        "resources": {"gpu_budget": 20},
    }
    config = campaign / job["config"]
    config.write_text(yaml.safe_dump({"backend": {"gpu_budget": 20}}))
    job["config_sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
    (campaign / "plan.json").write_text(json.dumps({"jobs": [job]}))
    (campaign / "status.json").write_text(
        json.dumps({"jobs": {job["id"]: "completed"}})
    )
    result = campaign / "results/case/mpc/results.json"
    result.parent.mkdir(parents=True)
    result.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "status": "ok",
                        "runtime": {"gpu_budget": 20},
                        "raw_metrics": {
                            "attempted_requests": 10,
                            "completed_requests": 10,
                        },
                    }
                ]
            }
        )
    )
    session = campaign / "session"
    session.mkdir()
    (session / "trace-report.json").write_text(
        json.dumps(
            {
                "prefill_gpus_per_worker": 4,
                "decode_gpus_per_worker": 8,
            }
        )
    )
    (session / "telemetry.jsonl").write_text(
        json.dumps(
            {
                "sampled_at_ms": 5000,
                "active_prefill_ids": ["p0"],
                "active_decode_ids": ["d0"],
                "starting_prefill_ids": ["p1"],
                "starting_decode_ids": [],
                "draining_prefill_ids": [],
                "draining_decode_ids": ["d1"] if draining else [],
            }
        )
        + "\n"
    )
    output = tmp_path / "exports"
    if draining:
        with pytest.raises(ValueError, match="GPU budget exceeded"):
            export(campaign, output)
        assert not list(output.glob("*.json"))
    else:
        export(campaign, output)
        saved = json.loads((output / "case-mpc.json").read_text())
        assert saved["budget_audit"]["peak_allocated_gpus"] == 16
        assert saved["source_sha256"] == hashlib.sha256(result.read_bytes()).hexdigest()
    assert result.is_file() and (session / "telemetry.jsonl").is_file()
