# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render the real policy/setup helpers for new and saved resource policies."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "policy,budget,label,key",
    [
        (
            {"gpu_budget_extra_workers": 3},
            36,
            "Static + 3 × max(worker GPUs)",
            "baseline-plus-3-max-workers",
        ),
        ({"gpu_budget_multiplier": 1}, 12, "1× Static", "1x"),
        ({"gpu_budget_multiplier": 2}, 24, "2× Static", "2x"),
    ],
)
def test_html_policy_and_setup_use_the_saved_report_contract(
    policy, budget, label, key
):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to execute the HTML's JavaScript helpers")
    template = Path(__file__).with_name("leaderboard.html").read_text()
    helpers = template.split("const resourceMultiplier=", 1)[1].split(
        "function methods(){", 1
    )[0]
    case = {"case_id": "test", "topology": "disagg", "regime": "LC", "size": "small"}
    setup = dict(
        initial_workers={"prefill": 1, "decode": 1},
        gpus_per_worker={"prefill": 4, "decode": 8},
        initial_gpus=12,
        baseline_gpus=12,
        gpu_budget=budget,
        min_workers={"prefill": 1, "decode": 1},
        max_workers={"prefill": (budget - 8) // 4, "decode": (budget - 4) // 8},
        worker_limits={"min_total": 2, "max_total": (budget - 8) // 4 + 1},
        min_gpus=12,
    )
    payload = dict(
        source="const resourceMultiplier=" + helpers,
        report={
            "resource_policy": policy,
            "pages": {"disagg": {"cases": [case]}},
            "experiment_setup": {"rows": {"test": setup}},
        },
    )
    program = """
const fs = require('node:fs');
const vm = require('node:vm');
const data = JSON.parse(fs.readFileSync(0, 'utf8'));
const result = vm.runInNewContext(data.source + '\\nJSON.stringify({html:experimentSetup(),key:resourcePolicyKey,text:resourcePolicyText})', {report:data.report,esc:String});
process.stdout.write(result);
"""
    completed = subprocess.run(
        [node, "-e", program],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
        timeout=10,
    )
    result = json.loads(completed.stdout)
    assert label in result["html"]
    assert f"<b>{budget}</b>" in result["html"]
    assert result["key"] == key
    assert "undefined" not in result["html"]
    if "gpu_budget_extra_workers" in policy:
        assert "plus 3 × the largest per-worker GPU cost" in result["text"]
        assert "3× Static" not in result["html"]
