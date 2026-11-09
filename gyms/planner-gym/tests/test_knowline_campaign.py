# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checks for continuation integrity and complete benchmark coverage."""

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import knowline_campaign as campaign


@pytest.mark.parametrize(
    "kind, retry",
    [
        ("TimeoutError", True),
        ("ReadError", True),
        ("ValueError", False),
        ("HTTPStatusError", False),
    ],
)
def test_server_failure_classification(kind, retry):
    state = {
        "state": "failed",
        "last_error": {
            "type": "RuntimeError",
            "message": f"Jev decision failed ({kind}); see decision log",
        },
    }
    assert campaign.is_server_failure(state) == retry
    state["state"] = "completed"
    assert not campaign.is_server_failure(state)


def test_hung_server_is_killed_and_reaped():
    from unittest.mock import Mock

    process = Mock()
    process.poll.return_value = None
    process.wait.side_effect = [subprocess.TimeoutExpired("server", 30), 0]
    campaign.stop_server(process)
    process.terminate.assert_called_once()
    process.kill.assert_called_once()
    assert process.wait.call_count == 2


def row(index):
    payload = {"model": campaign.MODEL, "state": {"tick": index}}
    return {
        "request": payload,
        "request_sha256": hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest(),
        "model": campaign.MODEL,
        "provider": "self_hosted",
        "status": "ok",
        "answers": {"choice": index},
        "targets": {"decode": index},
        "wall_latency_s": 1.2,
        "response_source": "api",
    }


class ContinuationTests(unittest.TestCase):
    def test_preserves_original_decisions_and_inference_latency(self):
        first = row(0)
        replay = {**first, "wall_latency_s": 0.001, "response_source": "replay"}
        second = row(1)
        self.assertEqual(
            campaign.merge_prefix([first], [replay, second]), [first, second]
        )

    def test_shorter_interrupted_replay_does_not_discard_progress(self):
        rows = [row(0), row(1)]
        self.assertEqual(campaign.merge_prefix(rows, rows[:1]), rows)

    def test_rejects_changed_observation_answer_or_action(self):
        for key in ("request_sha256", "answers", "targets"):
            changed = copy.deepcopy(row(0))
            changed[key] = "different"
            with self.assertRaises(ValueError):
                campaign.merge_prefix([row(0)], [changed])

    def test_only_incomplete_tail_is_discarded(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "decisions.jsonl"
            path.write_text(json.dumps(row(0)) + "\n" + '{"incomplete":')
            self.assertEqual(campaign.valid_prefix(path), [row(0)])
            path.write_text(json.dumps(row(0)) + "\n" + '{"corrupt":\n')
            with self.assertRaises(json.JSONDecodeError):
                campaign.valid_prefix(path)

    def test_rejects_modified_payload_or_paid_provider(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "decisions.jsonl"
            for key, value in [
                ("provider", "typesafe"),
                ("request", {"changed": True}),
            ]:
                changed = {**row(0), key: value}
                path.write_text(json.dumps(changed) + "\n")
                with self.assertRaises(ValueError):
                    campaign.valid_prefix(path)

    def test_selected_cases_keep_explicit_historical_worker_limits(self):
        policy = yaml.safe_load(
            (campaign.GYM / "configs/60traces/comparison.yaml").read_text()
        )
        policy["resources"] = {"gpu_budget_multiplier": 1}
        policy["worker_limits"] = {"min_total": 2, "max_total": 16}
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(campaign, "ROOT", Path(temp) / "campaign"),
            patch.object(campaign, "TRACE_DIR", Path(temp) / "traces"),
        ):
            campaign.prepare(["D-HT-S04", "D-LC-S04"], policy=policy)
            plan = json.loads((campaign.ROOT / "plan.json").read_text())
            self.assertEqual(len(plan["cases"]), 2)
            for job in plan["cases"]:
                config = yaml.safe_load(Path(job["config"]).read_text())
                self.assertEqual(
                    config["backend"]["worker_limits"], policy["worker_limits"]
                )
                self.assertEqual(
                    config["backend"]["gpu_budget"],
                    16 if job["case_id"].startswith("D-HT") else 12,
                )
            self.assertEqual(policy["knowline"]["max_calls"], 1000000)

    def test_all_sixty_configs_preserve_benchmark_contract(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(campaign, "ROOT", Path(temp) / "campaign"),
            patch.object(campaign, "TRACE_DIR", Path(temp) / "traces"),
        ):
            campaign.prepare()
            plan = json.loads((campaign.ROOT / "plan.json").read_text())
            self.assertEqual(len(plan["cases"]), 60)
            self.assertEqual(len({c["case_id"] for c in plan["cases"]}), 60)
            self.assertEqual(sum(c["duration_s"] == 604800 for c in plan["cases"]), 4)
            for job in plan["cases"]:
                generated = yaml.safe_load(Path(job["config"]).read_text())
                original = yaml.safe_load(
                    (
                        campaign.GYM / "configs/60traces" / f"{job['case_id']}.yaml"
                    ).read_text()
                )
                self.assertEqual(
                    generated["backend"]["gpu_budget"],
                    original["backend"]["gpu_budget"] + 24,
                )
                self.assertIn(
                    "plus 3 × the largest per-worker GPU cost", generated["description"]
                )
            with self.assertRaises(RuntimeError):
                campaign.prepare()


if __name__ == "__main__":
    unittest.main()
