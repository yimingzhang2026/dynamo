# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import fcntl
import queue
import sys
import tempfile
import threading
import time
import unittest
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_knowline as parallel
from run_knowline import assign_shards


class ParallelPlanTests(unittest.TestCase):
    def cases(self):
        return [
            {
                "case_id": f"{group}-{i:02}",
                "group": group,
                "duration_s": (i + 1) * 100,
                "bytes": i * 500,
            }
            for group in ("A-HT", "A-LC", "D-HT", "D-LC")
            for i in range(15)
        ]

    def test_sixty_cases_have_one_owner_and_two_balanced_node_queues(self):
        cases = self.cases()
        shards = assign_shards(cases)
        self.assertEqual(len(shards), 2)
        self.assertTrue(all(len(shard["case_ids"]) > 4 for shard in shards))
        owners = {
            case: shard["shard"] for shard in shards for case in shard["case_ids"]
        }
        self.assertNotEqual(owners["D-HT-14"], owners["D-LC-14"])
        assigned = [case for shard in shards for case in shard["case_ids"]]
        self.assertCountEqual(assigned, [job["case_id"] for job in cases])
        self.assertEqual(len(set(assigned)), 60)
        self.assertEqual(shards, assign_shards(list(reversed(cases))))

    def test_subset_has_no_unselected_case_or_duplicate(self):
        cases = self.cases()[:46]
        shards = assign_shards(cases)
        self.assertEqual(len(shards), 2)
        self.assertCountEqual(
            [case for shard in shards for case in shard["case_ids"]],
            [job["case_id"] for job in cases],
        )

    def test_duplicate_or_empty_selection_rejected(self):
        cases = self.cases()
        with self.assertRaises(ValueError):
            assign_shards([])
        cases[-1] = cases[0]
        with self.assertRaises(ValueError):
            assign_shards(cases)

    def test_one_case_per_node_has_two_independent_owners(self):
        cases = self.cases()[:2]
        shards = assign_shards(cases, lanes_per_node=1)
        self.assertEqual([shard["shard"] for shard in shards], [0, 1])
        self.assertTrue(all(len(shard["case_ids"]) == 1 for shard in shards))
        self.assertCountEqual(
            [shard["case_ids"][0] for shard in shards],
            [job["case_id"] for job in cases],
        )

    def test_invalid_lane_count_rejected(self):
        for lanes in (0, 5, True, 1.5):
            with self.subTest(lanes=lanes), self.assertRaises(ValueError):
                assign_shards(self.cases(), lanes_per_node=lanes)

    def test_invalid_node_count_rejected(self):
        for nodes in (0, -1, True, 1.5):
            with self.subTest(nodes=nodes), self.assertRaises(ValueError):
                assign_shards(self.cases(), nodes=nodes)

    def test_node_count_never_creates_empty_owners(self):
        self.assertEqual(len(assign_shards(self.cases()[:1], nodes=2)), 1)

    def test_second_case_owner_cannot_take_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.lock"
            with path.open("a") as first, path.open("a") as second:
                fcntl.flock(first, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(second, fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.fixture
def lane_fixture(tmp_path, monkeypatch):
    core = parallel.core
    monkeypatch.setattr(core, "ROOT", tmp_path)
    monkeypatch.setattr(core, "TRACE_DIR", tmp_path / "traces")
    core.TRACE_DIR.mkdir()
    states = {}
    jobs = queue.Queue()
    for i in range(10):
        case = f"case-{i}"
        (tmp_path / "cases" / case).mkdir(parents=True)
        (core.TRACE_DIR / f"{case}.jsonl").write_bytes(b"x")
        states[case] = {"state": "pending", "attempts": []}
        jobs.put({"case_id": case, "bytes": 1})
    monkeypatch.setattr(core, "read_state", lambda case: dict(states[case]))
    monkeypatch.setattr(
        core, "save", lambda path, state: states.__setitem__(path.parent.name, state)
    )
    monkeypatch.setattr(core, "harvest", lambda directory, state: [])
    monkeypatch.setattr(
        parallel.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=1900 * 1024**3),
    )
    servers = {}
    server_history = []

    class Server:
        stopped = False

        def poll(self):
            return 0 if self.stopped else None

        def terminate(self):
            self.stopped = True

        def wait(self, timeout=None):
            return 0

    def start(lane, cpus, allocation):
        assert cpus
        if lane in servers:
            assert servers[lane].stopped, "Stop the old server before replacement"
        servers[lane] = Server()
        server_history.append((lane, allocation, servers[lane]))
        return servers[lane], None  # No HTTP server or model is used by these tests.

    monkeypatch.setattr(core, "start_server", start)
    return SimpleNamespace(
        states=states,
        jobs=jobs,
        servers=servers,
        server_history=server_history,
        root=tmp_path,
    )


def test_two_lanes_drain_queue_once_reuse_servers_and_skip_completed(
    lane_fixture, monkeypatch
):
    fixture = lane_fixture
    fixture.states["case-0"]["state"] = "completed"
    barrier = threading.Barrier(2)
    seen_lanes = set()
    processed = []

    def run(job, lane, cpus, port, deadline, isolated):
        if lane not in seen_lanes:
            seen_lanes.add(lane)
            barrier.wait(timeout=5)
        processed.append(job["case_id"])
        fixture.states[job["case_id"]]["state"] = "completed"

    monkeypatch.setattr(parallel.core, "run_case", run)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                parallel.run_lane,
                lane,
                fixture.jobs,
                [lane],
                fixture.root,
                time.monotonic() + 600,
                False,
            )
            for lane in range(2)
        ]
        assert [future.result() for future in futures] == [[], []]
    assert Counter(processed) == Counter({f"case-{i}": 1 for i in range(1, 10)})
    assert len(fixture.servers) == 2
    assert len(fixture.server_history) == 2
    assert all(server.stopped for server in fixture.servers.values())
    assert all(state["state"] == "completed" for state in fixture.states.values())


def test_case_failure_does_not_discard_queued_work(lane_fixture, monkeypatch):
    fixture = lane_fixture

    def run(job, *args):
        case = job["case_id"]
        if case == "case-0":
            fixture.states[case]["state"] = "running"
            raise ValueError("test replay failure")
        fixture.states[case]["state"] = "completed"

    monkeypatch.setattr(parallel.core, "run_case", run)
    errors = parallel.run_lane(
        0, fixture.jobs, [0], fixture.root, time.monotonic() + 600, False
    )
    assert [error["case_id"] for error in errors] == ["case-0"]
    assert fixture.states["case-0"]["state"] == "failed"
    assert all(
        fixture.states[f"case-{i}"]["state"] == "completed" for i in range(1, 10)
    )
    assert len(fixture.servers) == 1
    assert len(fixture.server_history) == 2


def transport_failure(state):
    state.update(
        state="failed",
        last_error={
            "type": "RuntimeError",
            "message": "Jev decision failed (TimeoutError); see decision log",
        },
    )


def test_timeout_restarts_server_then_resumes_same_case(lane_fixture, monkeypatch):
    fixture = lane_fixture
    fixture.states["case-9"]["state"] = "completed"
    processed = []
    servers = []

    def run(job, *args):
        case = job["case_id"]
        processed.append(case)
        servers.append(fixture.servers[0])
        state = fixture.states[case]
        if len(processed) == 1:
            state["saved_decisions"] = 3059
            transport_failure(state)
        else:
            if case == "case-0":
                assert state["saved_decisions"] == 3059
                assert state["state"] == "pending_server_retry"
            state["state"] = "completed"

    monkeypatch.setattr(parallel.core, "run_case", run)
    assert (
        parallel.run_lane(
            0, fixture.jobs, [0], fixture.root, time.monotonic() + 600, False
        )
        == []
    )
    assert processed == ["case-0", "case-0"] + [f"case-{i}" for i in range(1, 9)]
    assert servers[0] is not servers[1]
    assert all(server is servers[1] for server in servers[2:])
    assert len(fixture.server_history) == 2
    assert len({path for _, path, _ in fixture.server_history}) == 2
    assert all(server.stopped for _, _, server in fixture.server_history)


def test_repeated_timeout_is_bounded_and_next_case_gets_fresh_server(
    lane_fixture, monkeypatch
):
    fixture = lane_fixture
    processed = []

    def run(job, *args):
        case = job["case_id"]
        processed.append(case)
        if case == "case-0":
            transport_failure(fixture.states[case])
        else:
            fixture.states[case]["state"] = "completed"

    monkeypatch.setattr(parallel.core, "run_case", run)
    errors = parallel.run_lane(
        0, fixture.jobs, [0], fixture.root, time.monotonic() + 600, False
    )
    assert processed.count("case-0") == 3
    assert errors[0]["type"] == "ServerRetryLimit"
    assert fixture.states["case-0"]["state"] == "failed"
    assert fixture.states["case-0"]["server_failures"] == 3
    assert len(fixture.server_history) == 4
    assert all(
        fixture.states[f"case-{i}"]["state"] == "completed" for i in range(1, 10)
    )


def test_expired_allocation_preserves_pending_server_retry(lane_fixture, monkeypatch):
    fixture = lane_fixture
    clock = [1000]
    monkeypatch.setattr(parallel.time, "monotonic", lambda: clock[0])

    def run(job, *args):
        transport_failure(fixture.states[job["case_id"]])
        clock[0] = 1500

    monkeypatch.setattr(parallel.core, "run_case", run)
    assert parallel.run_lane(0, fixture.jobs, [0], fixture.root, 1600, False) == []
    assert fixture.states["case-0"]["state"] == "pending_server_retry"
    assert fixture.jobs.qsize() == 9
    assert all(server.stopped for _, _, server in fixture.server_history)


def test_replay_timeout_recycles_server_without_retrying_in_same_allocation(
    lane_fixture, monkeypatch
):
    fixture = lane_fixture
    processed = []

    def run(job, *args):
        case = job["case_id"]
        processed.append(case)
        fixture.states[case]["state"] = (
            "pending_continuation" if case == "case-0" else "completed"
        )

    monkeypatch.setattr(parallel.core, "run_case", run)
    assert (
        parallel.run_lane(
            0, fixture.jobs, [0], fixture.root, time.monotonic() + 600, False
        )
        == []
    )
    assert processed.count("case-0") == 1
    assert len(fixture.server_history) == 2


def test_deadline_preserves_unstarted_queue(lane_fixture):
    fixture = lane_fixture
    assert (
        parallel.run_lane(
            0, fixture.jobs, [0], fixture.root, time.monotonic() + 100, False
        )
        == []
    )
    assert fixture.jobs.qsize() == 10
    assert not fixture.servers
    assert all(state["state"] == "pending" for state in fixture.states.values())


def test_server_failure_leaves_cases_pending(lane_fixture, monkeypatch):
    fixture = lane_fixture

    def fail(*args):
        raise RuntimeError("test model startup failure")

    monkeypatch.setattr(parallel.core, "start_server", fail)
    with pytest.raises(RuntimeError, match="startup failure"):
        parallel.run_lane(
            0, fixture.jobs, [0], fixture.root, time.monotonic() + 600, False
        )
    assert all(state["state"] == "pending" for state in fixture.states.values())


def test_missing_input_waits_for_continuation_without_blocking_other_cases(
    lane_fixture, monkeypatch
):
    fixture = lane_fixture
    (parallel.core.TRACE_DIR / "case-0.jsonl").unlink()
    monkeypatch.setattr(
        parallel.core,
        "run_case",
        lambda job, *args: fixture.states[job["case_id"]].update(state="completed"),
    )
    assert (
        parallel.run_lane(
            0, fixture.jobs, [0], fixture.root, time.monotonic() + 600, False
        )
        == []
    )
    assert fixture.states["case-0"]["state"] == "waiting_for_data"
    assert all(
        fixture.states[f"case-{i}"]["state"] == "completed" for i in range(1, 10)
    )


if __name__ == "__main__":
    unittest.main()
