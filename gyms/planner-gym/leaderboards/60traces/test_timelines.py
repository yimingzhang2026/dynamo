# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the time series of the Traces tab: reduction, identity and the page build."""

import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import pytest
from build_leaderboards import check_report, make_report, make_trace_view, write_report
from extract_timelines import (
    MAX_BUCKETS,
    SERIES,
    Reducer,
    bucket_grid,
    find_and_reduce,
    import_timelines,
    native_sample,
    timeline_jobs,
    trace_arrivals,
    validate_timeline,
)

ROOT = Path(__file__).parent
pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def snapshot(index, *, arriving, ttft=None, waiting=(0, 0), decode_ids=(0,)):
    """One native telemetry snapshot of a disaggregated fleet, 5 seconds long."""
    kind = "baseline" if index == 0 else "periodic"
    rows = lambda ids, queued: [  # noqa: E731
        {
            "worker_id": worker,
            "active_blocks": 10,
            "total_blocks": 100,
            "running_requests": 1,
            "waiting_requests": queued,
        }
        for worker in ids
    ]
    return {
        "sample_ordinal": index,
        "kind": kind,
        "interval_start_ms": max(index - 1, 0) * 5000.0,
        "sampled_at_ms": index * 5000.0,
        "traffic": {
            "duration_s": 5.0 if index else 0.0,
            "arriving_requests": arriving,
            "completed_requests": arriving,
            "avg_ttft_ms": ttft or 0.0,
            "avg_itl_ms": 10.0 if ttft else 0.0,
            "ttft_count": arriving if ttft else 0,
            "itl_count": arriving if ttft else 0,
        },
        "prefill_scheduler_metrics": rows([0], waiting[0]),
        "decode_scheduler_metrics": rows(decode_ids, waiting[1]),
        "prefill_interval_metrics": {"cache_hit_tokens": 0, "cache_total_tokens": 0},
        "decode_interval_metrics": {"cache_hit_tokens": 0, "cache_total_tokens": 0},
        "router_pending_prefill_requests": 1,
        "router_pending_decode_requests": 0,
        "active_prefill_ids": [0],
        "active_decode_ids": list(decode_ids),
        "starting_prefill_ids": [],
        "starting_decode_ids": [7] if len(decode_ids) > 1 else [],
        "draining_prefill_ids": [],
        "draining_decode_ids": [],
    }


# Twenty seconds: one quiet interval, then a burst served by a second decode worker.
SNAPSHOTS = [
    snapshot(0, arriving=0),
    snapshot(1, arriving=4, ttft=100.0),
    snapshot(2, arriving=12, ttft=300.0, waiting=(2, 5), decode_ids=(0, 1)),
    snapshot(3, arriving=0, decode_ids=(0, 1)),
    snapshot(4, arriving=4, ttft=100.0),
]
WIDTHS = {"prefill": 4, "decode": 8}


def telemetry_file(path, snapshots=SNAPSHOTS):
    path.write_text("".join(json.dumps(s) + "\n" for s in snapshots))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def job_for(telemetry_sha256, requests=20):
    return {
        "job_id": "T-mpc",
        "case_id": "T",
        "planner": "mpc",
        "trace_sha256": "trace",
        "duration_s": 20,
        "requests": requests,
        "gpus_per_worker": WIDTHS,
        "results": {"sha256": "result", "bytes": 1},
        "telemetry": {"sha256": telemetry_sha256, "samples": len(SNAPSHOTS)},
    }


@pytest.mark.parametrize(
    "duration,expected",
    [
        (1022.025, (5, 205)),
        (3536.999, (10, 354)),
        (7199.464, (15, 480)),
        (86400, (150, 576)),
        (604800, (1200, 504)),
    ],
)
def test_every_benchmark_window_fits_the_point_limit(duration, expected):
    assert bucket_grid(duration) == expected
    assert expected[1] <= MAX_BUCKETS and expected[0] % 5 == 0


def test_reduction_averages_gauges_over_time_and_latency_over_requests():
    reducer = Reducer(20, WIDTHS)
    for raw in SNAPSHOTS:
        reducer.add(native_sample(raw))
    series = reducer.series()
    assert reducer.width == 5 and reducer.samples == len(SNAPSHOTS)
    assert set(series) == set(SERIES)
    assert series["arrivals"] == [4, 12, 0, 4]
    # A bucket without completed requests has no latency, not a zero.
    assert series["ttft_ms"] == [100.0, 300.0, None, 100.0]
    # Waiting in every scheduler (2 + 5 + 5) plus the router's pending request.
    assert series["queued"] == [1.0, 13.0, 1.0, 1.0]
    assert series["kv_utilization"] == [0.1] * 4
    assert series["ready_decode"] == [1.0, 2.0, 2.0, 1.0]
    # Starting workers are provisioned: 1 prefill x 4 + 3 decode x 8.
    assert series["gpus"] == [12.0, 28.0, 28.0, 12.0]
    assert series["cache_reuse"] == [None] * 4


def test_coarser_buckets_weigh_latency_by_requests():
    reducer = Reducer(3500, WIDTHS)  # 10-second buckets: two snapshots in each
    for raw in SNAPSHOTS:
        reducer.add(native_sample(raw))
    series = reducer.series()
    assert reducer.width == 10
    assert series["arrivals"][:3] == [16, 4, None]
    assert series["ttft_ms"][:3] == [250.0, 100.0, None]  # (4 x 100 + 12 x 300) / 16
    assert series["queued"][:3] == [7.0, 1.0, None]  # (1 + 13) / 2: equal intervals


def test_saved_timeline_samples_reduce_like_native_telemetry():
    native, saved = Reducer(20, WIDTHS), Reducer(20, WIDTHS)
    for raw in SNAPSHOTS:
        sample = native_sample(raw)
        native.add(sample)
        # A saved result stores the same fields; a decision marker is not a sample.
        saved.add(dict(sample))
        saved.add({"decision_only": True, "timestamp_s": sample["timestamp_s"]})
    assert saved.series() == native.series() and saved.samples == native.samples


@pytest.mark.parametrize("trace_format", ["mooncake", "dynamo"])
def test_arrivals_use_the_replay_clock(tmp_path, trace_format):
    # Replay starts at the first arrival: 30 s into the window here.
    times = [30000, 30100, 34999, 35000, 49999.5]
    line = (
        (lambda t: {"request_id": "r", "timestamp": t, "input_length": 1})
        if trace_format == "mooncake"
        else (
            lambda t: {
                "schema": "dynamo.request.trace.v1",
                "event_time_unix_ms": 1,
                "request": {"request_id": "timestamp", "request_received_ms": t},
            }
        )
    )
    path = tmp_path / "trace.jsonl"
    path.write_text("".join(json.dumps(line(t)) + "\n" for t in times))
    reduced = trace_arrivals(path, 20)
    assert reduced["counts"] == [3, 1, 0, 1] and reduced["bucket_s"] == 5
    assert reduced["trace_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert reduced["first_arrival_ms"] == 30000


def test_telemetry_is_found_by_content_and_bound_to_its_result(tmp_path):
    raw = tmp_path / "campaign/runs/any-name/telemetry.jsonl"
    raw.parent.mkdir(parents=True)
    job = job_for(telemetry_file(raw))
    other = tmp_path / "campaign/runs/another/telemetry.jsonl"
    other.parent.mkdir()
    telemetry_file(other, SNAPSHOTS[:3])  # some other replay
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"jobs": [job]}))

    summary = find_and_reduce(manifest, [tmp_path / "campaign"], tmp_path / "out")
    assert summary["reduced"] == 1 and summary["missing"] == []
    value = json.loads((tmp_path / "out/T-mpc.json").read_text())
    assert value["verified"] == {
        "kind": "telemetry",
        "sha256": job["telemetry"]["sha256"],
    }
    assert validate_timeline(value, job) is None

    for change, reason in [
        ({"verified": {"kind": "telemetry", "sha256": "other"}}, "selected result"),
        ({"trace_sha256": "other"}, "another trace"),
        ({"samples": 4}, "sample count"),
        ({"sampled_arrivals": 19}, "every request"),
        ({"bucket_s": 10}, "time buckets"),
        ({"series": {**value["series"], "gpus": [1.0]}}, "wrong length"),
        ({"series": {**value["series"], "gpus": [-1.0, 1, 1, 1]}}, "non-negative"),
    ]:
        assert reason in validate_timeline({**value, **change}, job)


def test_telemetry_inside_a_session_archive_is_found(tmp_path):
    raw = tmp_path / "telemetry.jsonl"
    job = job_for(telemetry_file(raw))
    archive = tmp_path / "campaign/results/T/mpc/session.tar.gz"
    archive.parent.mkdir(parents=True)
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(raw, arcname="runs/0001-sim/telemetry.jsonl")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"jobs": [job]}))
    summary = find_and_reduce(manifest, [tmp_path / "campaign"], tmp_path / "out")
    assert summary["by_planner"] == {"mpc": "1 / 1"}
    assert not list((tmp_path / "out").glob("*.tmp"))


def test_telemetry_that_misses_requests_is_not_reduced(tmp_path):
    raw = tmp_path / "campaign/telemetry.jsonl"
    raw.parent.mkdir()
    job = job_for(telemetry_file(raw), requests=21)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"jobs": [job]}))
    summary = find_and_reduce(manifest, [tmp_path / "campaign"], tmp_path / "out")
    assert summary["reduced"] == 0 and summary["missing"] == ["T-mpc"]


@pytest.fixture(scope="module")
def report():
    return make_report(ROOT / "inputs")


def series_for(job):
    """A well-formed series for a real selected result (values are placeholders)."""
    width, count = bucket_grid(job["duration_s"])
    arrivals = [0] * count
    arrivals[0] = job["requests"]
    return {
        "verified": {"kind": "telemetry", "sha256": job["telemetry"]["sha256"]},
        "trace_sha256": job["trace_sha256"],
        "bucket_s": width,
        "samples": job["telemetry"]["samples"],
        "sampled_arrivals": job["requests"],
        "series": {
            **{key: [1.0] * count for key in SERIES},
            "arrivals": arrivals,
            "cache_reuse": [None] * count,
        },
    }


def test_trace_view_covers_every_trace_and_result(report):
    view = make_trace_view(ROOT / "inputs", report)
    cases = {c["case_id"]: c for p in report["pages"].values() for c in p["cases"]}
    assert set(view["arrivals"]) == set(cases) and len(cases) == 60
    assert all(
        sum(view["arrivals"][key]["counts"]) == case["requests"]
        for key, case in cases.items()
    )
    assert len(view["details"]) == 360
    # Static never scales; every result reports its average GPUs.
    assert all(view["details"][key + "-static"]["scale_events"] == 0 for key in cases)
    assert sum(view["timeline_runs"].values()) == len(view["timelines"])


@pytest.mark.parametrize("change", ["counts", "trace", "missing"])
def test_arrivals_must_be_the_benchmark_traces(tmp_path, report, change):
    shutil.copytree(ROOT / "inputs", tmp_path / "inputs")
    path = tmp_path / "inputs/arrivals.json"
    saved = json.loads(path.read_text())
    first = next(iter(saved["cases"]))
    if change == "counts":
        saved["cases"][first]["counts"][0] += 1
    elif change == "trace":
        saved["cases"][first]["trace_sha256"] = "0" * 64
    else:
        del saved["cases"][first]
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="Arrivals"):
        make_trace_view(tmp_path / "inputs", report)


def test_a_time_series_must_belong_to_the_selected_result(tmp_path, report):
    shutil.copytree(ROOT / "inputs", tmp_path / "inputs")
    job = next(j for j in timeline_jobs(report) if j["planner"] == "mpc")
    path = tmp_path / "inputs/timelines.json"

    path.write_text(json.dumps({"runs": {job["job_id"]: series_for(job)}}))
    view = make_trace_view(tmp_path / "inputs", report)
    assert list(view["timelines"]) == [job["job_id"]]
    assert view["timeline_runs"]["mpc"] == 1

    stale = series_for(job)
    stale["verified"]["sha256"] = "0" * 64  # the result was replaced
    path.write_text(json.dumps({"runs": {job["job_id"]: stale}}))
    with pytest.raises(ValueError, match="not the selected result"):
        make_trace_view(tmp_path / "inputs", report)
    # Importing drops it instead, and rejects reduced runs that do not verify.
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    (incoming / "stale.json").write_text(json.dumps({"job_id": job["job_id"], **stale}))
    (incoming / "good.json").write_text(
        json.dumps({"job_id": job["job_id"], **series_for(job)})
    )
    summary = import_timelines(incoming, tmp_path / "inputs")
    assert summary["runs"] == 1 and list(summary["rejected"]) == ["stale.json"]
    assert make_trace_view(tmp_path / "inputs", report)["timeline_runs"]["mpc"] == 1


def test_traces_page_is_written_checked_and_linked_only_when_built(tmp_path, report):
    view = make_trace_view(ROOT / "inputs", report)
    write_report(report, tmp_path / "with", trace_view=view)
    check_report(report, tmp_path / "with", trace_view=view)
    page = (tmp_path / "with/traces.html").read_text()
    assert '"selected": "trace"' in page and '"trace_page": true' in page
    assert '"trace_page": true' in (tmp_path / "with/index.html").read_text()
    proof = json.loads((tmp_path / "with/validation.json").read_text())["trace_view"]
    assert proof["arrival_series"] == 60
    assert proof["input_manifest"][0]["path"] == "inputs/arrivals.json"

    write_report(report, tmp_path / "without")
    assert not (tmp_path / "without/traces.html").exists()
    assert '"trace_page"' not in (tmp_path / "without/index.html").read_text()
    # The series never enter the scores: both builds export the same evidence.
    assert (tmp_path / "with/leaderboards.json").read_bytes() == (
        tmp_path / "without/leaderboards.json"
    ).read_bytes()
