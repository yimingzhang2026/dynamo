# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reduce saved traces and replay telemetry into the time series of the Traces tab.

Nothing here starts a simulation. From this directory:

    # Arriving requests, from the benchmark traces themselves.
    python extract_timelines.py arrivals --traces ../../data/60traces/reproduction/generated

    # Per-planner behavior, from the telemetry each selected result saved.
    python extract_timelines.py manifest --output timeline-manifest.json
    # ... on the machine that holds the raw results (standard library only):
    python3 extract_timelines.py reduce --manifest timeline-manifest.json \
        --search /path/to/campaign --output timelines
    # ... back here, with the timelines directory copied over:
    python extract_timelines.py import timelines
    python build_leaderboards.py

`reduce` binds every series to a selected result by content: a `results.json`
must have the result's SHA-256, a `telemetry.jsonl` (also inside a
`session.tar.gz`) must have the telemetry SHA-256 that result recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tarfile
import tempfile
from array import array
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).parent
SCHEMA_VERSION = 1
# One series has at most MAX_BUCKETS points. Widths are multiples of the
# 5-second telemetry interval, so every telemetry sample lies in one bucket.
BUCKET_WIDTHS_S = (5, 10, 15, 30, 60, 120, 150, 300, 600, 900, 1200, 1800, 3600)
MAX_BUCKETS = 600
# Series a reduced run carries; None marks a bucket without a measurement.
SERIES = (
    "arrivals",
    "completed",
    "queued",
    "ttft_ms",
    "itl_ms",
    "kv_utilization",
    "cache_reuse",
    "ready_prefill",
    "ready_decode",
    "gpus",
)
ARRIVAL_KEY = re.compile(rb'"(?:timestamp|request_received_ms)":\s*([0-9.eE+-]+)')


def bucket_grid(duration_s):
    """The (width, count) of the time buckets covering one trace window."""
    for width in BUCKET_WIDTHS_S:
        count = math.ceil(duration_s / width)
        if count <= MAX_BUCKETS:
            return width, count
    raise ValueError(f"No bucket width covers a {duration_s}-second trace")


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1 << 20):
            hasher.update(block)
    return hasher.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


# --- arrivals -----------------------------------------------------------------


def trace_arrivals(path, duration_s):
    """Count the requests of one trace per bucket of replay time.

    Replay starts its clock at the first arrival, so that is the origin here.
    """
    hasher = hashlib.sha256()
    times = array("d")
    with Path(path).open("rb") as stream:
        for line in stream:
            hasher.update(line)
            match = ARRIVAL_KEY.search(line, 0, 400) or ARRIVAL_KEY.search(line)
            if match is None:
                raise ValueError(f"{path}: a line has no arrival timestamp")
            times.append(float(match.group(1)))
    if not times:
        raise ValueError(f"{path}: empty trace")
    origin, last = min(times), max(times)
    width, count = bucket_grid(duration_s)
    counts = [0] * count
    for value in times:
        counts[min(int((value - origin) / 1000 / width), count - 1)] += 1
    return {
        "trace_sha256": hasher.hexdigest(),
        "requests": len(times),
        "bucket_s": width,
        "first_arrival_ms": origin,
        "last_arrival_ms": last,
        "counts": counts,
    }


def _case_arrivals(job):
    case_id, path, duration_s = job
    return case_id, trace_arrivals(path, duration_s)


def write_arrivals(traces, inputs, workers):
    cases = json.loads((inputs / "catalog.json").read_text())["cases"]
    jobs = [
        (c["case_id"], str(traces / (c["case_id"] + ".jsonl")), c["test_duration_s"])
        for c in cases
    ]
    missing = [job[1] for job in jobs if not Path(job[1]).is_file()]
    if missing:
        raise FileNotFoundError("Missing benchmark traces: " + ", ".join(missing[:5]))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        reduced = dict(pool.map(_case_arrivals, jobs))
    for case in cases:
        row = reduced[case["case_id"]]
        if (
            row["trace_sha256"] != case["trace_sha256"]
            or row["requests"] != case["scaled_test_requests"]
        ):
            raise ValueError(
                "Trace differs from the benchmark input: " + case["case_id"]
            )
    lines = ",\n".join(
        f"  {json.dumps(case_id)}: {json.dumps(reduced[case_id], separators=(',', ':'))}"
        for case_id in sorted(reduced)
    )
    save(
        inputs / "arrivals.json",
        f'{{\n "schema_version": {SCHEMA_VERSION},\n'
        ' "source": "Benchmark traces",\n'
        ' "time_origin": "first arrival, as in replay",\n'
        f' "cases": {{\n{lines}\n }}\n}}\n',
    )
    return reduced


# --- telemetry ----------------------------------------------------------------


def _count(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value) if math.isfinite(value) and value >= 0 else None


def _real(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _sum(rows, field):
    if rows is None:
        return None
    values = [_count(row.get(field)) for row in rows]
    return sum(values) if all(v is not None for v in values) else None


def _active_rows(rows, active_ids):
    """The scheduler rows of every active worker, or None if one is missing."""
    if rows is None or active_ids is None:
        return None
    active = {str(worker) for worker in active_ids}
    selected = [
        r
        for r in rows
        if r.get("worker_id") is not None and str(r["worker_id"]) in active
    ]
    covered = {str(r["worker_id"]) for r in selected}
    return selected if active.issubset(covered) else None


def native_sample(raw):
    """One `dynamo.replay.telemetry.v1` snapshot in the saved timeline's terms.

    The same reductions as `autoscaling_arena.runners.sims`, restricted to the
    fields the series need, so a raw telemetry file and the timeline of a saved
    result give the same series.
    """
    traffic = raw.get("traffic") if isinstance(raw.get("traffic"), dict) else {}
    at_ms = _real(raw.get("sampled_at_ms"))
    start_ms = _real(raw.get("interval_start_ms"))
    if at_ms is None or at_ms < 0:
        return None
    if start_ms is None or start_ms < 0:
        start_ms = at_ms

    def rows(role):
        value = raw.get(role + "_scheduler_metrics")
        if not isinstance(value, list):
            return None
        return [row for row in value if isinstance(row, dict)]

    def ids(name):
        value = raw.get(name)
        return list(value) if isinstance(value, (list, tuple)) else None

    fleet = {
        state + "_" + role: ids(f"{state}_{role}_ids")
        for state in ("active", "starting", "draining")
        for role in ("prefill", "decode")
    }
    known = all(value is not None for value in fleet.values())
    fleet = {key: value or [] for key, value in fleet.items()}
    pools = {}
    for role in ("prefill", "decode"):
        everything = rows(role)
        active = _active_rows(everything, fleet["active_" + role] if known else None)
        pools[role] = (everything, active)
    measured = all(e is not None and a is not None for e, a in pools.values())
    blocks = [
        (_sum(active, "active_blocks"), _sum(active, "total_blocks"))
        for _, active in pools.values()
    ]
    utilization = None
    if all(a is not None and t is not None for a, t in blocks):
        used, total = sum(a for a, _ in blocks), sum(t for _, t in blocks)
        if total > 0 and used <= total:
            utilization = used / total
    waiting = [
        _sum(e if measured else None, "waiting_requests") for e, _ in pools.values()
    ]
    pending = [_count(raw.get(f"router_pending_{role}_requests")) for role in pools]
    queued = (
        sum(waiting) + sum(pending)
        if all(v is not None for v in waiting + pending)
        else None
    )
    tokens = []
    for role in pools:
        interval = raw.get(role + "_interval_metrics")
        interval = interval if isinstance(interval, dict) else {}
        tokens.append(
            (
                _count(interval.get("cache_hit_tokens")),
                _count(interval.get("cache_total_tokens")),
            )
        )
    complete = all(h is not None and t is not None for h, t in tokens)
    ttft_count, itl_count = _count(traffic.get("ttft_count")), _count(
        traffic.get("itl_count")
    )
    kind = str(raw.get("kind", "periodic")).lower()
    return {
        "sample_kind": kind,
        "timestamp_s": at_ms / 1000,
        "window_start_s": min(start_ms, at_ms) / 1000,
        "interval_duration_s": _real(traffic.get("duration_s")),
        "arriving_requests": _count(traffic.get("arriving_requests")),
        "completed_requests": _count(traffic.get("completed_requests")),
        "ttft_sample_count": ttft_count,
        "mean_ttft_ms": _real(traffic.get("avg_ttft_ms")) if ttft_count else None,
        "itl_sample_count": itl_count,
        "mean_tpot_ms": _real(traffic.get("avg_itl_ms")) if itl_count else None,
        "total_queued_requests": queued,
        "active_kv_cache_utilization": utilization,
        "scheduler_cache_hit_tokens": sum(h for h, _ in tokens) if complete else None,
        "scheduler_cache_total_tokens": sum(t for _, t in tokens) if complete else None,
        "active_prefill_replicas": len(fleet["active_prefill"]),
        "active_decode_replicas": len(fleet["active_decode"]),
        "provisioned_prefill_replicas": sum(
            len(fleet[state + "_prefill"])
            for state in ("active", "starting", "draining")
        ),
        "provisioned_decode_replicas": sum(
            len(fleet[state + "_decode"])
            for state in ("active", "starting", "draining")
        ),
    }


class Reducer:
    """Bucket the telemetry samples of one run."""

    def __init__(self, duration_s, gpus_per_worker):
        self.duration_s = duration_s
        self.width, self.count = bucket_grid(duration_s)
        self.gpus_per_worker = gpus_per_worker
        self.samples = 0
        self.kinds = Counter()
        zeros = lambda: [0.0] * self.count  # noqa: E731
        self.total = {k: zeros() for k in ("arrivals", "completed", "hit", "tokens")}
        # Weighted sums and their weights, per averaged series.
        self.sums = {k: zeros() for k in SERIES[2:] if k != "cache_reuse"}
        self.weights = {k: zeros() for k in self.sums}
        self.seen = [False] * self.count

    def add(self, sample):
        if sample is None or sample.get("decision_only"):
            return
        at, start = _real(sample.get("timestamp_s")), _real(
            sample.get("window_start_s")
        )
        if at is None:
            return
        self.samples += 1
        self.kinds[str(sample.get("sample_kind"))] += 1
        start = at if start is None else start
        if start >= self.duration_s:
            return  # drain after the arrival window
        index = min(int(start / self.width), self.count - 1)
        self.seen[index] = True
        # A snapshot stands for its interval; the baseline has none.
        weight = max(_real(sample.get("interval_duration_s")) or at - start, 0) or 1e-9
        for key, field in (
            ("arrivals", "arriving_requests"),
            ("completed", "completed_requests"),
        ):
            self.total[key][index] += _count(sample.get(field)) or 0
        hit, tokens = (
            _count(sample.get("scheduler_cache_hit_tokens")),
            _count(sample.get("scheduler_cache_total_tokens")),
        )
        if hit is not None and tokens:
            self.total["hit"][index] += hit
            self.total["tokens"][index] += tokens

        def mean(key, value, by=weight):
            if value is not None and by:
                self.sums[key][index] += value * by
                self.weights[key][index] += by

        mean("queued", _count(sample.get("total_queued_requests")))
        mean("kv_utilization", _real(sample.get("active_kv_cache_utilization")))
        # Latency means are per request, so they weigh by their request counts.
        mean(
            "ttft_ms",
            _real(sample.get("mean_ttft_ms")),
            _count(sample.get("ttft_sample_count")),
        )
        mean(
            "itl_ms",
            _real(sample.get("mean_tpot_ms", sample.get("mean_itl_ms"))),
            _count(sample.get("itl_sample_count", sample.get("tpot_sample_count"))),
        )
        ready = [
            _count(sample.get(f"active_{role}_replicas"))
            for role in ("prefill", "decode")
        ]
        fleet = [
            _count(
                sample.get(
                    f"provisioned_{role}_replicas",
                    sample.get(f"active_{role}_replicas"),
                )
            )
            for role in ("prefill", "decode")
        ]
        mean("ready_prefill", ready[0])
        mean("ready_decode", ready[1])
        if None not in fleet:
            mean(
                "gpus",
                fleet[0] * self.gpus_per_worker["prefill"]
                + fleet[1] * self.gpus_per_worker["decode"],
            )

    def series(self):
        def averaged(key, digits):
            return [
                round(total / weight, digits) if weight else None
                for total, weight in zip(self.sums[key], self.weights[key])
            ]

        return {
            "arrivals": [
                int(v) if seen else None
                for v, seen in zip(self.total["arrivals"], self.seen)
            ],
            "completed": [
                int(v) if seen else None
                for v, seen in zip(self.total["completed"], self.seen)
            ],
            "queued": averaged("queued", 1),
            "ttft_ms": averaged("ttft_ms", 2),
            "itl_ms": averaged("itl_ms", 3),
            "kv_utilization": averaged("kv_utilization", 5),
            "cache_reuse": [
                round(hit / tokens, 4) if tokens else None
                for hit, tokens in zip(self.total["hit"], self.total["tokens"])
            ],
            "ready_prefill": averaged("ready_prefill", 2),
            "ready_decode": averaged("ready_decode", 2),
            "gpus": averaged("gpus", 2),
        }


def result_samples(path):
    """The timeline of a saved `results.json`, without loading the whole file."""
    try:
        import ijson
    except ImportError:  # a few GiB of memory for the largest results
        with Path(path).open("rb") as stream:
            yield from json.load(stream)["results"][0]["timeline"]
        return
    with Path(path).open("rb") as stream:
        yield from ijson.items(stream, "results.item.timeline.item", use_float=True)


def telemetry_samples(path):
    with Path(path).open("rb") as stream:
        for line in stream:
            yield native_sample(json.loads(line))


def reduce_run(job, kind, path):
    reducer = Reducer(job["duration_s"], job["gpus_per_worker"])
    for sample in (
        result_samples(path) if kind == "results" else telemetry_samples(path)
    ):
        reducer.add(sample)
    series = reducer.series()
    return {
        "job_id": job["job_id"],
        "case_id": job["case_id"],
        "planner": job["planner"],
        "verified": {"kind": kind, "sha256": job[kind]["sha256"]},
        "trace_sha256": job["trace_sha256"],
        "duration_s": job["duration_s"],
        "bucket_s": reducer.width,
        "samples": reducer.samples,
        "sample_kinds": dict(reducer.kinds),
        "sampled_arrivals": sum(v or 0 for v in series["arrivals"]),
        "series": series,
    }


def timeline_jobs(report):
    """The selected results, their fingerprints and what reduction needs."""
    setup = report["experiment_setup"]["rows"]
    jobs = []
    for page in report["pages"].values():
        for case in page["cases"]:
            widths = setup[case["case_id"]]["gpus_per_worker"]
            for planner, row in case["rows"].items():
                if row["status"] != "completed":
                    continue
                replay = row.get("runtime", {}).get("replay", {})
                job = {
                    "job_id": row["job_id"],
                    "case_id": case["case_id"],
                    "planner": planner,
                    "trace_sha256": case["trace_sha256"],
                    "duration_s": case["duration_s"],
                    "requests": case["requests"],
                    "gpus_per_worker": {
                        "prefill": widths.get("prefill", 0),
                        "decode": widths.get("decode", widths.get("aggregate", 0)),
                    },
                    "results": {
                        "sha256": row["source_sha256"],
                        "bytes": row["source_bytes"],
                    },
                }
                if replay.get("telemetry_sha256"):
                    job["telemetry"] = {
                        "sha256": replay["telemetry_sha256"],
                        "samples": replay.get("telemetry_sample_count"),
                    }
                jobs.append(job)
    return jobs


def selected_jobs(inputs):
    sys.path.insert(0, str(ROOT))
    from build_leaderboards import make_report

    return timeline_jobs(make_report(inputs, include_diagnostics=False))


def write_manifest(inputs, output):
    jobs = selected_jobs(inputs)
    save(
        output,
        json.dumps({"schema_version": SCHEMA_VERSION, "jobs": jobs}, indent=1) + "\n",
    )
    return jobs


def find_and_reduce(manifest, roots, output):
    """Reduce every selected run whose saved telemetry is under `roots`."""
    jobs = json.loads(Path(manifest).read_text())["jobs"]
    by_hash = {}
    for job in jobs:
        by_hash[("results", job["results"]["sha256"])] = job
        if "telemetry" in job:
            by_hash[("telemetry", job["telemetry"]["sha256"])] = job
    result_sizes = {job["results"]["bytes"] for job in jobs}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    written = {}

    def done(job):
        path = output / (job["job_id"] + ".json")
        if job["job_id"] in written:
            return True
        if path.is_file():
            saved = json.loads(path.read_text())
            if saved.get("verified", {}).get("sha256") in (
                job["results"]["sha256"],
                job.get("telemetry", {}).get("sha256"),
            ):
                written[job["job_id"]] = "kept"
                return True
        return False

    def consider(kind, path, shown):
        job = by_hash.get((kind, digest(path)))
        if job is None or done(job):
            return
        value = reduce_run(job, kind, path)
        if value["sampled_arrivals"] != job["requests"]:
            print(
                f"Rejected {shown}: telemetry covers {value['sampled_arrivals']} of {job['requests']} requests",
                flush=True,
            )
            return
        save(
            output / (job["job_id"] + ".json"),
            json.dumps(value, separators=(",", ":")) + "\n",
        )
        written[job["job_id"]] = shown
        print(f"Reduced {job['job_id']} from {shown}", flush=True)

    for root in roots:
        for current, directories, files in os.walk(root):
            directories.sort()
            for name in sorted(files):
                path = Path(current, name)
                try:
                    if name == "results.json" and path.stat().st_size in result_sizes:
                        consider("results", path, str(path))
                    elif name == "telemetry.jsonl":
                        consider("telemetry", path, str(path))
                    elif name == "session.tar.gz":
                        with tarfile.open(path) as archive:
                            for member in archive:
                                if not member.isfile() or not member.name.endswith(
                                    "telemetry.jsonl"
                                ):
                                    continue
                                with tempfile.NamedTemporaryFile(
                                    dir=output, suffix=".tmp"
                                ) as copy:
                                    source = archive.extractfile(member)
                                    while block := source.read(1 << 20):
                                        copy.write(block)
                                    copy.flush()
                                    consider(
                                        "telemetry", copy.name, f"{path}:{member.name}"
                                    )
                except (OSError, tarfile.TarError, ValueError, KeyError) as error:
                    print(
                        f"Skipped {path}: {type(error).__name__}: {error}", flush=True
                    )
    missing = [
        job["job_id"] for job in jobs if job["job_id"] not in written and not done(job)
    ]
    found = Counter(job["planner"] for job in jobs if job["job_id"] in written)
    summary = {
        "reduced": len(written),
        "expected": len(jobs),
        "by_planner": {
            p: f"{found[p]} / {n}"
            for p, n in sorted(Counter(j["planner"] for j in jobs).items())
        },
        "missing": missing,
    }
    save(output / "summary.json", json.dumps(summary, indent=1) + "\n")
    return summary


def validate_timeline(value, job):
    """Why a reduced run cannot stand for a selected result, or None."""
    width, count = bucket_grid(job["duration_s"])
    verified = value.get("verified", {})
    expected = job.get(verified.get("kind"), {}).get("sha256")
    series = value.get("series", {})
    if expected is None or verified.get("sha256") != expected:
        return "its source is not the selected result"
    if value.get("trace_sha256") != job["trace_sha256"]:
        return "it replays another trace"
    if value.get("bucket_s") != width or set(series) != set(SERIES):
        return "its time buckets or series differ from this version"
    if any(not isinstance(series[k], list) or len(series[k]) != count for k in SERIES):
        return "a series has the wrong length"
    if any(
        v is not None
        and not (
            isinstance(v, (int, float))
            and not isinstance(v, bool)
            and math.isfinite(v)
            and v >= 0
        )
        for k in SERIES
        for v in series[k]
    ):
        return "a series holds a value that is not a non-negative number"
    if (
        value.get("sampled_arrivals") != job["requests"]
        or sum(v or 0 for v in series["arrivals"]) != job["requests"]
    ):
        return "its telemetry does not cover every request of the trace"
    samples = job.get("telemetry", {}).get("samples")
    if samples is not None and value.get("samples") != samples:
        return "its telemetry sample count differs from the selected result"
    return None


def import_timelines(directory, inputs):
    """Merge verified reduced runs into `inputs/timelines.json`.

    Saved series whose result was since replaced are dropped; `directory` may be
    None to do only that.
    """
    jobs = {j["job_id"]: j for j in selected_jobs(inputs)}
    path = inputs / "timelines.json"
    runs = json.loads(path.read_text())["runs"] if path.is_file() else {}
    runs = {
        k: v
        for k, v in runs.items()
        if k in jobs and validate_timeline(v, jobs[k]) is None
    }
    rejected = {}
    for source in sorted(Path(directory).glob("*.json")) if directory else ():
        if source.name == "summary.json":
            continue
        value = json.loads(source.read_text())
        job = jobs.get(value.get("job_id"))
        reason = (
            "it is not a selected result"
            if job is None
            else validate_timeline(value, job)
        )
        if reason:
            rejected[source.name] = reason
            continue
        runs[job["job_id"]] = {
            k: value[k]
            for k in (
                "verified",
                "trace_sha256",
                "bucket_s",
                "samples",
                "sampled_arrivals",
                "series",
            )
        }
    lines = ",\n".join(
        f"  {json.dumps(key)}: {json.dumps(runs[key], separators=(',', ':'))}"
        for key in sorted(runs)
    )
    save(
        path,
        f'{{\n "schema_version": {SCHEMA_VERSION},\n "runs": {{\n{lines}\n }}\n}}\n',
    )
    covered = Counter(jobs[key]["planner"] for key in runs)
    return {
        "runs": len(runs),
        "expected": len(jobs),
        "by_planner": {
            p: f"{covered[p]} / {n}"
            for p, n in sorted(Counter(j["planner"] for j in jobs.values()).items())
        },
        "rejected": rejected,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    arrivals = commands.add_parser(
        "arrivals", help="Count arriving requests in the benchmark traces"
    )
    arrivals.add_argument(
        "--traces", type=Path, required=True, help="Directory holding <case>.jsonl"
    )
    arrivals.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    manifest = commands.add_parser(
        "manifest", help="List the selected results and their fingerprints"
    )
    manifest.add_argument("--output", type=Path, default=Path("timeline-manifest.json"))
    reduce = commands.add_parser(
        "reduce", help="Find, verify and reduce saved telemetry"
    )
    reduce.add_argument("--manifest", type=Path, required=True)
    reduce.add_argument(
        "--search",
        type=Path,
        nargs="+",
        required=True,
        help="Directories holding raw results",
    )
    reduce.add_argument("--output", type=Path, required=True)
    merge = commands.add_parser(
        "import", help="Verify reduced runs and merge them into the inputs"
    )
    merge.add_argument("directory", type=Path)
    for command in (arrivals, manifest, merge):
        command.add_argument("--inputs", type=Path, default=ROOT / "inputs")
    args = parser.parse_args()
    if args.command == "arrivals":
        reduced = write_arrivals(args.traces, args.inputs, args.workers)
        result = {
            "cases": len(reduced),
            "requests": sum(r["requests"] for r in reduced.values()),
        }
    elif args.command == "manifest":
        result = {
            "jobs": len(write_manifest(args.inputs, args.output)),
            "manifest": str(args.output),
        }
    elif args.command == "reduce":
        result = find_and_reduce(args.manifest, args.search, args.output)
    else:
        result = import_timelines(args.directory, args.inputs)
    print(json.dumps(result, indent=1), flush=True)


if __name__ == "__main__":
    main()
