# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reduce saved red-run timelines into auditable, bounded-size diagnostic evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import ijson

ROOT = Path(__file__).parent
FIELDS = [
    "active_prefill_replicas",
    "active_decode_replicas",
    "provisioned_prefill_replicas",
    "provisioned_decode_replicas",
    "queued_prefill_requests",
    "queued_decode_requests",
    "total_queued_requests",
    "router_pending_prefill_requests",
    "router_pending_decode_requests",
    "running_prefill_requests",
    "running_decode_requests",
    "prefill_active_kv_cache_utilization",
    "decode_active_kv_cache_utilization",
]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def reduce_timeline(path, duration):
    stats = {
        key: {"min": None, "max": None, "integral": 0, "covered_s": 0} for key in FIELDS
    }
    states = defaultdict(float)
    changes, largest, kinds = [], {}, Counter()
    first = previous = last = None
    arrivals = completions = 0
    arrival_peak = None
    samples = 0
    semantics = set()
    peak_queue_worker_states = None
    with path.open("rb") as stream:
        for sample in ijson.items(stream, "results.item.timeline.item", use_float=True):
            t = sample["timestamp_s"]
            row = {"time_s": t, **{k: sample.get(k) for k in FIELDS}}
            kinds[sample.get("sample_kind")] += 1
            semantics.add(sample.get("queue_telemetry_semantics"))
            samples += 1
            if first is None:
                first = row
            if previous:
                dt = max(0, min(t, duration) - min(previous["time_s"], duration))
                for key in FIELDS:
                    value = previous[key]
                    if isinstance(value, (int, float)):
                        stats[key]["integral"] += value * dt
                        stats[key]["covered_s"] += dt
                state = (
                    previous["active_prefill_replicas"],
                    previous["active_decode_replicas"],
                )
                states[state] += dt
                now = (row["active_prefill_replicas"], row["active_decode_replicas"])
                if state != now and t <= duration:
                    changes.append({"time_s": t, "from": state, "to": now})
            if t <= duration:
                for key in FIELDS:
                    value = row[key]
                    if isinstance(value, (int, float)):
                        stats[key]["min"] = (
                            value
                            if stats[key]["min"] is None
                            else min(stats[key]["min"], value)
                        )
                        if stats[key]["max"] is None or value > stats[key]["max"]:
                            stats[key]["max"] = value
                            largest[key] = row
                            if key == "queued_decode_requests":
                                peak_queue_worker_states = {
                                    "time_s": t,
                                    **{
                                        name: len(sample[name])
                                        for name in (
                                            "active_prefill_ids",
                                            "active_decode_ids",
                                            "starting_prefill_ids",
                                            "starting_decode_ids",
                                            "draining_prefill_ids",
                                            "draining_decode_ids",
                                        )
                                        if name in sample
                                    },
                                }
                if sample.get("sample_kind") in ("periodic", "final"):
                    arrivals += sample.get("arriving_requests") or 0
                    completions += sample.get("completed_requests") or 0
                    dt = sample.get("interval_duration_s") or 0
                    if dt > 0:
                        rps = (sample.get("arriving_requests") or 0) / dt
                        if arrival_peak is None or rps > arrival_peak["rps"]:
                            arrival_peak = {"rps": rps, "time_s": t, "window_s": dt}
                last = row
            previous = row
            if sample.get("is_final"):
                break
    for values in stats.values():
        values["mean"] = (
            values["integral"] / values["covered_s"] if values["covered_s"] else None
        )
    return {
        "samples": samples,
        "sample_kinds": dict(kinds),
        "queue_semantics": sorted(str(x) for x in semantics),
        "initial": first,
        "last_in_window": last,
        "replay_end_s": previous["time_s"] if previous else None,
        "stats": stats,
        "peaks": largest,
        "arrival_peak": arrival_peak,
        "peak_queue_worker_states": peak_queue_worker_states,
        "sampled_arrivals": arrivals,
        "sampled_completions": completions,
        "fleet_changes": len(changes),
        "first_changes": changes[:12],
        "last_changes": changes[-5:],
        "fleet_residence": [
            {"prefill": p, "decode": d, "seconds": seconds}
            for (p, d), seconds in sorted(states.items(), key=lambda x: -x[1])
        ],
    }


def extract_selected(manifest, output, results_root, knowline_campaign=None):
    """Reduce selected, hash-bound results without loading the complete report."""
    output.mkdir(parents=True, exist_ok=True)
    for item in json.loads(manifest.read_text())["records"]:
        key = item["job_id"]
        destination = output / (key + ".json")
        if destination.exists():
            previous = json.loads(destination.read_text())
            if previous["source_sha256"] == item["source_sha256"]:
                continue
        if item["planner"] == "knowline31b" and knowline_campaign:
            state = json.loads(
                (
                    knowline_campaign / "cases" / item["case_id"] / "state.json"
                ).read_text()
            )
            if state["state"] != "completed":
                raise ValueError("Cannot analyze an unfinished run: " + key)
            path = Path(state["result"])
        else:
            path = (
                results_root
                / "results"
                / item["case_id"]
                / item["planner"]
                / "results.json"
            )
        if digest(path) != item["source_sha256"]:
            raise ValueError("Selected result checksum mismatch: " + key)
        timeline = reduce_timeline(path, item["duration_s"])
        if not timeline["samples"]:
            raise ValueError("Selected result has no timeline: " + key)
        value = {
            **item,
            "source_path": f"results/{item['case_id']}/{item['planner']}/results.json",
            "timeline_available": True,
            "timeline": timeline,
        }
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temporary.replace(destination)
        print("Extracted " + key, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knowline-campaign", type=Path)
    parser.add_argument("--only", nargs="*")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Selected current results with hashes and scoring metadata",
    )
    parser.add_argument(
        "--output", type=Path, help="Directory for per-result timeline evidence"
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        help="Campaign directory containing results/<case>/<planner>/results.json",
    )
    parser.add_argument(
        "--refresh", action="store_true", help="Re-read cached source histories"
    )
    args = parser.parse_args()
    if args.manifest:
        if not args.output or not (args.results_root or args.knowline_campaign):
            parser.error("--manifest requires --output and a campaign directory")
        extract_selected(
            args.manifest, args.output, args.results_root, args.knowline_campaign
        )
        return
    from build_leaderboards import make_report

    report = make_report(ROOT / "inputs", include_diagnostics=False)
    evidence_path = ROOT / "inputs/diagnostic-evidence.json"
    evidence = (
        json.loads(evidence_path.read_text())
        if evidence_path.exists()
        else {"runs": {}}
    )
    expected = set()
    for page in report["pages"].values():
        for case in page["cases"]:
            for planner in report["core"]:
                row = (
                    report["static_latency"]["rows"][case["case_id"]]
                    if planner == "static"
                    else case["rows"][planner]
                )
                if row["status"] != "completed":
                    continue
                metrics = row["metrics"]
                red = (
                    metrics["good_rate"] < 0.2
                    or metrics["mean_ttft_ms"] / case["ttft_ms"] >= 5
                    or metrics["mean_itl_ms"] / case["itl_ms"] >= 5
                )
                key = case["case_id"] + "-" + planner
                if red:
                    expected.add(key)
                if not red or (args.only and key not in args.only):
                    continue
                if (
                    not args.refresh
                    and evidence["runs"].get(key, {}).get("source_sha256")
                    == row["source_sha256"]
                ):
                    print("Cached " + key, flush=True)
                    continue
                path = (
                    args.results_root
                    / "results"
                    / case["case_id"]
                    / planner
                    / "results.json"
                    if args.results_root
                    else ROOT / row["source_path"]
                )
                if planner == "knowline31b" and args.knowline_campaign:
                    directory = args.knowline_campaign / "cases" / case["case_id"]
                    state = json.loads((directory / "state.json").read_text())
                    path = (
                        directory / Path(state["result"]).parent.name / "results.json"
                    )
                if not path.exists():
                    raise FileNotFoundError("Provide --results-root to refresh " + key)
                if digest(path) != row["source_sha256"]:
                    raise ValueError("Selected result checksum mismatch: " + key)
                reduced = {
                    "source_path": "results/"
                    + case["case_id"]
                    + "/"
                    + planner
                    + "/results.json",
                    "source_sha256": row["source_sha256"],
                    "trace_sha256": case["trace_sha256"],
                    "duration_s": case["duration_s"],
                    "metrics": metrics,
                    "sla": {"ttft_ms": case["ttft_ms"], "itl_ms": case["itl_ms"]},
                    "static_gpu_hours": case["rows"]["static"]["metrics"]["gpu_hours"],
                }
                if planner == "static":
                    reduced.update(
                        timeline_available=False, fixed_allocation=row["runtime"]
                    )
                else:
                    reduced.update(
                        timeline_available=True,
                        timeline=reduce_timeline(path, case["duration_s"]),
                    )
                evidence["runs"][key] = reduced
                evidence_path.write_text(
                    json.dumps(evidence, indent=2, allow_nan=False) + "\n"
                )
                print("Extracted " + key, flush=True)

    if not args.only:
        evidence["runs"] = {k: v for k, v in evidence["runs"].items() if k in expected}
        evidence_path.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
