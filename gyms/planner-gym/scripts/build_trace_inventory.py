#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render the 60-case inventory from the versioned catalog and Match Configs."""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "A-HT": "Aggregated / high throughput",
    "A-LC": "Aggregated / low concurrency, latency focused",
    "D-HT": "Disaggregated / high throughput",
    "D-LC": "Disaggregated / low concurrency, latency focused",
}
EXPLANATION = """These are 60 experiment-specific test inputs derived from 11 source variants,
not 60 independent datasets. A case ID starts with A (aggregated) or D
(disaggregated), followed by HT or LC. The last component uses S for a small
deployment and L for a large deployment, followed by its case number. Each
experiment group has 10 small and 5 large cases. Deployment size is unrelated
to the S/L suffix used in some latency-profile names.

Aggregated workers perform both prefill (processing the prompt) and decode
(generating output tokens). Disaggregated deployments use separate prefill and
decode workers, so the two pools can use different parallelism and allocations.

HT uses attention data parallelism 8 and MoE expert parallelism 8, with TP 1,
and a 50 ms inter-token-latency (ITL) target. TTFT targets are 2 or 5 seconds,
depending on the case. LC uses attention DP 1 / expert parallelism 1: aggregate
and decode workers use TP 8, while disaggregated prefill workers use TP 4.
Its ITL target is 15 ms and TTFT target is 0.5 or 1.5 seconds. LC is the
low-concurrency/latency-focused regime, but these configs do not impose a
separate low-concurrency client cap: every engine has max_num_seqs=256, and
requests follow the case's scaled arrival trace. HT is not guaranteed to have
a larger arrival rate than LC for every source.

All four groups currently use fixed allocations, round-robin routing, zero
warmup and zero modeled KV-transfer delay. They measure the static serving
baseline, not a trained planner. Results rank by goodput per GPU: throughput
meeting the case's latency objectives, normalized by GPU usage. The groups
change engine parallelism, latency targets and calibrated input scale, so a
cross-group difference is not an isolated measurement of topology alone.

There is no randomized 80/20 split and no implemented RL train/validation/test
split. The planned history is for a future workload predictor: RAGPulse, Azure
and ART use days 1–4 as history and day 5 as test; BurstGPT uses days 1–7 as
history and days 8–14 as test. Qwen and Mooncake use their full short trace for
test and have no separate history. These history windows have not been
materialized as training files by the static preparation scripts. All 60
prepared files contain test traffic only, and each current config uses 0 s
warmup. Training or tuning on these test windows would make their evaluation
in-sample; choose a separate validation protocol before training an RL policy.

Duration means the offered trace window, not simulation wall time. Source
windows are relative to each source's origin; generated test timestamps are
rebased to zero. Short traces include their final timestamp; configs extend
the half-open measurement endpoint by one floating-point step when needed.
Sizes are uncompressed decimal MB (1 MB = 1,000,000 bytes). CSV retains exact
bytes, source-window seconds, hashes and engine settings. Scaling changes
request counts within the same window; alpha < 1 thins requests, alpha > 1
replicates them, with the frozen recipe's seed and jitter. It is not a train/test
ratio or an arrival-speedup setting.
"""


def inventory(root: Path = ROOT) -> tuple[list[dict], list[dict]]:
    catalog = json.loads((root / "configs/60traces/catalog.json").read_text())
    sources = catalog["sources"]
    rows = []
    for case in catalog["cases"]:
        ident = case["case_id"]
        config = yaml.safe_load((root / f"configs/60traces/{ident}.yaml").read_text())
        backend = config["backend"]
        allocation = backend["autoscalers"][0]["config"]
        group = "-".join(ident.split("-")[:2])
        replicas = (
            f"{allocation['num_prefill']}P + {allocation['num_decode']}D"
            if backend["topology"] == "disagg"
            else f"{allocation['num_decode']} aggregate"
        )
        engines = "; ".join(
            f"{role}: TP={engine['tp_size']}, attention-DP={engine['attention_dp_size']}, "
            f"MoE-TP={engine['moe_tp_size']}, EP={engine['moe_ep_size']}"
            for role, engine in backend["engines"].items()
        )
        recipe = case["preparation"]
        if (
            recipe["expected_output_bytes"] != case["expanded_test_bytes"]
            or recipe["expected_requests"] != case["scaled_test_requests"]
            or config["slo_profiles"][0]["ttft_ms"] != case["ttft_ms"]
            or config["slo_profiles"][0]["itl_ms"] != case["itl_ms"]
            or backend["gpu_budget"] != case["simulated_gpus"]
        ):
            raise ValueError(f"Catalog/config mismatch: {ident}")
        rows.append(
            {
                "case_id": ident,
                "experiment": group,
                "experiment_description": GROUPS[group],
                "source": case["source_id"],
                "dataset": case["dataset"],
                "deployment_size": case["deployment_size"],
                "test_duration_s": case["test_duration_s"],
                "test_window": case["test_label"],
                "source_test_start_s": case["test_start_s"],
                "source_test_end_s": case["test_end_s"],
                "planned_history": case["planner_warmup_label"],
                "planned_history_s": case["planner_warmup_s"],
                "training_materialized": False,
                "rl_training_split": "Not defined",
                "static_warmup_s": case["static_warmup_s"],
                "base_requests": case["base_test_requests"],
                "base_bytes": case["base_test_bytes"],
                "test_requests": case["scaled_test_requests"],
                "test_bytes": case["expanded_test_bytes"],
                "test_MB": case["expanded_test_bytes"] / 1e6,
                "scale_factor": case["scale_factor"],
                "seed": recipe["seed"],
                "jitter_s": recipe["jitter_seconds"],
                "trace_format": case["trace_format"],
                "block_size": recipe["block_size"],
                "simulated_GPUs": backend["gpu_budget"],
                "replicas": replicas,
                "engine_parallelism": engines,
                "TTFT_ms": case["ttft_ms"],
                "ITL_ms": case["itl_ms"],
                "input_file": f"data/60traces/reproduction/generated/{ident}.jsonl",
                "config_file": f"configs/60traces/{ident}.yaml",
                "sha256": case["trace_sha256"],
                "source_url": case["source_url"],
            }
        )
    if len(rows) != 60 or len({r["case_id"] for r in rows}) != 60:
        raise ValueError("Expected 60 unique cases")
    return sources, rows


def duration(seconds: float) -> str:
    if seconds == 86400:
        return "24 h"
    if seconds == 604800:
        return "7 days"
    return f"{seconds:.3f} s"


def split_label(row: dict) -> str:
    if row["planned_history_s"] == 345600:
        return "H1"
    if row["planned_history_s"] == 604800:
        return "H2"
    return "H0"


def render(sources: list[dict], rows: list[dict]) -> dict[str, str]:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    md = [
        "<!-- GENERATED by scripts/build_trace_inventory.py; do not edit. -->",
        "<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->",
        "<!-- SPDX-License-Identifier: Apache-2.0 -->",
        "",
        "# 60-case trace inventory",
        "",
        "[Filterable HTML](traces-60.html) · [Exact CSV](traces-60.csv)",
        "",
        EXPLANATION,
        "## Source test intervals",
        "",
        "| Source | Test window | Duration | Planned predictor history | Base requests | Base MB |",
        "|---|---|---:|---|---:|---:|",
    ]
    for source in sources:
        md.append(
            f"| {source['name']} | {source['test_label']} | {duration(source['test_duration_s'])} | {source['planner_warmup_label']} | {source['selected_test_requests']:,} | {source['base_bytes']/1e6:.6f} |"
        )
    md += [
        "",
        "History keys below: **H1** = days 1–4 planned history, day 5 test; **H2** = days 1–7 planned history, days 8–14 test; **H0** = no history, entire trace used for test. None is a materialized RL training split.",
        "",
    ]
    for group, description in GROUPS.items():
        selected = [row for row in rows if row["experiment"] == group]
        md += [
            f"## {group}: {description}",
            "",
            f"15 cases; {sum(r['test_requests'] for r in selected):,} test requests; {sum(r['test_bytes'] for r in selected)/1e9:.6f} GB.",
            "",
            "| Case | Source | Test duration | History/test | Requests | MB | Scale | GPUs | Fixed workers | TTFT / ITL ms |",
            "|---|---|---:|---|---:|---:|---:|---:|---|---:|",
        ]
        for row in selected:
            md.append(
                f"| {row['case_id']} | {row['source']} | {duration(row['test_duration_s'])} | {split_label(row)} | {row['test_requests']:,} | {row['test_MB']:.6f} | {row['scale_factor']:.8g} | {row['simulated_GPUs']} | {row['replicas']} | {row['TTFT_ms']} / {row['ITL_ms']} |"
            )
        md.append("")
    md += [
        "## Run or modify a case",
        "",
        "From `gyms/planner-gym/`:",
        "",
        "```bash",
        "python scripts/run_suite_config.py configs/60traces/suite.yaml --dry-run",
        "python scripts/run_suite_config.py configs/60traces/suite.A-HT.yaml --dry-run",
        "python scripts/run_match_config.py configs/60traces/A-LC-S08.yaml --validate-only --print-matrix",
        "```",
        "",
        "The same four group suites exist for `A-LC`, `D-HT` and `D-LC`. Generated inputs live at `data/60traces/reproduction/generated/<case>.jsonl`. Actual simulations require the native runtime; these dry-runs do not run a benchmark. No new simulation results are claimed.",
        "",
        "All 60 expanded files total 82,236,977 requests and 45,372,641,752 bytes. The 11 unique bases total 8,597,832 requests and 3,843,469,145 bytes. Repeated source/base sizes must not be summed across cases. A-LC-S08 / D-LC-S08 and A-LC-L04 / D-LC-L04 share identical input bytes; they remain different serving experiments.",
        "",
        "Provenance: `configs/60traces/catalog.json`, imported from the supplied reproduction package's `legal-inventory.json` and `cases.json`, plus each versioned Match Config. The frozen reproduction package remains under `data/60traces/reproduction/`.",
        "",
    ]
    columns = [
        "case_id",
        "experiment",
        "source",
        "deployment_size",
        "test_duration_s",
        "planned_history",
        "test_window",
        "test_requests",
        "test_MB",
        "scale_factor",
        "simulated_GPUs",
        "replicas",
        "TTFT_ms",
        "ITL_ms",
        "engine_parallelism",
    ]
    headers = [
        "Case",
        "Experiment",
        "Source",
        "Deployment",
        "Test seconds",
        "Planned history (not generated)",
        "Source test window",
        "Test requests",
        "Test MB",
        "Scale",
        "GPUs",
        "Fixed workers",
        "TTFT ms",
        "ITL ms",
        "Engine parallelism",
    ]
    body = []
    for row in rows:
        cells = []
        for key in columns:
            value = row[key]
            if key == "test_MB":
                value = f"{value:.6f}"
            if key == "test_requests":
                value = f"{value:,}"
            cells.append("<td>" + html.escape(str(value)) + "</td>")
        body.append(f'<tr data-group="{row["experiment"]}">' + "".join(cells) + "</tr>")
    paragraphs = "".join(
        "<p>" + html.escape(part) + "</p>" for part in EXPLANATION.split("\n\n")
    )
    page = (
        """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Planner Gym — 60 cases</title><style>
body{font:15px/1.5 system-ui,sans-serif;margin:0;color:#172538;background:#f4f6fa}main{padding:28px;max-width:1800px;margin:auto}h1{font-size:30px}p{max-width:1100px}header{background:white;padding:24px;border-radius:14px}label{display:inline-block;margin:16px 20px 16px 0}input,select{font:inherit;padding:8px;border:1px solid #8593a6;border-radius:6px}a{color:#07599e}.scroll{overflow:auto;max-height:75vh;background:white;border:1px solid #c7cfda}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:10px;border-bottom:1px solid #e0e5ed;white-space:nowrap}th{position:sticky;top:0;background:#1d3452;color:white;font-size:13px}tr:nth-child(even){background:#f7f9fc}td:first-child{font-weight:650}details{background:white;padding:16px;border-radius:12px;margin:20px 0}summary{font-weight:650;cursor:pointer}.notice{font-weight:600}#count{font-weight:600}@media print{.scroll{max-height:none;overflow:visible}label{display:none}th{position:static}main{padding:0}table{font-size:9px}}
</style></head><body><main><header><h1>Planner Gym: 60 experiment cases</h1><p>DeepSeek-V4-Pro · B300 SXM · 11 source variants · 82,236,977 requests · 45.373 GB expanded inputs</p><p class="notice">All cases contain test traffic. Static warmup is zero. Planned history is not a generated training set.</p><a href="traces-60.csv" download>Download exact CSV</a> · <a href="traces-60.md">Complete written guide</a></header><details open><summary>Understand the four experiments and train/test windows</summary>"""
        + paragraphs
        + """</details><label>Experiment <select id="group"><option value="">All four groups</option>"""
        + "".join(
            f'<option value="{group}">{group} — {html.escape(name)}</option>'
            for group, name in GROUPS.items()
        )
        + """</select></label><label>Search <input id="search" type="search" placeholder="Case or source name"></label><span id="count"></span><div class="scroll"><table><thead><tr>"""
        + "".join("<th>" + html.escape(label) + "</th>" for label in headers)
        + "</tr></thead><tbody>"
        + "".join(body)
        + """</tbody></table></div><p>MB uses decimal units. Exact byte sizes, SHA-256 hashes, source URLs and file paths are in the CSV. This inventory describes inputs and configurations, not newly measured simulation results.</p></main><script>
const rows=[...document.querySelectorAll('tbody tr')],group=document.querySelector('#group'),search=document.querySelector('#search');function update(){let count=0;for(const row of rows){const show=(!group.value||row.dataset.group===group.value)&&row.textContent.toLowerCase().includes(search.value.toLowerCase());row.hidden=!show;if(show)count++}document.querySelector('#count').textContent=`${count} / 60 cases`}group.addEventListener('change',update);search.addEventListener('input',update);update();
</script></body></html>"""
    )
    return {
        "traces-60.csv": stream.getvalue(),
        "traces-60.md": "\n".join(md),
        "traces-60.html": page + "\n",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="fail if generated inventory is stale"
    )
    args = parser.parse_args()
    for name, content in render(*inventory()).items():
        path = ROOT / "docs" / name
        if args.check:
            if not path.is_file() or path.read_text() != content:
                raise SystemExit(f"Inventory is stale: {path}")
        else:
            path.write_text(content)
    print(
        "60-case inventory checked." if args.check else "60-case inventory generated."
    )


if __name__ == "__main__":
    main()
