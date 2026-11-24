<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 60-trace results and analysis

**[The Planner Gym README explains every tab with figures](../../README.md#reading-each-tab)**
and shows the [current standings](../../README.md#current-standings). This page
covers what is behind the report: its inputs, validation and maintenance.

For the interactive report, open [index.html](index.html) from a local
checkout, or use GitHub's **Download raw file** button and open the saved file
in your browser. GitHub's repository file viewer displays HTML source rather
than running the report. The downloaded report works offline without a server.

Run `python build_leaderboards.py` to regenerate the six standalone pages:
`index.html`, `aggregated.html`, `disaggregated.html`, `latency.html`,
`slo-rankings.html` and `traces.html`. They work offline.

## Traces tab

`traces.html` shows one trace at a time: a leaderboard of the six planners on
that trace, ranked by goodput per GPU, with good rate, goodput, GPU hours,
average GPUs, mean/P95/P99 TTFT, mean ITL, P95 end-to-end latency, oscillations
and scale events; and the behavior of each planner over the time of the trace.
Its numbers are the saved results; its two series inputs never enter a score.

The charts share one time axis that can be zoomed and moved. Scroll over a
chart to zoom around the pointer. Drag to pick a period or, in "Drag to move"
or with Shift held, to move along the trace. The buttons and the arrow, `+`,
`-` and `0` keys do the same, and the strip above the charts shows the whole
trace with the period as a window that can be dragged or resized. Value axes fit
the period shown unless "Fit values to period" is cleared. A series has at most
600 points, one per time bucket (5 seconds to 20 minutes, by trace length), so
a zoom stops at four points and shows nothing shorter than a bucket.

`inputs/arrivals.json` holds the requests arriving per time bucket in each
benchmark trace, on the replay clock (replay starts at the first arrival). It is
checked against each trace's fingerprint and full request count. Rebuild it from
the expanded traces with:

```bash
python extract_timelines.py arrivals --traces ../../data/60traces/reproduction/generated
```

`inputs/timelines.json` holds, per run, the bucketed queue length, TTFT, ITL,
KV cache utilization, prefix cache reuse, ready workers and provisioned GPUs
from the telemetry that run saved. A run without an entry has no curve: the page
names the planners it has no series for, and draws Static's fixed fleet as
constants. A series is accepted only for the selected result: its source must
be the result file with the selected SHA-256, or the telemetry file with the
SHA-256 that result recorded, and its arrivals must add up to the trace's
requests. Replacing a result drops its series.

The series are reduced where the raw results are; no simulation is started:

```bash
python extract_timelines.py manifest --output timeline-manifest.json
# on the machine holding the raw results (Python 3.8+, standard library):
python3 extract_timelines.py reduce --manifest timeline-manifest.json \
    --search <campaign-directory> --output timelines
# back in this directory, with timelines/ copied over:
python extract_timelines.py import timelines
python build_leaderboards.py
```

`reduce` searches the given directories for `results.json`, `telemetry.jsonl`
and `session.tar.gz` files, matches them to selected results by content, and
writes `summary.json` listing the runs it found none for. `collect_timelines.sh
HOST REMOTE_PYTHON SEARCH_DIR...` runs these steps over SSH.

A trace has at most 600 points (150 s each for a 24-hour trace). Each point is
a mean over its time bucket: gauges are averaged over time, latency over the
requests that produced it.

## Inputs and validation

`inputs/results.json` contains one current record for each of 300 dynamic jobs.
Unfinished replacements contain no previous result or score. `plan.json` defines
the expected case/controller pairs.
The 60 Static records are in `static-latency.json`. During a rerun, unfinished
Static records remain pending without previous metrics. Every tab uses only
completed, validated runs for rankings, good rates and latency measurements.
Campaign preparation uses the current Static runtime and peak-memory measurements
in `static-latency.json` for execution resource estimates.
There are 360 main outcomes. Paid JEV is excluded.

`Cognitive Agent` is the leaderboard name of the scaling agent built around the
open KnowLine-31B decision model. Planner keys, saved run labels, result sources
and identity checks keep the model's name (`knowline31b`, `KnowLine-31B`,
`KnowLine`).

The builder validates trace fingerprints, complete request counts, SLOs,
measurement windows, worker limits, normalization and KnowLine model identity.
It rejects duplicates, missing case/controller records, partial traces and stale
diagnoses. Pending records have no result metrics.
Fingerprints identify the original result files; the compact extracts omit
execution locations and retired attempts.

Results keep the resource policy recorded in their `inputs/plan.json`.
This comparison uses a maximum GPU budget of **the original Static allocation
plus 3 × the largest per-worker GPU cost**. The starting fleet is unchanged.
Default worker bounds reserve one worker per active pool and impose no additional count cap.
The plan records baseline GPUs and maximum GPUs separately.
Import rejects configured or executed budgets that differ from
this campaign policy, and publication requires a resident-GPU audit.
Active, starting and draining workers all consume that budget. Retired results
are removed; pending replacements do not enter scores or ranks.

Within each trace, higher goodput per average simulated GPU wins. Each topology
averages ranks over cases completed by every compared planner, weighting traces
equally. This covers 30 cases per topology when all runs finish. Exact ties
receive average ranks and fractional win credit.

Latency cells combine good rate, mean TTFT/SLO and mean ITL/SLO. Red means good
rate below 20% or either latency ratio at least 5. Otherwise yellow means good
rate at most 50% or either ratio above 1. Missing values never become zero.
SLO rankings assign every red result rank 6 and rank the remaining planners by
the main goodput scores. `experiment-setup.json` supplies verified worker and
simulated GPU allocations.

`diagnostic-evidence.json` contains reduced queue/fleet histories where available.
`red-diagnoses.json` contains explanations tied to the current result hashes and
metrics. Explanations without a timeline are marked as inferences. The extractor
code is retained for new runs:

```bash
python extract_diagnostics.py --refresh --results-root ../../runs/60traces
```

To analyze new results, import the completed campaign with `import_results.py`.
Refresh diagnostic evidence, review/update the explanations, then rebuild.
New result hashes invalidate stale explanations. The saved evidence already
contains everything needed to rebuild the current pages; large raw reports
are unnecessary for that operation.

`export_completed.py <campaign-directory> <export-directory>` reduces successful
CPU results; add `--knowline` for KnowLine campaigns. It audits the preserved CPU
session telemetry or verifies an existing archive receipt. It requires a resident-GPU
audit and preserves the full-trace metrics and hashes.

`publish_collection.py <collection-directory>` verifies every file against the
CPU and KnowLine transfer manifests, validates all 360 results, reconstructs
portable inputs, refreshes diagnoses, and writes the pages. Its defaults
identify the source snapshot of the first `efficiency-plus3-20261106` campaign;
the saved results are later replays of it, published with `publish_latest.py`.
The bundled evidence is optional for HTML regeneration: `inputs/` is
self-contained.

`publish_latest.py <export-directory>` imports completed, GPU-audited results
incrementally and rebuilds the pages. A case/controller key has only its current
result. The publisher validates full request counts, trace hashes, SLOs, windows
and budget evidence. It refreshes analysis against the selected result hashes;
summary-only explanations are explicitly distinguished from timeline diagnoses.
To refresh queue and worker evidence on the machine holding the raw results, run
`extract_diagnostics.py --manifest <selected-results.json> --results-root <campaign> --output <diagnostics>`
(use `--knowline-campaign` for KnowLine). The manifest supplies each selected
result's job ID, case ID, planner, source and trace hashes, duration, metrics and
SLOs. Extraction verifies the raw result hash and never reruns a simulation.
Pass the exported directory to `publish_latest.py --diagnostics <diagnostics>`;
only evidence matching the selected result can replace its explanation.

The current exporter and campaign scheduler call `scripts/audit_gpu_usage.py`.
It checks resident GPUs from native telemetry, including active, starting and
draining worker IDs, and publication requires its audit receipt.

Run `python -m pytest -q test_leaderboards.py` for evidence and ranking checks.
`CHROME_ENDPOINT=http://127.0.0.1:9222 node check_browser.mjs` checks all tabs,
filters, dialogs, rank calculations, latency colors, the per-trace tables and
charts with their zoom and move controls (wheel, drag, touch, keys and buttons),
and mobile layouts.

`CHROME_ENDPOINT=http://127.0.0.1:9222 node capture_figures.mjs` captures the
figures of the Planner Gym README from the saved pages, into
`../../docs/images/leaderboard/`. Each figure is a region of one page with
numbered marks that the README explains. Run it after publishing results.
