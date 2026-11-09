<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 60-trace results and analysis

Run `python build_leaderboards.py` to regenerate the five standalone pages:
`index.html`, `aggregated.html`, `disaggregated.html`, `latency.html`,
and `slo-rankings.html`. They work offline.

`inputs/results.json` contains one current record for each of 300 dynamic jobs.
Unfinished replacements contain no previous result or score. `plan.json` defines
the expected case/controller pairs.
The 60 Static records are in `static-latency.json`. During a rerun, unfinished
Static records remain pending without previous metrics. Every tab uses only
completed, validated runs for rankings, good rates and latency measurements.
Campaign preparation uses the current Static runtime and peak-memory measurements
in `static-latency.json` for execution resource estimates.
There are 360 main outcomes. Paid JEV is excluded.

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
portable inputs, refreshes diagnoses, and writes all five pages. Its defaults
identify the current `efficiency-plus3-20261106` source snapshot. The bundled
evidence is optional for HTML regeneration: `inputs/` is self-contained.

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
filters, dialogs, rank calculations, latency colors and mobile layouts.
