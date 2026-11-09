<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Planner Gym: 60-trace benchmark

This directory contains the replay runners, controllers, KnowLine serving and
GPU campaign tools, result import, analysis, and offline HTML generation.
The saved tables contain all 360 results for cloudAI-MPC, Dynamo, KEDA, Reactive,
Static, and KnowLine-31B on 30 aggregated and 30 disaggregated traces. The shared
ceiling is **the original Static GPU allocation + 3 × the largest per-worker GPU
cost**. All planners start at that original fleet;
Static holds it fixed. Dynamic planners have a minimum of one worker per active
pool and no independent worker-count cap. Starting and draining workers count
against the shared ceiling.

## Saved results

The `efficiency-plus3-20261106` campaign has **360/360 completed results**:
six planners on 30 aggregated and 30 disaggregated traces. These tables are a
snapshot of [leaderboard-summary.csv](leaderboards/60traces/leaderboard-summary.csv),
using the `all` regime for each topology.

Planners are ranked within each trace by **goodput per average simulated GPU**
(higher is better), then ordered by **mean rank** across traces (lower is better).
Each trace has equal weight. Ties share rank and win credit, so wins can be
fractional. Mean good rate is the average per-trace fraction of attempted
requests meeting the configured SLOs; it is not the ranking metric. GPU-hours
are summed simulated serving allocations, excluding the physical GPUs used
for KnowLine decision inference. These are the main efficiency rankings;
the HTML report also includes separate SLO-adjusted rankings and latency diagnostics.

### Aggregated serving — 30 traces

Every planner completed all 30 traces; all 30 enter the comparison.

| Position | Planner | Mean rank ↓ | Wins ↑ | Mean good rate ↑ | Simulated GPU-hours |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | KnowLine-31B | 2.467 | 5.00 | 58.26% | 20,011.3 |
| 2 | cloudAI-MPC | 2.700 | 7.50 | 57.32% | 18,719.4 |
| 3 | Static | 2.733 | 10.50 | 63.93% | 27,718.9 |
| 4 | Dynamo | 3.733 | 5.00 | 81.57% | 30,488.8 |
| 5 | KEDA | 4.500 | 0.00 | 28.46% | 13,084.4 |
| 6 | Reactive | 4.867 | 2.00 | 21.26% | 10,952.0 |

### Disaggregated serving — 30 traces

Every planner completed all 30 traces; all 30 enter the comparison.

| Position | Planner | Mean rank ↓ | Wins ↑ | Mean good rate ↑ | Simulated GPU-hours |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | cloudAI-MPC | 2.483 | 12.00 | 76.74% | 18,802.0 |
| 2 | Dynamo | 3.017 | 6.50 | 62.30% | 15,490.7 |
| 3 | KnowLine-31B | 3.333 | 2.00 | 79.78% | 23,525.8 |
| 4 | Static | 3.433 | 5.83 | 65.53% | 23,897.4 |
| 5 | Reactive | 3.967 | 2.33 | 61.60% | 19,162.5 |
| 6 | KEDA | 4.767 | 1.33 | 64.09% | 21,337.6 |

Per-trace results: [aggregated CSV](leaderboards/60traces/aggregated-runs.csv)
and [disaggregated CSV](leaderboards/60traces/disaggregated-runs.csv).
The [summary CSV](leaderboards/60traces/leaderboard-summary.csv) also includes
separate HT and LC results. See [report details](leaderboards/60traces/README.md)
for scoring and validation.

### Open the interactive HTML report

The full [HTML leaderboard](leaderboards/60traces/index.html) is committed to
this branch. GitHub's repository file viewer shows HTML source; it does not
run the interactive dashboard inside the README.

To use the tabs, filters, latency matrices, and diagnostic dialogs, open
`leaderboards/60traces/index.html` from a local checkout in your browser.
Alternatively, follow the HTML link above, select **Download raw file** in
GitHub's file toolbar, and open the downloaded file. The report is self-contained
and works offline; no benchmark rerun or server is needed.

## Rebuild the saved tables

Python 3.11 or 3.12 is sufficient; replay and GPU inference are unnecessary.
The versioned `leaderboards/60traces/inputs/` contains the complete current scores,
GPU audits, configurations and reduced diagnostic evidence.

```bash
python leaderboards/60traces/build_leaderboards.py
```

To verify the saved HTML, CSV and JSON byte for byte without modifying them:

```bash
python leaderboards/60traces/build_leaderboards.py --check
```

See [reproducibility](docs/reproducibility.md) for the input/runtime pins and the
distinction between rebuilding these results and repeating model inference.

Open the generated `leaderboards/60traces/index.html` locally as described above.
The generator also writes topology tables, latency/SLO matrices, SLO-adjusted
rankings, JSON, and CSV exports.
[Report details](leaderboards/60traces/README.md) describe the scoring and inputs.
The README tables above are a saved snapshot; when publishing a new campaign,
refresh them from the `all` rows in `leaderboard-summary.csv`.

## Resources

| Work | CPU and RAM | Physical GPUs |
| --- | --- | --- |
| Rebuild saved HTML and analysis | 1 CPU, 1 GiB RAM | 0 |
| Full CPU replay campaign | Example policy: 8 CPU cores, 512 GiB RAM; up to 8 runs on distinct single-core slots, reserving 64 GiB | 0 |
| One KnowLine decision server | Budget 16 CPU cores; replay RAM remains subject to each trace's reservation | 2 × 80 GiB GPUs for bf16 weights |
| Parallel GPU campaign | Per node: 64 CPU cores, about 2 TiB RAM; 4 replay/server lanes with 384 GiB replay RSS guards | 8 × 80 GiB GPUs per node |

CPU and memory settings are configurable. The planner checks reservations and
available capacity before starting work; large jobs run alone when required.
These are execution profiles, not measured minimum requirements. The GPU
campaign defaults to two node queues with four concurrent traces per node.
The longest cases start on separate nodes; smaller cases fill each free lane,
reusing its loaded model. `--nodes` sets the node count when preparing a campaign.
Serving GPUs shown in the result tables are simulated allocations. Only the
KnowLine decision model needs physical GPUs.

## Run the benchmark

On Linux, install the native replay runtime and this package:

```bash
bash scripts/setup_simulation_env.sh
source .venv/bin/activate
python -m pip install -e '.[sim,jev,analysis,test]' numpy
```

Both Cargo workspaces use the optimized scheduler in `../../vendor/aisimulate-core`.
The runner requires `incremental-queue-stats-v1`. Replay streams request results
to JSONL, skips detailed per-tick planner charts, and retains lifecycle events
for exact GPU-hour scoring plus sampled telemetry for diagnosis. Full trace
windows, seeds, request counts, SLOs, and worker limits remain intact.

Prepare the inputs using [data/README.md](data/README.md), then:

```bash
python scripts/run_campaign.py cpu --workspace runs/benchmark
```

The CPU command prepares and executes all 300 CPU jobs on up to eight cores.
The same entry point starts all 60 KnowLine cases on two GPU nodes after model
setup, as described in [docs/comparison-60.md](docs/comparison-60.md). It checks
native source and binary identity before reusing a runtime. Server replacement,
memory isolation and allocation continuation are part of the maintained
scheduler. Completed cases are preserved; no separate recovery script is needed.

The decision contract is documented in [docs/knowline.md](docs/knowline.md).

## Collect and analyze a new campaign

```bash
python scripts/run_campaign.py collect --kind cpu --workspace runs/benchmark
python scripts/run_campaign.py collect --kind knowline --workspace runs/benchmark
python scripts/run_campaign.py publish --collection runs/benchmark/exports
```

Collect on the machine holding each campaign, then place both export directories
under one local `exports/` directory before publication. Collection requires
resident-GPU audits and complete trace measurements. The publisher accepts all
300 CPU and 60 KnowLine exports and refreshes analysis for their exact hashes.
Detailed timeline extraction remains available through `extract_diagnostics.py`.

The local `runs/efficiency-plus3-20261106/` retains the current campaign's 360
compact exports, transfer manifests and source/runtime identity receipts.
Large input traces and these local bundles are ignored by Git. Reconstruct
and verify the current saved tables through the same entry point:

```bash
python scripts/run_campaign.py publish
```

The versioned compact `inputs/` remain sufficient to rebuild HTML after a fresh
checkout. The expanded traces can be regenerated from `reproduction/` when a new
replay is required.

## Validate

```bash
python -m pytest -q tests leaderboards/60traces
```

Browser checks use `leaderboards/60traces/check_browser.mjs` with a local
Chrome DevTools endpoint supplied through `CHROME_ENDPOINT`.
