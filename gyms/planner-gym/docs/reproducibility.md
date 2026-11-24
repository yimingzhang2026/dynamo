<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Reproducing the saved benchmark

The saved results are 360 completed outcomes: 60 each for cloudAI-MPC, Dynamo,
KEDA, Reactive, Static and Cognitive Agent, on the plan of the
`efficiency-plus3-20261106` campaign. Its common maximum is the original Static
GPU allocation plus three times the largest worker GPU cost. Starting fleets
remain original; active, starting and draining workers all consume that
maximum. The fixed Static fleet does not grow.

All 360 outcomes were replayed again on 2026-11-23 and 2026-11-24 with the
source in this branch, and the saved results are these replays. Each replay
kept its telemetry, which the report's Traces tab reduces to curves; the raw
telemetry stays outside Git. Traces, case configurations, resource policy and
the native runtime build are those of the campaign. The cloudAI-MPC controller
is the one corrected on 2026-11-19 ([cloudAI-MPC](mpc.md)), whose 60 outcomes
were first replayed that day.

The replays also show how closely the earlier saved results reproduce. A run's
telemetry fingerprint is the SHA-256 of all its samples.

| Planner | Replays with the same telemetry | Change in goodput per GPU |
| --- | ---: | --- |
| cloudAI-MPC | 60 of 60 | none |
| KEDA | 60 of 60 | none |
| Reactive | 60 of 60 | none |
| Cognitive Agent | 60 of 60 | none |
| Static | no earlier fingerprint | none: all 60 scores are equal |
| Dynamo | 30 of 60 | the other 30: median 0.19%, at most 2.94% (`D-HT-L02`) |

Dynamo's differences change the ranking on one trace (`A-HT-S04`: Dynamo and
Static swap fifth and sixth place) and the winner on none. Their cause has not
been isolated.

## Rebuild and verify the current results

From `gyms/planner-gym`, with Python 3.11 or 3.12:

```bash
python leaderboards/60traces/build_leaderboards.py --check
```

This validates the committed compact evidence, rebuilds all six HTML pages and
the JSON/CSV analysis in a temporary directory, and compares every generated
byte to the saved files. It uses the saved generation timestamp for this
comparison. It needs no installed replay package, raw traces, GPU, model or
remote connection, and changes no saved files. Omit `--check` to regenerate
them with a new generation timestamp.

The versioned `leaderboards/60traces/inputs/` holds scores, trace fingerprints,
request counts, configurations, resident-GPU audit evidence and diagnostics.
Every topology includes all 30 shared cases. Exact rankings are reproducible
from these inputs independently of the ignored local transfer bundles.

## Repeat the experiment

Use the source in this branch, including `vendor/aisimulate-core` and its Python
binding patches. The branch is based on Dynamo
`45e30e07a4f4cef37ab7bfa739f7bee0c89b3057`; its checked-in Cargo locks and
`rust-toolchain.toml` pin the native dependency graph and toolchain. The native
runtime includes incremental queue statistics and completed-request feedback.
The launcher records source and binary hashes and rebuilds if either changes.
The model, controller parameters, seeds, observation/decision intervals, SLOs
and resource policy are defined in the current configs and controller code.

Follow [input preparation](../data/README.md) to reconstruct the 60 traces.
The content-addressed converters/scalers and `reproduction/cases.json` specify
their expected hashes, sizes and request counts; preparation verifies these
before use. The source-version fields in these frozen recipes describe input
construction, not the runtime for this campaign. Raw inputs stay outside Git.

Follow [campaign execution](comparison-60.md) for CPU replay and the two-node
GPU campaign. Both use `scripts/run_campaign.py`; server replacement, memory
isolation and continuation are integrated into its scheduler. Completed cases
are preserved. KnowLine uses `PelaAI/KnowLine-31B-Gen1` revision
`8f052a8130e599b53e5324c34f0f5ff90696e1c6` and the documented serving stack.

Rebuilding the saved leaderboard is exact. Fresh KnowLine inference is a new
measurement: the server runs at temperature 1, and GPU kernels and measured wall
latency can vary, so pinning the model and runtime does not promise identical
decisions or scores. The fresh campaign of 2026-11-23, on the same cluster, did
reproduce all 60 earlier runs sample for sample over 332,445 decisions; only
the measured wall latencies differed. Saved decision prefixes make continuation
strict within a campaign; the compact leaderboard evidence does not contain
every model answer.

Local source/unit checks and exact report reconstruction do not substitute for
executing the Linux replay and GPU serving pipeline on a fresh environment.
