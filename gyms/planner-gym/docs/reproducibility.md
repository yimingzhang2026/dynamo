<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Reproducing the saved benchmark

The saved `efficiency-plus3-20261106` campaign contains 360 completed outcomes:
60 each for cloudAI-MPC, Dynamo, KEDA, Reactive, Static and KnowLine-31B.
Its common maximum is the original Static GPU allocation plus three times the
largest worker GPU cost. Starting fleets remain original; active, starting and
draining workers all consume that maximum. The fixed Static fleet does not grow.

## Rebuild and verify the current results

From `gyms/planner-gym`, with Python 3.11 or 3.12:

```bash
python leaderboards/60traces/build_leaderboards.py --check
```

This validates the committed compact evidence, rebuilds all five HTML pages and
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
measurement: it uses temperature 1, and model sampling, GPU kernels and measured
wall latency can vary. Pinning the model and runtime does not promise identical
decisions or scores. Saved decision prefixes make continuation strict within a
campaign; the compact leaderboard evidence does not contain every model answer.

Local source/unit checks and exact report reconstruction do not substitute for
executing the Linux replay and GPU serving pipeline on a fresh environment.
