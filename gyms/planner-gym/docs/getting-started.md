<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Get started with the 60-trace suite

Use the [main guide](../README.md) for installation, CPU/RAM/GPU requirements,
replay, KnowLine inference, and HTML generation. The saved tables rebuild with
Python alone:

```bash
python leaderboards/60traces/build_leaderboards.py
```

For new runs, reconstruct the inputs using [data setup](../data/README.md),
install the optimized native runtime, and prepare the common campaign:

```bash
python scripts/prepare_comparison.py --output runs/60traces
python scripts/run_comparison.py runs/60traces/plan.json
```

Inspection does not run simulations. Add `--rolling --execute` and select the
controllers after checking the plan's resource reservations. GPU cluster
execution is described in [the campaign guide](comparison-60.md).
