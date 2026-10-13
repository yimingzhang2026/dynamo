<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Local benchmark inputs

The 60-case collection lives under `60traces/reproduction/`:

- `data/<source>/test-1x.jsonl`: 11 unscaled test inputs (3,843,469,145 bytes total).
- `generated/<case>.jsonl`: 60 expanded inputs (45,372,641,752 bytes total).
- `cases.json`: per-case scaling, seed, format and expected hashes.
- `source-assets.json` and `base-recipes.json`: pinned sources and conversions.
- `reference/`: historical static calibration results, not measurements from this checkout.

Run the versioned suite in `../configs/60traces/suite.yaml` from the Gym root.
For all source windows, planned history, case sizes and experiment assignments,
see [the inventory](../docs/traces-60.md).

Separate earlier history lives at `60traces/training/<source>/history-1x.jsonl`
with a checksum/provenance sidecar for each file. These are five unscaled
sources for training, separate from the 60 expanded test cases. See
[training history](../docs/training-history.md) for splits and usage.

## Reconstruct inputs

Place the supplied reproduction package at `data/60traces/reproduction/` and
install its `requirements-convert.txt` in the Gym environment. From the Gym root:

```bash
python data/60traces/reproduction/verify.py
python data/60traces/reproduction/fetch_and_convert.py --all
python scripts/prepare_60traces.py --all --jobs 2
python scripts/run_suite_config.py configs/60traces/suite.yaml --dry-run
```

The preparation driver adjusts RSS units on macOS while preserving the original
8 GiB guard and checks the exact expected SHA-256, byte count and request count.
Sources, expanded inputs, package references and runtime outputs stay local and
are Git-ignored. The suite configs and inventory are versioned with the code.
