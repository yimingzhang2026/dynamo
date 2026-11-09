<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Benchmark inputs

The versioned [reproduction recipes](../reproduction/) reconstruct the exact
60 input files. Raw and expanded traces remain outside Git.

From the Gym directory:

```bash
python -m pip install -r reproduction/requirements-convert.txt
python reproduction/fetch_and_convert.py --all
python scripts/prepare_60traces.py --all --jobs 2
python scripts/run_suite_config.py configs/60traces/suite.yaml --dry-run
```

Inputs are written to `data/60traces/reproduction/generated/<case>.jsonl`.
Preparation verifies SHA-256, byte count and request count against the recipes.
The collection contains about 42.3 GiB of expanded inputs. Leave additional
space for downloaded assets, conversion intermediates and request captures.

The frozen converters and scalers are referenced by content hashes; each is
needed to reproduce specific trace bytes. The readable case inventory is in
[docs/traces-60.md](../docs/traces-60.md). Dataset identities and licenses describe
the workloads, independently of the machine used to execute them.
