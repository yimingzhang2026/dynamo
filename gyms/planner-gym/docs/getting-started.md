<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Get started with the 60-case suite

Work in `gyms/planner-gym/`. See [data setup](../data/README.md) for download and
preparation. A lightweight environment can validate inputs and configs:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e '.[test]' -r data/60traces/reproduction/requirements-convert.txt
.venv/bin/python scripts/run_suite_config.py configs/60traces/suite.yaml --dry-run
```

Native simulations run on CPUs but require Dynamo's native replay bindings and
AISimulate. The reproduction package pins Linux, Python 3.11/3.12 and AISimulate
`0.13.0.dev202610060000000067`. Refer to its `software.json` and `README.md` for
the reference runtime. Config validation alone does not prove native compatibility.

With a compatible runtime installed, start with one small case:

```bash
python scripts/run_match_config.py configs/60traces/A-LC-S08.yaml
```

To execute and resume the complete campaign:

```bash
python scripts/run_suite_config.py configs/60traces/suite.yaml --output-dir runs/static-60
python scripts/run_suite_config.py configs/60traces/suite.yaml --output-dir runs/static-60 --resume
```

The largest reference replay exceeded 85 GiB RSS and 48 minutes. Check each
case's memory and timeout limits before selecting the full campaign. Resume
requires the same configs, inputs, runtime and output location.

Read [the experiment and trace inventory](traces-60.md) before comparing results.
