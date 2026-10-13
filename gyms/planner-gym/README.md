<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Planner Gym — 60-case planner experiments

This branch uses 60 static baseline cases for DeepSeek-V4-Pro on B300 SXM,
with vLLM 0.24.0 performance models. Each case specifies its own input, worker
allocation, engine parallelism and latency targets.

Start with the [trace inventory and experiment guide](docs/traces-60.md).
The [interactive inventory](docs/traces-60.html) can filter by experiment or source;
[CSV](docs/traces-60.csv) contains exact sizes and all case settings.

## Prepare and validate

From this directory:

```bash
source .venv/bin/activate
python scripts/run_suite_config.py configs/60traces/suite.yaml --dry-run
python scripts/run_match_config.py configs/60traces/A-LC-S08.yaml --validate-only --print-matrix
```

Inputs are stored in `data/60traces/reproduction/generated/`. The reproduction
package supplies pinned download/conversion recipes; see [data setup](data/README.md).
The local environment supports preparation and configuration validation.
Native replay also requires compatible Dynamo bindings and AISimulate; see
[getting started](docs/getting-started.md).

## Work on planners

- `src/autoscaling_arena/adapters/cloudai_mpc_v3.py`: MPC decisions.
- `src/autoscaling_arena/adapters/cloudai_rl_lstm.py`: aggregate RL decisions.
- `src/autoscaling_arena/adapters/cloudai_rl_disagg.py`: disaggregate RL decisions.
- `configs/60traces/`: the 60 static baseline configs and suite.
- `src/autoscaling_arena/match_runner.py`: match execution and reporting.
- `src/autoscaling_arena/suite_runner.py`: independent cases and resumable suites.

Copy a baseline config to a separate experiment directory before enabling a
planner. Adjust relative input/output paths, calibrate the planner for the
case's model and hardware, and train a compatible RL checkpoint when needed.
All baseline cases currently use zero warmup. Planned predictor-history windows
are documented in the inventory; training inputs have not been materialized.

Generic trace parsing, generation primitives, autoscalers and runner tests remain
available. Benchmark cases use explicit trace paths; the optional user workload
registry starts empty.

## Tests

```bash
python -m pytest tests -q
```

See [usage](docs/usage.md), [CloudAI adapters](docs/cloudai_autoscalers.md),
[MPC design](docs/cloudai_mpc_v3.md), and [replay integration](docs/replay-integration.md).
