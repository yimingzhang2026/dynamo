<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Planner Gym Usage Guide

Run these commands from `dynamo/gyms/planner-gym` after following the
[getting started guide](getting-started.md). For scoring definitions and
limitations, see the [Planner Gym README](../README.md).

See [the 60-case inventory](traces-60.md) for this branch's inputs and experiment groups.

## Reproducible suites and bounded execution

A suite references independent Match Configs, so each case can select its own
topology, engines, policy roster, workload and SLO. Paths resolve from the suite
file; case IDs must be unique and safe as directory names.

```yaml
schema_version: 1
name: controlled-comparison
cases:
  - id: combined
    config: A-LC-S08.yaml
    labels: {layout: aggregated}
  - id: split
    config: D-LC-S08.yaml
    labels: {layout: disaggregated}
```

```bash
python scripts/run_suite_config.py configs/60traces/suite.yaml --dry-run
python scripts/run_suite_config.py configs/60traces/suite.yaml --output-dir runs/controlled-suite
python scripts/run_suite_config.py configs/60traces/suite.yaml --output-dir runs/controlled-suite --resume
```

The baseline uses prepared traces and AIS engine timing; no model weights
are loaded. Select cases with repeatable `--case-id` arguments. Within each
case, optional `run_ids` select exact cells printed by `run_match_config.py
--print-matrix`. Suites run cases sequentially and delegate cell parallelism to
the Match Config, avoiding nested worker pools. External job arrays can shard
case IDs into separate output directories.

The suite writes `suite.json`, `comparison.csv`, and `index.html`, linking each
case's normalized JSON/HTML report. Referenced configs' publication destinations
are not invoked. A session lock rejects simultaneous writers.

Add execution controls to a Match Config:

```yaml
execution:
  max_parallel_runs: 2
  timeout_s: 900
  max_memory_mb: 4096
  fail_fast: false
```

These controls activate spawned workers. The timeout covers the guarded worker
lifetime; the memory limit uses sampled RSS across its process tree, in MiB.
It is not a hard OS memory reservation and can overshoot between samples.
Choose concurrency and per-worker limits together. A timeout, memory violation,
or worker crash produces a failed cell with a log. `fail_fast` stops launching
new cells while allowing already-running cells to finish. Interruption terminates
in-flight worker groups. Process supervision and session locking require POSIX.

Each result records setup/replay wall time, total guarded wall time and sampled
peak RSS for isolated workers. Replay wall time includes Planner preparation
inside the replay call. Default single-process execution retains wall timing
but has no isolated per-cell RSS measurement.

Single-match sessions also support explicit locations and resume:

```bash
python scripts/run_match_config.py configs/60traces/A-LC-S08.yaml --session-dir runs/one-session
python scripts/run_match_config.py configs/60traces/A-LC-S08.yaml --resume runs/one-session --overwrite
```

Successful cells have atomic checkpoints. Resume verifies replay settings,
input fingerprints, software identity and saved evidence; failed or incomplete
cells run again, with prior attempts archived. Timeout, memory and parallelism
budgets may change on retry. Changing the replay semantics or inputs requires
a new session. Resume uses the original session directory; copied or moved
sessions are rejected to avoid stale artifact references. Keep input files and
software immutable while execution is active.

## Scoring and request capture

Selected TTFT and ITL SLO bounds also become the Planner's control targets.
Omit duplicate targets from `planner_config`; explicit conflicting values are
rejected. E2E-only constraints remain scoring constraints because Planner's
control interface uses TTFT and ITL. Fixed/polynomial timing overrides are
available through `extra_args.timing_model`; external AIS identity remains
owned by the first-class model/engine settings. Planner with custom timing
requires `ais_bootstrap: false`.

```yaml
backend:
  replay:
    request_capture: jsonl
metrics:
  short_output_itl: skip
evaluations:
  defaults:
    measurement_window: {start_s: 0, end_s: 60}
```

Capture modes are `memory` (default), `jsonl`, and `summary`. JSONL mode keeps
terminal rows in a separate file and scores them in one streaming pass. It
avoids the Python request list; native replay still buffers rows before writing.
Summary mode omits request capture and uses native aggregates. With an ITL
constraint it requires `short_output_itl: skip`; explicit windows require
memory or JSONL. Truncated request capture fails scoring.

`fail` preserves the default AIPerf-style treatment of undefined ITL for a
one-token output; `skip` matches native replay for that case. Failed requests
still fail, and zero-token outputs cannot satisfy token-latency constraints.

The optional measurement window selects arrivals in `[start_s, end_s)` on the
replay clock after trace preparation and arrival speedup. Their eventual
outcomes include drain completions. Goodput and provisioned GPU-hours use the
same interval; starting and draining workers count until removed. Static fleets
may extend through a final idle tail. Dynamic windows cannot extend past replay
termination, because subsequent controller actions have not been simulated.
Original replay totals and full-replay latency context remain in the report.

## Preserve trace time while changing request count

An external Mooncake trace can declare `prepare` independently of arrival
speedup. Preparation is available for offline simulation and uses bounded-memory
disk sorting. Default settings preserve the input bytes exactly.

```yaml
evaluations:
  traces:
    - name: recorded-input
      path: input.jsonl
      format: mooncake
      block_size: 16
      presorted: true
      prepare:
        start_ms: 60000
        end_ms: 120000
        rebase: true
        request_scale: 2.5
        seed: 7
        jitter_ms: 1000
      warmup:
        path: input.jsonl
        prepare:
          start_ms: 0
          end_ms: 60000
          rebase: true
          request_scale: 2.5
          seed: 7
          jitter_ms: 1000
```

Bounds refer to original timestamps and use a half-open interval. Rebasing
subtracts the selected start. Scaling creates whole copies plus deterministic
fractional thinning, preserving input/output lengths and the time span. Added
copies receive independent hash namespaces. Optional copy-phase jitter is
bounded and tapers at interval edges; `session_id_field` shares that phase
within a session, without changing fractional selection to session sampling.

Preparation records source/output SHA-256, resolved window, counts and controls
in a sidecar manifest. It never overwrites the source or an existing output.
Namespaces are local to one prepared output: prepare a combined interval before
splitting if cross-window KV reuse is required. The `warmup` field feeds only
the builtin Planner's load predictor; it does not warm engine KV caches and is
not applied to rival policies. Warmup paths resolve from the Match Config.

## Bring your own recorded trace

Recorded traffic stays outside this repository. Match Configs accept one or
more user-supplied replay JSONL files and resolve relative paths from the
config file:

```yaml
evaluations:
  traces:
    - name: my-recorded-traffic
      path: ../../../private/traces/my-traffic.jsonl
      block_size: 512
      presorted: true
  defaults:
    max_requests: 1000
    arrival_speedup: 1.0
```

Each JSONL row must contain `timestamp`, `input_length`, `output_length`, and
`hash_ids`. Set `block_size` to the token span represented by each hash. Use
`presorted: true` for timestamp-sorted large traces so capped runs can stream
the first records without loading the whole file. The materialized trace is
fingerprinted in result provenance. External traces and transformed copies are
staged under a private system-temporary directory and deleted after execution;
their source paths are redacted from reports.

Simulation Match Configs also accept exact native Dynamo request-trace shards:

```yaml
evaluations:
  traces:
    - name: recorded-requests
      format: dynamo
      paths:
        - ../../../private/traces/requests-00.jsonl.gz
        - ../../../private/traces/requests-01.jsonl.gz
      # block_size: 16  # optional; omission uses embedded trace metadata
  defaults:
    max_requests: null
    arrival_speedup: 1.0
```

The ordered shard list is passed directly to Dynamo without sorting, merging,
conversion, capping, or rewriting. Exact native traces reject `max_requests`
and non-1 `arrival_speedup`; they are simulation-only because the real backend
drives AIPerf with Mooncake JSONL.

For a single simulation, pass the file directly:

```bash
python scripts/run_match.py \
  --trace /absolute/path/to/my-traffic.jsonl \
  --trace-block-size 512
```

## User guide: run a Match Config

A Match Config is a versioned YAML description of a complete comparison
matrix. It selects either the `sim` or `real` backend, the autoscalers or
endpoint names, model and engine configuration, recorded and synthetic
workloads, SLO targets, metrics, execution safeguards, and publication
destinations. Start from
[`configs/60traces/A-LC-S08.yaml`](../configs/60traces/A-LC-S08.yaml) or
a user-supplied live Match Config.
Use the [single-case quickstart](getting-started.md) for a small first run.

Simulation model and engine settings are first-class and control the replay:

```yaml
backend:
  type: sim
  topology: disagg
  gpu_budget: 32

  model:
    name: openai/gpt-oss-120b
    # Optional; defaults to name.
    ais_model_path: openai/gpt-oss-120b

  engines:
    common:
      system: h200_sxm
      backend: vllm
      backend_version: "0.19.0"
      tp_size: 1
      moe_tp_size: 1
      moe_ep_size: 1
      attention_dp_size: 1
      runtime:
        cold_start_delay_s: 60
      extra_args: {}
    # Optional partial overrides inherited from common.
    prefill:
      runtime:
        kv_transfer_bandwidth_gbps: 100
        kv_bytes_per_token: 131072
    decode: {}

  replay:
    # Simulated-time cadence; defaults to 5 seconds. null disables capture.
    telemetry_sample_interval_s: 5
```

Telemetry sampling has its own simulated-time schedule and does not need to be
a multiple of a Planner or rival autoscaler's tick duration. A positive value
can therefore sample every few seconds or minutes without changing controller
behavior. The default is 5 seconds so Match Config reports have useful charts;
set `telemetry_sample_interval_s: null` when the extra persisted observations
are not wanted. Every enabled simulation cell writes
`runs/<run-id>/telemetry.jsonl` beneath that session's configured artifact
root. The file contains one complete Dynamo telemetry snapshot per line and is
the only replay-telemetry ingestion boundary used by Arena reporting.

For an aggregated topology, use `engines.aggregate` instead of
`engines.prefill`/`engines.decode`. Engine GPU cost is derived as
`tp_size * attention_dp_size`; MoE TP and EP, when supplied, must describe the
same world size. GPU-budget validation weights prefill and decode replica
counts by their independently resolved engine costs. An engine's `backend`
selects DynoSim's scheduler semantics. AISimulate performance lookup defaults to that
backend/version but can be decoupled explicitly with `ais_backend` and
`ais_backend_version`; identity and accounting fields cannot be overridden
through `extra_args`. `cold_start_delay_s` models worker startup and defaults to
60 seconds for each engine role, including substrate presets. Set it to `0`
explicitly to model immediate startup, or set a measured delay. KV handoff
delay requires `kv_transfer_bandwidth_gbps` (GB/s) and `kv_bytes_per_token` together;
for a disaggregated deployment they normally belong on the prefill engine.
Replace the example bytes/token value with the model and KV-cache dtype being
simulated. Other low-level engine runtime behavior remains available through
`extra_args`; the three typed runtime fields cannot be duplicated there.

The compact legacy-compatible form `substrate: gpt_oss` remains available and
resolves to the same complete model and role-engine snapshot. A config must
choose either the preset shorthand or explicit `model` plus `engines`, so
there is never a hidden override order.

Validate a config and inspect its fully expanded matrix before starting work:

```bash
python scripts/run_match_config.py configs/60traces/A-LC-S08.yaml \
  --validate-only \
  --print-matrix
```

Run the simulation example from the Dynamo environment built above:

```bash
python scripts/run_match_config.py \
  configs/60traces/A-LC-S08.yaml
```

For a real match, first replace the example endpoint catalog entries with
already-running deployments, then run:

```bash
python scripts/run_match_config.py path/to/live-match.yaml
```

The nested real deployment metadata is easiest to read in
[`configs/endpoints.example.yaml`](../configs/endpoints.example.yaml); JSON
and YAML endpoint catalogs are accepted by both Match Configs and the endpoint CLI.

All relative paths—including `backend.planner_config`, `endpoint_catalog`,
`publish.artifact_root`, and JSON/HTML destination paths—resolve from the Match
Config's directory. Real autoscaler entries refer to endpoint catalog entries
by `name`; the runner
benchmarks those deployments but does not create or reconfigure them. Pass
`--overwrite` explicitly to replace an existing JSON or HTML destination.
`--autoscaler NAME` (repeatable) runs only the named roster entries, and
`--merge` folds those runs into the configured results JSON and HTML report
in place, replacing earlier runs of the same entries and keeping the rest,
so a new planner can join a published leaderboard without rerunning it.
Optional `autoscaler_type` and `declared_config` fields on a real entry are
recorded as controller provenance only. Each real endpoint's `model` is the
operational value passed to AIPerf. Its optional `declared_deployment` block
records unverified model, topology, and engine provenance; Arena does not
apply, discover, or verify that deployment configuration.

| Setting | Authority |
| --- | --- |
| Sim `backend.model` and `backend.engines` | Controls DynoSim |
| Sim `backend.planner_config` | Planner policy from a reusable file or inline mapping |
| Real endpoint `model` | Passed to AIPerf `--model` |
| Real endpoint `declared_deployment` | Unverified deployment provenance only |
| Real autoscaler `declared_config` | Unverified controller-policy provenance only |

Simulation configs can also pin `gpu_budget`, `router.mode`, Planner-specific
settings as an inline `planner_config` mapping or a reusable YAML/JSON path,
and replay controls (`ais_bootstrap` and `concurrency`). See
[`configs/planner.sim.example.yaml`](../configs/planner.sim.example.yaml) for a
reusable Planner policy that deliberately omits runner-owned topology, budget,
engine-cost, and report fields. Preset-based legacy configs may still use
`replay.model_name`; explicit configs must use `model.name`. Real configs can
set AIPerf's `executable`, `tokenizer`, `streaming`, `timeout_s`, and
`extra_args`. Optional top-level
`description` and string-valued `labels` travel with the resolved config and
run provenance in the result JSON.

`evaluations.workloads` selects registry entries directly,
`evaluations.traces` declares external recorded files, and
`evaluations.suites` adds `synthetic`, `recorded`, or `all`. Suite names are
also accepted directly in the workload list.
`evaluations.exclude` is applied after both are combined. Defaults such as
`seed`, `max_requests`, and `arrival_speedup` apply to every selected workload;
`evaluations.overrides` changes them for one workload.

An SLO threshold can be a scalar or a list. Lists within one profile expand as
a Cartesian product: `ttft_ms: [300, 500]` with `itl_ms: [50, 100]` produces
four targets. The complete run count is:

```text
autoscalers x workloads x expanded SLO targets x repetitions
```

`execution.max_runs` rejects an unexpectedly large matrix before execution,
and `fail_fast` controls whether a failed cell stops the remaining sweep.
Repetition zero uses the configured evaluation seed; later repetitions use
`seed + repetition`.
Static, reactive, and KEDA entries can set their polling and policy controls
directly. Reactive and KEDA configs include explicit `min_*`/`max_*` replica
bounds; KEDA also exposes thresholds, tolerance, and stabilization windows.
Omitted adapter settings are expanded to their effective defaults in the
resolved config, and maximum fleets are checked against `gpu_budget`.
`metrics.rank_by` chooses the leaderboard order; `metrics.include` selects the
reported context columns. Publication supports console, JSON, and standalone
HTML destinations, with raw diagnostics under `publish.artifact_root`:

```yaml
publish:
  artifact_root: ../runs/my-match/artifacts
  destinations:
    - type: console
    - type: json
      path: ../runs/my-match/results.json
    - type: html
      path: ../runs/my-match/report.html
```

The HTML report is one self-contained file per Match Config session. Its
leaderboard and eight aligned time lanes update together when you select an
expanded SLO/repetition configuration or workload. A shared checkbox legend
shows or hides each autoscaler across queue depth, TTFT, TPOT, KV utilization,
KV hit/cache reuse, active replicas, and provisioned GPUs. Arriving requests
come from the common workload trace or the replay's telemetry windows and are
shown once because every autoscaler receives the same open-loop traffic. TTFT
and TPOT are raw means from replay-owned windows, independent of policy-specific
observation handling; hover exposes the exact window bounds and saved
latency-sample count. The report timeline is projected from each run's persisted
`telemetry.jsonl`, and each saved result row retains the full traffic aggregate
and raw rank-level scheduler rows so additional charts can be derived without
rerunning the trace. For vLLM, the KV lane
distinguishes active-block pressure from physical residency, which also
includes inactive reusable blocks.
SGLang's legacy occupancy already includes all occupied radix pages, so its
active and physical lines are equal until the backend exposes that split. The
cache lane deliberately shows two different signals: router KV hit rate is a
request-weighted routing-overlap estimate, while scheduler cache reuse is the
backend-native ratio of interval hit tokens to observed cache tokens. They are
not substituted for each other. Older runs can fall back to a saved
completion-attributed timeline or a labeled whole-run aggregate. Capacity lines
show fleet observations at telemetry sample times, while hollow markers use
Planner details to show exact capacity requests at their controller decision
times. Provisioned GPUs include starting and draining replicas and weight
prefill/decode pools by their configured GPUs per worker.

Queue depth is the point-in-time sum of scheduler-waiting requests across all
live ranks plus router-pending requests. The total, scheduler, and router layers
are retained separately, as are the raw rank-level scheduler rows used to
derive them. Scheduled/in-flight work is excluded from queue depth. Legacy
artifacts retain their scheduler-only queue values and explicitly mark router
pending and the new KV time series unavailable instead of inferring zeros.

Open-loop simulation reports show the actual replay arrival schedule. A
closed-loop simulation ignores trace timestamps, so the report marks arrivals
unavailable rather than presenting the trace schedule as observed load. Online
matches retain the leaderboard, while replica/GPU lanes remain unavailable
until deployment telemetry is supplied.

Validation is strict: unknown fields, duplicate YAML keys or names, unsafe
path-bearing identifiers, unsupported backend metrics, missing endpoints, and
invalid replica/SLA ranges are rejected before a run begins.

Simulation runs enable per-request capture. Dynamo returns terminal request
records in `ReplayReport.per_request`, exposed through
`ArenaReplayResult.per_request` for SLO scoring. A sim Match Config still
performs one deterministic replay per expanded SLO target so the configured
runtime SLA, result identity, and artifact set remain aligned. The real backend
likewise runs one AIPerf process per endpoint, workload, and SLO target.

Saved results can be rendered again without replaying traffic:

```bash
python scripts/render_saved_report.py \
  /path/to/results.json \
  --out /path/to/report.html \
  --normalized-out /path/to/normalized-report.json
```

The renderer accepts the normalized `results.json` produced by a Match Config
run. It uses the saved timeline and metrics without reopening source shards,
rereading telemetry JSONL, or invoking Dynamo. The raw JSONL remains in the session artifact tree for audit
and future projections.

## User guide: run one simulated match

`run_match.py` is the quickest simulation smoke test. Its CLI currently exposes
the `static` and `reactive` adapters.

```bash
export TRACE_DIR=../../traces

python scripts/run_match.py \
  --autoscaler static \
  --trace "$TRACE_DIR/mooncake/1000.jsonl" \
  --num-prefill 4 \
  --num-decode 1 \
  --ttft-ms 2000 \
  --itl-ms 50 \
  --report-json runs/static-4p1d-trace-report.json
```

The terminal summary reports mean/p99 TTFT, mean ITL, request throughput,
simulation ticks, and scaling-event count. Compare against the reactive policy
by changing `--autoscaler static` to `--autoscaler reactive`.

Useful options:

- `--trace`: any replay JSONL trace with the fields described below.
- `--substrate`: select a bundled model/engine/hardware preset.
- `--num-prefill`, `--num-decode`: initial worker counts.
- `--ttft-ms`, `--itl-ms`: populate the substrate `PlannerConfig` and
  diagnostics. The current static/reactive adapters ignore them, so these flags
  do not change their decisions or redefine scorecard profiles.
- `--report-json`: write the AIPerf-style trace report.

## Understand the outputs

| Path | Producer | Contents |
| --- | --- | --- |
| `runs/traces/*.jsonl` | Trace generator and leaderboard scripts | Materialized synthetic traces |
| `runs/leaderboard.json` | Offline leaderboard default | One full scorecard per policy/workload pair |
| `planner_reports/arena_<policy>.html` and `.log.jsonl.gz` | Single simulated match | Dynamo diagnostics, relative to the shell working directory |
| `planner_reports/_leaderboard_run.html` and `.log.jsonl.gz` | Offline sweep | Dynamo diagnostics; the fixed filename is overwritten across matches |
| `runs/online/<endpoint>/<workload>_<profile>/` | Online endpoint runner | Raw nested AIPerf artifacts |
| `runs/online_leaderboard.json` | Online endpoint runner default | Parsed metrics, return codes, artifact paths, and stderr tails |
| Configured `publish.artifact_root/<session>/` | Match Config runner | Generated synthetic traces, per-run diagnostics, raw `runs/<run-id>/telemetry.jsonl` streams, and AIPerf artifacts; external traces use ephemeral private staging |
| Configured JSON destination | Match Config runner | Resolved config, expanded matrix, provenance, normalized metrics, raw metrics, and failures |
| Configured HTML destination | Match Config runner | Standalone interactive leaderboard and aligned workload/latency/capacity timelines |

## Extend the Arena

For the experimental TypeSafe Jev decision engine and its five-policy synthetic
pilot, see [the Jev guide](jev.md). It uses the same Planner engine interface
and reports hosted-controller overhead separately from simulation metrics.

The CloudAI MPC and offline-RL planners (`cloudai-mpc`, `cloudai-rl`) and the
configuration requirements are described in
[the CloudAI autoscalers guide](cloudai_autoscalers.md). Use the static 60-case
suite as the reference before enabling these planners.

### Add an offline autoscaler

An adapter implements three methods:

1. `initial_tick(start_s)` schedules its first observation.
2. `async tick(scheduled_tick, tick_input)` returns `PlannerEffects`, including
   the desired replica counts and next tick.
3. `async shutdown()` releases policy resources.

Register its factory in an `AutoscalerSpec`:

```python
from autoscaling_arena.leaderboard import AutoscalerSpec

spec = AutoscalerSpec(
    name="my-policy",
    factory=lambda planner_config, capabilities: MyAutoscaler(capabilities),
    start_prefill=1,
    start_decode=1,
)
```

The factory receives the shared `PlannerConfig` and derived
`WorkerCapabilities`. Use `run_leaderboard(autoscalers=[spec], ...)` from a
Python driver, or add the policy to `default_autoscalers()`. Do not encode model
or hardware behavior in the adapter; that belongs to the shared substrate.

### Score a custom SLO

```python
from autoscaling_arena.scorecard import SLOProfile, scorecard

chat = SLOProfile(name="chat", ttft_ms=500, itl_ms=75)
result = scorecard(report, profiles=(chat,))
```

For live runs, pass custom `SLOProfile` objects to the Python
`run_endpoint_leaderboard()` API. The supplied endpoint CLI only selects from
the three built-in profiles.

### Main Python modules

| Module | Purpose |
| --- | --- |
| `autoscaling_arena.workloads` | Workload registry, axes, and trace materialization |
| `autoscaling_arena.datasets` | Generic Dynamo request-trace validation and shard discovery |
| `autoscaling_arena.adapters` | Static, reactive, and KEDA/HPA policy adapters |
| `autoscaling_arena.runners.sims` | One DynoSim match via `run_arena_replay()` |
| `autoscaling_arena.runners.real` | Endpoint loading, AIPerf execution, and result parsing |
| `autoscaling_arena.scorecard` | SLO goodput, efficiency, and stability metrics |
| `autoscaling_arena.leaderboard` | Offline sweep orchestration and table formatting |
| `autoscaling_arena.match_config` | Strict YAML schema, validation, and matrix expansion |
| `autoscaling_arena.match_runner` | Config-driven sim/real execution and publication |
| `autoscaling_arena.html_report` | Standalone interactive leaderboard and time-series report rendering |

## Development and tests

Install the test extra:

```bash
python -m pip install -e '.[test]'
```

The scorecard tests are standalone:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q tests/test_scorecard.py
```

Run the pure-Python suite without writing caches:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q
```

## Troubleshooting

### `ModuleNotFoundError: dynamo` or `dynamo._core`

The offline scripts are running under the wrong Python environment or the
Dynamo bindings have not been built. Activate that Dynamo virtual environment
and verify the import command in
[simulation setup](getting-started.md). Installing
`autoscaling-arena` alone does not install Dynamo.

### `unexpected keyword argument 'telemetry_jsonl_path'`

The active Dynamo binding predates the replay telemetry bridge, or Python is
loading a binding built from a different checkout. Rebuild the native runtime from the same Dynamo checkout and run the capability check in
[Replay integration](replay-integration.md). Dynamo #12334 alone does not
provide the telemetry-file contract. Note that the public
`dynamo.replay.run_trace_replay()` API exposes `telemetry_options`; the direct
file parameters belong to the low-level `_core.run_mocker_trace_replay()`
binding.

### AISimulate cannot resolve the default model/system/backend

Verify that the installed AISimulate performance data supports the selected
model, system, and backend version. Use the AISimulate version pinned by the
same Dynamo checkout. For legacy simulation commands, select a bundled
combination with `--substrate`. For Match Config runs, set the first-class
`backend.model` and `backend.engines` fields.

### `aiperf` is not found

Install AIPerf in the active environment, or provide the executable explicitly:

```bash
python scripts/run_endpoint_bench.py \
  --aiperf-bin /absolute/path/to/aiperf \
  --endpoints configs/endpoints.local.yaml
```

### An endpoint row has null metrics or `[FAILED rc=N]`

Open the result JSON and inspect `returncode`, `stderr_tail`, and `artifact_dir`.
Confirm that the URL is reachable, the model name and endpoint type are
correct, and the tokenizer is available. Use a fresh `--artifact-root` while
debugging so a previous summary cannot be mistaken for the current run.

### A live sweep takes longer than expected

Online replays run serially and in real time. Estimate the lower bound as:

```text
endpoints x workloads x profiles x trace duration
```

Most synthetic traces are 180 seconds. Start with one endpoint, one workload,
and one profile.

## Current limitations

- The built-in roster is Planner, KEDA/HPA port, reactive, and static. Ray
  Serve, llm-d, and an offline oracle are not implemented.
- The online runner has no deployment, first-class authentication/header,
  parallelism, cost-telemetry, or controller-telemetry layer; Match Config
  `aiperf.extra_args` can pass supported AIPerf authentication flags.
- Match Config results carry resolved provenance in JSON and can be rendered as
  standalone HTML, but there is no persistent run database or cross-run query
  layer.
- The legacy offline leaderboard uses fixed diagnostic filenames that are
  overwritten during a sweep; Match Config runs use isolated session/run paths.
- The legacy offline and endpoint CLIs accept registry workload names; use a
  Match Config for a matrix containing user-supplied recorded traces.
