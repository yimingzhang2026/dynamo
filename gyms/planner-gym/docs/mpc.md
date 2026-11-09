<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# cloudAI-MPC

MPC is the Gym's model predictive controller. Select `type: mpc` and use `cloudAI-MPC`
as its display name. The implementation is `adapters/mpc.py`, exported as
`MPCAutoscaler`; controller tests live in `tests/test_mpc.py`.

## Decision model

The capacity model combines prefill time, decode step time and the KV-cache
concurrency limit. For aggregate workers, the sustainable request rate is:

```text
c_sat = min(0.9 * worker_kv_tokens / attention_dp / (isl + osl), rank_max_num_seqs)
worker_mu = attention_dp / (prefill_seconds + osl * decode_step_seconds(c_sat) / c_sat)
```

Capacity is evaluated at saturation. Offered load also enters the latency
estimate, while a bounded correction learns scheduler effects from observed
completions during saturated intervals. Arrival-rate forecasts use coarse
bins, an exponentially weighted level, change-point detection and significant
linear trends. Token lengths and cache-hit rates use smoothed observations.
No pretrained policy checkpoint is required.

A pending replica costs GPUs immediately but contributes service capacity only
after its cold-start delay. Candidate plans cover that delay plus `horizon`
control intervals. The controller maximizes an approximation of good requests
per GPU-second using a Dinkelbach objective: predicted good requests minus
`rho * GPU-seconds`, with `rho` tracking measured recent efficiency. Both TTFT
and ITL enter the predicted good-request fraction.

Replay supplies exact cumulative terminal, completed and joint-SLO success
counts plus integrated GPU-seconds. The controller computes `rho` from the
trailing `rho_window_s` completion window once outcomes exist, including failed
requests and GPUs occupied by starting/draining workers. Until then it uses its
capacity model. This online window is not the final arrival-cohort score:
completion delays can cross decision intervals.

A deterministic reservoir retains at most 64 paired TTFT/mean-ITL observations
per telemetry interval. MPC keeps at most 256 latency residual pairs from the
last 60 seconds, comparing outcomes with the previous interval's prediction.
It learns only when the predicted and observed fleet agree and the sample has
no starting or draining workers. Candidate joint success uses these paired
residuals, blending with the model's smooth prior until 32 samples arrive.
One-token outputs obey the scoring profile's `short_output_itl` rule.
The correction remains an approximation: completed requests can have started
under an earlier fleet, and observed residuals need not transfer perfectly to
an untried fleet. Performance must be checked on validation history.

Scale-up uses the mean forecast. Scale-down requires no pending replicas and a
smaller fleet that remains below 75% utilization, half the TTFT limit and the
calibrated ITL limit under
the 80th-percentile forecast for a full cold-start interval. A queue-spike
response accounts for capacity already starting. The initial fleet stays in
place until a positive-duration traffic observation is available; an empty
startup snapshot is not evidence of zero demand.

In disaggregated mode the controller evaluates separate prefill and decode
counts within the run's GPU budget and total-worker limits. The series
throughput is `min(prefill_count * prefill_rate,
decode_count * decode_rate)`. The model includes KV-transfer cost, per-pool
pending replicas and the GPU cost of each pool. Attention-DP throughput and
load use each engine's configured parallelism. Shared transfer bandwidth is
not multiplied by the rank count.

The feasible set enforces `prefill_count * prefill_gpu_cost + decode_count *
decode_gpu_cost <= backend.gpu_budget` together with pool and total-worker bounds.
The same set governs normal optimization, backlog recovery and reallocation.
If no candidate clears the backlog within the horizon, recovery selects the
feasible pair with the smallest remaining queue. A pool exchange may bypass
the ordinary scale-down hold when neither pool has pending orders and the
coupled plan improves without worsening queue or latency beyond the current
plan/SLO bounds.

## Configuration

| Setting | Purpose |
|---|---|
| `forward_model` | `ais` for the configured performance model; `roofline` for an analytical approximation |
| `slo_ttft_ms`, `slo_itl_ms` | Controller latency targets; explicitly match the case's scoring profile |
| `poll_interval_s`, `horizon` | Decision cadence and planning horizon beyond cold start |
| `min_replicas`, `max_replicas` | Candidate bounds; applied to each pool in disaggregated mode |
| `cold_start_s` | Replica startup delay; explicitly match the engine's runtime setting |
| `scale_up_penalty`, `scale_down_penalty` | Optional switching costs |
| `rho_min`, `rho_window_s` | Efficiency weight floor (default `1e-6`) and measured trailing window; also the startup model's smoothing window |
| `reactive_queue_threshold` | Threshold for the queue-spike floor |
| `gpus_per_worker` | Aggregate/decode worker cost, filled from the backend if omitted |
| `prefill_gpus_per_worker`, `kv_transfer_gbps`, `kv_bytes_per_token` | Disaggregated prefill cost and transfer parameters, filled from the engine where available |

Match Config defaults are a 5-second cadence, horizon 5, bounds 1–8, and
controller targets 300 ms TTFT / 50 ms ITL. Switching penalties default to
1.5 for scale-up and 10 for scale-down through Match Config; set both to zero
when the scale-down rule should supply all hysteresis. The controller's
cold-start default is 30 seconds. These are generic defaults, not the settings
of every benchmark case. Set targets, startup delay and replica limits
explicitly when adding MPC to a case.

In disaggregated mode, separate AIS sessions predict prefill and decode using
each pool's deployed role and engine parallelism. Aggregate workers reuse one
session for both phases. This preserves the TP4 prefill / TP8 decode identities
in the low-concurrency cases.

The Match runner supplies feasible allocations from the resolved case settings.
GPU limits are not a controller-wide constant: new 60-trace comparisons use
the original Static GPU allocation plus three times the largest worker's GPU
cost, giving ceilings of 36, 40, 104, 136 or 152 GPUs shared by all controllers.
The common replay adapter also enforces this budget during transitions. Starting
and draining workers retain their GPUs. A rebalance reduces one pool first and
admits growth only as those GPUs become free; a feasible final pair alone does
not make a simultaneous replacement affordable.

The static suite in `configs/60traces/` remains the baseline. Adding an MPC
entry creates a separate dynamic experiment; it does not run automatically.

The updated controller requires five-second native telemetry with
`completion_feedback.schema_version == 1`. Rebuild the native replay runtime
from this checkout before a new run; older binaries fail with an explicit
rebuild message. The Match runner passes the scorer's short-output policy to
the controller and records the `paired-slo-feedback-v2` policy identity.
