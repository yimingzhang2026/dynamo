<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# MPC

MPC is the Gym's model predictive controller. Select `type: mpc` and use `MPC`
as its display name. The implementation is `adapters/mpc.py`, exported as
`MPCAutoscaler`; controller tests live in `tests/test_mpc.py`.

## Decision model

The capacity model combines prefill time, decode step time and the KV-cache
concurrency limit. For aggregate workers, the sustainable request rate is:

```text
c_sat = min(0.9 * max_kv_tokens / (isl + osl), max_num_seqs)
mu = 1 / (prefill_seconds + osl * decode_step_seconds(c_sat) / c_sat)
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
`rho * GPU-seconds`, with `rho` tracking recent efficiency. Both TTFT and ITL
enter the predicted good-request fraction.

Scale-up uses the mean forecast. Scale-down requires no pending replicas and a
smaller fleet that remains below 75% utilization and half the TTFT limit under
the 80th-percentile forecast for a full cold-start interval. A queue-spike
floor accounts for capacity already starting.

In disaggregated mode the controller evaluates separate prefill and decode
counts. The series throughput is `min(prefill_count * prefill_rate,
decode_count * decode_rate)`. The model includes KV-transfer cost, per-pool
pending replicas and the GPU cost of each pool.

## Configuration

| Setting | Purpose |
|---|---|
| `forward_model` | `ais` for the configured performance model; `roofline` for an analytical approximation |
| `slo_ttft_ms`, `slo_itl_ms` | Controller latency targets; explicitly match the case's scoring profile |
| `poll_interval_s`, `horizon` | Decision cadence and planning horizon beyond cold start |
| `min_replicas`, `max_replicas` | Candidate bounds; applied to each pool in disaggregated mode |
| `cold_start_s` | Replica startup delay; explicitly match the engine's runtime setting |
| `scale_up_penalty`, `scale_down_penalty` | Optional switching costs |
| `rho_min`, `rho_window_s` | Efficiency weight floor and smoothing window |
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

The static suite in `configs/60traces/` remains the baseline. Adding an MPC
entry creates a separate dynamic experiment; it does not run automatically.
