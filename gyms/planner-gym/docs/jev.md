<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Jev decision-engine experiment

The experimental `jev` adapter implements Dynamo Planner's `EngineProtocol`.
It replaces the decision engine inside Gym replay: Jev's selections directly
produce `PlannerEffects.scale_to`. The native Planner remains a separate baseline.
This does not configure a deployed Dynamo Planner service.

## Configure an experiment

Copy a config from `configs/60traces/` into a separate experiment directory,
update its relative paths and add a `jev` policy with explicit replica limits.
Provide `TYPESAFE_API_KEY` through the environment. Hosted policy calls require
an explicit experiment plan and API-call budget. The static baseline has no
hosted policy calls.

## Decision contract

One Choice question per pool selects from the current expected replica count
and a bounded step up/down. Questions share one state and one HTTP request.
Code constructs valid targets; Match Config rejects combined pool maxima above
the GPU budget. The default step is one replica and the cadence is 15 simulated
seconds. Starting workers are included in the expected count.

State includes aggregate queue/token backlog, KV utilization, ready and expected
workers, traffic shape and rate, worker GPU costs/cold-start delay, SLO objectives,
and four prior snapshots. Missing measurements remain null. Request content,
worker identifiers, prefix hashes, future arrivals, and baseline decisions are
not sent. The tick interface supplies no measured TTFT/ITL; their limits are
objectives, not observations. Aggregated topology asks only for the decode pool.

Default `min_confidence: 0` exposes Jev's choices without an uncalibrated cutoff.
Setting a threshold explicitly makes low-confidence answers hold their pool's
current expected target. Default `failure_mode: raise` fails on a timeout, API
error, malformed answer, or pinned-model mismatch. `failure_mode: hold` is a
separate reliability experiment; it holds both targets and logs the error.
Exhausting `max_calls` always stops the run.

## Evidence and interpretation

Each Jev run writes `jev-decisions.jsonl` containing the exact request, request
digest, validated answers/probabilities/confidence, returned model, available token
usage, wall latency, selected targets, and error categories. Match results include
the artifact path and a `runtime.jev` overhead summary. Exception text and error
response bodies are excluded; no credentials enter these artifacts.

**API wall time does not advance replay time.** The first experiment evaluates
decision quality with ideal actuation timing. Controller latency is reported
separately and cannot be interpreted as already included in TTFT or GPU-hours.
Delayed-action simulation or a live deployment is needed for that comparison.

Read goodput/GPU together with good rate, latency, GPU-hours, and oscillation.
Freeze prompts and settings before testing held-out workloads. Repeat Jev cells
to measure variability, and report failed or gated decisions. Tests using local
HTTP fixtures validate integration only; they provide no evidence of Jev's
autoscaling quality.
