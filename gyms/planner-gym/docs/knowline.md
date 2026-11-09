<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KnowLine-31B decision model

The benchmark serves `PelaAI/KnowLine-31B-Gen1` locally at revision
`8f052a8130e599b53e5324c34f0f5ff90696e1c6`. The replay adapter uses the
`jev` protocol name internally, with `provider: self_hosted`; the result name is
`KnowLine-31B`. It sends no paid API calls. Installation, resource requirements,
parallel node queues and continuation commands are in [comparison-60.md](comparison-60.md).

## Joint fleet actions

Each decision asks one Choice question about the joint fleet. Options specify
absolute prefill/decode worker counts, weighted GPU cost, and the GPUs needed
immediately. The candidate set intersects bounded per-pool steps with the shared
GPU and worker limits. Workers remain indivisible: adding a TP4 prefill worker
uses four GPUs; adding a TP8 decode worker uses eight.

Active, starting and draining workers occupy the budget. A reduction cannot
fund growth until its workers release their GPUs. Confidence gating holds the
whole joint fleet atomically. A single feasible fleet needs no model call.

The objective is requests satisfying every configured SLO per GPU-second.
The prompt supplies recent measured SLO success, goodput/GPU, TTFT and ITL,
resident allocation, and before/after outcome windows for the previous action.
Completion-window observations can include requests started under earlier
fleets; they are distinct from the final arrival-cohort leaderboard score.
Missing observations remain null. Future arrivals and request content are
excluded.

The policy identifier is `joint-fleet-measured-efficiency-v2`. Five-second
native telemetry supplies `completion_feedback.schema_version == 1`; missing,
stale or inconsistent feedback fails the run. Model decisions occur every
15 simulated seconds. The full-trace configuration allows up to 1,000,000 calls.

## Serving and evidence

The endpoint accepts `POST /v1/systemone` and returns the pinned model identity,
choice distributions, selected choices and confidence. `download_knowline.py`
pins serving source and weights; `serve_knowline.sh` starts the local server.
The adapter bypasses ambient HTTP proxies and never falls back to a hosted
provider. Authentication, if needed, comes from a separate environment variable.

Each run records exact requests, validated answers, targets, inference latency
and errors in `jev-decisions.jsonl`, with a `runtime.jev` summary. Continuations
replay the saved prefix and reject changed observations or answers. Replayed
decisions retain their original inference latency and do not call the model.

API wall time does not advance replay time. The tables measure decision quality
under ideal actuation timing; physical model GPU use and call latency are
reported separately from simulated serving costs. Timeouts, malformed answers,
model mismatches and exhausted call budgets fail explicitly.
