<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Non-learning planner comparison

The experiment policy is [comparison.yaml](../configs/60traces/comparison.yaml).
The existing 60 static cases remain the reference. MPC, Dynamo, KEDA and Reactive
each run all 60 cases. JEV runs the 28 short cases, with **600 calls per run** and a
15-second simulated control interval. No one-day or seven-day case calls JEV.
This gives **268 new runs** and at most **16,800 JEV calls** in one complete attempt.
Retries are separate attempts and are never automatic.

Each of A-HT, A-LC, D-HT and D-LC has 67 new runs plus 15 existing static rows.
The seven JEV cases in each group are S05, S06, S07, S08, S09, L03 and L04.
The longest selected arrival window is approximately two hours: around 480
decision opportunities at 15-second intervals, leaving room below the 600-call
ceiling for drain time. Exhausting the ceiling fails the run; it does not silently
substitute another policy or freeze the last decision.

## Worker limits and the static reference

The dynamic range is **2–16 total worker targets** for every case and planner:

| Experiment | Worker targets | Simulated GPU cost |
|---|---|---|
| A-HT / A-LC | 2–16 aggregate workers | 8 per worker; 16–128 total |
| D-HT | P >= 1, D >= 1, 2 <= P+D <= 16 | 8 per worker; 16–128 total |
| D-LC | P >= 1, D >= 1, 2 <= P+D <= 16 | 4 per P, 8 per D; 12–124 total |

The small static cases specify a fixed two-worker allocation. The large static
cases specify sixteen workers. Those are starting/reference allocations, not
the dynamic range. The comparison increases small-case headroom while keeping
the exact input scaling, engine identity, SLO, zero warmup and initial allocation.
The original static YAML files are unchanged. A larger ceiling does not allocate
all those GPUs: scoring uses actual allocation over time.

`backend.worker_limits` supplies a shared allocation layer for all five planners.
Per-pool limits allow every asymmetric split, including 1P+15D and 15P+1D.
If a controller proposes an infeasible fleet, the allocation layer selects the
closest feasible integer pair by squared distance, then least change from the
current fleet, then fewer workers, then lexicographic order. GPU limits and each
controller's per-pool bounds also apply. Adjusted decisions appear in
`worker-limit-decisions.jsonl`; raw JEV decisions remain in `jev-decisions.jsonl`.
This is an explicit benchmark allocation constraint, not a change to upstream
Dynamo's native policy. The limit applies to desired targets; draining workers
can temporarily remain physically present during a transition.

MPC receives each case's actual TTFT/ITL objectives and zero startup delay.
Its AIS capacity model uses separate deployed-role identities for prefill and
decode, including LC's TP4 prefill / TP8 decode. Native runtime and AIS compatibility require validation on
the execution host; configuration validation alone is not a replay test.

## Parallel execution setup

Use a **Linux CPU host with 32 physical cores and 256 GiB RAM**, the pinned
AISimulate version and compatible Dynamo native bindings. No physical B300 GPUs
are required for simulation. The Mac can run Linux ARM native replay inside
Docker Desktop with a smaller, explicitly partial campaign (see below).

The selected initial setup is:

- At most **four independent run processes**, each bound to **two physical cores**.
  The launcher picks one allowed logical CPU per physical core, avoiding SMT
  double-counting. Each Match Config has `max_parallel_runs: 1`.
- Reserve **32 GiB RAM** for the OS, supervisors and headroom. Each run reserves
  `max(8, ceil(2 * reference_peak_RSS_GiB + 2))` GiB. Batches must fit both RAM
  and CPU reservations. These are conservative estimates, not measured dynamic
  peaks. A run exceeding its RSS guard fails and must be investigated.
- Runs reserving **64 GiB or more run alone**. The largest historical static run
  used 85.7 GiB; its initial dynamic reservation is 174 GiB. A 48 GiB machine
  cannot safely cover the full suite with this policy, even sequentially.
- At most **one JEV run at once**. API waits consume wall time, and the 15-second
  control interval is virtual time, not a wall-clock rate limit. This serializes
  JEV requests across runs; provider quotas must still be respected. There are
  no automatic API retries.
- Numeric-library threads are capped at one; Rayon and Tokio at two. CPU affinity
  also limits native child processes. Each job has separate configs, sessions,
  logs and results. Do not start a second launcher for the same campaign.

From the Gym directory, prepare and inspect only:

```bash
.venv/bin/python scripts/prepare_comparison.py
.venv/bin/python scripts/run_comparison.py runs/60traces-comparison/plan.json
```

The default plan has 268 jobs in 79 conservative batches, including 16 exclusive
large-memory runs. A batch finishes before the next begins. This trades some
CPU utilization for predictable memory usage. Measure real resource use on the
Linux host before increasing concurrency or lowering reservations.

To inspect another host's capacity without executing:

```bash
.venv/bin/python scripts/prepare_comparison.py \
  --memory-gib 48 --physical-cores 18 --output runs/60traces-comparison-48gib
```

That profile leaves 20 runs blocked: four planners each on A-HT-S04, A-LC-L02,
D-HT-S04, D-HT-L01 and D-HT-L02. Partial execution requires `--allow-blocked`;
oversized jobs remain recorded as pending in `status.json`.

Only when ready on Linux, explicitly add `--execute` to `run_comparison.py`.
It verifies config hashes, CPU/RAM capacity and available memory, applies CPU
affinity and thread caps, and verifies each trace hash before replay. Existing
sessions/results/logs are not overwritten. A failed batch stops the campaign;
inspect its logs and account for JEV calls already spent before retrying.
No benchmark or hosted API call is made by preparation or plan inspection.

After investigating a failed batch, `--resume-pending` continues only jobs whose
saved status is pending. It never retries failed or completed jobs. Add
`--keep-going` to record individual failures and continue independent batches;
the final exit status remains nonzero if any job failed. `--planners mpc planner
keda reactive` selects a subset while retaining other jobs as pending. An
exclusive campaign lock prevents overlapping launchers; `--wait-for-lock`
queues a later launcher, such as a JEV-only continuation, behind the active one.

### Mac execution

`comparison.mac.yaml` reserves up to three simultaneous jobs, two Linux VM CPU
slots per job and one JEV job at a time. The intended Docker Desktop VM has
32 GiB RAM; the dedicated container is capped at 30 GiB and six CPUs. Planning
uses 28 GiB, with another 2 GiB reserved inside that budget. This leaves 16 GiB
of the Mac's 48 GiB outside the VM and room for resident runtime overhead plus
free supervisor headroom. CPU affinity separates jobs inside Linux; it does not
pin Docker's virtual CPUs to specific Apple physical cores.

The launcher checks cgroup v2 CPU and memory limits, as well as Linux's reported
capacity. Resource estimates and complete trace windows remain unchanged.
With the bundled reference measurements this schedules 220 runs in 98 batches:
48 each for MPC, Dynamo, KEDA and Reactive, and all 28 eligible JEV runs. The
remaining 48 runs (12 cases for each non-JEV planner) remain pending for RAM.
Use the container's Python environment, not the macOS `.venv`:

```bash
/opt/venv/bin/python scripts/prepare_comparison.py \
  --policy configs/60traces/comparison.mac.yaml --output runs/60traces-mac
/opt/venv/bin/python scripts/run_comparison.py \
  runs/60traces-mac/plan.json --allow-blocked --execute
```

First validate the installed native runtime with `scripts/smoke_comparison.py
--include-jev --execute`. These are synthetic eight-request tests, kept outside
the benchmark results. `--include-jev` makes live API calls with at most ten
calls per smoke run. Campaign JEV runs retain the 600-call cap. Store the key
outside the repository and inject `TYPESAFE_API_KEY` through the process
environment. Never place it in a Match Config, plan, log or result.

The Mac environment uses the matched `ai-dynamo` / `ai-dynamo-runtime`
`1.6.0.dev20261013` Linux ARM wheels with `aisimulate==0.13.0.dev2`. The
Planner Gym package is editable from this branch. The upstream nightly runtime
is distinct from the historical static runtime; installed package versions are
recorded in session provenance. Do not relabel the historical scores as a rerun.

## Four reports and fair ranking

Each future group HTML report should have a full 15-case ranking for Static,
MPC, Dynamo, KEDA and Reactive, plus a shared-seven-case ranking that includes
JEV. Mark JEV's eight excluded cases as not evaluated. Do not compare JEV's
seven-case average against another planner's fifteen-case average.

Retain static scores as reference evidence. Their recorded Dynamo SHA differs
from this branch: verify runtime, metric and trace compatibility before treating
them as interchangeable measurements in the new leaderboard. No static replay
is added to the 268-run plan, and no ranking is produced before results exist.
