<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Running all 60 traces

[comparison.yaml](../configs/60traces/comparison.yaml) is the common execution
policy. MPC, Dynamo, KEDA, Reactive, Static and KnowLine each cover all 60 cases.
Paid JEV is excluded.

## CPU replay

The default policy budgets 8 CPU cores and 512 GiB RAM. It reserves
64 GiB for the host and allows up to 8 single-core jobs. Numeric, Rayon and
Tokio thread pools are capped at one. Each job has a separate session, config,
request stream and result. Override `--physical-cores` and `--memory-gib` when
preparing a plan; edit the policy to adjust concurrency and per-run cores.

Memory reservations use the saved reference peak: at least 8 GiB, otherwise
`ceil(2 * peak_RSS_GiB + 2)`. Jobs reserving 256 GiB or more run alone. These
estimates are checked against available RAM; the process-tree RSS guard fails
a job that exceeds its budget. A larger host does not change trace contents.

```bash
python scripts/run_campaign.py cpu --workspace runs/benchmark
```

The entry point prepares the 300 CPU jobs, builds or verifies the current native
runtime, and runs the rolling queue. Free CPU slots are filled as jobs finish.
Reinvoking it resumes pending work and preserves completed results. Genuine
failures stay explicit. A campaign lock prevents overlapping launchers.

## KnowLine on a GPU cluster

The model is `PelaAI/KnowLine-31B-Gen1` at revision
`8f052a8130e599b53e5324c34f0f5ff90696e1c6`. The pinned serving stack is
PyTorch `2.13.0+cu126`, Transformers `5.12.1`, and Accelerate `1.15.0`.
Use a separate serving environment so its GPU dependencies do not alter replay.

```bash
python3.12 -m venv runs/benchmark/knowline/serve-env
runs/benchmark/knowline/serve-env/bin/pip install torch==2.13.0+cu126 \
  --index-url https://download.pytorch.org/whl/cu126
runs/benchmark/knowline/serve-env/bin/pip install transformers==5.12.1 accelerate==1.15.0 requests huggingface_hub
runs/benchmark/knowline/serve-env/bin/python scripts/download_knowline.py \
  runs/benchmark/knowline/models/KnowLine-31B-Gen1-8f052a8
```

Use the same entry point on the GPU cluster:

```bash
python scripts/run_campaign.py knowline --workspace runs/benchmark \
  --account YOUR_ACCOUNT --partition YOUR_PARTITION
```

This prepares all 60 cases, divides them into two queues with four model/replay
lanes per node, and submits the two-node array. The two largest traces go to
separate nodes. A submission receipt prevents duplicate submission.
Native runtime verification and any rebuild happen inside the compute allocation.
The two nodes share a setup lock and reuse the same validated build.

Each node needs 64 CPU cores, approximately 2 TiB RAM and eight 80 GiB GPUs.
This schedules at most two nodes (16 GPUs), with up to eight concurrent traces.
Each node owns a queue of cases. The longest cases are assigned first to
separate nodes, then remaining work is balanced between them. Free lanes claim
the next case from their node's queue while reusing the loaded model server.
Specify the account and partition using your cluster's `sbatch` options.
`PLANNER_TRACE_DIR` can override the expanded input directory.

Each node queue has one owner and four disjoint CPU/GPU lanes. Four-hour allocations
reserve a 3h40m work window. A dependent successor continues unfinished cases,
up to 16 allocations per shard, and is cancelled when the shard finishes.
Saved decisions retain their original inference latency. Continuation replays
from the trace start and rejects any changed observation, answer or action;
replayed decisions do not count as new inference. Two allocations without
progress mark a case as needing a longer allocation. Memory-limited cases
continue alone with a 1400 GiB guard. Other failures stay explicit.

```bash
python scripts/run_campaign.py status --workspace runs/benchmark
python scripts/run_campaign.py collect --kind knowline --workspace runs/benchmark
```

The standard scheduler replaces failed model-server processes before continuing
saved decision prefixes. It also handles interrupted allocations and isolates
memory-heavy cases. These fixes are in the main implementation; no patch or
recovery package is applied. Collection and publication use the same entry
point; see the [benchmark guide](../README.md).

## Benchmark constraints

The shared maximum GPU allowance is **the original Static allocation plus
3 × the largest per-worker GPU cost** in that case.
Static and every dynamic planner start at the original fleet. Static holds it
fixed. Small cases allow 36 or 40 simulated GPUs; large cases allow 104, 136 or
152. Minimums are one aggregate worker or one prefill plus one decode worker.
Per-pool maxima reserve the other pool's minimum, and both pools share the GPU
ceiling. There is no independent 16-worker cap. Aggregated and HT workers use
8 GPUs each; LC prefill uses 4 and LC decode uses 8. These costs are simulated.
Draining workers remain charged until their lifecycle ends. `hold_final` charges
the last fleet over any idle remainder of the offered window.

Rankings use goodput per average simulated GPU, with equal trace weight and
average ranks for exact ties. Physical decision-model GPU cost and call latency
are recorded separately from simulated serving scores.
