# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extract auditable scoring evidence without retaining replay histories.

Requires ijson. This script reads saved results only; it never runs a simulation.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import ijson
from ijson.common import ObjectBuilder

TOP_FIELDS = {"match", "summary", "provenance"}
ROW_FIELDS = {
    "run_id",
    "backend",
    "autoscaler",
    "autoscaler_type",
    "workload",
    "sla",
    "seed",
    "status",
    "metrics",
    "raw_metrics",
    "execution",
    "error",
}
RUNTIME_FIELDS = {
    "gpu_budget",
    "planner_config",
    "model",
    "topology",
    "worker_limits",
    "replay",
    "jev",
    "autoscaler_config",
    "initial_replicas",
}
EVALUATION_FIELDS = {
    "workload",
    "seed",
    "repetition",
    "max_requests",
    "arrival_speedup",
    "trace_block_size",
    "trace",
    "sla",
}


class HashingReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def read(self, size=-1):
        data = self.stream.read(size)
        self.digest.update(data)
        return data


def compact(path: Path) -> tuple[dict, str]:
    """Stream the complete JSON and retain only metadata and scoring fields."""
    report = {"results": []}
    current = None
    builder = None
    depth = 0
    destination = None
    key = None
    before = path.stat()
    with path.open("rb") as stream:
        reader = HashingReader(stream)
        for prefix, event, value in ijson.parse(reader, use_float=True):
            if builder is not None:
                builder.event(event, value)
                depth += (event in ("start_map", "start_array")) - (
                    event in ("end_map", "end_array")
                )
                if depth == 0:
                    destination[key] = builder.value
                    builder = None
                continue
            if prefix == "results.item" and event == "start_map":
                current = {}
                report["results"].append(current)
                continue
            destination = None
            if prefix in TOP_FIELDS:
                destination, key = report, prefix
            elif prefix.startswith("results.item.") and current is not None:
                field = prefix[len("results.item.") :]
                if field in ROW_FIELDS:
                    destination, key = current, field
                elif field.startswith("runtime.") and field[8:] in RUNTIME_FIELDS:
                    destination, key = current.setdefault("runtime", {}), field[8:]
                elif (
                    field.startswith("evaluation.") and field[11:] in EVALUATION_FIELDS
                ):
                    destination, key = current.setdefault("evaluation", {}), field[11:]
            if destination is not None:
                if event in ("start_map", "start_array"):
                    builder = ObjectBuilder()
                    builder.event(event, value)
                    depth = 1
                elif event in ("string", "number", "boolean", "null"):
                    destination[key] = value
        fingerprint = reader.digest.hexdigest()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Result changed while being read: {path}")
    if builder is not None or not report["results"]:
        raise ValueError(f"Incomplete or empty result: {path}")
    return report, fingerprint
