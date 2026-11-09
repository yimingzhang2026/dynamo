# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional user-registered workloads; benchmark inputs use explicit Match Config paths.

The 60-case benchmark is selected through configs/60traces/suite.yaml. Keeping
an empty registry avoids implicitly selecting or generating benchmark traffic.
"""

from autoscaling_arena.workloads.generator import Workload

WORKLOADS: dict[str, Workload] = {}


def get_workload(name: str) -> Workload:
    if name not in WORKLOADS:
        raise KeyError(
            f"unknown workload {name!r}; use an explicit trace in a Match Config"
        )
    return WORKLOADS[name]


def list_workloads() -> list[str]:
    return list(WORKLOADS)
