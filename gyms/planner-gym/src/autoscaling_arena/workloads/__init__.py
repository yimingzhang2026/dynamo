# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic trace validation and optional user-defined workload generation."""

from autoscaling_arena.workloads import axes
from autoscaling_arena.workloads.generator import Workload, validate_mooncake_trace
from autoscaling_arena.workloads.registry import WORKLOADS, get_workload, list_workloads

__all__ = [
    "axes",
    "Workload",
    "validate_mooncake_trace",
    "WORKLOADS",
    "get_workload",
    "list_workloads",
]
