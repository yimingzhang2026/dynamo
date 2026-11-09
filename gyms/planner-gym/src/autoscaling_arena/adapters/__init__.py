# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay controllers, with MPC's optional numerical dependencies loaded lazily."""

from __future__ import annotations

import importlib
from typing import Any

from autoscaling_arena.adapters.keda import KedaAutoscaler
from autoscaling_arena.adapters.planner import planner_engine_factory
from autoscaling_arena.adapters.reactive import ReactiveAutoscaler
from autoscaling_arena.adapters.static import StaticAutoscaler

_LAZY_EXPORTS = {
    "MPCAutoscaler": "autoscaling_arena.adapters.mpc",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value  # cache so later lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
    "StaticAutoscaler",
    "ReactiveAutoscaler",
    "KedaAutoscaler",
    "MPCAutoscaler",
    "planner_engine_factory",
]
