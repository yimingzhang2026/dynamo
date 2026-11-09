# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate and merge an exact saved prefix of self-hosted model decisions."""

import hashlib
import json


def valid_prefix(path, *, model):
    if not path.exists():
        return []
    rows = []
    with path.open() as stream:
        for line in stream:
            if not line.endswith("\n"):
                break
            row = json.loads(line)
            if row["status"] != "ok":
                break
            if row.get("model") != model or row.get("provider") != "self_hosted":
                raise ValueError("Unexpected decision identity")
            expected = hashlib.sha256(
                json.dumps(row["request"], sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
            if row["request_sha256"] != expected:
                raise ValueError("Decision request digest differs")
            rows.append(row)
    return rows


def merge_prefix(previous, observed):
    for old, new in zip(previous, observed):
        if any(
            old[key] != new[key] for key in ("request_sha256", "answers", "targets")
        ):
            raise ValueError("Continuation changed an already recorded decision")
    return previous + observed[len(previous) :]
