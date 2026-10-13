#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare earlier, unscaled workload history without changing any test input."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import tempfile
from collections import Counter
from pathlib import Path
from typing import TextIO

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "data/60traces/reproduction"
OUTPUT = ROOT / "data/60traces/training"
SPEC = ROOT / "configs/60traces/history-sources.json"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(path: Path, expected: str) -> None:
    if digest(path) != expected:
        raise ValueError(f"Checksum mismatch: {path}")


class HistoryWriter:
    """Validate every row while recording the uncompressed input and statistics."""

    def __init__(self, stream: TextIO, ident: str, block: int, end_s: float):
        self.stream, self.ident, self.block, self.end_s = stream, ident, block, end_s
        self.count = self.input_tokens = self.output_tokens = self.max_hash = 0
        self.first_ms = None
        self.last_ms = -1.0
        self.daily: Counter[int] = Counter()
        self.next_opaque = 1

    def opaque(self, input_length: int) -> list[int]:
        count = math.ceil(input_length / self.block)
        start = self.next_opaque
        self.next_opaque += count
        if self.next_opaque >= 2**32:
            raise ValueError("History exceeds the native uint32 block namespace")
        return list(range(start, self.next_opaque))

    def emit(
        self,
        timestamp_ms: float,
        input_length: int,
        output_length: int,
        hashes: list[int],
        source_row: int | str,
        extra: dict | None = None,
    ) -> None:
        timestamp_ms = round(float(timestamp_ms), 6)
        input_length, output_length = int(input_length), int(output_length)
        if not math.isfinite(timestamp_ms) or not 0 <= timestamp_ms < self.end_s * 1000:
            raise ValueError(
                f"Row outside training history: {source_row}, {timestamp_ms}"
            )
        if timestamp_ms < self.last_ms or min(input_length, output_length) < 0:
            raise ValueError(f"Unsorted or negative row: {source_row}")
        if len(hashes) != math.ceil(input_length / self.block):
            raise ValueError(f"Block count mismatch: {source_row}")
        if any(not isinstance(h, int) or not 0 <= h < 2**32 for h in hashes):
            raise ValueError(f"Invalid block ID: {source_row}")
        row = {
            "request_id": f"{self.ident}:history:r{source_row}",
            "timestamp": timestamp_ms,
            "input_length": input_length,
            "output_length": output_length,
            "hash_ids": hashes,
            "source_row": source_row,
            **(extra or {}),
        }
        self.stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        if self.first_ms is None:
            self.first_ms = timestamp_ms
        self.last_ms = timestamp_ms
        self.count += 1
        self.input_tokens += input_length
        self.output_tokens += output_length
        self.max_hash = max(self.max_hash, max(hashes, default=0))
        self.daily[int(timestamp_ms // 86400000)] += 1
        if self.count % 1000000 == 0:
            print(f"{self.ident}: {self.count:,} history requests written", flush=True)

    def statistics(self) -> dict:
        if not self.count:
            raise ValueError(f"No history rows selected for {self.ident}")
        return {
            "requests": self.count,
            "first_arrival_ms": self.first_ms,
            "last_arrival_ms": self.last_ms,
            "daily_request_counts": [
                self.daily[i] for i in range(math.ceil(self.end_s / 86400))
            ],
            "mean_isl": self.input_tokens / self.count,
            "mean_osl": self.output_tokens / self.count,
            "total_input_tokens": self.input_tokens,
            "total_output_tokens": self.output_tokens,
            "trace_block_size": self.block,
            "max_hash_id": self.max_hash,
            "namespace_stride": self.max_hash + 1,
        }


def csv_history(writer: HistoryWriter, spec: dict, source_dir: Path) -> dict:
    """Select before sorting; retain source indices so split audits are possible."""
    source = spec["source"]
    asset = source["source_files"][0]
    path = source_dir / asset["file"]
    verify(path, asset["sha256"])
    azure = writer.ident.startswith("azure-")
    columns = (
        ["TIMESTAMP", "ContextTokens", "GeneratedTokens"]
        if azure
        else ["Timestamp", "Request tokens", "Response tokens"]
    )
    pieces = []
    offset, origin, minimum = 0, None, math.inf
    for chunk in pd.read_csv(path, usecols=columns, chunksize=250000):
        if azure:
            timestamps = (
                pd.to_datetime(chunk[columns[0]], utc=True, format="mixed")
                .dt.as_unit("ns")
                .astype("int64")
                .to_numpy()
            )
            if origin is None:
                origin = int(timestamps.min())
            relative = (timestamps - origin) / 1e9
        else:
            timestamps = chunk[columns[0]].to_numpy(dtype=float)
            origin = source["full_trace_first_timestamp"]
            relative = timestamps - origin
        minimum = min(minimum, timestamps.min())
        mask = (relative >= 0) & (relative < writer.end_s)
        if mask.any():
            pieces.append(
                (
                    relative[mask],
                    chunk[columns[1]].to_numpy(dtype=np.int64)[mask],
                    chunk[columns[2]].to_numpy(dtype=np.int64)[mask],
                    np.arange(offset, offset + len(chunk), dtype=np.int64)[mask],
                )
            )
        offset += len(chunk)
    if minimum != origin or offset != source["full_trace_requests"]:
        raise ValueError(f"Source origin/row-count differs for {writer.ident}")
    if not pieces:
        raise ValueError(f"No history found in {path}")
    timestamps, inputs, outputs, indices = [
        np.concatenate([p[i] for p in pieces]) for i in range(4)
    ]
    del pieces
    for index in np.argsort(timestamps, kind="stable"):
        writer.emit(
            timestamps[index] * 1000,
            inputs[index],
            outputs[index],
            writer.opaque(int(inputs[index])),
            int(indices[index]),
        )
    return {
        "source_files": [asset],
        "source_full_rows_scanned": offset,
        "source_origin": int(origin) if azure else float(origin),
        "source_origin_unit": "Unix nanoseconds" if azure else "source seconds",
        "hash_assumption": "Source publishes no block hashes; request-unique opaque IDs assume zero prefix reuse.",
    }


def rag_history(writer: HistoryWriter, spec: dict, bundle: Path) -> dict:
    helper_path = bundle / "scripts/frozen_converters/ragpulse_converter.py"
    verify(helper_path, spec["helper_sha256"])
    module_spec = importlib.util.spec_from_file_location(
        "history_ragpulse_converter", helper_path
    )
    if module_spec is None or module_spec.loader is None:
        raise ImportError(str(helper_path))
    helper = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(helper)
    sources = {}
    for asset in spec["assets"]:
        path = bundle / "work/source-inputs/source/ragpulse" / asset["file"]
        verify(path, asset["sha256"])
        sources[asset["file"]] = path
    helper.SOURCES = sources
    rows, _ = helper.build_base(writer.block)
    for row in rows:
        if 0 <= row["source_timestamp_s"] < writer.end_s:
            writer.emit(
                row["source_timestamp_s"] * 1000,
                row["input_length"],
                row["output_length"],
                row["hash_ids"],
                row["source_row"],
                {
                    key: row[key]
                    for key in (
                        "source_timestamp_s",
                        "source_session_id",
                        "unknown_tail_tokens",
                    )
                },
            )
    return {
        "source_files": spec["assets"],
        "helper_sha256": spec["helper_sha256"],
        "hash_assumption": "Frozen cumulative component approximation; not original token identities.",
    }


def art_history(writer: HistoryWriter, spec: dict, source_dir: Path) -> dict:
    mapping = {}
    projections = []
    for asset in [*spec["assets"], spec["boundary_projection"]]:
        path = source_dir / asset["projection_file"]
        sidecar = (
            asset
            if "projection_sha256" in asset
            else json.loads(path.with_suffix(".source.json").read_text())
        )
        if sidecar["url"] != asset["url"]:
            raise ValueError(f"Projection source mismatch: {path}")
        verify(path, sidecar["projection_sha256"])
        projections.append(sidecar)
        offset = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
            for row in batch.to_pylist():
                timestamp = row["timestamp_ms"]
                # Pinned calendar files can include a boundary row; use the actual timestamp.
                if 0 <= timestamp < writer.end_s * 1000:
                    hashes = []
                    for value in row["hash_ids"]:
                        key = (type(value).__name__, str(value))
                        if key not in mapping:
                            mapping[key] = len(mapping) + 1
                        hashes.append(mapping[key])
                    writer.emit(
                        timestamp,
                        row["input_length"],
                        row["output_length"],
                        hashes,
                        f"{path.stem}:{offset}",
                        {"source_request_id": row["request_id"]},
                    )
                offset += 1
        if offset != sidecar["rows"]:
            raise ValueError(f"Projection row count mismatch: {path}")
        print(f"art-chat: converted {path.name}", flush=True)
    return {
        "source_files": projections,
        "hash_assumption": "Native 256-token prefix identity preserved by a bijection fitted only on history. Training and test hash namespaces are independent; reset caches between replays.",
    }


def prepare(
    ident: str, spec: dict, bundle: Path, output: Path, verify_only: bool = False
) -> dict:
    destination = output / ident / "history-1x.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    sidecar = destination.with_suffix(".jsonl.manifest.json")
    recipe_sha = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    with destination.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists():
            metadata = json.loads(sidecar.read_text())
            verify(destination, metadata["output_sha256"])
            if metadata["recipe_sha256"] != recipe_sha:
                raise ValueError(
                    f"History recipe changed; refusing to overwrite {destination}"
                )
            print(
                f"{ident}: existing history verified ({metadata['requests']:,} requests)",
                flush=True,
            )
            return metadata
        if verify_only:
            raise FileNotFoundError(destination)
        # Compare to versioned test boundaries, without reading test requests into training.
        catalog = json.loads((ROOT / "configs/60traces/catalog.json").read_text())
        test = next(s for s in catalog["sources"] if s["id"] == ident)
        if spec["history_start_s"] != 0 or spec["history_end_s"] > test["test_start_s"]:
            raise ValueError(f"Overlapping history/test windows: {ident}")
        with tempfile.TemporaryDirectory(
            prefix=".history-", dir=destination.parent
        ) as directory:
            temporary = Path(directory) / destination.name
            with temporary.open("w") as stream:
                writer = HistoryWriter(
                    stream, ident, spec["source"]["block_size"], spec["history_end_s"]
                )
                if ident == "ragpulse":
                    provenance = rag_history(writer, spec, bundle)
                elif ident == "art-chat":
                    provenance = art_history(
                        writer, spec, bundle / "work/source-inputs/source"
                    )
                else:
                    provenance = csv_history(
                        writer, spec, bundle / "work/source-inputs/source"
                    )
            metadata = {
                "schema": "planner-gym.training-history.v1",
                "dataset": ident,
                "kind": "workload traffic, not RL transitions",
                "alpha": 1,
                "window_start_seconds": spec["history_start_s"],
                "window_end_seconds": spec["history_end_s"],
                "duration_seconds": spec["history_end_s"] - spec["history_start_s"],
                "window_end_exclusive": True,
                "test_start_seconds": test["test_start_s"],
                "test_end_seconds": test["test_end_s"],
                "validation_split": "Not yet defined; reserve a chronological part of history before tuning.",
                "format": "mooncake",
                "timestamps": "Milliseconds from source origin; idle gaps preserved, no compression.",
                "sorted_by_timestamp": True,
                "hash_namespace_scope": "Independent replay; do not concatenate with other sources or test traces without remapping.",
                "recipe_sha256": recipe_sha,
                "converter_sha256": digest(Path(__file__)),
                "output_sha256": digest(temporary),
                "output_bytes": temporary.stat().st_size,
                **writer.statistics(),
                **provenance,
            }
            temporary.replace(destination)
            sidecar.write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"{ident}: ready, {metadata['requests']:,} requests, {metadata['output_bytes'] / 1e9:.3f} GB",
        flush=True,
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--all", action="store_true")
    selector.add_argument("--dataset", action="append")
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--export-index",
        type=Path,
        help="Write the portable inventory of selected, verified histories",
    )
    args = parser.parse_args()
    specs = json.loads(SPEC.read_text())
    selected = list(specs) if args.all else args.dataset
    unknown = set(selected) - specs.keys()
    if unknown:
        parser.error(
            "No independent history defined for: " + ", ".join(sorted(unknown))
        )
    histories = {}
    for ident in selected:
        metadata = prepare(
            ident, specs[ident], args.bundle, args.output, args.verify_only
        )
        input_path = (args.output / ident / "history-1x.jsonl").resolve()
        histories[ident] = {
            **metadata,
            "input_file": str(
                input_path.relative_to(ROOT)
                if input_path.is_relative_to(ROOT)
                else input_path
            ),
        }
    if args.export_index:
        args.export_index.write_text(
            json.dumps({"schema_version": 1, "histories": histories}, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
