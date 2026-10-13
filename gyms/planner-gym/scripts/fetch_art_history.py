#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download the five metadata columns of pinned ART Chat history days 1–4."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import fsspec
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "data/60traces/reproduction"
COLUMNS = ("request_id", "timestamp_ms", "input_length", "output_length", "hash_ids")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(asset: dict, destination: Path) -> dict:
    """Preserve row order and bounded memory; omit message bodies entirely."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    sidecar = destination.with_suffix(".source.json")
    if destination.exists():
        metadata = json.loads(sidecar.read_text())
        if (
            metadata["url"] != asset["url"]
            or digest(destination) != metadata["projection_sha256"]
        ):
            raise ValueError(f"Existing projection differs: {destination}")
        print(f"Verified {destination.name}", flush=True)
        return metadata

    fs = fsspec.filesystem("http", client_kwargs={"raise_for_status": True})
    with fs.open(
        asset["url"], "rb", block_size=65536, cache_type="readahead"
    ) as stream:
        parquet = pq.ParquetFile(stream)
        metadata, size = parquet.metadata, stream.size
    if size != asset["upstream_bytes"]:
        raise ValueError(f"Pinned remote size changed: {asset['url']}")

    def read_group(index: int):
        with fs.open(
            asset["url"], "rb", block_size=65536, cache_type="readahead", size=size
        ) as stream:
            return pq.ParquetFile(stream, metadata=metadata).read_row_group(
                index, columns=list(COLUMNS), use_threads=False
            )

    temporary = destination.with_suffix(".partial")
    try:
        with ThreadPoolExecutor(max_workers=6) as pool, ExitStack() as stack:
            writer = None
            # Submit only six groups at a time, avoiding an unbounded result queue.
            for begin in range(0, metadata.num_row_groups, 6):
                indices = range(begin, min(begin + 6, metadata.num_row_groups))
                for table in pool.map(read_group, indices):
                    if writer is None:
                        writer = stack.enter_context(
                            pq.ParquetWriter(temporary, table.schema)
                        )
                    writer.write_table(table)
                if begin % 30 == 0:
                    print(
                        f"{destination.name}: {min(begin + 6, metadata.num_row_groups)}/{metadata.num_row_groups} row groups",
                        flush=True,
                    )
        result = {
            **asset,
            "rows": metadata.num_rows,
            "row_groups": metadata.num_row_groups,
            "columns_downloaded": list(COLUMNS),
            "messages_column_downloaded": False,
            "upstream_sha256_verified": False,
            "verification": "Pinned revision and remote size; local projection SHA-256. Full-file LFS hash is provenance only, since only selected columns are downloaded.",
            "projection_sha256": digest(temporary),
            "projection_bytes": temporary.stat().st_size,
        }
        temporary.replace(destination)
        sidecar.write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    args = parser.parse_args()
    spec = json.loads((ROOT / "configs/60traces/history-sources.json").read_text())
    for asset in spec["art-chat"]["assets"]:
        fetch(
            asset, args.bundle / "work/source-inputs/source" / asset["projection_file"]
        )


if __name__ == "__main__":
    main()
