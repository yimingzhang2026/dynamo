#!/usr/bin/env python3
"""Create a reproducible, open-loop RAGPulse-derived Mooncake workload.

The input is component IDs, not original token sequences. This deliberately
models concatenated opaque chunks; it does not claim exact original KV reuse.
All large outputs should be placed on /dev/shm, not the nearly-full root disk.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import heapq
import json
import math
from pathlib import Path
import statistics
import time

DAY = 86400
REVISION = "7da286becf0f049b2bcb1e5a11d9ba8eb638eff4"
SOURCE_DIR = Path("/tmp/nvopt-traffic-research-20260928")
LENGTH_DIR = Path("/dev/shm/nvopt-traffic-stats-20260928")
SOURCES = {
    "0_trace.jsonl": SOURCE_DIR / "ragpulse-trace.jsonl",
    "1_sys_prompt.jsonl": SOURCE_DIR / "ragpulse-sys.jsonl",
    **{f"{i}_{name}.jsonl": LENGTH_DIR / f"ragpulse-{i}.jsonl" for i, name in
       [(2, "passages"), (3, "history"), (4, "user_input"), (5, "web_search")]},
}
# Prefer the preserved upstream bytes bundled next to this script. The second
# candidate also makes the scratch copy reproducible after /tmp is cleared.
# --raw-dir remains an explicit override, and the legacy downloaded paths are
# retained as the last fallback for existing callers.
for _candidate in (
    Path(__file__).resolve().parent / "source",
    Path("/home/hongkuanz/Projects/nvopt-ragpulse-baselines-20260928/traffic/source"),
):
    if all((_candidate / name).is_file() for name in SOURCES):
        SOURCES = {name: _candidate / name for name in SOURCES}
        break


def records(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def build_base(block_size=64):
    """Return first five days, with block IDs consistent across every split."""
    lengths = {}
    for name, path in SOURCES.items():
        if name == "0_trace.jsonl":
            continue
        for row in records(path):
            keys = [k for k in row if k != "token_length"]
            assert len(keys) == 1
            ident, length = row[keys[0]], row["token_length"]
            assert ident not in lengths or lengths[ident] == length
            assert length >= 0
            lengths[ident] = length

    blocks, result = {}, []
    previous_time = -1.0
    for source_row, row in enumerate(records(SOURCES["0_trace.jsonl"])):
        timestamp = float(row["timestamp"])
        assert timestamp >= previous_time, "source trace is not chronological"
        previous_time = timestamp
        if timestamp >= 5 * DAY:
            break
        assert tuple(row["hash_ids"]) == (
            "sys_prompt", "passages_ids", "history", "web_search", "user_input"
        )
        chunks = [(ident, lengths[ident]) for ids in row["hash_ids"].values()
                  for ident in ids]
        residual = row["input_length"] - sum(length for _, length in chunks)
        assert 0 <= residual <= 12
        # Upstream concatenates chunk token lists. Preserve recorded ISL by
        # appending row-specific opaque tokens for the otherwise missing 0-12.
        if residual:
            chunks.append((-(source_row + 1), residual))
        hashes, segments, remaining, parent = [], [], block_size, 0
        for ident, length in chunks:
            offset = 0
            while offset < length:
                take = min(remaining, length - offset)
                segments.append((ident, offset, take))
                offset += take
                remaining -= take
                if remaining == 0:
                    key = (parent, tuple(segments))
                    parent = blocks.setdefault(key, len(blocks) + 1)
                    hashes.append(parent)
                    segments, remaining = [], block_size
        if segments:
            key = (parent, tuple(segments))
            parent = blocks.setdefault(key, len(blocks) + 1)
            hashes.append(parent)
        assert len(hashes) == math.ceil(row["input_length"] / block_size)
        result.append({
            "source_row": source_row,
            "source_timestamp_s": timestamp,
            "source_session_id": row["session_id"],
            "input_length": row["input_length"],
            "output_length": row["output_length"],
            "hash_ids": hashes,
            "unknown_tail_tokens": residual,
        })
    return result, len(blocks) + 1


def prefix_reuse(rows, block_size, seen=None):
    """Full-block contiguous-prefix reuse, infinite cache; inputs only."""
    seen = set() if seen is None else set(seen)
    hit_tokens = total_tokens = 0
    for row in rows:
        total_tokens += row["input_length"]
        full = row["input_length"] // block_size
        for ident in row["hash_ids"][:full]:
            if ident not in seen:
                break
            hit_tokens += block_size
        seen.update(row["hash_ids"][:full])
    return hit_tokens / total_tokens if total_tokens else 0.0, seen


def describe(rows, begin, end, block_size):
    seconds = end - begin
    counts = collections.Counter(int((r["source_timestamp_s"] - begin) // 60)
                                 for r in rows)
    minute_rates = [counts[i] / 60 for i in range(int(seconds // 60))]
    hourly = [sum(begin + h * 3600 <= r["source_timestamp_s"] <
                  begin + (h + 1) * 3600 for r in rows) / 3600
              for h in range(int(seconds // 3600))]
    reuse, _ = prefix_reuse(rows, block_size)
    return {
        "begin_s": begin, "end_s_exclusive": end, "duration_s": seconds,
        "requests": len(rows), "sessions": len({r["source_session_id"] for r in rows}),
        "mean_isl": statistics.mean(r["input_length"] for r in rows),
        "mean_osl": statistics.mean(r["output_length"] for r in rows),
        "mean_rps": len(rows) / seconds,
        "min_rps_60s": min(minute_rates), "max_rps_60s": max(minute_rates),
        "hourly_rps": hourly,
        "theoretical_prefix_reuse_full_blocks_cold": reuse,
        "first_source_arrival_s": rows[0]["source_timestamp_s"],
        "last_source_arrival_s": rows[-1]["source_timestamp_s"],
        "unknown_tail_tokens_histogram": dict(sorted(collections.Counter(
            r["unknown_tail_tokens"] for r in rows).items())),
    }


def phase(copy_id, seed, jitter_s, session_id=None):
    """Stable cohort/session phase, independent of total number of copies."""
    key = f"{seed}:copy:{copy_id}"
    if session_id is not None:
        key += f":session:{session_id}"
    value = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    return jitter_s * (2 * (value + 0.5) / 2**64 - 1)


def arrival(row, copy_id, seed, jitter_s, phase_mode="session"):
    t = row["source_timestamp_s"]
    in_day = t % DAY
    taper_s = max(60.0, 2.0 * jitter_s)
    taper = min(1.0, in_day / taper_s, (DAY - in_day) / taper_s)
    # Each original session remains chronological (derivative >= 0.5), even
    # across the day boundary. Distinct sessions may reorder within 60 seconds.
    session_id = row.get("source_session_id", "boundary-probe") if phase_mode == "session" else None
    return t + phase(copy_id, seed, jitter_s, session_id) * taper


def stream_scaled(rows, stride, copies, begin_s, end_s, jitter_s=30.0, seed=20260928,
                  phase_mode="session"):
    """Merge a bounded temporal halo; emit globally sorted ms timestamps.

    Namespaces are disjoint, while within-copy prefix identities remain exact
    for this opaque-chunk model. Session IDs stay in the source index sidecar:
    passing them to replay would impose turn dependencies absent in this trace.
    """
    assert copies >= 1 and jitter_s >= 0
    assert copies * stride - 1 <= 2**32 - 1, "native static loader requires uint32 hashes"
    assert 0 <= begin_s < end_s <= 5 * DAY
    # Include halo for shifted requests near a user-selected short-window edge.
    selected = [r for r in rows if begin_s - jitter_s <= r["source_timestamp_s"] < end_s + jitter_s]
    def emit(item):
        timestamp, copy_id, index = item
        row = selected[index]
        return {
            "request_id": f"ragpulse:c{copy_id}:r{row['source_row']}",
            "timestamp": round((timestamp - begin_s) * 1000, 6),
            "input_length": row["input_length"],
            "output_length": row["output_length"],
            "hash_ids": [ident + copy_id * stride for ident in row["hash_ids"]],
        }

    assert phase_mode in {"copy", "session"}
    heap = []
    for index, row in enumerate(selected):
        # Future source records can shift backwards by at most jitter_s.
        safe_before = row["source_timestamp_s"] - jitter_s
        while heap and heap[0][0] < safe_before:
            yield emit(heapq.heappop(heap))
        for copy_id in range(copies):
            timestamp = arrival(row, copy_id, seed, jitter_s, phase_mode)
            if begin_s <= timestamp < end_s:
                heapq.heappush(heap, (timestamp, copy_id, index))
    while heap:
        yield emit(heapq.heappop(heap))


def prepare(args, rows, stride):
    args.output.mkdir(parents=True, exist_ok=True)
    days = [describe([r for r in rows if d * DAY <= r["source_timestamp_s"] < (d + 1) * DAY],
                     d * DAY, (d + 1) * DAY, args.block_size) for d in range(5)]
    warmup = [r for r in rows if r["source_timestamp_s"] < 4 * DAY]
    evaluation = [r for r in rows if r["source_timestamp_s"] >= 4 * DAY]
    _, seen = prefix_reuse(warmup, args.block_size)
    warm_reuse, _ = prefix_reuse(evaluation, args.block_size, seen)
    provenance = []
    for name, path in SOURCES.items():
        provenance.append({"source": f"https://github.com/flashserve/RAGPulse/blob/{REVISION}/data/{name}",
                           "local_path": str(path), "bytes": path.stat().st_size,
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    report = {
        "source_revision": REVISION, "source_files": provenance,
        "block_size": args.block_size, "namespace_stride": stride,
        "days": days,
        "warmup": describe(warmup, 0, 4 * DAY, args.block_size),
        "evaluation": describe(evaluation, 4 * DAY, 5 * DAY, args.block_size),
        "eval_theoretical_prefix_reuse_with_4day_input_cache": warm_reuse,
        "five_day_requests": len(rows),
        "timestamp_origin": "trace start; README describes 12:00:00 but does not identify timezone",
        "evaluation_duration_s": DAY,
        "conversion": "ordered opaque chunk slices packed into blocks; IDs chained to preceding prefix; 0-12 unlocated tokens appended as row-unique tail",
        "limitations": ["Original token IDs, separators and their positions are unavailable.",
                        "Reuse within distinct chunk IDs is unknown and deliberately not invented.",
                        "Appending unknown tokens matches recorded lengths but is a synthetic approximation.",
                        "Output tokens cannot be linked to later input chunks from the released metadata.",
                        "Open-loop arrivals: session membership preserved in sidecar; no completion dependencies inferred."],
    }
    (args.output / "split_stats.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output / "source_index.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps({k: v for k, v in row.items() if k != "hash_ids"}, separators=(",", ":")) + "\n")
    with (args.output / "warmup_hourly.jsonl").open("w") as f:
        for hour in range(96):
            selected = [r for r in warmup if hour * 3600 <= r["source_timestamp_s"] < (hour + 1) * 3600]
            f.write(json.dumps({"timestamp_s": hour * 3600, "duration_s": 3600,
                                "requests": len(selected), "request_rate": len(selected) / 3600,
                                "input_tokens": sum(r["input_length"] for r in selected),
                                "output_tokens": sum(r["output_length"] for r in selected)}) + "\n")
    print(json.dumps({"report": str(args.output / "split_stats.json"),
                      "requests": len(rows), "eval_requests": len(evaluation), "stride": stride}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "generate"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--raw-dir", type=Path,
                        help="optional directory with upstream canonical 0_trace.jsonl .. 5_web_search.jsonl names")
    parser.add_argument("--split", choices=["eval", "warmup", "full5"], default="eval")
    parser.add_argument("--copies", type=int, default=1)
    parser.add_argument("--jitter-s", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--phase-mode", choices=["session", "copy"], default="session",
                        help="session avoids a fixed cohort order causing round-robin affinity; copy reproduces v1 pilots")
    parser.add_argument("--window-start-s", type=float, default=0.0,
                        help="relative to selected split; 0 is beginning of eval day, not its first request")
    parser.add_argument("--window-duration-s", type=float)
    args = parser.parse_args()
    assert args.block_size > 0
    if args.raw_dir is not None:
        for name in SOURCES:
            SOURCES[name] = args.raw_dir / name
    start = time.perf_counter()
    rows, stride = build_base(args.block_size)
    if args.command == "prepare":
        prepare(args, rows, stride)
        return
    bounds = {"eval": (4 * DAY, 5 * DAY), "warmup": (0, 4 * DAY), "full5": (0, 5 * DAY)}
    lower, upper = bounds[args.split]
    begin = lower + args.window_start_s
    end = upper if args.window_duration_s is None else begin + args.window_duration_s
    assert lower <= begin < end <= upper
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = input_tokens = 0
    output_sha256 = hashlib.sha256()
    first = last = None
    with args.output.open("w", buffering=1024 * 1024) as f:
        for row in stream_scaled(rows, stride, args.copies, begin, end, args.jitter_s, args.seed, args.phase_mode):
            if first is None:
                first = row["timestamp"]
            last = row["timestamp"]
            line = json.dumps(row, separators=(",", ":")) + "\n"
            f.write(line)
            output_sha256.update(line.encode())
            count += 1
            input_tokens += row["input_length"]
    elapsed = time.perf_counter() - start
    manifest = {"output": str(args.output), "requests": count, "bytes": args.output.stat().st_size,
                "sha256": output_sha256.hexdigest(),
                "source_revision": REVISION, "block_size": args.block_size, "namespace_stride": stride,
                "copies": args.copies, "jitter_s": args.jitter_s, "seed": args.seed,
                "phase_mode": args.phase_mode,
                "source_window_start_s": begin, "source_window_end_s_exclusive": end,
                "evaluation_duration_s": end - begin, "mean_rps": count / (end - begin),
                "first_arrival_ms": first, "last_arrival_ms": last, "generation_wall_seconds": elapsed,
                "session_index": "source_index.jsonl; request_id encodes copy and source row; namespace is (copy_id, source_session_id)",
                "copies_have_disjoint_hashes": True,
                "native_u32_token_storage_lower_bound_bytes": input_tokens * 4}
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
