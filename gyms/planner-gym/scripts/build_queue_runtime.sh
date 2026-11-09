#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# CPU-only native build; all traces/planners share the vendored scheduler.
set -euo pipefail
if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 DYNAMO_SOURCE PYTHON OUTPUT_DIR [BUILD_JOBS]" >&2
  exit 2
fi
source_dir=$(cd "$1" && pwd -P)
python_bin=$2
mkdir -p "$3"
build_dir=$(cd "$3" && pwd -P)
export CARGO_BUILD_JOBS=${4:-4}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$build_dir/target}
export PYO3_PYTHON=$python_bin
test -f "$source_dir/vendor/aisimulate-core/src/engine/scheduler/vllm/core.rs"
mkdir -p "$build_dir/source"
if [[ "$source_dir" != "$build_dir/source" ]]; then
  tar -C "$source_dir" -cf - Cargo.toml Cargo.lock rust-toolchain.toml lib .cargo vendor |
    tar -C "$build_dir/source" -xf -
fi
"$python_bin" - "$build_dir/source" <<'PY'
from pathlib import Path
import sys
root=Path(sys.argv[1])
manifest=root/'lib/bindings/python/Cargo.toml'
text=manifest.read_text().replace('dynamo-llm = { path = "../../llm" }',
    'dynamo-llm = { path = "../../llm", default-features = false }')
manifest.write_text(text)
PY
cd "$build_dir/source"
cargo test --release --manifest-path vendor/aisimulate-core/Cargo.toml \
  --lib engine::scheduler::vllm
cargo build --release --manifest-path lib/bindings/python/Cargo.toml \
  --lib --features ais-forward-pass --no-default-features
mkdir -p "$build_dir/overlay/dynamo"
cp "$CARGO_TARGET_DIR/release/lib_core.so" "$build_dir/overlay/dynamo/_core.abi3.so.next"
mv "$build_dir/overlay/dynamo/_core.abi3.so.next" "$build_dir/overlay/dynamo/_core.abi3.so"
env PYTHONPATH="$build_dir/overlay" "$python_bin" - "$build_dir" <<'PY'
import hashlib,json,platform,sys
from pathlib import Path
from dynamo import _core
base=Path(sys.argv[1])
assert _core.REPLAY_QUEUE_STATISTICS_REVISION=='incremental-queue-stats-v1'
def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
record={'revision':_core.REPLAY_QUEUE_STATISTICS_REVISION,
    'library':_core.__file__,'binary_sha256':sha(Path(_core.__file__)),
    'queue_source_sha256':sha(base/'source/vendor/aisimulate-core/src/engine/scheduler/vllm/core.rs'),
    'architecture':platform.machine(),'libc':platform.libc_ver(),
    'features':['ais-forward-pass'],'cpu_only':True}
(base/'build-manifest.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
PY
