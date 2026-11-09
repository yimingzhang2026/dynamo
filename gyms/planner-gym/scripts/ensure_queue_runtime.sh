#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 DYNAMO_SOURCE PYTHON BUILD_DIR [BUILD_JOBS]" >&2
  exit 2
fi
scripts=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
if env -u PYTHONPATH "$2" "$scripts/install_queue_runtime.py" --check 2>/dev/null; then
  exit 0
fi
library="$3/overlay/dynamo/_core.abi3.so"
if [[ -f "$library" ]] && env -u PYTHONPATH "$2" "$scripts/install_queue_runtime.py" --library "$library"; then
  exit 0
fi
bash "$scripts/build_queue_runtime.sh" "$@"
env -u PYTHONPATH "$2" "$scripts/install_queue_runtime.py" --library "$library"
