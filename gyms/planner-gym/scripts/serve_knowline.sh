#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 MODEL_DIRECTORY [SERVING_PYTHON [PORT]]" >&2
  exit 2
fi
model_dir="$(cd "$1" && pwd -P)"
serve_python="${2:-python3}"
port="${3:-18080}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
exec "$serve_python" "$model_dir/knowline_server.py" \
  --model "$model_dir" --backend hf --temperature 1 --workers 1 \
  --host 127.0.0.1 --port "$port" --temperatures "$model_dir/temperatures.json" \
  --box-prompt gemma --box-coords yxyx
