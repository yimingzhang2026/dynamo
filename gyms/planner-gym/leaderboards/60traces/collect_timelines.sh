#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Reduces the saved telemetry of the selected results on the machine that holds
# the raw results, brings the series back and rebuilds the pages. No simulation
# is started; the remote side only reads, and writes into its own directory.
#
#   bash collect_timelines.sh HOST REMOTE_PYTHON SEARCH_DIR [SEARCH_DIR...]
#   bash collect_timelines.sh user@replay-host /opt/planner/runtime/bin/python /opt/planner
#
# TIMELINE_PYTHON     local interpreter (default: python3)
# TIMELINE_REMOTE_DIR remote working directory (default: timeline-extract in the home directory)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd -P)"
host="${1:?Usage: bash collect_timelines.sh HOST REMOTE_PYTHON SEARCH_DIR [SEARCH_DIR...]}"
python="${2:?Give the Python interpreter on the remote machine}"
shift 2
[[ $# -gt 0 ]] || { echo "Give at least one remote directory to search" >&2; exit 2; }
remote="${TIMELINE_REMOTE_DIR:-timeline-extract}"
for value in "$python" "$remote" "$@"; do
    [[ "$value" =~ ^[a-zA-Z0-9_./~-]+$ ]] || { echo "Remote paths must contain no shell metacharacters: $value" >&2; exit 2; }
done
local_python="${TIMELINE_PYTHON:-python3}"
ssh_options=(-o ConnectTimeout=20 -o ServerAliveInterval=30)
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

"$local_python" "$here/extract_timelines.py" manifest --output "$work/timeline-manifest.json"
ssh "${ssh_options[@]}" "$host" "mkdir -p $remote"
rsync -tz "$here/extract_timelines.py" "$work/timeline-manifest.json" "$host:$remote/"
# Reduced runs stay on the remote side, so repeating this skips them.
ssh "${ssh_options[@]}" "$host" "cd $remote && $python extract_timelines.py reduce --manifest timeline-manifest.json --search $* --output timelines"
rsync -rtz "$host:$remote/timelines/" "$work/timelines/"
"$local_python" "$here/extract_timelines.py" import "$work/timelines"
"$local_python" "$here/build_leaderboards.py"
