#!/usr/bin/env bash
#SBATCH --job-name=planner-knowline
#SBATCH --exclusive
#SBATCH --mem=1900G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
set -euo pipefail
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export RAYON_NUM_THREADS=1 TOKIO_WORKER_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset TYPESAFE_API_KEY
gym_dir="${PLANNER_GYM:?Set PLANNER_GYM to the absolute Planner Gym directory}"
shard=${1:-${SLURM_ARRAY_TASK_ID:?Expected array task or explicit shard}}
cd "$gym_dir"
exec "${PLANNER_PYTHON:-$gym_dir/.venv/bin/python}" "$gym_dir/scripts/run_campaign.py" worker \
  --workspace "${PLANNER_CAMPAIGN_WORKSPACE:?Set the campaign workspace}" \
  --shard "$shard" --round "${2:-1}"
