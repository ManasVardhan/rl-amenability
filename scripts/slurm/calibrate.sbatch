#!/bin/bash
# scripts/slurm/calibrate.sbatch
# Difficulty calibration (transfer-rulings T22, T23): one model per array task, every
# difficulty candidate, 8 samples per item with the GRPO rollout sampling.
# Task i runs ROSTER[i] of scripts/calibrate_difficulty.py (9 models). Launch:
#   sbatch --array=0-8 scripts/slurm/calibrate.sbatch
# Sampling only, no training, so any bf16 GPU will do (as for the pilot, T18);
# never v100 or p100. Results land in results/calibration/<model>.json, written
# after each candidate; a rerun of a task resumes from the cached candidates,
# except candidates written before T23 (no per-item category counts, so no
# shaped group signal), which it recomputes with the same seeds.
#SBATCH --job-name=amen-calib
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|l40s|a40
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/calibrate_%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

OUT_DIR="${OUT_DIR:-results/calibration}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is unset: launch with --array=0-8}"

# Same environment as probe_array.sbatch: weights from the scratch cache, never
# the Hub, and vLLM's PyTorch sampler instead of FlashInfer's (T17).
SCRATCH_DIR="${SCRATCH:-/scratch1/${USER:-$(id -un)}}"
HF_HOME_SET_BY_USER="${HF_HOME:+yes}"
export HF_HOME="${HF_HOME:-$SCRATCH_DIR/hf}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0

cmd=(uv run python -m scripts.calibrate_difficulty --model-index "$TASK_ID" --out-dir "$OUT_DIR")
if [ -n "$EXTRA_ARGS" ]; then
  # shellcheck disable=SC2206
  cmd+=($EXTRA_ARGS)
fi

if [ -n "${DRY_RUN:-}" ]; then
  echo "[dry run] task $TASK_ID: ${cmd[*]}"
  exit 0
fi

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv: command not found (searched PATH and \$HOME/.local/bin): install it with curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi
if [ -z "$HF_HOME_SET_BY_USER" ] && [ ! -d "$SCRATCH_DIR" ]; then
  echo "scratch directory $SCRATCH_DIR does not exist: set SCRATCH or HF_HOME" >&2
  exit 1
fi
if [ ! -d "$HF_HOME" ]; then
  echo "HF_HOME=$HF_HOME does not exist: run scripts/slurm/prefetch.sbatch first" >&2
  exit 1
fi

# No `module load cuda` (T14); module purge keeps the login node's modules out.
module purge
export PATH="$HOME/.local/bin:$PATH"
mkdir -p logs
echo "task $TASK_ID on $(hostname): ${cmd[*]}"
"${cmd[@]}"
