#!/bin/bash
# scripts/slurm/prefetch.sbatch
# CPU-only: sync the environment and download every roster model to scratch once.
#SBATCH --job-name=amen-prefetch
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/prefetch_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
# SLURM jobs inherit the submitting shell's PATH, and a shell without
# ~/.local/bin on it (where the uv installer puts uv) fails with
# "uv: command not found". Do not depend on it.
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv: command not found (searched PATH and \$HOME/.local/bin): install it with curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi
# $SCRATCH is unset on CARC login nodes, so default to the personal scratch
# directory rather than requiring it.
SCRATCH_DIR="${SCRATCH:-/scratch1/${USER:-$(id -un)}}"
if [ -z "${HF_HOME:-}" ] && [ ! -d "$SCRATCH_DIR" ]; then
  echo "scratch directory $SCRATCH_DIR does not exist: set SCRATCH or HF_HOME" >&2
  exit 1
fi
export HF_HOME="${HF_HOME:-$SCRATCH_DIR/hf}"
mkdir -p logs "$HF_HOME"

# No `module load cuda`: this node never touches a GPU, and on the GPU nodes the
# pip torch and vllm wheels bundle the CUDA runtime and only need the driver.

# vllm lives in the "gpu" extra, not the base dependencies, so a GPU run needs
# BOTH extras installed. Syncing here, on the CPU node, means the array tasks
# find a ready environment. Do not "simplify" this back to a single extra.
uv sync --extra dev --extra gpu

uv run python -m scripts.prefetch_models --out results/transfer/prefetch.json
