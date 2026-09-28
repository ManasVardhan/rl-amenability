#!/bin/bash
#SBATCH --job-name=amenability-stage0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/stage0_%j.out

set -euo pipefail
module purge
# No `module load cuda` (transfer-rulings T14): the pip torch and vllm wheels
# bundle the CUDA runtime and only need the driver.
cd "$SLURM_SUBMIT_DIR"

# vllm lives in the "gpu" extra, not the base dependencies (it broke `uv sync` on
# non-Linux dev machines when it was a base dependency), so a GPU run needs BOTH
# extras installed. Do not "simplify" this back to a single extra.
uv sync --extra dev --extra gpu

uv run python -m scripts.run_stage0 --out "results/stage0_report.md"
