#!/bin/bash
# scripts/slurm/probe_array.sbatch
# One probe job per array task. Line (SLURM_ARRAY_TASK_ID + 1) of $JOBS_FILE is
# "model suite seed". Launch with, e.g.:
#   sbatch --array=0-29%8 scripts/slurm/probe_array.sbatch
# Training jobs run on an 80 GB A100 (transfer-rulings T22): SmolLM2-1.7B ran
# out of memory at step 1 on a 40 GB card in the smoke probes. CARC feature
# names are a100-40gb and a100-80gb. A pre-eval-only job (EXTRA_ARGS=--pre-only)
# may override both lines on the sbatch command line, e.g.
#   --gres=gpu:1 --constraint="a100|l40s|a40"
# (see scripts/slurm/README.md, section 2). No --account line: anakano_429 is the
# default account.
#SBATCH --job-name=amen-probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/probe_%A_%a.out

set -euo pipefail
# Resolve JOBS_FILE, WORK_DIR and the log directory against the submit dir.
cd "${SLURM_SUBMIT_DIR:-.}"

JOBS_FILE="${JOBS_FILE:-results/transfer/jobs.txt}"
WORK_DIR="${WORK_DIR:-results/transfer}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is unset: launch with --array}"

if [ ! -f "$JOBS_FILE" ]; then
  echo "job list not found: $JOBS_FILE (generate it with uv run --no-sync python -m scripts.make_transfer_jobs, from a shell where uv works)" >&2
  exit 1
fi
line=$(sed -n "$((TASK_ID + 1))p" "$JOBS_FILE")
if [ -z "$line" ]; then
  echo "no job on line $((TASK_ID + 1)) of $JOBS_FILE" >&2
  exit 1
fi
read -r MODEL SUITE SEED <<< "$line"

# Weights live on scratch (the prefetch put them there) and the array must never
# reach out to the Hub: thirty concurrent downloads on a shared cluster is how a
# batch dies of rate limiting. $SCRATCH is unset on CARC login nodes, so default
# to the personal scratch directory rather than requiring it.
SCRATCH_DIR="${SCRATCH:-/scratch1/${USER:-$(id -un)}}"
HF_HOME_SET_BY_USER="${HF_HOME:+yes}"
export HF_HOME="${HF_HOME:-$SCRATCH_DIR/hf}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# FlashInfer JIT-compiles its top-p sampler and needs nvcc, which CARC compute
# nodes lack. vLLM's built-in PyTorch sampler draws from the same temperature /
# top-p distribution, applied identically to every model, so the protocol is
# unchanged (transfer-rulings T17).
export VLLM_USE_FLASHINFER_SAMPLER=0

cmd=(uv run python -m scripts.run_probe --model "$MODEL" --suite "$SUITE" --run-seed "$SEED" --work-dir "$WORK_DIR")
if [ -n "$EXTRA_ARGS" ]; then
  # shellcheck disable=SC2206
  cmd+=($EXTRA_ARGS)
fi

if [ -n "${DRY_RUN:-}" ]; then
  echo "[dry run] task $TASK_ID: ${cmd[*]}"
  exit 0
fi

# SLURM jobs inherit the submitting shell's PATH, and a shell without
# ~/.local/bin on it (where the uv installer puts uv) kills every task with
# "uv: command not found". Do not depend on it. Checked after the dry-run exit
# so a laptop dry run without uv still works.
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

# No `module load cuda`: the pip torch and vllm wheels bundle the CUDA runtime
# and only need the driver, and a mismatched toolkit module can shadow the
# bundled libraries. module purge keeps the login node's modules out.
module purge
# module purge leaves ~/.local/bin alone; re-prepending is harmless insurance.
export PATH="$HOME/.local/bin:$PATH"
mkdir -p logs
echo "task $TASK_ID on $(hostname): ${cmd[*]}"
"${cmd[@]}"
