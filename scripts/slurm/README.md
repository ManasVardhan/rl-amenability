# CARC launch sequence for the transfer batch

All commands from the repository root on a CARC Discovery login node. The account
is `anakano_429`, which is the default, so no `--account` flag is needed.

`$SCRATCH` is not set on CARC login nodes. The scripts default the scratch
directory to `${SCRATCH:-/scratch1/$USER}` and put the Hugging Face cache at
`$HF_HOME`, default `<scratch>/hf`. Set `SCRATCH` or `HF_HOME` to override.

## Before the first run

1. Confirm the scratch path exists: `ls -d /scratch1/$USER`.
2. Confirm `uv` is on the PATH of a compute node, not only the login node:

       srun --partition=main --time=00:05:00 --pty bash
       which uv
       exit

3. Only if the smoke probe (step 3) fails with a CUDA library error: run
   `module avail cuda` and add the matching `module load` line after
   `module purge` in `probe_array.sbatch`. The scripts load no CUDA module on
   purpose: the pip torch and vllm wheels bundle the CUDA runtime and only need
   the driver.

## Once

    export HF_HOME=${SCRATCH:-/scratch1/$USER}/hf
    ls -d /scratch1/$USER           # must print the directory
    uv run hf auth login            # the account that accepted the Llama 3.2 and Gemma 3 licences
    mkdir -p logs

## 1. Prefetch (CPU, ~30 min)

    sbatch scripts/slurm/prefetch.sbatch
    # wait; then:
    grep -c '"ok": true' results/transfer/prefetch.json    # must print 10

## 2. Calibration pilot (20 GPU jobs, pre-eval only, ~10 min each)

    uv run python -m scripts.make_transfer_jobs --pilot
    JOBS_FILE=results/transfer/jobs_pilot.txt EXTRA_ARGS=--pre-only \
      sbatch --array=0-19%8 scripts/slurm/probe_array.sbatch
    # results land in results/transfer/preeval/

## 3. Smoke probes (2 GPU jobs)

    uv run python -m scripts.make_transfer_jobs --only-model smollm2-1.7b qwen2.5-0.5b --out results/transfer/jobs_smoke.txt
    JOBS_FILE=results/transfer/jobs_smoke.txt sbatch --array=0,1 scripts/slurm/probe_array.sbatch
    # lines 0 and 1 are the countdown seed-0 jobs for the two models.
    # Then read results/transfer/meta/*.json: peak_train_bytes and wall_seconds.

## 4. The batch (30 GPU jobs, ~1 h each)

    uv run python -m scripts.make_transfer_jobs
    sbatch --array=0-29%8 scripts/slurm/probe_array.sbatch

If step 3 showed `peak_train_bytes` above 36e9 for either model, a 40 GB A100 is
too tight; request the 80 GB nodes instead:

    sbatch --constraint=a100-80gb --array=0-29%8 scripts/slurm/probe_array.sbatch

Jobs whose telemetry already exists (the smoke probes) exit in seconds.

## Monitoring and reruns

    squeue -u $USER
    grep -l Traceback logs/probe_*.out
    # rerun failed tasks by id (keep --constraint if step 4 used it):
    sbatch --array=4,17 scripts/slurm/probe_array.sbatch

## 5. Analysis (CPU, seconds)

    uv run python -m scripts.analyze_transfer --work-dir results/transfer
    # writes results/transfer/report.md, report.json, leaderboard_preview.md

## Dry run (any machine, no GPU, SCRATCH need not be set)

    SLURM_ARRAY_TASK_ID=3 DRY_RUN=1 bash scripts/slurm/probe_array.sbatch
