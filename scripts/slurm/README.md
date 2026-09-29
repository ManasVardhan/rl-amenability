# CARC launch sequence for the transfer batch

All commands from the repository root on a CARC Discovery login node. The account
is `anakano_429`, which is the default, so no `--account` flag is needed.

`$SCRATCH` is not set on CARC login nodes. The scripts default the scratch
directory to `${SCRATCH:-/scratch1/$USER}` and put the Hugging Face cache at
`$HF_HOME`, default `<scratch>/hf`. Set `SCRATCH` or `HF_HOME` to override.

## Before the first run

This is the setup that worked on CARC Discovery. Do not run `uv sync` on the
login node: its per-user thread cap makes uv's thread pool crash with EAGAIN.

1. Install uv and put it on PATH in `~/.bash_profile`. If `~/.bash_profile`
   does not exist, create it. If you also have `~/.profile` or `~/.bashrc`, add
   `[ -f ~/.bashrc ] && . ~/.bashrc` (or `. ~/.profile`) to it, since bash
   reads only `~/.bash_profile` when it exists:

       curl -LsSf https://astral.sh/uv/install.sh | sh
       echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bash_profile
       source ~/.bash_profile && which uv

   The launchers prepend `$HOME/.local/bin` to PATH themselves, but
   `make_transfer_jobs` and `hf` run from your shell.

2. Confirm the scratch path exists and point uv's cache at it:

       ls -d /scratch1/$USER
       echo 'export UV_CACHE_DIR=/scratch1/$USER/uv-cache' >> ~/.bash_profile && source ~/.bash_profile

3. Create the venv on uv-managed Python. The default `python` is a spack
   module, and the jobs run `module purge`, so a venv built on it breaks:

       uv python install 3.11
       uv venv --python 3.11 --python-preference only-managed

4. Sync on a compute node with capped concurrency (run from the repo root;
   `$PWD` expands on the login node):

       srun --partition=main --cpus-per-task=4 --mem=16G --time=01:00:00 --pty bash -lc "cd $PWD && UV_CONCURRENT_DOWNLOADS=4 UV_CONCURRENT_INSTALLS=2 UV_CONCURRENT_BUILDS=1 RAYON_NUM_THREADS=4 uv sync --extra dev --extra gpu"

5. Gated models: Llama-3.2-1B and gemma-3-1b-pt need access requested on their
   Hugging Face model pages, with the account behind your token, or the
   prefetch fails with 403 GatedRepoError. Check which account that is:

       uv run --no-sync hf auth whoami

6. Confirm `uv` survives `module purge` on a compute node, exactly as
   `probe_array.sbatch` runs it (it purges modules before `uv run`):

       srun --partition=main --time=00:05:00 --pty bash -lc 'module purge; which uv && uv --version'

   If `which uv` prints nothing after the purge, `uv` came from a module and
   every array task would fail with exit 127. Either install uv to
   `~/.local/bin` with the installer (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
   or remove `module purge` from `probe_array.sbatch`, and record which one in
   `docs/decisions/transfer-rulings.md`.

7. After section 1 below (the prefetch syncs the `gpu` extra), confirm the synced
   environment imports torch and vllm after the purge. Run this from the
   repository root; `$PWD` expands on the login node to the repo path:

       srun --partition=main --time=00:10:00 --pty bash -lc "cd $PWD && module purge && uv run python -c 'import torch, vllm; print(torch.__version__)'"

8. Only if the pilot (section 2) or the smoke probes (section 3) fail with a CUDA
   library error: run `module avail cuda` and add the matching `module load`
   line after `module purge` in `probe_array.sbatch`. The scripts load no CUDA
   module on purpose: the pip torch and vllm wheels bundle the CUDA runtime and
   only need the driver.

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

The pilot alone may take any bf16-capable GPU to shorten the queue wait
(transfer-rulings T18). Override the script's A100 request on the sbatch line:

    JOBS_FILE=results/transfer/jobs_pilot.txt EXTRA_ARGS=--pre-only \
      sbatch --gres=gpu:1 --constraint="a100|l40s|a40" --array=0-19%8 scripts/slurm/probe_array.sbatch

Never v100 or p100 (no bf16). The smoke probes and the batch stay on A100, so
every model's training protocol runs on the same hardware.

## 3. Smoke probes (2 GPU jobs)

    uv run python -m scripts.make_transfer_jobs --only-model smollm2-1.7b qwen2.5-0.5b --out results/transfer/jobs_smoke.txt
    JOBS_FILE=results/transfer/jobs_smoke.txt sbatch --array=0,1 scripts/slurm/probe_array.sbatch
    # lines 0 and 1 are the countdown seed-0 jobs for the two models.
    # Then read results/transfer/meta/*.json: peak_train_bytes and wall_seconds.

## 4. The batch (30 GPU jobs, ~1 h each)

    uv run python -m scripts.make_transfer_jobs
    sbatch --array=0-29%8 scripts/slurm/probe_array.sbatch

If section 3 showed `peak_train_bytes` above 36e9 for either model, a 40 GB A100 is
too tight; request the 80 GB nodes instead:

    sbatch --constraint=a100-80gb --array=0-29%8 scripts/slurm/probe_array.sbatch

Jobs whose telemetry already exists (the smoke probes) exit in seconds.

## Monitoring and reruns

    squeue -u $USER
    # tasks that exit within seconds: check this first (uv missing, or the
    # manifest was never generated)
    grep -l "command not found\|job list not found" logs/probe_*.out logs/prefetch_*.out
    grep -l Traceback logs/probe_*.out
    # rerun failed batch tasks by id (keep --constraint if section 4 used it):
    sbatch --array=4,17 scripts/slurm/probe_array.sbatch

If tasks fail at vLLM engine warmup with `Could not find nvcc` raised from
flashinfer: the launchers export `VLLM_USE_FLASHINFER_SAMPLER=0`
(transfer-rulings T17) so the sampler never JIT-compiles, which means some other
FlashInfer kernel is JIT-compiling. The fallback is to point `CUDA_HOME` at a
CUDA toolkit (a CARC cuda module from `module avail cuda`, or the venv's
`nvidia-cuda-nvcc` package) and record a ruling in
`docs/decisions/transfer-rulings.md`.

    grep -l "Could not find nvcc" logs/probe_*.out

Task ids index the manifest the job was submitted with. To rerun a pilot or
smoke task, repeat the original submission's `JOBS_FILE=` and `EXTRA_ARGS=`
prefix, otherwise the ids index the batch manifest and run a different job:

    JOBS_FILE=results/transfer/jobs_pilot.txt EXTRA_ARGS=--pre-only \
      sbatch --array=4,17 scripts/slurm/probe_array.sbatch

## 5. Analysis (CPU, seconds)

    uv run python -m scripts.analyze_transfer --work-dir results/transfer
    # writes results/transfer/report.md, report.json, leaderboard_preview.md

## Dry run (any machine, no GPU, SCRATCH need not be set)

    uv run python -m scripts.make_transfer_jobs --out /tmp/jobs.txt
    SLURM_ARRAY_TASK_ID=3 DRY_RUN=1 JOBS_FILE=/tmp/jobs.txt bash scripts/slurm/probe_array.sbatch
