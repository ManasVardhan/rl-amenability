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

## 2. Calibration pilot (20 GPU jobs, pre-eval only, ~30 min each)

Measured on the first pilot: about 30 min per pre-eval job (300 items x 64
samples = 19,200 generations at about 12 per second on an A100 40GB), not the
10 min first estimated.

Before rerunning the pilot after the prompt change (transfer-rulings T19), delete
every pre-eval result made with the old prompts. The pipeline skips any job
whose output file exists, so a stale file is silently reused:

    rm results/transfer/preeval/falcon3-1b-base-countdown.json
    ls results/transfer/preeval/    # must list nothing from before T19


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

`probe_array.sbatch` requests an 80 GB A100 by default (`--constraint=a100-80gb`,
transfer-rulings T22). The first smoke run (job 12633929) used the 40 GB cards,
where SmolLM2-1.7B ran out of memory at step 1. Its outputs used the 512-token
completion cap and the old buckets, so delete them before any rerun or the
batch; the pipeline skips every job whose telemetry exists:

    rm -f results/transfer/meta/*.json results/transfer/manifests/*.json results/transfer/telemetry/*.json
    rm -rf results/transfer/probes/*

Then:

    uv run python -m scripts.make_transfer_jobs --only-model smollm2-1.7b qwen2.5-0.5b --out results/transfer/jobs_smoke.txt
    JOBS_FILE=results/transfer/jobs_smoke.txt sbatch --array=0,1 scripts/slurm/probe_array.sbatch
    # lines 0 and 1 are the countdown seed-0 jobs for the two models.
    # Then read results/transfer/meta/*.json: peak_train_bytes_excl_vllm_pool and wall_seconds.

Do not add a `--constraint` to this line: the script's a100-80gb is the
training hardware for every model.

GRPO generates its rollouts with a vLLM engine colocated with the trainer, in
sleep mode, stopping at the first `</answer>` (transfer-rulings T21), with a
768-token cap (T22). For each smoke job, check:

    # the stop works in training: a few characters at most after the first
    # </answer> (a loop would show hundreds), and most completions carry it
    uv run --no-sync python -c "import json,glob; [print(f, m['generation_backend'], m['generation_stop'], m['max_completion_length'], m['completion_stats']) for f in glob.glob('results/transfer/manifests/*.json') for m in [json.load(open(f))]]"
    # generation_backend must be vllm_colocate; generation_stop must be </answer>;
    # max_completion_length must be 768
    # the colocated engine and the trainer were released before the post-eval
    # (issue #33): allocated_after_train_bytes near zero, under 1e9
    grep -h allocated_after_train_bytes results/transfer/meta/*.json
    # no engine start-up failure and no traceback
    grep -l "less than desired GPU memory utilization\|Traceback" logs/probe_*.out

TRL's per-step log lines should show `completions/mean_length` well under 768.
Its `completions/clipped_ratio` reads near 1.0 on the probe suites and is not a
failure: it counts every completion that does not end in EOS, and one ended by
the stop string does not.

Memory figures (T22): `peak_train_bytes` is torch's `max_memory_allocated`
over training, and it over-reports. vLLM's sleep mode releases the engine's
physical memory but torch still counts the engine's pool as allocated, so the
raw figure is the trainer's peak plus the whole pool (Gemma-3-1B reported
45.6e9 on a 42.4e9-byte card). Read `peak_train_bytes_excl_vllm_pool` (raw
peak minus `vllm_sleep_pool_bytes`) as the trainer-phase peak. If the job dies
with a CuMemAllocator or expandable-segments error, check that
`PYTORCH_CUDA_ALLOC_CONF` is unset: vLLM's sleep mode cannot run with
`expandable_segments:True`.

## 4. The batch (30 GPU jobs, over 1 h each)

Do not launch before the difficulty calibration (section 6) has selected the
training-pool candidates under the T22 selection rule.

Each batch job runs two pre-eval-sized evaluations (pre and post, about 30 min
each at the pilot's measured rate) plus 60 GRPO steps, so expect well over the
first estimate of 1 h. The array requests 3 h; read the smoke probes'
`wall_seconds` before submitting and raise `--time` if they come close.

    uv run python -m scripts.make_transfer_jobs
    sbatch --array=0-29%8 scripts/slurm/probe_array.sbatch

Every batch job runs on an 80 GB A100 (the script's default constraint, T22);
there is no 40 GB variant of the batch. Jobs whose telemetry already exists
(the smoke probes) exit in seconds.

## Monitoring and reruns

    squeue -u $USER
    # tasks that exit within seconds: check this first (uv missing, or the
    # manifest was never generated)
    grep -l "command not found\|job list not found" logs/probe_*.out logs/prefetch_*.out
    grep -l Traceback logs/probe_*.out
    # rerun failed batch tasks by id (the script keeps the 80 GB constraint):
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
