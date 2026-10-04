# Probe Transfer Execution Rulings

Decisions taken during execution of
`docs/superpowers/plans/2026-09-27-probe-transfer.md`, in the order they were
made, in the format of `stage0-rulings.md`: what was decided, why, and the cost if
the decision turns out to be wrong. Numbered T1, T2, ... to keep them distinct from
the Stage 0 rulings R1 to R38.

**T1. Graph path buckets are (8, 10, 12) nodes with n // 3 extra edges and minimum
distance 3, not the spec's first draft of (6, 8, 10).**
A design-time simulation of an informed guesser (starts at the source, ends at the
target, random distinct middle) gave pass@32 of 0.94 / 0.62 / 0.34 at (6, 8, 10)
with n // 2 extra edges and distance >= 2, so the easy bucket's breadth would have
been mostly luck. At (8, 10, 12), n // 3, distance >= 3 the implemented guesser
measures 0.21 / 0.10 / 0.03 (mean 0.113; 300 items, seed 0, k = 32), and
`test_informed_guesser_stays_well_below_saturation` holds the bound. The design
simulation's 0.20 / 0.06 / 0.01 was optimistic at 10 and 12 nodes; the measured
rates are still inside the bound (< 0.3 per bucket, < 0.15 on average). Made before
any model was run.
*Cost if wrong:* the suite may be too hard at 0.5B, in which case the calibration
pilot shows pass@32 near zero on every bucket and the bound contingency is
(7, 9, 11), where the easiest bucket guesses at 0.37.

**T2. Task 1's trainer-release test runs with the cyclic collector disabled and no
gc.collect().**
The plan's test called `gc.collect()` before checking the trainer weakref. But once
`run_grpo` returns, the trainer survives only inside a reference cycle (trainer ->
callbacks -> TelemetryCallback -> model_getter closure -> trainer), and
`gc.collect()` frees exactly that kind of cycle. The planned test therefore passed
before the fix and could not demonstrate anything. The test now disables the
collector and asserts the trainer is gone the moment `run_grpo` returns; it failed
before the fix and passes after it. The implementation comment was also corrected
to name the cycle, instead of claiming that `del trainer` frees nothing. One
residual gap is known: the explicit `gc.collect()` inside `_release_gpu` would free
the cycle by itself, so no test pins the closure break separately.
*Cost if wrong:* one test is stricter than the plan asked for; no behaviour change.

**T3. Code documentation states the measured guessability, not the design
simulation.**
Task 2 measured the implemented guesser, and the numbers differ from the design
simulation at 10 and 12 nodes (see T1). The `graphpath` module docstring and T1
both carry the measured 0.21 / 0.10 / 0.03, so a reader never meets two sets of
figures for the same bound.
*Cost if wrong:* wording only.

**T4. The Graph suite's guessability bound covers edge-blind guessing only; breadth
saturation is detected at the pilot rather than designed out now.**
Task 2's review measured two stronger guessers. One that ignores the edges but only
guesses paths of length 4 or 5 scores pass@32 of 0.50 / 0.24 / 0.23. A random walk
along the listed edges with no revisits saturates pass@32 at 1.0 in every bucket.
So for any model that can read the edge list, Graph breadth (pass@32 minus pass@1)
may saturate, which makes the conversion denominator effectively 1 minus pass@1.
No generator change is made before any model has been run: whether real models
behave like the edge-following walker is an empirical question the pilot answers
cheaply. Instead the analysis adds a per-(model, suite) `breadth_saturated` flag
(pre-probe pass@32 > 0.95 in every bucket) beside the existing floored and
saturated flags, and in Task 10 step 3, Graph breadth saturated for 3 or more
models is a STOP-and-rule condition, like the floored case. The candidate remedy
is larger buckets. `prereg/transfer.md` records the flag and the stop condition.
*Cost if wrong:* one extra column, and possibly a bucket change after the pilot at
about 1 GPU-hour.

**T5. Probe/target disjointness on the transfer path is structural, not checked at
runtime.**
`run_probe` does not register suites in a `SuiteRegistry`. The probe can only reach
suites through the catalog, which excludes target suites by design. Graph task IDs
have the form `probe/graphpath/...` and are registered alongside Countdown in the
suite's own tests, and the transfer batch never loads GSM8K. Registering at runtime
would mean downloading GSM8K in every probe job for no protective gain.
*Cost if wrong:* a future catalog suite that reads target data would not be caught
at runtime; the catalog's docstring is the guard.

**T6. The per-job CLI refuses non-protocol flags in the batch directory.**
The telemetry cache key is (model, suite, run seed); it ignores probe steps,
learning rate and item seed. A smoke run with `--steps 2`, a recalibration run
with `--lr 3e-6`, or a run with `--item-seed 1`, written into the default
`results/transfer`, would later be reused by the batch as if it were the 60-step
protocol run. `scripts/run_probe.py` now exits with status 2 when any of those
flags is non-default and `--work-dir` is the default batch directory. The guard
lives in the CLI rather than in the cache key so that `run_probe`'s key and Stage
0's key stay unchanged.
*Cost if wrong:* a direct library call to `run_probe` with a custom config and the
batch directory is still unguarded. After an lr recalibration (spec section 8),
the old telemetry in `results/transfer` is stale and must be rerun with `--force`,
which Task 10 step 4 of the plan already requires.

The rulings T7 to T11 were made on 2026-09-28 in response to the whole-branch
review, before any batch data existed. T7 to T10 amend `prereg/transfer.md`.

**T7. The test-retest pair has the same completeness floor as the transfer pair.**
The decision rule gated only the two seed-0 arms at 8 of 10 models. The Countdown
replicate had no floor, so a batch in which only three seed-1 jobs finished would
still produce a test-retest rho, and that rho would be allowed to pass the model
into a transfer verdict. A reliability ceiling measured on three models means
nothing. `transfer_stats` now reports `n_retest`, the number of models with a
score in both Countdown s0 and s1 after failures and no-breadth exclusions, and
`decide` returns INCONCLUSIVE_INCOMPLETE when either count is below 8. The
completeness checks run first, before any statistic is read.
*Cost if wrong:* a batch with seven good replicates is inconclusive and needs its
failed replicates rerun, at about 1 GPU-hour each.

**T8. Each pairwise statistic re-normalises the score over that pair's common
models.**
`amenability_score` is relative to the roster it is computed over (z-scores of
conversion and retention, then a zero-advantage gate). The analysis scored each
arm over whatever loaded in that arm, then correlated the overlap. If a model
failed on Graph only, Countdown was z-scored over ten models and Graph over nine,
and because the gate multiplies the z-scores, a different roster can change the
ranks, not only the scale. The rank comparison would then mix a roster effect
into the answer. The spec's claim that per-suite z-scoring cannot affect a
Spearman statistic holds only without the gate. So for the transfer rho and the
test-retest rho, each arm is now scored separately over the models present in
both arms of that pair, and those vectors are correlated. Per-feature rhos use
raw features and only need the same pairing. A test builds two work directories,
one with a model missing from one arm and one with it removed from every arm, and
requires identical transfer statistics; it failed before the change (rho -0.117
against 0.100). The leaderboard preview keeps per-arm scores over each arm's full
scored set, and is labelled to say so.
*Cost if wrong:* the leaderboard and the statistics use different normalisation
sets. Both are labelled, so the difference is visible rather than hidden.

**T9. A statistic whose input is constant is undefined, not rho 0.**
The shared statistics helper maps a NaN Spearman to 0.0. For a constant input
vector, for example every model scoring identically on one suite, that turns "no
information" into a transfer rho of 0, which the rule would read as
DOMAIN_SPECIFIC, or into a test-retest rho of 0, read as a reliability failure.
Both are real-sounding conclusions drawn from nothing. `correlate` now returns an
undefined marker when either paired vector has zero range; the report prints
"undefined (constant input), n=..." for it, including for per-feature and static
rows, and `decide` maps an undefined transfer or test-retest rho to a new
decision, INCONCLUSIVE_DEGENERATE, checked after completeness. The shared helper
in `eval/stats.py` is not changed, because Stage 0 depends on it.
*Cost if wrong:* one more decision class in the pre-registration.

**T10. Telemetry is validated against its meta file.**
The telemetry cache key is (model, suite, run seed) and ignores learning rate,
probe steps and item seed. T6 guards the CLI, but nothing guarded the analysis: a
direct library call with a custom config, or the stale 1e-6 telemetry left in the
batch directory after an lr recalibration, would be scored as if it were the
protocol run. `load_arm` now reads each job's meta file and treats the job as a
named failure when the meta is missing or unreadable, when its learning_rate,
probe_steps or item_seed differs from `ProbeConfig()`'s defaults and item seed 0,
or when its number of step records differs from probe_steps. The failure message
names the field and both values.
*Cost if wrong:* after a legitimate lr recalibration the analysis accepts only
telemetry at the recalibrated `ProbeConfig` default, which is exactly what spec
section 8 already pins, so old runs must be rerun with `--force`.

**T11. Three hardening fixes on the batch's critical path.**
First, `run_probe` writes telemetry, meta and pre-eval files to a temporary file
in the same directory and renames it over the target, so a job killed by walltime
or quota mid-write leaves the previous file or nothing, never a truncated file
that a later run would reuse from the cache. Second, before training the CUDA
peak-memory counter is reset, and the meta records `peak_train_bytes` (the
training peak alone) and `allocated_after_train_bytes` (what is still allocated
when training returns, which is the evidence for #33's trainer release);
`peak_memory_bytes` stays the whole job's peak. Without a GPU all three are null.
Third, the prefetch now requires at least one `*.safetensors` file in each
snapshot and, where `model.safetensors.index.json` exists, every shard it names,
because a config and tokenizer load fine from a snapshot whose weights never
arrived, and that failure would otherwise first appear on a GPU node. The review
also asked for report wording: the leave-one-out row prints its N, the static
comparator row says it is descriptive and includes breadth-excluded models, and
the leaderboard gains a pre-probe breadth column per suite.
*Cost if wrong:* a slightly larger fix wave before the batch; no protocol change.

**T12. The scratch directory defaults to `/scratch1/$USER`.**
The plan required `$SCRATCH` for the Hugging Face cache, but `$SCRATCH` is unset on
CARC login nodes, so `${SCRATCH:?}` would have aborted every job. The launchers now
use `${SCRATCH:-/scratch1/$USER}` as the scratch directory and default `HF_HOME` to
its `hf` subdirectory, following CARC's personal scratch convention. Outside a dry
run they fail fast with a message naming SCRATCH and HF_HOME when that directory
does not exist, and the array also fails fast when `HF_HOME` itself is missing.
A dry run skips the checks, so it still works on a laptop.
*Cost if wrong:* the owner confirms the path with `ls -d /scratch1/$USER`; one
environment variable to change.

**T13. The array requests any A100; the smoke probe decides on 80 GB.**
`probe_array.sbatch` requests `--gres=gpu:a100:1`, which lands on a 40 GB or an
80 GB card. Only sixteen 80 GB GPUs exist on Discovery, so requiring them up front
would lengthen the queue for jobs that may fit in 40 GB. If the smoke probe's
`peak_train_bytes` exceeds 36e9, the batch adds `--constraint=a100-80gb` (the
feature name confirmed from sinfo). H1's test allocation landed on a 40 GB card,
and SmolLM2-1.7B with full-parameter fp32 AdamW plus a reference model is
estimated at 35 to 40 GB, so the constraint is the likely outcome. No `--account`
line is needed: anakano_429 is the default account.
*Cost if wrong:* the smoke probe runs out of memory on a 40 GB node and is rerun
with the constraint.

**T14. The launchers load no CUDA module.**
The plan's `module load cuda/12.4` is dropped from the new launchers and from
`stage0.sbatch`; `module purge` stays in the array script. The launch README's
first-run checklist confirms that `uv` and the synced environment still work
after `module purge` on a compute node; if `uv` turns out to come from a module,
the owner either installs it to `~/.local/bin` or removes the purge, and records
which here. The pip torch and vllm wheels bundle the CUDA
runtime and need only the driver (H1 found driver 580, which supports CUDA 12 and
13 runtimes). The module name was never verified on Discovery, and a mismatched
toolkit can shadow the bundled libraries. The launch README tells the owner to run
`module avail cuda` only if the smoke probe fails with a CUDA library error.
*Cost if wrong:* one module line added back after the first smoke probe.

**T15. Probe meta is written before telemetry, and result files keep umask modes.**
The telemetry file is the cache marker, but `run_probe` wrote it before the meta
file, so a job killed between the two writes left telemetry without meta; every
rerun hit the cache, never wrote meta, and T10 reported "missing meta" until a
manual `--force`. Meta is now written first and telemetry last, so telemetry is
the commit and an interrupted job retrains. Separately, `_write_json`'s temp file
came from `mkstemp` with mode 0600, which `os.replace` kept, making every result
owner-only; it is now chmodded to `0o666 & ~umask` before the rename, the mode a
plain open would give. The umask is read once at import, because reading it
means setting it, and a per-write read would open a process-wide umask-0 window.
*Cost if wrong:* none to the protocol; both are write-path changes with tests.

**T16. The launchers no longer depend on the submitting shell's PATH, and a missing manifest fails loudly.**
Written from real CARC failures. uv lives at `$HOME/.local/bin/uv`, but some of
the owner's shells lacked `~/.local/bin` on PATH, and SLURM jobs inherit the
submitting shell's PATH, so a prefetch job died with `uv: command not found`.
Both `probe_array.sbatch` and `prefetch.sbatch` now prepend `$HOME/.local/bin`
to PATH and exit 1 with a clear message if `command -v uv` still finds nothing
(in the array script after the dry-run exit, so laptop dry runs need no uv, and
before `module purge`, with the export repeated after the purge). Separately, a
job manifest whose generating command had failed on the same missing uv was
never written, and every array task exited on a bare `sed: can't read` error;
the array script now checks `$JOBS_FILE` exists first, dry run included, and
names the command that generates it. The launch README's first-run checklist
now records the setup that worked: uv on PATH via `~/.bash_profile`, uv cache on
scratch, a venv on uv-managed Python 3.11 (the default `python` is a spack
module that `module purge` removes), `uv sync` on a compute node with capped
concurrency (the login node's thread cap crashed it), and the gated-repo check.
*Cost if wrong:* none to the protocol; if uv lives elsewhere the job fails in
seconds with a message naming the fix.

**T17. vLLM uses its built-in PyTorch sampler, not FlashInfer's.**
Written from a real CARC failure: all 16 pilot tasks died at vLLM engine warmup
with `RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda'
doesn't exist`, raised from FlashInfer's JIT path during
`compile_or_warm_up_model`. Our `SamplingParams` use `top_p=0.95`, which routes
through FlashInfer's top-p sampler, and FlashInfer JIT-compiles it with nvcc;
CARC compute nodes have no CUDA toolkit on PATH (T14 loads no CUDA module). The
launchers (`probe_array.sbatch`, `stage0.sbatch`) and the Modal image now set
`VLLM_USE_FLASHINFER_SAMPLER=0`, so vLLM uses its precompiled PyTorch
top-p/top-k sampler. It draws from the same distribution (temperature 1.0,
top_p 0.95), applied uniformly to every model and suite, so the protocol is
unchanged. Probe meta and pre-eval files record the variable's value as
`vllm_use_flashinfer_sampler`; the analysis records it only and gates on nothing.
*Cost if wrong:* if another FlashInfer JIT op (attention) also needs nvcc, the
fallback is `CUDA_HOME` pointed at the venv's `nvidia-cuda-nvcc` package or a
CARC cuda module, recorded in a further ruling.

**T18. The calibration pilot may run on a100, l40s or a40; training stays on A100.**
The pre-only calibration pilot may be submitted with
`--gres=gpu:1 --constraint="a100|l40s|a40"` (bf16-capable GPUs, never v100 or
p100) to shorten queue waits. The smoke probes and the transfer training batch
stay on `--gres=gpu:a100:1`, so every model's protocol runs on the same
hardware. Pilot numbers are calibration, not scored, and tiny cross-GPU numeric
differences cannot meaningfully flip the floored or saturated flags.
*Cost if wrong:* a borderline calibration flag differs on rerun.

**T19. The probe prompts use the TinyZero base-model completion scaffold.**
Written from a real CARC pilot result, before any batch data. Falcon3-1B-Base
scored pre-probe pass@64 = 0.003 on Countdown and exactly 0 on the easiest
(3-number) bucket. Sampled completions showed no attempt at any problem: the
prompts ended in a bare instruction ("Put only the expression inside
<answer></answer> tags."), which a base model reads as the start of a web
document and continues with forum rules or FAQ sections. The pilot was measuring
format compliance, not capability, and a GRPO probe on those prompts would get
all-zero reward. Both probe suites now use the TinyZero Countdown template (also
used by Gandhi et al. 2025, "Cognitive Behaviors that Enable Self-Improving
Reasoners"): a "conversation between User and Assistant" preamble, the task as
the User turn with "Show your work in <think> </think> tags. And return the final
answer in <answer> </answer> tags, for example ...", and a prompt that ends in
"Assistant: Let me solve this step by step.\n<think>\n", so the model's
continuation is the reasoning and then the answer. Countdown keeps its semantics:
the wording says "each number must be used exactly once" (TinyZero's "can only be
used once" is looser than our verifier, which requires every number), and the
example is `(1 + 2) / 3`, whose value 1 is below every target (10 to 400). Graph
path uses the same framing and opener with the example `X -> Y -> Z`; node labels
stop at L, so no item can contain those nodes. Neither example can verify for
any item, and tests prove that scoring the whole prompt as if it were a
completion gives 0 on all 300 items of each suite.
The prompt scaffold was preferred over two alternatives. (a) Prefilling
"<answer>" as the end of the prompt forces format but suppresses the reasoning
that RLVR amplifies, so the probe would measure a different behaviour from the
one the target training rewards. (b) Few-shot prompting inflates pre-scores by
an amount that depends on how well each model in-context learns, which is a
model-dependent confound in a ranking study, and it lengthens every prompt.
The Countdown verifier now tolerates exactly one trailing "= <target>" inside
the answer tag, for example `<answer> (20 - 9) + 4 = 15 </answer>`: such an
answer is a true statement that a strict parser rejected only for its form.
Any other "=" (a wrong right-hand side, a chain such as `e = 15 = 15`, or the
target on the left) still scores 0, and the stripped expression must still use
every number exactly once and equal the target. Graph path's verifier is
unchanged; it already tolerated spaces around "->" and inside the tags, which
the new example uses. Every consumer scores the completion alone: vLLM returns
only generated text to `evaluate_passk`, TRL passes completions separately from
prompts to the reward function, and the SFT records pair the prompt with the
text that follows it. `by_prompt` attribution still holds, since every prompt is
still unique. Prompt lengths are at most 157 (Countdown) and 192 (Graph) tokens
under the SmolLM2 tokenizer, inside `EntropyProbe`'s 256-token truncation. The
pass@k budget of 768 new tokens leaves room for the reasoning block; GRPO uses
TRL's default `max_completion_length` of 512, which this ruling records but does
not change, and a truncated completion earns reward 0. Item generation consumes
the same random stream as before, so the items' numbers, graphs and answers,
and T1's guessability figures, are unchanged; only the prompt text changed.
The stale pilot result `results/transfer/preeval/falcon3-1b-base-countdown.json`
on CARC must be deleted before the pilot is rerun, because `run_probe` skips any
pre-eval or probe job whose output file already exists. Any other pre-eval or
telemetry file produced with the old prompts must be deleted (or the job rerun
with `--force`) for the same reason. `prereg/transfer.md` is amended.
*Cost if wrong:* the scaffold suits base models, and an instruct model in the
roster would see an unfamiliar plain-text conversation instead of its chat
template; the roster is base models only, so this is not expected to bite. A
base model that finishes its answer may continue the transcript with an invented
next "User:" turn and a second answer tag, and the verifier takes the last tag;
if pilot completions show this, the remedy is a stop string or truncation at the
next turn, recorded in a further ruling. The trailing-"=" tolerance makes
Countdown slightly more lenient than TinyZero's reward, uniformly across models.

**T20. Probe completions end at the first `</answer>`, and the verifiers score the
first answer.**
Written from a real GPU check of the T19 scaffold, before any batch data.
Falcon3-1B-Base (temperature 1.0, top_p 0.95, max_tokens 768) now engages and
emits the tags, but it does not stop after its first `</answer>`. It loops
think/answer blocks until the token cap:
"</think>\n<answer>\nThe equation 4 + 20 = 25\n</answer>\n<think>\n4 + 20 = 24
...</think>\n<answer>\n20 + 9 = 29...</think>\n<answer>\n20 - 4 = 16, so 16 more
is needed...</think>\n<answer>\n20 - 4 = 16 ..." and so on to max_tokens. Both
verifiers took the LAST `<answer>` match, so they scored the degenerate loop
text rather than the model's first answer, and every sample spent its full
768-token budget (300 items x 64 samples per pre-eval job, plus GRPO's
completions). This is the failure T19's cost-if-wrong anticipated.
The change has two parts.
(1) Stop at the first `</answer>`, per suite. `ProbeSuite.stop` is
`ANSWER_STOP = "</answer>"` for both probe suites, and `run_probe` passes it to
pass@k and to GRPO. pass@k: `_vllm_generate` builds `SamplingParams` from
`sampling_kwargs`, which adds `stop=["</answer>"]` and
`include_stop_str_in_output=True`; `evaluate_passk` also truncates every
completion just after the first stop string before verifying, so the contract
holds for any generator. GRPO: the reward function scores each completion
truncated just after its first `</answer>`. SFT: `build_rejection_dataset` takes
`stop` and cuts every candidate there before verifying, deduplicating and
storing it as a target, so SFT never teaches the loop (no runner calls it yet;
a caller for a probe suite must pass the suite's stop). gsm8k is not a probe
suite and has no tags: every stop parameter defaults to None, `RealRunner`'s
gsm8k pass@k and GRPO calls pass none, and their generation and scoring are
byte-for-byte unchanged.
(2) Both extractors (`extract_expression`, `extract_path`) take the FIRST
`<answer>...</answer>` match instead of the last.
Both parts, not one. The stop alone leaves every path where it is not applied
(GRPO, below; any future caller that forgets the stop) scoring the loop. The
first-match rule alone makes scoring correct but keeps paying for the loop on
every sample, and lets the loop's tokens dominate SFT targets. With both, a
completion is judged on its first answer whichever path produced it.
The vLLM pitfall: vLLM strips stop strings from the returned text by default.
With `stop` but without `include_stop_str_in_output=True`, no completion would
contain `</answer>`, the `<answer>(.*?)</answer>` regex would never match, and
every probe score would be exactly 0. `test_sampling_kwargs_keep_the_stop_string_in_the_output`
holds the flag; the post-hoc truncation in `evaluate_passk` cannot repair a
stripped tag, so the flag is load-bearing.
GRPO does not stop generation. The installed TRL is 1.13.0 (uv.lock), and
`run_grpo` uses TRL's transformers generate path (`use_vllm` defaults to
False). `GRPOConfig.generation_kwargs` is forwarded to transformers'
`GenerationConfig`, which accepts `stop_strings`, but TRL 1.13 calls
`model.generate` without `tokenizer=`, and transformers raises in its
`StopStringCriteria` setup when stop strings are set without a tokenizer. So
the stop cannot be expressed cleanly, and the reward function truncates
instead: GRPO generation still runs to `max_completion_length` (TRL default
512, `GRPOConfig.max_completion_length`, not pinned here; pinning it is an open
owner decision), the loop tokens stay in the completion mask and share the
group's advantage, and only the reward ignores them. The run manifest records
`completion_stop` and `generation_runs_to_cap: true`. Switching GRPO to TRL's
vLLM generation (`use_vllm=True` with `generation_kwargs={"stop": ["</answer>"],
"include_stop_str_in_output": True}`) would stop it, but it changes the training
backend and its memory profile and needs a GPU check, so it is not done here.
The T19 tolerance of one trailing "= <target>" is unchanged. An answer with
prose, such as "The equation 4 + 20 = 25", still scores 0, as intended. Any
pre-eval or telemetry file produced before this ruling was scored on the last
answer and must be deleted (or rerun with `--force`), because `run_probe`
reuses cached outputs. `prereg/transfer.md` is amended.
*Cost if wrong:* a model that writes a provisional answer, then corrects itself
in a later answer block, is scored on the provisional one; that is the price of
not scoring the loop, and it applies uniformly across models. In GRPO the
reward stays correct but the policy gradient still pushes on loop tokens, so
the loop may be reinforced or suppressed in proportion to the first answer's
reward; the probe measures that dynamic as it is. If the 512-token GRPO cap
truncates first answers often, the remedy is the vLLM path above or pinning
`max_completion_length`, recorded in a further ruling.
