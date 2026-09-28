# Probe Transfer: Design

**Date:** 2026-09-27
**Status:** Approved design, implementation plan at
`docs/superpowers/plans/2026-09-27-probe-transfer.md`
**Supersedes:** step 1 of the session handoff in issue #34
**Parent spec:** `docs/superpowers/specs/2026-09-22-rl-amenability-benchmark-design.md`

## 1. Why this comes before Stage 0

Issue #34 reframed the project from a research artifact into a developer product:
the only input is the model, the task and protocol are ours and fixed, and scores are
pre-computed into a leaderboard. That framing rests on one empirical claim that
nothing in the repository has tested: **amenability is a property of the model, not of
the model-and-probe-task pair.** If a model's probe score changes rank when the probe
task family changes, a single leaderboard number is not defensible and the product is
per-domain. Everything downstream (Stage 0's positive control, the over-SFT fixture in
#28, the product surface) is shaped by the answer, so it is asked first, and it is
asked with the cheapest experiment that can answer it.

## 2. Two questions, one of which is being asked

The handoff blurs two different transfer questions. They have different costs and
different evidence:

| | Question | Needs ground truth? | Cost |
|---|---|---|---|
| **Q1, probe invariance** | Does model M's probe score depend on which probe suite measures it? | No | ~30 GPU-hours |
| **Q2, target transfer** | Does the score predict full-run gain equally on two different target tasks? | Yes, two full arms | ~2x Stage 0 |

Q1 is necessary for Q2 and an order of magnitude cheaper. **This design answers Q1.**
Q2 is deferred to Stage 1, where it becomes "add a second target task" and only
matters if Q1 passes.

## 3. The experiment

Probe every model in the 10-model roster with the existing GRPO probe protocol on two
probe suites, and measure whether the resulting score vectors agree.

| Arm | Suite | Run seed | Runs |
|---|---|---|---|
| A | Countdown (existing) | 0 | 10 |
| B | Graph path (new) | 0 | 10 |
| A' | Countdown | 1 | 10 |

Arm A' is a **test-retest replicate**: same items, same protocol, different training and
sampling seed. It gives the reliability ceiling that the transfer statistic must be
read against. Without it, a transfer rho of 0.5 cannot be distinguished from a noisy
probe, and "amenability is domain-specific" would be confused with "the probe is
unstable." Test-retest reliability is also the product's single most important number.

The protocol is the pre-registered Stage 0 probe protocol, unchanged: 60 GRPO steps,
8 generations, 4 prompts per optimiser step, lr 1e-6, beta 0.04, temperature 1.0,
full-parameter training, 300 probe items across three difficulty buckets, pass@k at
k in {1, 8, 32, 64} from 64 samples before and after. Identical for every model and
both suites. The one permitted protocol change is described in section 8.

### What the batch produces as by-products

The same ~30 GPU-hours does five jobs. This is why the roster is probed in full before
any over-SFT variant is produced or any 600-step full run is started:

1. The Q1 answer.
2. Probe difficulty calibration (#21) on all ten models, not just the 0.5B one.
3. The first real GPU exercise of the whole probe path, so #33 and whatever else is
   wrong surfaces at one GPU-hour per failure rather than twenty.
4. Pre-probe pass@k on every roster model, which are baselines 2 and 3 of the
   pre-registered four and the static-capability comparator in section 5.
5. The first leaderboard-shaped table: ten models, features, two domains.

## 4. The second probe suite: graph path-finding

### Requirements any second suite had to meet

- Verifiable by a deterministic, sandboxed verifier. No LLM judge.
- Procedurally generated from a seed. No dataset download, no contamination question.
- Difficulty-bucketed so the zero-advantage rate carries information at every size.
- Learnable at 0.5B to 1.7B in 60 GRPO steps: pass@32 nonzero, pass@1 not saturated.
- **Answer space large enough that pass@32 by guessing stays well below 1.0.** This
  rules out anything multiple-choice and any puzzle whose answer is a small
  assignment: breadth is `pass@32 - pass@1`, and a 33% guess rate makes pass@32
  saturate at 1.0 for every model, destroying the conversion-rate denominator.
- Tokenizer-fair. Character-level string transformations were rejected because
  tokenizers differ in how they split words, which would inject a tokenizer confound
  into the very comparison the experiment makes.
- A different domain from arithmetic search (Countdown) and from grade-school word
  problems (GSM8K).

### Why graph path-finding

It has the **same reward structure as Countdown**: search for a witness, and a
verifier checks the witness, with any valid witness accepted. Holding the reward
structure constant while changing the domain is what makes a ranking disagreement
attributable to domain rather than to reward sparsity or shape. Node labels are single
uppercase letters, so it is tokenizer-fair. Difficulty is controlled by node count.

### Definition

- **Instance.** An undirected connected graph on `n` nodes labelled `A`, `B`, ...,
  built as a random spanning tree plus `n // 3` extra random edges, with a source and
  target at shortest-path distance >= 3. Buckets: `n` in `(8, 10, 12)`, 100 items each.
  These numbers come from a simulation run at design time, not from taste: an
  informed guesser (starts at the source, ends at the target, fills the middle with
  random distinct nodes, 32 tries) scores pass@32 of 0.94 on 6-node graphs with
  `n // 2` extra edges and distance >= 2, which would make the easy bucket's breadth
  mostly luck. At `(8, 10, 12)`, `n // 3`, distance >= 3 the same guesser scores
  0.21 / 0.10 / 0.03. If the calibration pilot shows the suite is too hard at the
  small end (section 7), the bound contingency is `(7, 9, 11)`, where the easiest
  bucket guesses at 0.37, recorded as a ruling.
- **Prompt.** Lists the edges as `A-B, B-C, ...`, names the source and target, and
  asks for a path as node names joined by `->` inside `<answer></answer>` tags. Same
  tag convention as Countdown.
- **Answer field** (verifier-side, not shown to the model): `edges|source|target`.
- **Verifier.** Extracts the last `<answer>` block, splits on `->`, and accepts iff the
  first node is the source, the last is the target, every consecutive pair is an edge
  of the graph, and no node repeats (a simple path, so a model cannot inflate length
  by cycling). Anything unparseable is a 0.
- **Guessability bound, enforced by a test.** The informed-guesser baseline above,
  32 samples per item over 100 items per bucket, must score pass@32 < 0.3 in every
  bucket and < 0.15 averaged over the suite. This is the property that keeps the
  breadth denominator informative, and it is asserted, not assumed. The bound
  covers edge-blind guessing only: a random walk along the listed edges saturates
  pass@32 (see `docs/decisions/transfer-rulings.md`, T4).
- **Registry.** Suite name `probe_graphpath`, task IDs `probe/graphpath/{seed}/{idx}`.
  The suite's tests register it in a `SuiteRegistry` alongside `probe_countdown`. On
  the transfer path, disjointness is structural, not checked at runtime (see
  `docs/decisions/transfer-rulings.md`, T5).

## 5. Analysis and the pre-registered decision rule

Committed to `prereg/transfer.md` before the batch runs. Not hash-frozen: the freeze
mechanism belongs to Stage 0 and the product reframe replaces it with metric
versioning. The git history of `prereg/transfer.md` is the audit trail.

### Statistics

All from the existing `amenability.eval.stats`: Spearman rho, percentile bootstrap CI,
permutation p-value, leave-one-out. N = 10 for every statistic.

| Statistic | Computed between | Purpose |
|---|---|---|
| **Transfer rho** | score(Countdown, s0) and score(Graph, s0) | The Q1 answer |
| **Test-retest rho** | score(Countdown, s0) and score(Countdown, s1) | Reliability ceiling |
| Per-feature transfer rho | each of `conversion_rate`, `retention_factor`, `zero_advantage_rate` across suites | Which feature is domain-sensitive if transfer fails |
| Static comparator rho | pre-probe pass@32 across suites | Does the dynamics signal transfer beyond what raw capability transfers? |
| Leave-one-out transfer rho | as transfer rho, each model held out | No single model carries the result |

Scores are `amenability_score` applied per suite across the ten roster models, so each
suite's score vector is z-scored within that suite. Spearman is rank-based, so the
per-suite z-scoring does not affect the transfer statistic.

### Decision rule

Read the transfer rho against the test-retest rho, never bare. A transfer rho near
the test-retest rho means the probe transfers as well as it can be measured to.

| Transfer rho | Permutation p | Decision |
|---|---|---|
| >= 0.7 | < 0.05 | **Invariant.** A single score is defensible. Stage 0 proceeds as designed. |
| 0.4 to 0.7 | any | **Partial.** Leaderboard reports a composite plus per-domain columns. Stage 0 proceeds; #28 unchanged. |
| < 0.4 | any | **Domain-specific.** The product is per-domain. #28 and Stage 0 are re-planned per domain before any further GPU spend. |

Two overriding conditions:

- If **test-retest rho < 0.5**, the probe is too noisy to answer Q1 at N = 10. The
  decision is "inconclusive: probe reliability", and the next step is a protocol
  change (more steps, more items, or seed averaging), not a transfer verdict.
- If fewer than **8 of 10** models complete both s0 arms, the decision is
  "inconclusive: rerun failures". Failed jobs are listed by name in the report; they
  are never silently dropped, and rho is never computed on a subset without N printed
  beside it.

### What will not happen

- No change to the score formula, the feature definitions or the protocol after seeing
  any result from the batch, except as bound in section 8 before the batch.
- No choosing between Countdown and Graph as "the" probe based on which gives the
  nicer leaderboard.
- No reading the transfer rho without the test-retest rho beside it.

## 6. Architecture

### New

| Path | Responsibility |
|---|---|
| `src/amenability/suites/graphpath.py` | Generator, verifier, random-walker baseline for the guessability test |
| `src/amenability/suites/catalog.py` | Name-to-(generator, verifier) lookup for probe suites, so runners and analysis never hard-code a suite |
| `src/amenability/probe/run.py` | `run_probe(...)`: pre-eval, GRPO, post-eval, telemetry write, checkpoint deletion. The one probe implementation. |
| `scripts/run_probe.py` | Launcher-agnostic per-job CLI over `run_probe`, with `--pre-only` for the calibration pilot |
| `scripts/make_transfer_jobs.py` | Writes the job manifest (one line per model, suite, seed) |
| `scripts/prefetch_models.py` | Resolves and downloads every roster model once, on a CPU node; resolves #20 |
| `scripts/analyze_transfer.py` | Section 5's statistics, calibration table, leaderboard preview, report |
| `scripts/slurm/prefetch.sbatch`, `scripts/slurm/probe_array.sbatch` | CARC launchers |
| `prereg/transfer.md` | The decision rule, committed before the batch |
| `docs/decisions/transfer-rulings.md` | Rulings T1... made during execution, with cost-if-wrong |

### Changed

- `scripts/real_runner.py`: `RealRunner.probe` delegates to `run_probe`. Behaviour
  preserved; Stage 0 and the transfer batch share one probe implementation.
- `src/amenability/training/grpo.py`: releases trainer GPU memory before returning
  (#33).
- Telemetry cache keys carry suite and seed:
  `telemetry/{model}-{suite}-grpo-s{seed}.json`. The current key
  `{variant}-grpo.json` would silently return Countdown telemetry for a Graph request.
- Item seed and run seed are separate parameters. Items are generated from
  `item_seed = 0` for every arm; the run seed varies. Today one `seed` does both, which
  would make the test-retest replicate use different items.

### Data flow

```
prefetch.sbatch  ->  $HF_HOME populated, hf_ids verified (#20)
make_transfer_jobs.py  ->  results/transfer/jobs.txt  (30 lines: model suite seed)
probe_array.sbatch  -x  run_probe.py --pre-only   ->  results/transfer/preeval/{model}-{suite}.json   (pilot)
probe_array.sbatch  -x  run_probe.py              ->  results/transfer/telemetry/{model}-{suite}-grpo-s{seed}.json
analyze_transfer.py  ->  results/transfer/report.md, report.json, leaderboard_preview.md
```

Each array task runs exactly one job in its own process. Process exit is what
guarantees GPU memory release between jobs; #33's fix covers the train-to-eval
transition inside a job.

### Storage

Probe checkpoints are deleted after post-eval unless `--keep-checkpoint` is passed.
What survives per job: the run manifest and the telemetry JSON, a few hundred KB.
`HF_HOME` points at scratch storage; ten roster models at fp32 are roughly 35 GB,
which does not belong in a home directory.

## 7. Compute

| Phase | Runs | GPU-hours | Purpose |
|---|---|---|---|
| Prefetch | 1 CPU job | 0 | Downloads, resolves #20 |
| Calibration pilot | 20 pre-eval only | ~2 | Per-bucket pass@k per model, both suites; fix buckets before spending on training |
| Smoke probe | 1 to 3 | ~2 | Largest model first: peak-memory check, #33 check, nonzero-gain check (section 8) |
| Transfer batch | 30 | ~30 | The experiment |
| Contingency | | ~6 | Reruns, protocol pilot if section 8 triggers |
| **Total** | | **~40** | |

Per-probe estimate: pre-eval 5 to 10 min (19,200 vLLM completions on a 1B model),
60 GRPO steps at 20 to 40 s each, post-eval 5 to 10 min. Roughly one GPU-hour.
Embarrassingly parallel; the wall clock is set by the array concurrency limit and the
queue, not by the work.

**Memory.** Full-parameter training with fp32 master weights, AdamW state and a
reference model at 1.7B is roughly 35 to 40 GB before activations. This fits an
A100-80GB comfortably and an A100-40GB marginally. The smoke probe runs the largest
roster model (SmolLM2-1.7B) first and records `torch.cuda.max_memory_allocated`; if it
exceeds 36 GB on a 40 GB card, the array requests 80 GB nodes. Halving memory by
loading in bf16 is **not** an option: pure-bf16 AdamW changes training numerics and
would be a silent protocol change.

**Launcher.** CARC SLURM is primary, per the owner's decision. Modal remains the
documented fallback (`scripts/modal_app.py`) for the case #19 anticipates, an
allocation under ~400 GPU-hours. Because the per-job CLI is launcher-agnostic, the
Modal fallback for this batch is a small fan-out wrapper, not a rewrite.

## 8. The one permitted protocol change, bound in advance

The transfer test needs the probe to *move*. If 60 steps at lr 1e-6 produce no
measurable gain on any model, `conversion_rate` is ~0 everywhere and the score is
noise; the transfer rho would then be a measurement of nothing.

The smoke probe on Qwen2.5-0.5B (Countdown) must show `post pass@1 > pre pass@1`. If it
does not:

1. Run the same probe at lr 3e-6 and 5e-6 on that one model only (2 GPU-hours).
2. Pin the **smallest** lr that shows gain, for every model and both suites.
3. Record the change as a ruling with the three measured gains, and update
   `prereg/stage0.md`'s "Fixed quantities" before Stage 0's freeze.

This is protocol calibration on a single model before any transfer or ground-truth
data exists, which is the legitimate window for it. It is bound here so that it
cannot become a post-hoc search over learning rates after the batch is seen. No other
protocol parameter is touched.

## 9. Human checkpoints

The executing session stops and asks at each of these. None can be done by an agent.

1. **CARC allocation** (#19). Run on the login node and paste the output:
   `myaccount` (or the CARC equivalent), `sinfo -p gpu -o "%N %G %f"`, and the
   project's quota on scratch. The plan needs: GPU-hours available, which A100 variants
   exist and their SLURM feature names, and the scratch path.
2. **Gated weights.** Accept the licences for `meta-llama/Llama-3.2-1B` and
   `google/gemma-3-1b-pt` on Hugging Face with the account whose token is on CARC.
   Without this, two of ten prefetches fail and the batch is N = 8 before it starts.
3. **Batch launch.** After the pilot and smoke results are in the tree, approve the
   30-run array. This is the ~30 GPU-hour commitment.

## 10. Out of scope, deliberately

Each of these changes depending on Q1's answer. Planning them now means planning two
versions of each and discarding one.

- #28, the over-SFT variant production path.
- The product-surface pass: leaderboard shape, `score_version`, changelog.
- Dispositions of #29, #30, #31, #32 under the product framing. #34 already records
  the intended direction for each.
- Any change to the freeze mechanism or `FROZEN_PATHS`.
- The SFT-arm probe. Only the GRPO probe is wired into the runner today and only the
  GRPO probe is needed for Q1.
- Q2, target transfer.

## 11. Limitations of this experiment, stated up front

- N = 10. The transfer CI will be wide. The decision rule uses the point estimate with
  a permutation p, and the test-retest ceiling, precisely because a CI at N = 10 cannot
  do the work alone.
- Two probe domains, both witness-search tasks. A pass says the score is invariant
  across these two; it does not say it is invariant across every task family. A
  third, structurally different suite is a Stage 1 question.
- One run seed per arm except Countdown. The test-retest reliability is measured on
  Countdown and assumed comparable for Graph.
- Batching is by sequences, not tokens, inherited from Stage 0 and disclosed there.
- The probe protocol's learning rate may be recalibrated once, as bound in section 8.
