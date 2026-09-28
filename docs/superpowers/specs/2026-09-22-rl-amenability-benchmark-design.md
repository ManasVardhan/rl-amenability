# RL-Amenability Benchmark: Design

**Date:** 2026-09-22
**Status:** Approved design, pending implementation plan
**Scope tier:** Tier M (~355 GPU-hours)

## 1. Problem

Selecting a base model to post-train is currently guesswork. Practitioners pick by
pretraining benchmark scores (MMLU, GSM8K pass@1), but those measure what a model can
already do, not how much it will gain from RLVR-style post-training. The two dissociate
badly: Qwen2.5-Math-7B gains 21 to 27% on MATH-500 from random or incorrect rewards
while Llama3.1/3.2 and OLMo2 gain nothing from identical training.

This project builds a benchmark that scores a base model by how much it will gain from
post-training, per algorithm, and validates that score against real post-training runs.

## 2. Prior art and positioning

### Closest prior work

**RuDE, "On Predicting the Post-training Potential of Pre-trained LLMs"**
(arXiv 2605.11978, USTC / Alibaba / NUS, May 2026). Predicts post-training performance
from base models using rubric-based discriminative evaluation over a 4C taxonomy
(Competence, Content, Control, Compliance). Reports Pearson > 0.90 across 16 models from
4B to 1T. Training-free, single scalar output.

Its stated limitations are this project's openings:
- No public code or benchmark release, so the result is unreplicable.
- Single scalar for "post-training" as a monolith, no per-algorithm prediction.
- Bounded by the quality of the response generator used to build pairs (Gemini-3-Pro).
- Explicitly cannot model training dynamics such as entropy collapse or reward hacking.

### Other relevant work

| Work | Relation |
|---|---|
| MATH-Beyond (arXiv 2510.11653) | Task-side benchmark of problems base models cannot solve at pass@1024. Measures RL headroom in the task, not amenability in the model. |
| Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes? (arXiv 2504.12491) | Largely negative result for perplexity as a predictor at fixed model size. |
| Spurious Rewards | Demonstrates amenability is model-specific and set at pretraining (code-reasoning traces). Primary motivating evidence. |
| When RL Fails after SFT (arXiv 2606.09932) | Model plasticity is lost to SFT overtraining and can be partially restored. Basis for the positive control arm. |
| SFT Overtraining Predicts Rank Inversion via Entropy Collapse (arXiv 2606.18487) | Model rankings invert depending on preceding SFT. Motivates measuring dynamics, not statics. |
| Observational scaling laws | Predicts post-training technique gains from cheap capability metrics at zero training cost. |
| LogME, LEEP, H-score | Classic transferability estimation. The pre-LLM version of this question. |
| SteerEval, Steer-Bench, AxBench, CLaS-Bench | Inference-time controllability of aligned models. A different quantity, cited to disambiguate. |

### Gap this project fills

No public work provides an algorithm-conditional amenability score, an open calibration
corpus of controlled post-training runs, or a predictor that measures training dynamics
rather than proxying them.

## 3. Conceptual spine

RLVR does not add capability. It converts latent breadth into realized reliability:
RL-trained models win at low sampling budgets while base models catch up at high k.

**Amenability is therefore defined as conversion efficiency**: how effectively a fixed
training budget turns pass@k breadth the model already has into pass@1 reliability, and
how much breadth remains unconverted.

This definition makes the headroom confound part of the metric rather than a nuisance,
because it measures a normalized rate rather than a raw delta, and it gives the score a
mechanism to point at rather than being a fitted black box.

## 4. Scope decisions (locked)

| Decision | Choice |
|---|---|
| Gain type predicted | Capability gain on verifiable tasks (RLVR-style). Not behavioral alignment, not safety durability. |
| Predictor class | Micro-probe: a short, real, fixed-budget RL/SFT run. Not training-free. |
| Calibration scale | 10 models in the 0.5B to 1.7B nominal size class (0.3 to 1.8B actual parameter counts), controlled runs only (no public checkpoint pairs). |
| Algorithms | GRPO and rejection-sampling SFT. No PPO (expected ~0.95 correlation with GRPO), no DPO. |
| Primary claim | Causal: the probe detects experimentally induced amenability differences. |
| Secondary claim | Observational: N=10 cross-model ranking, labeled preliminary. |

## 5. The two non-negotiable design constraints

A short RL run trivially predicts a long RL run, because learning curves are smooth.
"Reward at step 80 predicts reward at step 600" is not a finding. Two constraints
prevent the project from producing a tautology:

1. **Probe tasks must be disjoint from target tasks.** Enforced in code by a suite
   registry that refuses overlapping task IDs, not by convention. The claim is then
   about general plasticity, not about one run's momentum.
2. **The probe must beat a naive-extrapolation baseline.** Fit a curve to the probe
   trace and extrapolate. The interesting cases are exactly where that fails: entropy
   collapse, late plateaus, and rank inversion between models.

## 6. Architecture

### 6.1 The probe

A fixed-budget training stress test with identical protocol across all models.

**Probe suite.** Verifiable tasks disjoint from target tasks, difficulty-graded into
buckets so every model size has some solvable and some unsolvable items. Candidates:
Countdown-style arithmetic puzzles (controllable difficulty, verifiable, learnable at
this scale per TinyZero), constrained string/format transformations, and a small graded
logic slice. No overlap with GSM8K or MATH.

**Protocol.** Identical lr, KL coefficient, group size, temperature, and step budget for
every model. Batches sized by tokens, not sequences, for tokenizer comparability.
Approximately 60 to 80 GRPO steps and 150 SFT steps, targeting roughly 10% of the full
run budget. A pre-probe measurement pass computes pass@1, pass@8, pass@32, pass@64 on
both probe and target suites.

**Full-parameter training only. No LoRA.** LoRA constrains how far weights can move,
which is the quantity under measurement. Using it would make the benchmark measure the
adapter rather than the model.

**Output.** A telemetry feature vector, not a scalar.

### 6.2 Probe telemetry

Features chosen because each maps to a documented failure mode:

| Feature | Failure mode detected |
|---|---|
| Zero-advantage group rate | GRPO dead-signal condition: all samples in a group score identically, so no gradient exists. Measures whether RL has anything to learn from at all. |
| Entropy trajectory (initial, slope, collapse step) | Entropy collapse, the documented cause of post-SFT rank inversion. |
| pass@32 minus pass@1, before and after probe | The conversion itself: did breadth become reliability, and how much remains. pass@32 is used throughout the score for cost reasons; pass@64 is measured at endpoints only, as a check that the k=32 truncation does not change conclusions. |
| KL drift per unit reward gain | Plasticity. How far the policy must move to buy improvement. |
| Early and mid reward slope | The naive signal, retained explicitly so its marginal contribution can be measured. |
| Gradient norm trajectory | Effective-gradient loss identified in over-SFT'd checkpoints. |

### 6.3 Stack and repository layout

Python. TRL `GRPOTrainer` plus vLLM for generation and evaluation. Not verl: at 0.5B to
1.7B the scale machinery is unnecessary and its operational overhead on a shared SLURM
cluster is a real tax.

```
rl-amenability/
  src/amenability/
    probe/        suite.py  grpo_probe.py  sft_probe.py  telemetry.py
    groundtruth/  run_grpo.py  run_sft.py  targets.py
    scoring/      features.py  score.py  baselines.py
    eval/         passk.py  stats.py
    registry/     models.yaml
  configs/
  scripts/
  results/
  prereg/
  docs/
```

The probe harness and ground-truth runner share a trainer wrapper but are separate entry
points, so the probe cannot see target-task data.

## 7. Ground-truth pipeline

### 7.1 Model roster (10 models, 7 families)

| Family | Models |
|---|---|
| Qwen (4) | Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen2.5-Math-1.5B, Qwen3-0.6B-Base |
| Llama (1) | Llama-3.2-1B |
| Gemma (1) | Gemma-3-1B-pt |
| OLMo (1) | OLMo-2-1B |
| SmolLM (1) | SmolLM2-1.7B |
| Falcon (1) | Falcon3-1B-Base |
| StableLM (1) | StableLM-2-1.6B |

Qwen is capped at 4 of 10 deliberately. Given that Spurious Rewards is literally
"Qwen behaves unlike everything else," a Qwen-heavy roster risks measuring family
membership rather than amenability.

Qwen2.5-Math-1.5B is included on purpose. It is the Spurious Rewards outlier, and a
score that fails to flag it as anomalously RL-amenable has failed its most famous test.

### 7.2 Positive control arm

Two base models (Qwen2.5-0.5B and Llama-3.2-1B), each SFT'd at two over-training token
budgets beyond the base, producing six checkpoints with an a priori known amenability
ordering: degradation should be monotonic in SFT token count as entropy depletes.

This provides a falsification test independent of any correlation study. If the probe
cannot separate a fresh base from its over-SFT'd sibling, the probe is broken. Two bases
rather than one preempts the objection that entropy collapse is a Qwen artifact.

### 7.3 Algorithm arms

- **GRPO**: 600 steps on the target task, checkpointed every 200 steps.
- **SFT**: rejection-sampling SFT (STaR-style) on traces the model generates itself,
  filtered by verifier. 1500 steps, checkpointed every 500, so the 150-step SFT probe is
  the same 10% fraction of budget as the GRPO probe. Explicitly **not** distillation from a larger teacher. Teacher
  distillation would make the SFT arm measure the teacher-student gap rather than the
  student's plasticity, collapsing the SFT-versus-RL dissociation claim.

### 7.4 Target tasks

GSM8K as the single training target for Tier M. It is verifiable, and 0.5B models score
nonzero so nothing floors. MATH-500 evaluated at endpoints only as a transfer check.
Code tasks (HumanEval+/MBPP+) deferred to a future tier. MATH-Beyond deferred: at this
scale it likely returns near-zero across the board, giving no variance to correlate
against.

### 7.5 Outcome variable

- **Primary:** conversion ratio, Δpass@1 divided by (base pass@32 minus base pass@1).
- **Secondary:** raw Δpass@1, so readers who reject the normalization can still read the
  table.

Both computed from the full curve, not the endpoint, because ceilings and plateaus are
the interesting part.

### 7.6 Seeds

Three seeds on a 2-model subset to establish a run-to-run noise floor, single seed
elsewhere. Without this, "this model is unamenable" cannot be distinguished from "this
run was unlucky."

### 7.7 Compute budget

| Component | GPU-hours |
|---|---|
| GRPO full runs, 10 models | 110 |
| Rejection-sampling SFT arm | 30 |
| Seed replicates (2 models x 1 extra) | 22 |
| Positive control (2 bases x 2 over-SFT levels) | 52 |
| Probes, 14 model variants x 2 algorithms | 14 |
| Eval sweeps | 26 |
| Subtotal | 254 |
| Padded 1.4x | ~355 |

GPU-hours, not wall-clock. The workload is embarrassingly parallel. On 4 concurrent
GPUs this is roughly 4 to 6 weeks including SLURM queue waits. Modal is a fallback at
roughly $600 on L40S if the CARC allocation is short, and is specifically worth using
for Stage 0 so the de-risk is not blocked on queue access.

**Action item before any code: verify the actual CARC GPU allocation.**

## 8. Scoring and validation

### 8.1 The N=10 constraint

A learned regressor cannot be fit on ten data points with six features. This is not a
statistical nicety; it is guaranteed overfitting. The scoring function must be specified
before the ground truth is seen.

### 8.2 Pre-registration

The score formula, baselines, gate thresholds, and analysis plan are written to
`prereg/` and committed to git with a timestamp before any full GRPO run starts. The
commit hash is referenced in the paper.

This costs nothing, is rare in this literature, converts small N from a fatal weakness
into a defensible design, and prevents Stage 0 from quietly becoming a hyperparameter
search over score definitions.

### 8.3 Score definition

Two factors, both probe-measured, gated by a third:

- **Conversion rate:** Δpass@1 during the probe divided by available breadth at start
  (base pass@32 minus base pass@1).
- **Retention factor:** sustainability of that conversion, derived from entropy slope
  and KL drift per unit reward gain. This is the term that predicts where the curve will
  bend rather than where it currently sits.
- **Signal-availability gate:** the zero-advantage group rate. If most GRPO groups
  produce identical rewards there is no gradient, and the score floors regardless of the
  other terms.

Combined as `score = gate * mean(z(conversion_rate), z(retention_factor))`, where
`gate` is 1 minus the zero-advantage group rate, clipped to [0, 1]. Equal weights,
z-scored across the roster, fixed a priori. Not tuned. Tuning at N=10 fits noise.

The score consumes three of the six telemetry features. The remaining three (early and
mid reward slope, gradient norm trajectory) are recorded as diagnostics and used in the
failure-mode analysis, but are deliberately excluded from the score: reward slope is the
naive-extrapolation baseline and must stay outside the thing being compared against it.

### 8.4 Baselines (pre-registered)

The score must beat all of these or the contribution is not real:

1. Naive curve extrapolation from the probe reward trace. **The one that matters.**
2. Base pass@64 alone (the literature's current best cheap metric).
3. Base pass@1 alone.
4. Model parameter count.

### 8.5 Statistics

- Spearman as primary, not Pearson, since the use case is ranking models.
- BCa bootstrap confidence intervals on every reported coefficient.
- Permutation test for significance rather than table lookup.
- Leave-one-out validation to show no single model carries the result.
- No bare correlation coefficients anywhere in the paper.

### 8.6 Confound controls (reported as first-class results)

- **Headroom:** partial correlation controlling for base pass@1, showing the score is
  not merely detecting low starting points.
- **Family:** family-stratified analysis, showing the score is not merely detecting
  Qwen.

Both will be underpowered at N=10. Report them with that stated rather than omitting
them and letting a reviewer raise it.

### 8.7 Noise floor

Seed replicates give run-to-run variance in the outcome variable. This caps the
correlation anyone could achieve and is reported explicitly. If seed variance is large
relative to cross-model variance, that is itself a publishable finding about the
stability of RL-readiness claims in this literature.

## 9. Stages and kill gates

### Stage 0: positive control. ~130 GPU-hours, 1 to 2 weeks

Build the probe harness, produce the over-SFT variants, run six full GRPO runs, probe
all six.

- **Gate A:** does the probe recover the known amenability ordering within each base?
  Threshold pre-registered at: correct ordering in both families, and pooled Spearman
  rho >= 0.7.
- **Gate B:** does it beat naive curve extrapolation on those six?

Failing both means the probe design is wrong, at a cost of 130 hours rather than 355.
Failing only B means the probe works but is not cheap enough, and the paper pivots
toward failure-mode detection.

### Stage 1: cross-model sweep. ~230 GPU-hours, 2 to 3 weeks

The remaining eight models (the two control bases are already done), plus the SFT arm
and seed replicates.

- **Gate C:** does the cross-model Spearman CI exclude zero?

If yes, the full three-claim paper. If no, Stage 0's causal result plus the corpus still
stand, and the paper is written as "amenability is real and measurable, but cross-model
prediction at this scale is harder than the literature suggests," which is a legitimate
and useful negative result given that RuDE's 0.90 is unreplicable.

### Stage 2: analysis and writing. 2 to 3 weeks, no GPU

Including roughly two weeks of harness construction before Stage 0, the project runs
about two and a half months end to end, with the first go/no-go inside the first month.

## 10. Release artifacts

- **Probe harness**, pip-installable, one command per model. This is what other people
  actually use, and what makes the work cited rather than merely read.
- **Frozen pre-registration file**, hash referenced in the paper.
- **Corpus:** training curves, entropy traces, probe telemetry, per-checkpoint eval
  results for all 14 model variants (10 roster models plus 4 over-SFT'd control variants).
- **Over-SFT'd checkpoint variants.** Nobody has released controlled plasticity-degraded
  checkpoints.
- **Static leaderboard page.**

## 11. Venue

Primary target is an OpenReview workshop submission (ICLR 2027 workshops, deadlines
around February 2027). arXiv is treated as contingent rather than load-bearing, since
endorsement has blocked earlier work and should be resolved separately. The repository
and leaderboard go public independently of either.

## 12. Limitations, stated up front

- Validated on the 0.5B to 1.7B nominal size class only, meaning actual parameter
  counts from 0.49B (Qwen2.5-0.5B) to 1.71B (SmolLM2-1.7B). No claim about models
  above 2B.
- Single training target (GSM8K). No claim about code, agentic, or multi-task settings.
- Two algorithms (GRPO, rejection-sampling SFT). No claim about PPO or DPO.
- N=10 observational plus N=6 causal. Observational results are preliminary by design.
- Verifiable-reward capability gain only. No claim about behavioral alignment, safety
  durability, or inference-time steerability.

## 13. Open items before implementation

1. Verify the actual CARC GPU allocation. Determines whether Stage 0 runs on CARC or
   Modal.
2. Confirm base-model availability and licensing for all 10 roster entries, in
   particular that true base (non-instruct) weights are published for Falcon3-1B,
   StableLM-2-1.6B, and Gemma-3-1B-pt.
3. Pilot the probe suite for difficulty calibration: confirm that at 0.5B some Countdown
   buckets are solvable and some are not, so the zero-advantage rate is informative
   rather than saturated.
4. Fix the exact over-SFT token budgets for the positive control arm so that the induced
   degradation is large enough to detect but not so large that the model is destroyed.
