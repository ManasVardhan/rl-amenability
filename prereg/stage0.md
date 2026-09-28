# Stage 0 Pre-Registration

Frozen before any Stage 0 ground-truth run. The commit hash of this file is cited in the paper.

## Hypotheses

- **H1 (Gate A):** The amenability score recovers the known ordering of RL amenability
  across over-SFT'd variants, within both control families. Amenability degrades
  monotonically with SFT token budget.
- **H2 (Gate B):** The amenability score predicts the full-run outcome better than naive
  extrapolation of the probe reward trace.

## Fixed quantities

- Score: `gate * mean(z(conversion_rate), z(retention_factor))`,
  `gate = clip(1 - zero_advantage_rate, 0, 1)`. Equal weights, not tuned.
- Outcome variable (primary): conversion ratio,
  `(post_pass@1 - pre_pass@1) / (pre_pass@32 - pre_pass@1)` measured on the target suite
  after the 600-step full run.
- Outcome variable (secondary): raw `post_pass@1 - pre_pass@1`.
- Probe budget: 60 GRPO steps, 150 SFT steps. Full budget: 600 GRPO steps, 1500 SFT steps.
- Breadth k = 32. pass@64 computed at endpoints only.

## What the freeze mechanism does and does not guarantee

The freeze is a SHA-256 manifest of the analysis files, stored at `prereg/FROZEN.json` in
this same repository and committed to git. `verify_freeze` runs before any ground-truth
run and refuses to proceed if a hashed file has changed.

What it guarantees: the analysis code cannot drift accidentally or incrementally between
the freeze and the run. An edit to the score, the baselines, the gates or the roster
after freezing halts the pipeline with a named violation rather than silently changing a
result. Because the manifest is committed, it also records a git-visible point in history,
and the manifest names the commit that was HEAD when the freeze was taken.

What it does NOT guarantee: it is not cryptographic protection against a determined
operator. The manifest is unsigned and lives in a repository writable by whoever can edit
the analysis code, so deleting the manifest, editing a file and re-freezing would produce
a self-consistent state that passes verification. Defeating the freeze therefore requires
a deliberate, git-visible sequence of actions rather than a silent edit, and that is the
property being relied on. The guarantee is procedural and auditable, not cryptographic.

Anyone auditing this work should inspect the git history of `prereg/FROZEN.json` and
confirm that the commit it names precedes the commits containing Stage 0 results.

### Which files are covered

The manifest covers the analysis DEFINITIONS only:

- `src/amenability/scoring/score.py`
- `src/amenability/scoring/baselines.py`
- `src/amenability/scoring/gates.py`
- `src/amenability/registry/models.yaml`
- `prereg/stage0.md`

It does NOT cover the orchestration and execution code, principally
`scripts/run_stage0.py` and `scripts/real_runner.py`. Those files determine which
checkpoints are probed, which suites are used, and how results are grouped before the
gates see them, so they influence outcomes even though they compute no score themselves.
They are deliberately left unfrozen because they are operational code that legitimately
changes for logging, command-line arguments and defect fixes, and freezing them would
create pressure to unfreeze casually, which would weaken the mechanism more than the
residual risk it removes.

The consequence is that an auditor must read the git history of the orchestration code
alongside the manifest, not the manifest alone. Two specific properties worth checking
there: that both pre-registered control bases were used, which `run_stage0.py` asserts at
runtime, and that the probe and target suites remained disjoint, which the suite registry
enforces by raising on any overlapping task identifier.

## Gate thresholds

- **Gate A passes** when the score's ordering is correct within both control families
  AND pooled Spearman rho between score and known ordering is >= 0.7.

  How the pooling works, because this is a load-bearing methodological choice.
  Scores are pooled across families only AFTER within-family MEAN-CENTRING: each
  family's scores have that family's mean subtracted before the two families are
  concatenated and the single pooled rho is computed. Centring is necessary because
  `amenability_score` z-scores across the whole roster, so two families can separate
  in absolute score level; pooling raw scores would let family membership dominate
  the pooled statistic and fail this gate on a result whose ordering is perfectly
  correct within each family. Verified: correctly ordered families occupying disjoint
  score ranges give pooled rho 0.478 under raw pooling and 0.956 under centred
  pooling.

  The BINDING condition is the within-family ordering check. The pooled rho is a
  graded SECONDARY statistic, and centring is what keeps it informative: within-family
  ranking would also remove the family offset, but it would force rho to exactly 1.0
  for any correct ordering, making the 0.7 threshold vacuous and collapsing the
  conjunction to a single criterion. Under centred pooling, rho still varies with
  within-family spacing, so a correct but compressed ordering measures lower (0.837 in
  a verified case) and can in principle fail a higher threshold. Consequence to state
  plainly: because rho retains spacing information, it is sensitive to how unevenly
  the checkpoints are spaced and not only to whether they are ordered correctly.
- **Gate B passes** when `|rho(score, outcome)| > |rho(naive_extrapolation, outcome)|`
  on the six control checkpoints.

## Decisions bound in advance

- Gate A fails and Gate B fails: the probe design is wrong. Stop and redesign.
- Gate A passes and Gate B fails: the probe works but is not cheap enough. Pivot the
  paper toward failure-mode detection rather than prediction efficiency.
- Both pass: proceed to Stage 1.

## What will not happen

- No reweighting of the score after seeing ground truth.
- No addition or removal of features from the score.
- No change of outcome variable after seeing results.
- No dropping of models from the roster except for a documented infrastructure failure.

## Known properties of the frozen analysis code

These were established during implementation, before any ground-truth run. They are
recorded because a reader reproducing this work needs them, and because stating them
now prevents them being presented later as post-hoc discoveries.

### Score normalisation

- `zscore` uses numpy's default `ddof=0`, that is the POPULATION standard deviation, not
  the sample standard deviation.
- Scoring is RELATIVE ACROSS THE ROSTER. `amenability_score` z-scores across the list it
  is given, so a model's score depends on which other models it was scored alongside.
  Consequences: a single-model roster always scores exactly 0.0, and adding or removing a
  model retroactively changes every other score in that call. Scores from different
  roster calls are therefore not comparable. In particular, Stage 0's six control
  checkpoint scores are not on the same scale as any later cross-model roster's scores.
- `zscore` returns all zeros when the standard deviation falls below 1e-12. This is a
  discontinuity: crossing that threshold flips the output from all-zeros to bounded
  values. It is only reachable when inputs differ at around the 1e-12 absolute scale,
  which does not describe genuinely distinct models.
- The score rejects non-finite inputs rather than propagating them. This guard is input
  validation, not part of the registered formula.

### Retention factor saturation

`kl_per_gain` divides total KL by `max(reward_gain, 1e-6)`. `reward_gain` is
`rewards[-1] - rewards[0]` over the RECORDED step trace, so the floor engages for any
probe whose FINAL recorded reward does not exceed its first. That is the precise
condition: it is a property of the first and last recorded steps, not of the trace's
overall trend. Where the floor engages, `kl_per_gain` becomes very large and
`retention_factor` collapses toward zero. Two consequences:

- A small reward decline and a large reward decline are indistinguishable in
  `retention_factor`; both saturate near zero.
- A perfectly FLAT reward trace also saturates, not only a declining one. Verified: a
  healthy run with flat reward and stable entropy yields a retention factor of about
  2e-5.

A defect that made this condition apply UNIVERSALLY was found and fixed before the
freeze, and is recorded here because it changes how this section should be read.
`transformers`' `Trainer._finalize_training` calls `self.log(metrics)` after training
ends, which fired the telemetry callback one extra time with no preceding reward pass
and appended a spurious final step whose `mean_reward` was 0.0. That phantom step made
`rewards[-1] - rewards[0]` negative for every probe, so the floor engaged and
`retention_factor` collapsed for every variant regardless of its real reward trace. The
callback now skips a log that carries no training-step rewards, distinguished by whether
any rewards were actually buffered rather than by whether the mean is zero, since a
legitimate step can score every completion 0.0. Without that fix the saturation
described here would not have been a limitation affecting some runs; it would have
silently affected all of them.

This matters for Gate A, because heavily over-SFT'd control variants are the ones most
likely to show flat or declining probe reward, and the score's ability to rank them then
rests on `conversion_rate` alone. The formula is deliberately NOT being changed to
address this: it was fixed before any results were seen, and Gate A's outcome is what
will reveal whether the saturation actually bites. If Gate A fails, this limitation is
the first thing to examine, and it must be reported as an instrumental explanation rather
than treated as evidence about the models.

### Non-finite telemetry is rejected, for a specific reason

`extract_features` raises on any non-finite entropy, KL, gradient norm or reward value.
This is not routine hygiene. Without it, a diverged run whose entropy measurement went
non-finite would be scored as having PERFECT entropy retention, because
`max(0.0, -nan)` evaluates to 0.0 rather than nan in Python, giving an entropy term of
1.0, the maximum. The failure mode was silent and biased in the favourable direction,
awarding the best possible retention to precisely the runs that diverged.

### Baseline extrapolation may exceed the accuracy range

`naive_extrapolation` fits `y = a + b*log(1+x)` without constraint and is deliberately
NOT clamped to [0,1], so it can return values slightly above 1.0. Clamping was
considered and rejected: the baseline is compared to the score only by RANK, via
Spearman, so an overshoot preserves ordering and costs nothing, whereas clamping would
saturate multiple models at exactly 1.0, create artificial ties, and weaken the
baseline's ability to discriminate. A weakened baseline would make the score look better
by comparison, biasing the comparison in this work's favour. This decision is safe ONLY
because the consumer is rank-based; if any later analysis compares these values as
magnitudes, it must be revisited.

### Batching is by sequences, not tokens

The design spec asks for batches "sized by tokens, not sequences, for tokenizer
comparability". TRL's `GRPOConfig` offers no token-based batching, and building it was
judged out of scope before any run. The batch shape is therefore PINNED by sequence
count instead: `per_device_train_batch_size = num_generations`, so one device batch holds
exactly one prompt's completion group, and `gradient_accumulation_steps = 4`
(`PROMPTS_PER_STEP`), so four groups are accumulated per optimiser step. Leaving these
unset, as the code originally did, means TRL's defaults apply and the effective protocol
silently depends on the number of processes; pinning them makes the protocol identical
across models and machines. Every run records the shape it used in its run manifest.

The deviation's cost is the one the spec's wording anticipated: models whose tokenizers
differ in average tokens per completion see different token counts per optimiser step, so
the comparison across tokenizers is not token-normalised. This is a stated limitation of
Stage 0, not a silent choice, and it must be reported as such.

### Confidence intervals are percentile, not BCa

The design spec asks for BCa intervals. Percentile intervals are used instead, because
BCa's acceleration term is jackknife-estimated and degenerates when bootstrap resamples
produce constant vectors, which happens at N=6 and yields NaN intervals. Percentile
intervals are wider and more conservative, which is the correct direction to err at this
sample size. Any publication arising from this work must say percentile, not BCa.

### Rank correlation of degenerate inputs

- `_rho` converts scipy's NaN, returned when an input has no variance, to 0.0, meaning no
  detected association. Applied uniformly to the point estimate, every bootstrap
  resample, every permutation and every leave-one-out fit. A bootstrap resample
  degenerates only when all n drawn indices are identical, with probability
  n*(1/n)^n, which is 0.0129% at n=6 and negligible above that. Converting to 0.0 rather
  than propagating NaN is necessary, not merely convenient: a percentile over a
  NaN-containing array returns NaN, so one degenerate resample would otherwise destroy
  the whole interval.
- `partial_spearman` returns 0.0 when either residual array has a standard deviation
  below 1e-9. Without this guard it returns 1.0 when the control fully explains the
  signal, which is the reverse of the truth, because nothing survives a control that
  explains everything. This matters specifically for the headroom confound control, whose
  purpose is to detect exactly that situation.

## Resolved before the freeze

The freeze must not be executed until these are settled, because each one changes frozen
content:

- Confirmation that true base, non-instruction-tuned weights exist for every roster
  model. An instruction-tuned checkpoint has already been post-trained, so its measured
  amenability is not the base-model property this work claims to measure.
- Probe-suite difficulty calibration at the smallest model size. The zero-advantage rate
  is a scoring input and only carries information if some difficulty buckets are solvable
  and some are not; if none are solvable the rate saturates and the score's gate term
  floors for every model.
- The over-SFT token budgets for the positive control. Gate A's validity depends on those
  checkpoints having a genuinely ordered amenability.
