# Probe Transfer Pre-Registration

Committed before the transfer batch runs. Not hash-frozen: the freeze mechanism
belongs to Stage 0, and the product reframe (issue #34) replaces it with metric
versioning. The git history of this file is the audit trail; the batch's telemetry
commits must postdate the commit that adds this file.

## Amendment of 2026-09-28

Made on 2026-09-28, before any batch data existed, in response to the
whole-branch review of the implementation. No model had been run on either
suite. Four changes, each recorded as a ruling in
`docs/decisions/transfer-rulings.md` (T7 to T10) and edited into the sections
below:

- The test-retest pair has the same completeness floor as the transfer pair (T7).
- Every pairwise statistic re-normalises the score over that pair's common
  models (T8).
- A statistic whose input is constant is undefined, and an undefined transfer
  or test-retest rho is its own inconclusive decision (T9).
- Telemetry is validated against its meta file; a mismatch is a named job
  failure (T10).

## Question

Q1, probe invariance: does a model's probe score depend on which probe suite
measures it? Ten roster models, two suites (Countdown, Graph path), one seed
replicate on Countdown. No ground truth is involved.

## Fixed quantities

- Protocol: the Stage 0 probe protocol, unchanged. 60 GRPO steps, 8 generations,
  4 prompts per optimiser step, lr 1e-6, beta 0.04, temperature 1.0, 300 items in
  three buckets, pass@k at (1, 8, 32, 64) from 64 samples.
- Score: `amenability_score` from `src/amenability/scoring/score.py`, applied per
  suite across the ten roster models.
- Common-set scoring (amended, T8): the score is relative to the roster it is
  computed over, and its zero-advantage gate means a change of roster can change
  the ranks, not only the scale. So for each pairwise statistic, the transfer rho
  (Countdown s0 against Graph s0) and the test-retest rho (Countdown s0 against
  Countdown s1), each arm is scored separately over the models present in BOTH
  arms of that pair, and those two vectors are correlated. Per-feature rhos use
  raw features and only need the same pairing. The leaderboard preview keeps
  per-arm scores over each arm's full scored set and says so.
- Accepted telemetry (amended, T10): a job's telemetry counts only if its meta
  file exists and records learning_rate, probe_steps and item seed equal to the
  protocol above (`ProbeConfig()` defaults, item seed 0) and a number of step
  records equal to probe_steps. A missing, unreadable or mismatched meta makes
  the job a named failure, like a missing telemetry file.
- Items: generated from item seed 0 for every arm. Run seed 0 for the two suite
  arms, run seed 1 for the Countdown replicate.
- Statistics: `spearman_with_ci` (percentile bootstrap, permutation p, 10,000
  resamples) and `loo_spearman` from `src/amenability/eval/stats.py`.

## Statistics reported

| Statistic | Between | Purpose |
|---|---|---|
| Transfer rho | score(Countdown, s0), score(Graph, s0) | The answer |
| Test-retest rho | score(Countdown, s0), score(Countdown, s1) | Reliability ceiling |
| Per-feature transfer rho | conversion_rate, retention_factor, zero_advantage_rate across suites | Which feature is domain-sensitive |
| Static comparator rho | pre-probe pass@32 across suites | Does the dynamics signal transfer beyond raw capability? |
| Leave-one-out transfer rho | as transfer rho, each model held out | No single model carries it |

The calibration table also reports, per model and suite, whether the suite is
floored (pass@32 < 0.02 in every bucket), saturated (pass@1 > 0.95 in every
bucket) or breadth-saturated (pass@32 > 0.95 in every bucket). If Graph breadth
is saturated in 3 or more models, the batch stops before launch for a ruling
(see `docs/decisions/transfer-rulings.md`, T4). This is decided before any model
has run.

Bucket changes decided at the calibration pilot (Countdown floored, the Graph
(7, 9, 11) contingency, or a Graph breadth-saturation remedy) are pre-batch
calibration. Each is recorded as a ruling before the batch launches, and none is
a protocol change after data.

## Decision rule

Evaluated in this order; the first matching row is the decision.

| Condition | Decision |
|---|---|
| Fewer than 8 of 10 models have both seed-0 arms scored | INCONCLUSIVE_INCOMPLETE: rerun failures |
| Fewer than 8 of 10 models have both Countdown arms (s0 and s1) scored (amended, T7) | INCONCLUSIVE_INCOMPLETE: rerun failures |
| Transfer rho or test-retest rho undefined because one of its input vectors is constant (amended, T9) | INCONCLUSIVE_DEGENERATE: the statistic carries no information; report which, do not read it as rho 0 |
| Test-retest rho < 0.5 | INCONCLUSIVE_RELIABILITY: the probe is too noisy at N=10; change the protocol, do not read the transfer rho |
| Transfer rho >= 0.7 and permutation p < 0.05 | INVARIANT: single score defensible; Stage 0 proceeds as designed |
| Transfer rho >= 0.4 | PARTIAL: composite plus per-domain columns; Stage 0 proceeds |
| Otherwise | DOMAIN_SPECIFIC: per-domain product; #28 and Stage 0 re-planned per domain before further GPU spend |

The transfer rho is always reported beside the test-retest rho, never bare. No
rho is computed on a subset of models without its N printed beside it. A
statistic with a constant input is printed as "undefined (constant input)" with
its N, never as 0.

A no-breadth exclusion: a model whose pre-probe pass@32 equals its pass@1 on a
suite has no conversion ratio (`extract_features` raises). It is excluded from
that suite's arm by name, and N for every rho involving that arm drops accordingly.

## The one permitted protocol change

If the Qwen2.5-0.5B Countdown smoke probe shows post pass@1 <= pre pass@1, lr 3e-6
and 5e-6 are tried on that one model only, the smallest lr showing gain is pinned
for every model and both suites, and the change is recorded as a ruling with the
three measured gains. Bound here so it cannot become a post-hoc search after the
batch is seen. No other protocol parameter changes.

## What will not happen

- No change to the score formula, feature definitions or protocol after seeing
  any batch result, other than the lr rule above, which is decided before the batch.
- No choosing between Countdown and Graph as "the" probe by which gives the nicer
  leaderboard.
- No dropping a model from the analysis except for a named, documented job failure
  (including a meta mismatch, T10) or a documented no-breadth exclusion, both
  listed in the report.
- No reading the transfer rho without the test-retest rho beside it.
