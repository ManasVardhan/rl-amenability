# Probe Transfer Pre-Registration

Committed before the transfer batch runs. Not hash-frozen: the freeze mechanism
belongs to Stage 0, and the product reframe (issue #34) replaces it with metric
versioning. The git history of this file is the audit trail; the batch's telemetry
commits must postdate the commit that adds this file.

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

## Decision rule

Evaluated in this order; the first matching row is the decision.

| Condition | Decision |
|---|---|
| Fewer than 8 of 10 models have both seed-0 arms scored | INCONCLUSIVE_INCOMPLETE: rerun failures |
| Test-retest rho < 0.5 | INCONCLUSIVE_RELIABILITY: the probe is too noisy at N=10; change the protocol, do not read the transfer rho |
| Transfer rho >= 0.7 and permutation p < 0.05 | INVARIANT: single score defensible; Stage 0 proceeds as designed |
| Transfer rho >= 0.4 | PARTIAL: composite plus per-domain columns; Stage 0 proceeds |
| Otherwise | DOMAIN_SPECIFIC: per-domain product; #28 and Stage 0 re-planned per domain before further GPU spend |

The transfer rho is always reported beside the test-retest rho, never bare.

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
  or a documented no-breadth exclusion, both listed in the report.
- No reading the transfer rho without the test-retest rho beside it.
