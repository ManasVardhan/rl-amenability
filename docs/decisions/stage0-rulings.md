# Stage 0 Execution Rulings

Decisions taken during subagent-driven execution of
`docs/superpowers/plans/2026-09-22-stage0-probe-harness.md`, in the order they
were made. Each names what was decided, why, and what it costs if the decision
turns out to be wrong.

Most of these are corrections to defects in the plan itself, surfaced either by
the pre-flight conflict scan or by per-task review. That is the expected
pattern: the plan was written in one pass, and review is where its gaps show.

## Pre-flight (before any code)

A conflict scan over all 18 tasks, checking every pair that shares a file or an
interface, and every task against its own text.

**R1. Feature branch, not a git worktree.**
Work happens on `stage0-probe-harness` in place. The repo is new,
single-purpose, has no concurrent work, and worktree indirection would
complicate SLURM and Modal path resolution in Task 18.
*Cost if wrong:* `main` is not independently checkoutable during execution.

**R2. Task 17 must call `baseline_naive`, not re-implement extrapolation.**
The plan text duplicated a logic block Task 12 already owns. The spec treats
naive extrapolation as one pre-registered baseline with one definition.
*Cost if wrong:* none material; prevents future drift between two copies.

**R3. `spearman_with_ci` gains an optional `n_perm` defaulting to `n_boot`.**
As written it ran 10,000 permutations regardless of the `n_boot` a caller
passed, so Task 15's gates would burn 20-40s per suite run for no statistical
benefit.
*Cost if wrong:* slightly less precise permutation p-value inside gate
evaluation, where only the point estimate is used.

**R4. Task 17's `main()` constructs `Stage0Config()` and passes it to `RealRunner(config)`.**
The plan showed `RealRunner()` with no arguments in Task 17 while Task 18
defined `__init__(self, config, ...)` with `config` required. Straight
TypeError at launch.
*Cost if wrong:* a TypeError caught immediately.

## Task 1: scaffolding and roster

**R5. `vllm` moves to a `gpu` optional extra.**
Declared as a base dependency, it made `uv sync --extra dev` fail on every
non-Linux machine. Verified that torch resolves fine on darwin and vllm is the
sole blocker. Import-time audit across all 18 tasks: `vllm`, `trl` and
`datasets` are imported lazily inside functions and are never needed for the
CPU test suite; only `torch` and `transformers` are needed at module level.
*Cost if wrong:* GPU environments must use `uv sync --extra dev --extra gpu`,
carried into Task 18's sbatch and Modal scripts.

**R6. The size-band check keeps bounds 0.3-1.8 and gets an honest message.**
The test asserted `0.3 <= params_b <= 1.8` while its failure message claimed a
"0.5-1.7B band". Two roster entries breach a literal reading: Qwen2.5-0.5B is
0.49B and SmolLM2-1.7B is 1.71B. But "0.5B" and "1.7B" are nominal size-class
names, not exact parameter counts, and the bound exists as a compute-budget
guard against a 7B model entering the roster. Fixed the lying message rather
than rejecting two valid models, and corrected the spec's wording so the
paper's limitations section is accurate.
*Cost if wrong:* the roster admits a model between 0.3B and 0.5B that a
stricter reading would exclude.

**R7. `models.yaml` joins `FROZEN_PATHS` (applies to Task 14).**
`load_registry` enforces no roster invariants at runtime, so a hand-edited
roster is undetectable. `prereg/stage0.md` already commits in prose to "no
dropping of models from the roster except for a documented infrastructure
failure"; freezing the file is what makes that enforceable rather than
decorative.
*Cost if wrong:* a legitimate roster change requires a documented unfreeze,
which is the intended behaviour.

## Task 2: suite registry

**R8. Add an atomicity regression test, taking the file past the plan's five-test list.**
Review hand-traced `register()` as genuinely atomic but showed the tests would
pass unchanged against a naive single-pass implementation that mutates then
raises. Spec section 5 makes probe/target disjointness a code-enforced
invariant; an invariant whose only protection is untested code regresses
silently on the first readability refactor.
*Cost if wrong:* one extra test of about six lines.

**R9. The reviewer's `local`-set Minor is incorrect; no change made.**
It claimed the `local` set was redundant with `self._seen`. It is not: `local`
catches duplicate task IDs within a single batch, which `self._seen` cannot,
precisely because the validation pass deliberately does not write to
`self._seen` before validating. Removing it would break a test and reopen the
atomicity hole.
*Cost if wrong:* none, code unchanged.

## Task 3: countdown suite

**R10. Commits in this repo carry no AI attribution trailers.**
A harness-level instruction tells agents to append `Co-Authored-By: Claude ...`
and `Claude-Session: ...`; the repo owner's `CLAUDE.md` forbids exactly that.
Standardised on `CLAUDE.md`, being the owner's explicit standing preference
about their own repository. Known inconsistency: two early commits
(`7aa120e`, `86d2da4`) carry the trailers because the harness instruction was
followed before the conflict surfaced. Not rewritten unilaterally.
*Cost if wrong:* the branch lacks session-provenance trailers on most commits.

## Task 5: pass@k evaluator

**R11. Add three input guards to `evaluate_passk`.**
`zip(items, completions)` truncates silently on length mismatch, and duplicate
task IDs collapse the prompts dict so every later completion is attributed to
the wrong item. Review judged this unreachable because `SuiteRegistry` enforces
uniqueness; that reasoning does not hold. Task 18's target-side evaluation uses
an items list from `load_gsm8k(...)` that is never registered, so the
uniqueness guarantee does not cover it. Safe today only because dataset indices
happen to be unique. A silent misattribution in the measurement layer would
corrupt every downstream number while every test stayed green.
*Cost if wrong:* raises on malformed input instead of silently producing wrong
numbers.

**R12. Fix the `pass_at_k` docstring.**
It claimed the value is "computed stably in log space"; it is a running product
of ratios. Cosmetic as code, but a false statement in a codebase whose numbers
go into a paper.
*Cost if wrong:* none.

**Declined:** clamping the estimator output to [0,1]. The range is
mathematically guaranteed by the guard branches, and a clamp would mask a
genuine estimator bug rather than reveal one.

## Task 6: telemetry and features

**R14. `extract_features` raises on an empty steps list.**
It previously returned a `FeatureVector` carrying
`zero_advantage_rate = nan` from `np.mean([])`, with only a RuntimeWarning. That
NaN is a scoring input, and because every NaN comparison is False it would not
even surface as an outlier during gate ranking: the run would produce something
that looks like a result and is not one. Confirmed reachable, not theoretical:
Task 9's runner returns records that accumulate only inside `on_log`, so a run
raising before its first logging step hands Task 18 an empty list.
*Cost if wrong:* a probe that legitimately produced no logging steps errors
instead of scoring, which is intended.

**R15. Do not change the `kl_per_gain` floor; document its saturation (applies to Task 14).**
`max(reward_gain, 1e-6)` makes every run whose reward declined collapse to
`retention_factor` near zero, so a small decline and a large decline are
indistinguishable. This is a real resolution loss and could matter for Gate A,
since heavily over-SFT'd control variants are the ones most likely to decline.
Not redesigning a pre-registered formula mid-execution on speculation: the
semantics are defensible, `conversion_rate` separates those variants
independently, and Gate A's outcome is what will reveal whether saturation
bites. Documenting before the freeze is honest; changing the formula after
seeing Stage 0 results would not be.
*Cost if wrong:* Gate A fails for an instrumental rather than a real reason,
diagnosable from this note.

**R13. Commit `uv.lock`.**
The plan never mentioned it. Reproducibility is a stated release artifact, and
TRL, transformers and vllm all change RL-relevant defaults between minor
versions, so an unpinned environment could shift Stage 0 results without any
change to this repository.
*Cost if wrong:* a 1.3MB file in git history.

## Task 7: entropy measurement

**R16. Add mock-model coverage for `EntropyProbe.measure()`.**
The plan tested only the pure `mean_next_token_entropy` function, leaving
tokenization, the device move, the `no_grad` wrapper and the train/eval restore
unverified. The restore is the dangerous one: a bug there does not fail a test,
it silently leaves the model in eval mode and disables dropout for every
subsequent training step. A measurement instrument corrupting the run it
measures, invisibly.
*Cost if wrong:* four extra tests using fakes.

## Task 9: GRPO probe runner

**R17. Add a run manifest to `run_grpo`.**
This fixes two findings at once. Review found `checkpoint_schedule` was dead
code, never called from anywhere including all downstream task briefs.
Separately, the plan's own Global Constraints say "Every run is seeded and every
seed is recorded in the run manifest. Reproducibility is a release artifact" and
NO task implements a run manifest. That is a spec requirement with no
implementing task, which the plan self-review was supposed to catch and did not.
Writing a manifest from `run_grpo` satisfies the constraint and gives
`checkpoint_schedule` a real consumer, so the function stops being decorative.
*Cost if wrong:* one small JSON per run, which the spec already asked for.

**R18. Close the LoRA guard gap with a post-construction assertion.**
The guard is airtight only against `run_grpo(spec, peft_config=...)` and is
bypassed entirely by a caller-supplied `trainer_factory`, since TRL's
`GRPOTrainer` accepts `peft_config` itself. Full-parameter-only is a research
requirement, not a style preference: LoRA constrains how far weights can move
and that is the measured quantity. Asserting after construction that the model
is not PEFT-wrapped covers both paths.
*Cost if wrong:* a legitimate PEFT experiment would have to remove the
assertion, which is the intended friction.

**R19. Guard duplicate prompt text in `build_reward_fn`.**
Same class as R11 but a different collision axis: prompt text, not task_id.
Confirmed reachable in `generate_countdown`, which draws numbers with no
collision check, though harmless there because the answer is a pure function of
the prompt. Nothing stops a future suite from producing identical prompts with
different answers, where rollouts would be silently mis-scored.
*Cost if wrong:* a suite with genuinely duplicate prompts must deduplicate
before training.

**R20. Add a batch-level match-rate guard.**
If NO prompt in a reward call matches a known item, raise. This is the one that
would have saved a whole run: a systemic prompt mismatch (templating,
whitespace, a prompt-format change) would otherwise score every completion 0.0,
and the run would complete and report a clean zero-amenability result rather
than an error. A single unmatched prompt still scores 0.0; only a total miss
raises. TRL was verified to pass raw prompts so this should never fire, which is
precisely why it is cheap insurance.
*Cost if wrong:* a legitimate all-unmatched batch raises instead of scoring
zero, which is the intended behaviour.

**R21. Accept the implementer's deviation on the match-rate guard's condition.**
The original instruction was self-contradictory and the implementer was right to
flag it rather than pick a side silently. Their resolution changes the condition
to `len(prompts) > 1`, so a genuine multi-completion batch with zero matches
raises while the single-unknown-prompt contract is preserved. That is sound:
GRPO batches are always a multiple of `num_generations` and group-relative
advantage is undefined for a group of one, so a real batch is never size 1. The
guard still fires on every realistic systemic-mismatch scenario, which is the
failure mode R20 exists to catch.
*Cost if wrong:* a size-1 batch with an unknown prompt scores 0.0 instead of
raising, which is unreachable in GRPO.

## Task 10: SFT runner

**R22. Extend R18 to the SFT arm.**
`sft.py` carried the same LoRA bypass gap that R18 fixed in `grpo.py`: it checked
only the `peft_config` parameter, with no post-construction
`hasattr(model, "peft_config")` check. Since `run_sft` calls `trainer_factory()`
with no arguments, an injected factory fully controls construction and TRL's
`SFTTrainer` accepts `peft_config` directly. The post-construction PEFT check
must exist in both training runners or full-parameter-only is enforced on the RL
arm and merely advisory on the SFT arm, which would be an incoherent invariant
for a benchmark whose entire premise is measuring how far weights move.
*Cost if wrong:* the same intended friction as R18, now on both arms.

## Task 11: amenability score

**R23. Reject non-finite telemetry at the feature boundary.**
`extract_features` already raises on empty steps per R14; it must equally raise
when any step carries a non-finite entropy, kl, grad_norm or reward, naming the
model and step. This is the root cause and the only place the diagnosis is
legible. Guarding inputs is NOT tuning the pre-registered formula: the formula
assumes finite inputs and the `max(0.0, -entropy_slope)` expression itself is
untouched.
*Cost if wrong:* a diverged run raises instead of silently scoring 1.0
retention, which is the entire point.

**R24. Add defence in depth at the score boundary.**
`amenability_score` raises if any of the three scoring inputs is non-finite. Two
independent guards are justified because this is the number the paper reports and
the failure is silent in the favourable direction. This crosses a task boundary:
`telemetry.py` is Task 6's deliverable and Task 6 is closed. Fixing it inside
Task 11's fix round is deliberate, because respecting task boundaries at the cost
of leaving a known silent mis-scoring in the tree would be the wrong trade. Both
files are still pre-freeze, so this is the correct and last window for the change.
*Cost if wrong:* one redundant guard on a code path that should already be clean.

**R25. Do NOT clamp `naive_extrapolation` to [0,1]; document the overshoot in the
pre-registration instead.**
Clamping looks like obvious hygiene and would actively bias the research result
in this work's favour. The baseline is only ever compared to the score by RANK,
through Spearman, so an overshoot to 1.0076 preserves ordering and costs nothing.
Clamping, by contrast, would saturate several models at exactly 1.0, create
artificial ties, and destroy the baseline's ability to discriminate between them.
A weaker baseline makes the score look better by comparison. Leaving the
overshoot is the honest choice, and it is only safe to leave because the consumer
is rank-based; if any later analysis compares these values as magnitudes rather
than ranks, this decision must be revisited.
*Cost if wrong:* a baseline value outside the nominal accuracy range appears in
the results table and needs a footnote.

## Task 13: bootstrap statistics

**R26. Guard degenerate residuals in `partial_spearman`.**
If either residual array has a standard deviation below 1e-9, return 0.0. Rank
values are O(n) so genuine residual variation is many orders of magnitude above
that, while the degenerate case sits at 1e-15. Verified: the guard returns 0.0
for the fully-explained case and leaves legitimate cases untouched.
*Cost if wrong:* a genuine association whose residuals are pathologically small
reports no association.

**R27. Replace the degenerate partial-correlation test with three that actually
exercise the property.**
Use seeded independent perturbations rather than a self-comparison. Verified
numerically: a shared driver gives raw rho 0.877 collapsing to partial -0.178; a
genuine association with an unrelated control gives raw 0.990 and partial 0.990,
correctly preserved; and the fully-explained case returns 0.0 under the R26
guard. The original test could not demonstrate the property at all, because after
residualising a self-comparison both sides are the same numerical noise.
*Cost if wrong:* two extra tests.

## Task 14: freeze mechanism

**R28. Accept the implementer's `pyproject.toml` change, `pythonpath = ["src", "."]`.**
It was necessary, not cosmetic. The plan's pyproject set `pythonpath = ["src"]`
only, but `prereg/` sits at the repository root rather than under `src`, so
`import prereg` failed with ModuleNotFoundError even with a correct
implementation. Another cross-task edit (`pyproject.toml` is Task 1's
deliverable) and another gap the plan self-review missed: a package was specified
whose location was incompatible with the path configuration already specified.
The change is one line and additive, so it cannot break the existing src-based
imports.
*Cost if wrong:* the repository root is importable in tests, which is standard
for a project with root-level tooling packages.

**R29. Record the git HEAD commit SHA in the freeze manifest.**
This makes the code honour a commitment `prereg/stage0.md` already makes in
prose, and it is the difference between a pre-registration that can be checked
and one that must be trusted. With the SHA recorded, the paper cites exactly what
the tooling captured and any reader can verify that commit precedes the data
commits in history. Must degrade gracefully when git is unavailable rather than
failing the freeze.
*Cost if wrong:* one extra field in a JSON file, and a subprocess call at freeze
time.

**R30. Disclose the freeze mechanism's actual guarantee in `prereg/stage0.md`.**
The document did not state the mechanism's limits. This matters for research
integrity rather than for code: describing a same-repo, unsigned hash manifest as
a "hash-frozen pre-registration" without qualification would oversell it, since
anyone who can edit `score.py` can delete `FROZEN.json` and re-freeze. The honest
framing is that it prevents accidental and incremental drift, and creates a
git-visible audit trail, but provides no cryptographic guarantee against a
determined operator. Stating that plainly costs nothing and is the difference
between a credible methods section and an overclaim a reviewer will catch.
*Cost if wrong:* a paragraph acknowledging a limitation that is obvious to anyone
who reads the code anyway.

## Task 15: gates

**R31. Change Gate A's pooling to within-family ranks. SUPERSEDED by R34.**
Raw pooling was PROVEN to fail on a correct result: two correctly ordered
families occupying disjoint score levels gave pooled rho 0.478 against a 0.7
threshold. The change was made to match the intent the code comment already
documented. Justification, stated carefully because this modifies a
pre-registered criterion: (a) the file was NOT yet frozen, which is exactly the
window in which such a change is legitimate, and the last one; (b) it makes the
implementation match its own documented intent, written before any results
existed, so it is not a new criterion; (c) NO ground-truth data exists anywhere
in this project, so this cannot be results-driven tuning, which is the specific
harm pre-registration guards against; (d) the alternative is shipping a primary
gate already proven to fail on correct results in the most likely scenario and
disclosing that as a known likely failure, which is worse.

The honest consequence, disclosed rather than hidden: under within-family RANK
pooling with two equal-size families of three checkpoints, a correct ordering
always yields rho = 1.0, so the pooled-rho condition becomes largely redundant
with the ordering check rather than independent evidence. That defect is what R34
fixes.
*Cost if wrong:* Gate A becomes insensitive to systematic cross-family
disagreement in score level, which raw pooling would have caught. A real loss,
but not what Gate A was designed to test, and recoverable since Stage 0
telemetry is retained.

**R34. Pool within-family CENTRED scores rather than within-family RANKS.
Supersedes R31's mechanism, not its reasoning.**
Adopted from the final whole-branch review, whose fix is better than R31's.
Measured: ranking returns 1.000 for every correct case, which makes the
pre-registered rho threshold vacuous and collapses the conjunction to a single
criterion, a loosening in the hypothesis's favour. Mean-centring fixes the
family-offset defect equally well (0.956 against the broken 0.478) while
preserving graded information, so a compressed family drops to 0.837 and an
inverted family falls far below the threshold. Both conditions of the conjunction
stay meaningful. R31's rank-based fix solved the defect but damaged the gate;
centring solves it without that cost. R31's legitimacy argument (pre-freeze, no
ground-truth data anywhere, matching documented intent) applies unchanged.
*Cost if wrong:* rho now varies with score spacing, so a genuinely correct but
very unevenly spaced result sits closer to the threshold.

## Task 17: Stage 0 orchestration

**R32. Assert BOTH pre-registered control bases are present.**
Carried to Task 17 rather than a Task 16 fix round. Nothing prevented
`variant_plan` being called with a single base, so a partial single-family
invocation could be silently mistaken for full primary-claim evidence. That
matters because the whole reason for two families is to rule out the objection
that entropy collapse is a Qwen-specific artifact, which the Spurious Rewards
literature makes a live concern. The guard belongs in the consumer, Task 17's
`run_stage0`, not in the plan function whose flexibility the unit tests
legitimately use. Landing it in Task 17's dispatch is cheaper than a fix round
and puts it where the pipeline actually decides what counts as a primary-claim
run.
*Cost if wrong:* `run_stage0` refuses to proceed with a deliberately reduced
control arm, which would then require an explicit code change rather than
passing silently.

## Task 18: launch paths

**R33. Document the frozen/unfrozen boundary in `prereg/stage0.md` rather than
expanding `FROZEN_PATHS`.**
`run_stage0.py` contains the gate wiring and family grouping yet is not frozen,
so editing it after the freeze would be undetectable. Adding it, and Task 18's
`real_runner.py`, to `FROZEN_PATHS` was considered and rejected: the orchestrator
and runner are operational plumbing that will legitimately need changes for
logging, CLI arguments and bug fixes, and freezing them would create steady
pressure to unfreeze casually, which corrodes the mechanism's meaning far more
than the residual risk. What the freeze must protect is the analysis DEFINITIONS
(score, baselines, gates, roster, plan), and those are covered. The honest
response is the same as R30's: state the boundary precisely rather than let a
reader assume the freeze covers everything, and tell an auditor to review the
orchestrator's git history alongside the manifest.
*Cost if wrong:* an undetected post-freeze orchestrator edit remains possible,
mitigated by disclosure and by git history being the audit trail.

## Post-review rulings (R35-R38)

**R35. Accept both of the fix-wave implementer's beyond-instruction changes.**
Neither was scope creep; both prevented a false statement standing in the repo.
(a) The phantom-step fix broke two tests in `tests/test_grpo_runner.py` whose fake
trainers never called the reward function at all. The implementer made them call it, as a
real GRPO step does. Its observation is the lesson worth keeping: the test doubles were
UNFAITHFUL to the real trainer, and that unfaithfulness is why the phantom step survived
eighteen task reviews and 164 passing tests. A fake that does not model the interaction
under test cannot protect the property it appears to cover.
(b) A mandated comment asserted the sequence-batching deviation was recorded in the
pre-registration when it was not. Rather than write a comment that lied, the implementer
added the disclosure and made it true.
*Cost if wrong:* two fakes exercise one more interaction; the prereg carries one more
disclosed deviation.

**R36. R34's claim that "both conditions of the conjunction stay meaningful" is FALSE, and the remedy is disclosure rather than a threshold change.**
Exhaustive enumeration, independently reproduced with 200,000 randomised
correctly-ordered draws across family scales spanning 1e-3 to 1e2, shows the attainable
pooled rho when both families of three order correctly is exactly {0.8367, 0.9562}. The
minimum is 0.8367, so `rho >= 0.7` can never fail while the ordering check passes. Gate A
is effectively a one-part test.
The threshold stays at 0.7. Raising it to 0.9 would make the compressed case bind, but it
would introduce an unstudied instrumental failure mode where a correct ordering with weak
within-family separation fails the gate; 0.9 sits between the only two attainable values,
making it a threshold fitted to the attainable set rather than motivated by the science;
and this gate has already been changed twice, which is where tuning should stop and
documenting should start. Tracked as issue #31 for the owner to decide.
*Cost if wrong:* Gate A is a one-part test at Stage 0, which it has effectively always
been; disclosure makes that honest rather than changing it.

**R37. `telemetry.py` belongs in `FROZEN_PATHS` (parked, issue #32).**
`conversion_rate` and `retention_factor` are defined there, and they are two of the three
quantities the score z-scores. R33 asserts the freeze covers the analysis definitions; it
does not, and the review fix wave edited exactly that file's `conversion_rate`.
*Cost if wrong:* the two scored feature definitions stay editable after the freeze without
detection, and R33's disclosure is inaccurate.

**R38. `run_grpo` must release trainer GPU memory before vLLM initialises (parked, issue #33).**
Same defect class the fix wave closed on the vLLM side. Torch's caching allocator holds
the training reservation when the probe's vLLM engine runs its init memory check at 0.85
utilisation in the same process.
*Cost if wrong:* the first real Stage 0 run OOMs at the first probe. Costs a queue slot,
corrupts no result.
