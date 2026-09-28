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
