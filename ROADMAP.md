# ROADMAP.md — Post-Acceptance Hardening Plan
## IBR Frequency Estimator Benchmark (SGSMA Camera-Ready)

**Purpose.** This roadmap is structured for Claude Code (or any agentic coder) to pick up one ticket at a time, execute, and verify. Each ticket is self-contained: it states the problem, the files involved, the fix, and *machine-checkable* acceptance criteria. Dependencies are explicit.

**Context.** The paper has been accepted. This hardening pass exists because the reproduction attempt surfaced several silent bugs and methodology soft-spots that a Reviewer 2 on a journal extension would catch. The goal is to either (a) confirm the camera-ready numbers are defensible, or (b) regenerate them from a fixed pipeline and update the manuscript values.

**Execution protocol.**
- Work one ticket at a time. Do not batch.
- Each ticket ends with either a passing test/assertion or a concrete diagnostic output pasted into the ticket's "Evidence" field.
- Do not modify scenario parameters, metric definitions, or evaluation windows without an explicit ticket authorizing it.
- If a ticket's investigation reveals the premise is wrong, close the ticket with `WONTFIX + justification` rather than silently dropping it.

---

## Epic 0 — Diagnostic Baseline (must finish before anything else)

### T-000 — Establish ground truth on sample-rate handling
**Priority:** 🔴 blocker
**Depends on:** nothing
**Files:** `src/scenarios/base.py`, `src/scenarios/ieee_single_sinwave.py`, `src/analysis/monte_carlo_engine.py`

**Problem.** Diagnostic confirmed `FS_PHYSICS = 1e6`, `1/DT_DSP = 1e4`. `Scenario.generate()` produces signals at 1 MHz. The engine calls `Scenario.run(**params)` which *may or may not* decimate before passing `sc.v` to estimators whose `step_vectorized()` hardcodes `self.dt = DT_DSP = 1e-4`. If no decimation, every DSP estimator silently operates on the wrong time base.

**Action.**
1. Run this diagnostic and paste the output in the Evidence field:
   ```bash
   python -c "
   import sys; sys.path.insert(0, 'src')
   from scenarios.ieee_single_sinwave import IEEESingleSinWaveScenario
   g = IEEESingleSinWaveScenario.generate(duration_s=0.1, freq_hz=60.0, noise_sigma=0.0, seed=42)
   r = IEEESingleSinWaveScenario.run(duration_s=0.1, freq_hz=60.0, noise_sigma=0.0, seed=42)
   print(f'generate(): N={len(g.v)}, dt={g.t[1]-g.t[0]:.2e}, fs={1/(g.t[1]-g.t[0]):.0f}')
   print(f'run()     : N={len(r.v)}, dt={r.t[1]-r.t[0]:.2e}, fs={1/(r.t[1]-r.t[0]):.0f}')
   "
   ```
2. Open `src/scenarios/base.py` and read `Scenario.run()`. Document whether it decimates.
3. Read `src/analysis/monte_carlo_engine.py::_run_estimator` and confirm whether `sc.v` reaches `step_vectorized` at 1 MHz or 10 kHz.

**Acceptance criteria.** A comment-block at the top of `monte_carlo_engine.py` explicitly states which sample rate reaches each estimator and whether this matches each estimator's internal `self.dt`. No code changes yet — just documented truth.

**Evidence field:**

```
CLOSED: MISMATCH CONFIRMED (2026-04-12)

Diagnostic output:
  generate(): N=100001, dt=1.00e-06, fs=1,000,000
  run()     : N=100001, dt=1.00e-06, fs=1,000,000

Findings:
1. Scenario.run() calls generate() with no decimation — signals remain at 1 MHz.
2. MonteCarloEngine._run_estimator() passes sc.v (1 MHz) directly to
   est.step_vectorized(v) with NO decimation.
3. All estimators use DT_DSP = 1e-4 s (10 kHz) as self.dt internally —
   a factor-100 time-base mismatch.
4. calculate_all_metrics() called with hardcoded fs_dsp=10000.0 even though
   signal is at 1 MHz — evaluation window sample counts are 100× larger than intended.
5. Smoke tests (test_dedicated_smoke_no_mc_test.py:127-138) DO correctly
   decimate (if dt_real < 1e-5 → decimate by 100) — smoke results are valid.
6. MonteCarloEngine does NOT decimate — this is the bug addressed by T-100.

Comment block added to src/analysis/monte_carlo_engine.py documenting all findings.
Commit: e11e3db "T-000: Document sample-rate mismatch in monte_carlo_engine.py"
```

---

### T-001 — Build minimal smoke test
**Priority:** 🔴 blocker
**Depends on:** T-000
**Files:** new file `src/tests/smoke_test_minimal.py`

**Problem.** Current `test_dedicated_smoke_test.py` runs Optuna tuning + 50 MC runs + dashboards for every estimator. A smoke test should finish in seconds and pinpoint wiring failures.

**Action.** Create a smoke test that:
1. Verifies `FS_PHYSICS` / `DT_DSP` consistency (or documents the ratio).
2. For each estimator in the canonical 11-method set: on a clean 60 Hz sine (no noise, 1.5 s):
   - runs `step_vectorized(sc.v)`, asserts output is finite and correct length.
   - runs `step(z)` in a loop over the first 2000 samples and asserts `step` and `step_vectorized` agree within 0.05 Hz.
   - asserts `|mean(f_hat[last 50%]) - 60.0| < 0.1 Hz`.
3. For every metric in `metrics.py`: runs `calculate_all_metrics` on identical vectors (`f_hat == f_true`) and asserts every accuracy metric is 0.
4. Prints a table: `estimator | status | mean_err | max_err | step==step_vec | has_nan | time_ms`.

**Acceptance criteria.**
- Smoke test completes in < 30 s.
- Output table identifies exactly which estimators pass and which fail.
- Any estimator failing is flagged with a non-zero exit code.

**Evidence field:** *(paste table output)*

---

## Epic 1 — Core Benchmark Correctness

### T-100 — Resolve sample-rate mismatch in estimator path
**Priority:** 🔴 critical
**Depends on:** T-000, T-001
**Files:** `src/analysis/monte_carlo_engine.py`, `src/estimators/base.py`, potentially every `estimators/*.py`

**Problem.** If T-000 shows that `sc.v` reaches `step_vectorized` at 1 MHz, estimators operate on the wrong `dt`. Three possible fixes, pick one explicitly:

**Option A — decimate in engine.** Before calling `_run_estimator`, downsample `sc.v` by `int(FS_PHYSICS * DT_DSP)` (with anti-alias filter). Update `f_true` likewise. This matches the "dual-rate simulation" claim in the paper.

**Option B — sync `dt` in engine.** Compute `dt_actual = sc.t[1] - sc.t[0]`, set `est.dt = dt_actual`, call `est._update_internals()` (where applicable). Signal stays at 1 MHz but estimators adapt.

**Option C — decimate in `Scenario.run()`.** Make `run()` always return 10 kHz output. `generate()` stays at `FS_PHYSICS` for ground-truth fidelity.

**Recommendation:** Option C is cleanest and matches the paper's claim that "fs_DSP = 10 kHz … reproduces the sampling and acquisition constraints of commercial protection relays." Use a Butterworth anti-alias filter + `scipy.signal.decimate`.

**Action.**
1. Decide between A/B/C and document the choice in `CLAUDE.md`.
2. Implement. Anti-alias filter must be zero-phase (`filtfilt`) only if applied to ground truth; causal for `v` to reflect relay reality.
3. Add a sanity assertion at engine entry: `assert abs(sc.t[1]-sc.t[0] - est.dt) < 1e-10`.

**Acceptance criteria.**
- Smoke test (T-001) passes for IpDFT, TFT, Prony, SOGI-PLL on a clean 60 Hz sine (mean error < 0.01 Hz).
- If Option C chosen: `sc.v` length after `run()` equals `duration_s * 10000`.
- A line is added to `CLAUDE.md` §4 stating the chosen resolution.

**Evidence field:** *(smoke test output after fix)*

---

### T-101 — Eliminate `step()` vs `step_vectorized()` divergence
**Priority:** 🔴 critical
**Depends on:** T-100
**Files:** `src/estimators/prony.py`, `src/estimators/ipdft.py`, other estimators with both paths

**Problem.** `Prony.step()` computes every sample; `Prony.step_vectorized()` computes every 10th sample (`if i % 10 == 0`). Same pattern suspected in IpDFT's Numba core. This means scalar-path unit tests and vector-path benchmarks measure different algorithms.

**Action.**
1. Choose a single source of truth for the decimation policy (either "compute every sample" or "compute every N samples"). Document in class docstring.
2. Make both paths identical: either both decimate or neither does.
3. Add a unit test in `src/tests/test_scalar_vs_vector.py` that asserts `np.allclose(est.step_vectorized(v), [est_copy.step(z) for z in v], atol=1e-6)` for every estimator that exposes both paths.

**Acceptance criteria.**
- Unit test passes for all estimators in the canonical 11-method set.
- CLAUDE.md §11 adds: "Never allow `step` and `step_vectorized` to implement different algorithms."

**Evidence field:** *(pytest output)*

---

### T-102 — Make Prony deterministic (fix RNG inside Numba core)
**Priority:** 🟠 major
**Depends on:** T-100
**Files:** `src/estimators/prony.py`

**Problem.** `_prony_core` calls `np.random.standard_normal(N)` inside `@njit`. Numba's RNG is per-thread, not seeded by `base_seed`. Monte Carlo runs for Prony are **not reproducible**. Additionally, occasional eigenvalue jitter from this dither produces spurious NaN/outlier frequencies that look like non-convergence.

**Action.**
1. Replace the stochastic dither with a deterministic one. Options:
   - Pass a pre-generated noise vector as a function argument (seeded outside Numba).
   - Use `1e-9 * np.arange(N) / N` (deterministic, no statistical properties needed — we just want numerical well-posedness).
2. Rerun smoke test on a pure sine 1000 times; Prony RMSE std should be exactly 0.

**Acceptance criteria.**
- `prony.py::_prony_core` contains no `np.random.*` calls.
- Reproducibility test: running the MC for Prony twice with the same `base_seed` produces bit-identical `summary_df`.

**Evidence field:** *(output of `df1.equals(df2)`)*

---

### T-103 — Fix ZCD nominal frequency default (60 Hz, not 50 Hz)
**Priority:** 🟠 major
**Depends on:** none
**Files:** `src/estimators/zcd.py`

**Problem.** `ZCDEstimator.__init__(nominal_f=50.0)` and `default_params(){"nominal_f": 50.0}`. The entire project targets 60 Hz (`F_NOM = 60.0` in `common.py`). Every other estimator defaults to 60 Hz. ZCD is the outlier; this biases `structural_latency_samples()` by 20%.

**Action.**
1. Change both defaults to 60.0 Hz.
2. Audit all other estimators for the same issue (grep for `nominal_f: float = 50`).

**Acceptance criteria.**
- `grep -rn "nominal_f.*=.*50" src/estimators/` returns zero hits.
- Smoke test ZCD mean error on 60 Hz sine < 0.05 Hz.

**Evidence field:** *(grep output)*

---

### T-104 — Correct SOGI-PLL structural latency
**Priority:** 🟠 major
**Depends on:** none
**Files:** `src/estimators/sogi_pll.py`

**Problem.** `structural_latency_samples()` returns 0 with the comment "Un PLL no tiene latencia estructural fija." This is wrong: the loop has a well-defined settling time (the `settle_time` param). When Optuna tunes `settle_time` up to 250 ms, the transient leaks into the "steady state" metric window and inflates RMSE.

**Action.**
1. Return `int(round(self.settle_time / self.dt))` or a small multiple thereof (2× is standard for "settled").
2. Consider whether the same issue exists in the plain PLL estimator.

**Acceptance criteria.**
- SOGI-PLL RMSE on the Ramp scenario improves or stays the same (should not worsen — we're trimming more transient, not less).
- No other test regresses.

**Evidence field:** *(before/after RMSE on Ramp, reported from smoke test)*

---

## Epic 2 — Metric Robustness

### T-200 — Guard RoCoF/RNAF metrics against phase discontinuities
**Priority:** 🟠 major
**Depends on:** none
**Files:** `src/analysis/metrics.py`

**Problem.** `m9_m10_rfe_metrics` computes `np.gradient(f_hat)` and `np.gradient(f_true)`. For phase-jump scenarios (D, E), `f_true` is piecewise constant (RoCoF = 0), but `f_hat` has an impulse at the jump. RFE_max is dominated by a single-sample numerical artifact. The "true" instantaneous RoCoF at a step is a Dirac — undefined.

**Action.**
1. Accept an optional `event_mask: np.ndarray[bool]` argument in `m9_m10_rfe_metrics` that excludes a ±N-sample neighborhood of known events.
2. Alternatively, switch the aggregation from `np.max` to `np.percentile(..., 99.5)` to reject single-sample outliers, and document this choice.
3. Propagate the mask through `calculate_all_metrics`. Scenarios should expose `event_samples` in their `meta` dict.

**Acceptance criteria.**
- Running the D scenario on a perfect oracle (f_hat = f_true shifted by a 1-sample step) returns RFE_max ≤ 0.1 Hz/s instead of 1e4 Hz/s.
- Paper table values for RFE_max change at most 5% on scenarios A/B/C (no events) and may change significantly on D/E (by design).

**Evidence field:** *(before/after values on a synthetic test)*

---

### T-201 — Report Ttrip resolution limit
**Priority:** 🟡 minor
**Depends on:** T-200
**Files:** `src/analysis/metrics.py`, paper §IV-B

**Problem.** `m5_trip_risk_s` returns `n_violations * dt`. At 10 kHz, the resolution is 0.1 ms. The paper reports `Ttrip = 0.6 ms` for RA-EKF — that's 6 samples, right at the measurement floor. A journal extension reviewer will ask whether this is resolvable.

**Action.**
1. In `calculate_all_metrics`, add output field `m5_trip_risk_resolution_s = dt`.
2. In the paper § IV-B, add one sentence: "Ttrip is quantized to dt = 0.1 ms; differences below ≈1 ms should be interpreted as within measurement resolution."

**Acceptance criteria.**
- Field present in `benchmark_results.json`.
- Paper diff contains the sentence.

**Evidence field:** *(git diff of paper section)*

---

### T-202 — Warm up Numba JIT before timing
**Priority:** 🟡 minor
**Depends on:** none
**Files:** `src/analysis/monte_carlo_engine.py`

**Problem.** `time.perf_counter` wraps the full first call to `step_vectorized`, which includes Numba JIT compilation (up to several seconds). `m13_cpu_time_us` is inflated on whichever estimator runs first. Paper claims "averaged over 10,000 iterations" but the engine measures once per MC run.

**Action.**
1. In `_run_estimator`, before the timed block, call `est.step_vectorized(v[:100])` and then `est.reset()` to trigger JIT without counting it.
2. Alternatively, use `est.step_vectorized(np.zeros(100))` + reset as the warmup.
3. Verify CPU measurements are stable across the first 3 MC runs (was previously monotonically decreasing).

**Acceptance criteria.**
- CPU time for the first MC run is within 20% of the 10th MC run for every estimator.
- `paper_ready_numbers.txt` `TIME_PER_SAMPLE_US` values change by < 5% (sanity: the ratio Pareto should be stable).

**Evidence field:** *(coefficient of variation of CPU time across 10 runs, before/after)*

---

## Epic 3 — Smoke Test / Tuning Pipeline Fixes

### T-300 — Fix wrong parameter names in `SEARCH_SPACES`
**Priority:** 🟡 minor
**Depends on:** T-001
**Files:** `src/tests/test_dedicated_smoke_test.py` or new `src/tests/smoke_test_full.py`

**Problem.** `SEARCH_SPACES` lambdas suggest parameters that don't exist on the estimator `__init__`:
- `IPDFT`: suggests `n_cycles` (not an init param; only `cycles` is)
- `PLL`: suggests `kp_scale`, `ki_scale` (unlikely init params)
- `ZCD`: suggests `window_size`, `avg_window` (neither is an init param)
- `TFT`: suggests `window_size` (check source)

The `{k: v for k, v in suggested.items() if k in init_params}` guard silently drops them, so Optuna "optimizes" parameters that never reach the estimator. Every trial returns the same RMSE; `best_trial` is arbitrary.

**Action.**
1. For each estimator, inspect `__init__` signature and rewrite the search space with only real params.
2. In the Optuna objective, warn (or raise) when `applied < suggested`:
   ```python
   dropped = set(suggested) - set(applied)
   if dropped:
       raise ValueError(f"{est_name}: search space contains unknown params {dropped}")
   ```

**Acceptance criteria.**
- Running the smoke test with the fail-loud version raises no `ValueError` for any estimator.
- At least one non-default parameter value is applied per tuned estimator (confirmed by logging `applied` dict).

**Evidence field:** *(log showing applied params per estimator)*

---

### T-301 — Verify Optuna tuning actually improves RMSE
**Priority:** 🟡 minor
**Depends on:** T-300
**Files:** same as T-300

**Problem.** Even with correct search spaces, if tuning doesn't improve over defaults, either the search space is too narrow or the objective is noisy.

**Action.**
1. For each estimator, log `rmse_default` (using `default_params()`) and `rmse_tuned` (using `study.best_trial`) on the same scenario and seed.
2. Assert `rmse_tuned <= rmse_default * 1.1` (tuning should not make things worse by more than 10%).
3. If the assertion fails, widen search range or increase `N_TRIALS_TUNING`.

**Acceptance criteria.**
- Log table `estimator | rmse_default | rmse_tuned | improvement_%` produced.
- No estimator shows tuning regression > 10%.

**Evidence field:** *(log table)*

---

## Epic 4 — Manuscript Consistency (post-reproduction)

### T-400 — Add Monte Carlo mean ± std to every headline number
**Priority:** 🟠 major
**Depends on:** T-100 through T-200 complete
**Files:** `paper/Sections/C4_Simulation_Results/main.tex`, `src/analysis/paper_table_builder.py` (if exists)

**Problem.** Tables III and IV report single numbers. `MonteCarloEngine` runs N=30. Reviewer 2 on a journal extension will ask for dispersion. The headline "275× improvement" is especially exposed: it may be based on a single favorable phase-jump timing.

**Action.**
1. For every reported metric, compute mean and std across 30 MC runs.
2. Render as `X.XXX ± X.XXX` in tables.
3. Recompute headline ratios (275×, 4.7×, 12.6×, 1.59×, 3.3×) using mean values; verify each still exceeds its CLAUDE.md §6 threshold.

**Acceptance criteria.**
- All JSON claim paths in CLAUDE.md §6 thresholds verified.
- Tables render mean ± std.
- Abstract/Conclusions numbers match Tables within last sig fig.

**Evidence field:** *(LaTeX diff + verification log)*

---

### T-401 — Reconcile "grid search" vs Optuna wording
**Priority:** 🟡 minor
**Depends on:** none
**Files:** `paper/Sections/C3_Methods/main.tex` (or wherever tuning is described)

**Problem.** Paper §III-C says "structured scenario-wise grid search over the hyperparameter ranges in Table II." Code uses Optuna TPE, which is Bayesian, not grid search.

**Action.** Replace "grid search" with "Tree-structured Parzen Estimator (TPE) Bayesian optimization, N=20 trials per estimator–scenario pair" or whatever the current `N_TRIALS_TUNING` value is.

**Acceptance criteria.** Paper text matches code behavior.

**Evidence field:** *(diff)*

---

### T-402 — Soften or defend the TFT-vs-IpDFT ramp claim
**Priority:** 🟡 minor
**Depends on:** T-100
**Files:** `paper/Sections/C2_Related_Work/main.tex`, §II-A

**Problem.** §II-A claims "TFT adopts a dynamic Taylor-series phasor model that improves RoCoF tracking [relative to IpDFT]." Table III shows TFT RMSE 0.0726 vs IpDFT 0.0729 on the Ramp — a tie, not an improvement. Either the TFT implementation isn't leveraging the dynamic terms, the ramp is too gentle to expose the difference, or the claim is overstated.

**Action.**
1. After T-100 fix, rerun Ramp and re-check TFT vs IpDFT.
2. If still tied: soften §II-A to "TFT is theoretically better suited to RoCoF tracking; in our benchmark the difference is not statistically significant at +5 Hz/s."
3. Optional: add a +10 Hz/s ramp to the MC space to expose the gap.

**Acceptance criteria.** §II-A claim either empirically supported or reworded.

**Evidence field:** *(Ramp results with 30-run std)*

---

### T-403 — Disclose or remove the SOGI-FLL "correction" mention
**Priority:** 🟡 minor
**Depends on:** none
**Files:** paper §IV-A

**Problem.** §IV-A says "SOGI-FLL results incorporate the corrected FLL adaptation law (Section III)" but Section III doesn't clearly describe what was wrong. Looks like a late-stage bugfix, which invites "what else was broken?" from reviewers.

**Action.** Either:
- (a) Add 2 sentences in §III-A explaining the ω-scaling error and the correction (more defensible).
- (b) Remove the mention; just report current numbers (less defensible but cleaner).

**Acceptance criteria.** Paper is internally consistent.

**Evidence field:** *(diff)*

---

### T-404 — Fix method count inconsistency in abstract
**Priority:** 🟡 minor
**Depends on:** none
**Files:** abstract, §I, §II-A, Table I

**Problem.** Abstract says "nine principal estimators … plus two legacy baselines" (= 11). CLAUDE.md §1 lists 12 methods. Table I has 11 rows. Tables III/IV show 8 and 7 respectively.

**Action.** Align the count everywhere. Per CLAUDE.md ticket D-004, change to "twelve estimators spanning four algorithmic families" and remove the 9+2 phrasing. Explain in § IV-A why Tables III and IV show subsets (e.g., "TKEO excluded due to RMSE > 8 Hz on all scenarios").

**Acceptance criteria.** Numbers in abstract, §I, §II, §III, §IV, and Conclusions agree.

**Evidence field:** *(grep of all number mentions)*

---

### T-405 — Classify PI-GRU cost honestly
**Priority:** 🟡 minor
**Depends on:** none
**Files:** Table V, paper text

**Problem.** Table V reports PI-GRU at ≈136,000 µs/sample and leaves the Class column with "—". Reviewer will say "of course it's slow, you're running eager-mode Python with no batching."

**Action.** Add a footnote: "PI-GRU inference performed in unoptimized PyTorch eager mode without batching. Production deployments using TorchScript/ONNX and batched windowing would reduce per-sample cost by ≥100×; however, this remains outside the scope of embedded relay protection without dedicated hardware acceleration."

**Acceptance criteria.** Footnote present; PI-GRU exclusion from Tables III/IV justified.

**Evidence field:** *(diff)*

---

## Epic 5 — Release Hygiene

### T-500 — Pin reproducibility envelope
**Priority:** 🟡 minor
**Depends on:** T-102, T-202
**Files:** `requirements.txt`, `CLAUDE.md`

**Problem.** Numba, NumPy, SciPy, PyTorch versions affect reproducibility (especially with T-102's RNG changes).

**Action.**
1. Pin exact versions in `requirements.txt`.
2. Record Python version and OS in `benchmark_results.json` meta.
3. Add a single-command reproduction instruction in `README.md`.

**Acceptance criteria.** A fresh clone + `pip install -r requirements.txt` + `python main.py` on a different machine reproduces `benchmark_results.json` headline ratios within 5%.

**Evidence field:** *(reproduction log from a clean VM)*

---

### T-501 — Tag camera-ready commit
**Priority:** 🟡 minor
**Depends on:** all tickets above closed
**Files:** git

**Action.**
```bash
git tag -a v1.1-hardened -m "Post-acceptance hardening: T-000 through T-500"
git push --tags
```

Also create `RESULTS_CHANGELOG.md` documenting every headline number that changed from the accepted version, with attribution to the ticket that caused the change.

**Acceptance criteria.** Tag pushed; changelog complete.

---

## Execution Order (fast path)

```
T-000 → T-001 → T-100 → T-101 → T-102 → T-103 → T-104
                  │
                  └── T-200 → T-201
                  └── T-202
                  └── T-300 → T-301
                  └── T-400 ← (requires T-100..T-200)
                  └── T-401, T-402, T-403, T-404, T-405 (paper-only, parallel)
                  └── T-500 → T-501
```

**Stop conditions.**
- If T-000 shows the benchmark is fundamentally correct (decimation already happens somewhere) → T-100 becomes "add assertion" only; escalate T-101, T-102, T-200 as the real work.
- If T-100 changes headline numbers by > 20%, halt all paper tickets until co-authors review; the paper may need an errata rather than just a journal extension.

---

## Ticket Template (for adding new items)

```
### T-XXX — <short title>
**Priority:** 🔴 critical | 🟠 major | 🟡 minor
**Depends on:** T-YYY, T-ZZZ
**Files:** <paths>

**Problem.** <one paragraph>

**Action.**
1. <step>
2. <step>

**Acceptance criteria.**
- <machine-checkable statement>
- <machine-checkable statement>

**Evidence field:** *(to be filled on close)*
```