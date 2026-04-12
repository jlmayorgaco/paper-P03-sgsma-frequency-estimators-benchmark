# CLAUDE.md — IBR Frequency Estimator Benchmark / SGSMA Camera-Ready
## Scientific Integrity and Execution Protocol

---

## 0. Project identity

**Research domain:** Dynamic frequency estimation in low-inertia / IBR-dominated power systems.
**Audience:** Graduate-level signal processing + power systems researchers. Every decision must be defensible to a domain expert. Assume the reader knows Kalman filters, DFT theory, PLL loop dynamics, IEC 60255-118-1, and Monte Carlo methodology.

**Benchmark:** Twelve estimators across four algorithmic families, evaluated over five stress scenarios with scenario-wise grid-search tuning, dual-rate simulation (1 MHz physics / 10 kHz DSP, RATIO=100), and Monte Carlo (N=30).

**Paper target:** SGSMA 2026 conference. LaTeX source under `paper/Sections/`. Final artifacts drive tables, figures, and all quantitative claims.

---

## 1. Twelve benchmark methods (paper-facing set)

| Label | Class | Family |
|---|---|---|
| `EKF2` / RA-EKF | Augmented Kalman `[θ,ω,A,ω̇]` + event-gating + innovation-driven Q | Model-based |
| `EKF` | ClassicEKF `[θ,ω,A]` | Model-based |
| `UKF` | Unscented Kalman `[θ,ω,A]` | Model-based |
| `PLL` | SRF-PLL with PI controller | Loop-based |
| `SOGI` | SOGI-FLL with ω-normalization fix + IIR bandpass + FastRMS normalizer | Loop-based |
| `IpDFT` | Tunable-cycle Hann DFT + Jacobsen complex spectral interpolation | Window-based |
| `TFT` | 2nd-order Taylor-Fourier transform, Hann-weighted | Window-based |
| `RLS` | Recursive least squares, AR(2) bandpass pre-whitening | Adaptive recursive |
| `RLS-VFF` | Variable forgetting factor RLS | Adaptive recursive |
| `Teager` | Teager-Kaiser energy operator | Adaptive recursive |
| `Koopman-RKDPmu` | Koopman operator, window-based | Data-driven |
| `PI-GRU` | Physics-informed GRU (weights: `pi_gru_pmu.pt`) — excluded from MC | Data-driven |

**Monte Carlo runs 11 methods** (PI-GRU excluded — fixed model, no MC benefit).
**LKF permanently excluded.** RMSE 17-21 Hz on all scenarios (degenerate AR(2)). Must not appear in any paper-facing output, MC analysis, or plot.

---

## 2. Five evaluation scenarios

| ID | JSON key | Key disturbance | Duration | IEC scope |
|---|---|---|---|---|
| A | `IEEE_Mag_Step` | +10% voltage step at t=0.5s, σ=0.001 | 1.5 s | IEC compliant |
| B | `IEEE_Freq_Ramp` | +5 Hz/s RoCoF from t=0.3–1.0s, σ=0.001 | 1.5 s | IEC compliant |
| C | `IEEE_Modulation` | AM 10%@2Hz + σ=0.001 | 1.5 s | IEC compliant |
| D | `IBR_Nightmare` | +60° phase jump + 5th harmonic + 32.5Hz interharmonic + σ=0.005 | 1.5 s | IBR-specific |
| E | `IBR_MultiEvent_Classic` | +40°/+80° jumps + −6 Hz/s RoCoF + 5th+7th harmonics + impulsive noise | 5.0 s | IBR-specific |

**Critical naming:** Scenario E JSON key is `IBR_MultiEvent_Classic` — never `IBR_MultiEvent`. All JSON lookups, table references, and paper claim checks must use the exact key `IBR_MultiEvent_Classic`.

---

## 3. Metrics

| Metric | Definition |
|---|---|
| `RMSE` | √(mean(e²)) over evaluation window (Hz) |
| `T_trip` / `TRIP_TIME_0p5` | Cumulative time \|e\|>0.5 Hz in evaluation window (s) |
| `MAX_PEAK` | max(\|e\|) in evaluation window (Hz) |
| `SETTLING` | First time \|e\|<0.2 Hz and remains below (s) |
| `TIME_PER_SAMPLE_US` | CPU µs/sample, mean of N=20 reps via `time.process_time()` |

**Evaluation window:** `start_idx = max(1500 samples = 150 ms fixed, structural_samples)`.
The baseline is `int(0.15 * FS_DSP) = 1500 samples = 150 ms` — a **fixed duration**, not 15% of signal length. All disturbance events in all scenarios fall well within the evaluation window. The docstring at `estimators.py:~1205` incorrectly says "15% of samples" — this is a documentation error only; the calculation is correct.

---

## 4. Canonical execution

```
Canonical directory:  src/
Entry point:          python main.py
Config:               USE_BAYESIAN_TUNING = False   (src/estimators.py:25)
EKF/UKF R bound:      p_ekf_r = np.logspace(-5, 0, 12)  (src/main.py:590)  ← max R = 1.0
MC:                   N_MC_RUNS = 30, seeds 2000+i  (src/main.py:143)
Timing:               N_COST_REPS = 20  (src/main.py:83)
RNG:                  SEED = 42 (tuning); MC seeds independent
```

**T-100 sample-rate resolution (Option C — implemented 2026-04-12):**
`Scenario.generate()` produces signals at `FS_PHYSICS = 1 MHz` for physics fidelity.
`Scenario.run()` decimates `t`, `v`, and `f_true` by `RATIO = 100` (simple `[::RATIO]` slicing)
before returning, so all callers receive 10 kHz data matching each estimator's `DT_DSP = 1e-4 s`.
`MonteCarloEngine.run_once()` asserts `dt_actual == 1e-4` after each `scenario.run()` call.
`generate()` output is never passed directly to estimators; always go through `run()`.

**Run command:**
```bash
cd /c/Users/walla/Documents/Github/paper-P03-sgsma-frequency-estimators-benchmark/src
python main.py 2>&1 | tee ../run_final_$(date +%Y%m%d_%H%M%S).log
```

`src/` is the only code directory. No other execution path is valid.

---

## 5. Key output artifacts

All outputs land under `src/` when `main.py` is run from `src/`. The `results/` directory at the repo root is a **stale copy from a previous run with a different working directory — do not use it for paper updates.**

| Artifact | Canonical location | Purpose |
|---|---|---|
| `benchmark_results.json` | `src/figures_estimatores_benchmark_test5/` | **Master results — all claims verification** |
| `paper_ready_numbers.txt` | `src/figures_estimatores_benchmark_test5/` | Auto-extracted paper claim values |
| `issues_log.txt` | `src/figures_estimatores_benchmark_test5/` | ISSUES_RESOLVED checklist output |
| `Fig1_Scenarios_Final.pdf/png` | `src/figures_estimatores_benchmark_test5/` | Publication Figure 1 |
| `Fig2_Mega_Dashboard.pdf/png` | `src/figures_estimatores_benchmark_test5/` | Publication Figure 2 |
| `results_raw/raw_mc.json` | `src/results_raw/` | Per-run Monte Carlo data |
| `results_raw/<scenario>/<method>.json` | `src/results_raw/` | Per-method detailed traces |
| `scientific_analysis_extended.json` | `src/figures_estimatores_benchmark_test5/` (inferred) | Q1-Q12 + extended analyses |
| `figures_scientific/` | `src/figures_scientific/` | Q1-Q12 analysis figures |
| `Fig*.pdf` (LaTeX-referenced) | `paper/Figures/Plots_And_Graphs/` | Copied after run — must be fresh |

**Timestamp rule:** If `benchmark_results.json` timestamp predates the most recent code change in `src/`, the file is stale and must be regenerated.

---

## 6. Key paper claims to verify from fresh JSON

All JSON paths are relative to `benchmark_results.json → results.<scenario>.methods.<method>`.

| Claim | JSON path | Threshold |
|---|---|---|
| 275× trip-risk (EKF vs EKF2, Scen D) | `IBR_Nightmare.EKF.TRIP_TIME_0p5 / IBR_Nightmare.EKF2.TRIP_TIME_0p5` | >200× |
| 4.7× RMSE (EKF vs EKF2, Scen D) | `IBR_Nightmare.EKF.RMSE / IBR_Nightmare.EKF2.RMSE` | >3× |
| 12.6× ramp (PLL vs EKF2, Scen B) | `IEEE_Freq_Ramp.PLL.RMSE / IEEE_Freq_Ramp.EKF2.RMSE` | >8× |
| 1.59× EKF vs EKF2 (Scen B) | `IEEE_Freq_Ramp.EKF.RMSE / IEEE_Freq_Ramp.EKF2.RMSE` | >1.2× |
| 3.3× trip (PLL vs EKF2, Scen E) | `IBR_MultiEvent_Classic.PLL.TRIP_TIME_0p5 / IBR_MultiEvent_Classic.EKF2.TRIP_TIME_0p5` | >2× |

**Red flag:** If `EKF.TRIP_TIME_0p5 == 0` for Scen D after the run → R constraint fix failed. Verify `p_ekf_r = np.logspace(-5, 0, 12)` at `src/main.py:590`.

**SOGI paradox:** Code at `src/main.py:1312-1329` detects whether SOGI achieves lower RMSE than EKF2 in `IBR_Nightmare`. If the anomaly fires, paper §IV-B must address it explicitly — do not hide it.

---

## 7. LaTeX hardcoded values to update after run

All file paths are relative to `paper/Sections/`. JSON source paths are `benchmark_results.json → results.<scenario>.methods.<method>.<field>`.

| File | Current string | JSON source |
|---|---|---|
| `C4_Simulation_Results/main.tex:21` | `0.0113 Hz`, `12.6×` | `IEEE_Freq_Ramp.EKF2.RMSE`, ratio |
| `C4_Simulation_Results/main.tex:77` | `T_trip=0.165 s`, `RMSE=0.325 Hz` (EKF) | `IBR_Nightmare.EKF.TRIP_TIME_0p5`, `IBR_Nightmare.EKF.RMSE` |
| `C4_Simulation_Results/main.tex:80` | `RMSE=0.0695 Hz`, `T_trip=0.6 ms` (RA-EKF) | `IBR_Nightmare.EKF2.RMSE`, `IBR_Nightmare.EKF2.TRIP_TIME_0p5 × 1000` |
| `C4_Simulation_Results/main.tex:81` | `4.7×`, `275×` | ratios above |
| `C4_Simulation_Results/main.tex:76` | `0.481 Hz`, `2.91 Hz` (IpDFT) | `IBR_Nightmare.IpDFT.RMSE`, `IBR_Nightmare.IpDFT.MAX_PEAK` |
| `C5_Conclusions/main.tex:24` | `T_trip=0.471 s` (PLL) | `IBR_MultiEvent_Classic.PLL.TRIP_TIME_0p5` |
| `C5_Conclusions/main.tex:33-34` | `275×`, `0.165 s`, `64 µs/sample` | ratios above + `IBR_MultiEvent_Classic.EKF2.TIME_PER_SAMPLE_US` |
| Table I `tab:ieee_results` | all cells | `IEEE_Mag_Step`, `IEEE_Freq_Ramp`, `IEEE_Modulation` — 8 methods × 3 scenarios |
| Table II `tab:risk_metrics` | all cells | `IBR_MultiEvent_Classic` — RMSE, TRIP_TIME_0p5, TIME_PER_SAMPLE_US |
| `C4_Simulation_Results/main.tex` Fig 2g caption | `0.506 Hz` (RA-EKF peak) | `IBR_Nightmare.EKF2.MAX_PEAK` |

---

## 8. Non-negotiable rules

1. **Paper follows artifacts.** If fresh results differ from manuscript, update the manuscript — never the reverse.
2. **`src/` is the only code.** Any other directory is suspect.
3. **No silent estimator failure.** If any estimator returns RMSE=60.0 Hz (PI-GRU fallback) or `optimal_params = "PI-GRU FAILED"`, exclude it from all paper-facing outputs and flag it.
4. **Latency must be honest.** Koopman and window methods have structural output latency = `window_samples / FS_DSP` (up to 100 ms for Koopman). Must be disclosed in paper.
5. **SOGI-FLL is non-standard.** Implements IIR bandpass pre-filter + FastRMS normalizer — not textbook SOGI-FLL. Label it "bandpass-pre-filtered SOGI-FLL" in paper.
6. **Every number traces to a JSON key.** If you cannot state the exact `benchmark_results.json` path for a manuscript number, it cannot go in the paper.
7. **Use `IBR_MultiEvent_Classic` everywhere.** The alias `IBR_MultiEvent` does not exist as a JSON key and will silently fail all lookups.
8. **Canonical results only.** `results/benchmark_results.json` at the repo root is a stale artifact. Only `src/figures_estimatores_benchmark_test5/benchmark_results.json` is canonical.

---

## 9. Session startup checklist

- [ ] Working directory is `src/`
- [ ] `USE_BAYESIAN_TUNING = False` in `src/estimators.py:25`
- [ ] `p_ekf_r = np.logspace(-5, 0, 12)` in `src/main.py:590` (max R = 1.0)
- [ ] No `LKF_Estimator` / `tune_lkf` in `src/main.py` (functional code, not comments)
- [ ] `src/pi_gru_pmu.pt` exists (not `pi_gru_pmu_v2.pt`)
- [ ] `src/figures_estimatores_benchmark_test5/benchmark_results.json` timestamp — is it from the current run or stale?
- [ ] Do NOT use `results/benchmark_results.json` at repo root — it is stale
- [ ] State the current project phase before touching any file

---

## 10. Ticket roadmap

Status: `[DONE]` `[READY]` `[BLOCKED: needs run]` `[REVIEW — paper edit]`

---

### EPIC A — Code integrity

#### A-010 `[READY]` Fix stale inline comments in src/main.py
- `src/main.py:140`: comment says "N_MC = 10 is sufficient" — actual value is N_MC_RUNS = 30. Update comment.
- `src/main.py:~152`: `run_monte_carlo_all` docstring says "13 estimators" — LKF removed, now 11. Update docstring.

#### A-011 `[READY]` Delete stale `results/` directory at repo root
`results/` was generated by a run with the wrong working directory. It contains `benchmark_results.json` timestamped 2026-03-30 that does NOT correspond to the canonical output path. Delete `results/` or clearly mark it with a `STALE_DO_NOT_USE.txt` to prevent accidental use for paper updates.

---

### EPIC B — Benchmark execution

#### B-001 `[READY]` Full canonical benchmark run
```bash
cd /c/Users/walla/Documents/Github/paper-P03-sgsma-frequency-estimators-benchmark/src
python main.py 2>&1 | tee ../run_final_$(date +%Y%m%d_%H%M%S).log
```
**Expected runtime:** 30–90 min (grid search + 30 MC runs × 5 scenarios).
**Abort conditions:** any RMSE > 10 Hz, crash before all 5 scenarios, `src/figures_estimatores_benchmark_test5/benchmark_results.json` not updated.
**Watch for in log:**
- `[REPRO-1] EKF best params ... R=` → must be ≤ 1.0
- `[REPRO-2] Verifying key paper claims` → must show `✅ All key paper claims verified`
- `Regression assertions: ALL PASS`

#### B-002 `[BLOCKED: needs B-001]` Post-run sanity checks
Minimum pass criteria (read from `src/figures_estimatores_benchmark_test5/benchmark_results.json`):
- `IBR_Nightmare.EKF.TRIP_TIME_0p5 > 0.05 s`
- `IEEE_Freq_Ramp.IpDFT.RMSE < 0.5 Hz`
- `paper_claims_verification.verified == true` in JSON
- Zero NaN or inf in any RMSE field
- `PI-GRU.optimal_params` is NOT `"PI-GRU FAILED"` in any scenario
- `EKF.tuned_params.R ≤ 1.0` in all scenarios

#### B-003 `[BLOCKED: needs B-001]` Verify scientific analysis artifacts regenerated
- `src/figures_scientific/scientific_summary.txt` — fresh timestamp
- `src/figures_scientific/paper_tables_E1_E2_E3.txt` — fresh timestamp
- `src/figures_estimatores_benchmark_test5/scientific_analysis_extended.json` — fresh timestamp
- All Q1-Q12 figures in `src/figures_scientific/`

#### B-004 `[BLOCKED: needs B-001]` Copy publication figures to LaTeX directory
```bash
# Run from repo root:
cp src/figures_estimatores_benchmark_test5/Fig1_Scenarios_Final.pdf paper/Figures/Plots_And_Graphs/
cp src/figures_estimatores_benchmark_test5/Fig2_Mega_Dashboard.pdf paper/Figures/Plots_And_Graphs/
# Verify timestamps:
ls -lh paper/Figures/Plots_And_Graphs/Fig*.pdf
```

---

### EPIC C — Manuscript tables

#### C-001 `[BLOCKED: needs B-001]` Update Table I — IEEE standard scenarios
**File:** `paper/Sections/C4_Simulation_Results/main.tex`, `\label{tab:ieee_results}`
**Source:** `benchmark_results.json → results.IEEE_Mag_Step / IEEE_Freq_Ramp / IEEE_Modulation`
**Metrics:** RMSE (Hz) and MAX_PEAK (Hz). 8 methods × 3 scenarios. 3 sig figs, bold best per column.

#### C-002 `[BLOCKED: needs B-001]` Update Table II — IBR stress metrics
**File:** `paper/Sections/C4_Simulation_Results/main.tex`, `\label{tab:risk_metrics}`
**Source:** `benchmark_results.json → results.IBR_MultiEvent_Classic`
**Metrics:** RMSE, TRIP_TIME_0p5 (s), TIME_PER_SAMPLE_US (µs). EKF2 and EKF must be separate rows.

#### C-003 `[BLOCKED: needs B-001]` Update Table III — CPU complexity
**Source:** `TIME_PER_SAMPLE_US` from `IBR_MultiEvent_Classic` for all methods.

---

### EPIC D — Manuscript text and claims

#### D-001 `[BLOCKED: needs B-001]` Update §IV-A ramp claims
**File:** `paper/Sections/C4_Simulation_Results/main.tex:21`
Replace `0.0113 Hz` with `IEEE_Freq_Ramp.EKF2.RMSE` and recompute `12.6×` and `1.59×` from fresh JSON.

#### D-002 `[BLOCKED: needs B-001]` Update §IV-B Nightmare scenario values
**File:** `paper/Sections/C4_Simulation_Results/main.tex:76-81`

| Replace | With |
|---|---|
| `T_trip=0.165 s` (EKF) | `IBR_Nightmare.EKF.TRIP_TIME_0p5` |
| `RMSE=0.325 Hz` (EKF) | `IBR_Nightmare.EKF.RMSE` |
| `RMSE=0.0695 Hz` (RA-EKF) | `IBR_Nightmare.EKF2.RMSE` |
| `T_trip=0.6 ms` (RA-EKF) | `IBR_Nightmare.EKF2.TRIP_TIME_0p5 × 1000` |
| `4.7×` | `IBR_Nightmare.EKF.RMSE / IBR_Nightmare.EKF2.RMSE` |
| `275×` | `IBR_Nightmare.EKF.TRIP_TIME_0p5 / IBR_Nightmare.EKF2.TRIP_TIME_0p5` |
| `0.481 Hz`, `2.91 Hz` (IpDFT) | `IBR_Nightmare.IpDFT.RMSE`, `IBR_Nightmare.IpDFT.MAX_PEAK` |
| `0.506 Hz` (RA-EKF peak, Fig 2g caption) | `IBR_Nightmare.EKF2.MAX_PEAK` |

#### D-003 `[BLOCKED: needs B-001]` Update Conclusions hardcoded values
**File:** `paper/Sections/C5_Conclusions/main.tex:24,33-34`
- `T_trip=0.471 s` (PLL) → `IBR_MultiEvent_Classic.PLL.TRIP_TIME_0p5`
- `275×`, `0.165 s` → from D-002
- `64 µs/sample` → `IBR_MultiEvent_Classic.EKF2.TIME_PER_SAMPLE_US`

#### D-004 `[REVIEW — paper edit]` Reconcile method count in Introduction
**File:** `paper/Sections/C1_Introduction/main.tex:31`
**Current:** "nine principal estimators... plus two legacy baselines" = 11, but code has 12.
**Fix:** Change to "twelve estimators spanning four algorithmic families" and remove the 9+2 phrasing. Propagate to Conclusions and Methods section headings.

---

### EPIC E — Scientific defensibility

#### E-001 `[REVIEW — paper edit]` Disclose SOGI-FLL non-standard front-end
**File:** `paper/Sections/C3_Methods/main.tex`
Line ~20 partially acknowledges bandpass pre-filtering. Add explicitly: implementation uses IIR bandpass pre-filter + FastRMS adaptive normalizer — not directly comparable to the textbook SOGI-FLL in [ref]. Label as "bandpass-pre-filtered SOGI-FLL" throughout.

#### E-002 `[REVIEW — paper edit]` Disclose Koopman structural output latency
**File:** `paper/Sections/C3_Methods/main.tex` or results caption
Koopman `smooth_win = window_samples` → up to 100 ms structural output delay at 10 kHz. Add one sentence noting this structural latency for the tuned configuration. Reviewers will catch an undisclosed 100 ms delay in a benchmarked estimator.

#### E-003 `[REVIEW — paper edit]` Fix evaluation window description in paper
**File:** `paper/Sections/C4_Simulation_Setup/main2.tex` (see A-009)
If any text says "15% warm-up" → change to "150 ms warm-up window (`max(150 ms, structural latency)`)". Also fix `src/estimators.py:~1205` docstring (says "15% of samples", should say "150 ms fixed = 1500 samples at 10 kHz").

#### E-004 `[REVIEW — paper edit]` Verify IEC 60255-118-1 compliance heatmap
After run: check `scientific_analysis_extended.json` Q12 IEC sufficiency output. Confirm RA-EKF F* annotation (peak≈0.506 Hz for ≈0.6 ms) survives fresh run. Update Fig 2g caption if value changes (source: `IBR_Nightmare.EKF2.MAX_PEAK`).

#### E-005 `[REVIEW — pre-run check]` Verify PI-GRU did not silently fail
After run: confirm `benchmark_results.json` shows `optimal_params != "PI-GRU FAILED"` for all 5 scenarios.
**Why critical:** failure fallback outputs 60.0 Hz constant → RMSE≈0 for near-nominal scenarios (A, C, D), making a crashed model appear best. Pretrained model file is `src/pi_gru_pmu.pt` (not `pi_gru_pmu_v2.pt`).

#### E-006 `[REVIEW — paper edit]` Explain SOGI-FLL scenario paradox
**File:** `paper/Sections/C4_Simulation_Results/main.tex` §IV-B discussion
SOGI ranks last on Ramp (RMSE≈2.1 Hz) but competitive on Nightmare. Detected and logged by code at `src/main.py:1312-1329`. Add 1-2 sentences: "SOGI-FLL's bandpass pre-filtering suppresses the apparent frequency deviation under the phase jump but cannot track sustained linear RoCoF — explaining the divergent ranking across Scenarios B and D."

---

### EPIC F — Final consistency and submission

#### F-000 `[REVIEW — paper edit]` State N_MC=30 in methods section
**File:** `paper/Sections/C4_Simulation_Setup/main2.tex` (see A-009 — must create this file)
Add: "Monte Carlo robustness analysis uses N=30 independent runs per estimator per scenario with fixed tuned parameters and noise seeds 2000–2029 (tuning seed: 42)."

#### F-001 `[BLOCKED: needs D-001–D-003 + A-009]` LaTeX full build — clean compile
```bash
cd /c/Users/walla/Documents/Github/paper-P03-sgsma-frequency-estimators-benchmark/paper
pdflatex index && bibtex index && pdflatex index && pdflatex index
```
Pass criteria: zero `undefined reference`, zero `citation undefined`, no `overfull \hbox` on table rows, within SGSMA page limit.
**Pre-condition:** `paper/Sections/C4_Simulation_Setup/main2.tex` must exist (ticket A-009).

#### F-002 `[BLOCKED: needs F-001]` Final paper ↔ JSON consistency audit
Every number in abstract, Tables I–III, §IV, and Conclusions must be traceable to a specific JSON key in `src/figures_estimatores_benchmark_test5/benchmark_results.json`. Read conclusions against Table II. Verify all `\ref{}` labels resolve.

#### F-003 `[BLOCKED: needs F-002]` Git commit, tag, and archive
```bash
git add paper/Sections/ src/figures_estimatores_benchmark_test5/benchmark_results.json
git add src/figures_estimatores_benchmark_test5/scientific_analysis_extended.json
git commit -m "Camera-ready: fresh run results, updated tables and all claims verified"
git tag v1.0-camera-ready
```

---

## 11. What Claude must never do in this project

- Preserve a hardcoded manuscript number because "it looks about right" — always read the JSON key.
- Treat `benchmark_results.json` as valid if its timestamp predates the most recent code change.
- Use `results/benchmark_results.json` at repo root — it is a stale artifact from a mislocated run.
- Use `IBR_MultiEvent` as a JSON key — the correct key is `IBR_MultiEvent_Classic`.
- Edit paper tables before `src/figures_estimatores_benchmark_test5/benchmark_results.json` is regenerated from the fixed code.
- Accept RMSE=60.0 Hz or `"PI-GRU FAILED"` results without flagging and excluding.
- Change scenario parameters, metric definitions, or evaluation windows without a commit message explaining the scientific rationale.
- Silently skip a failed estimator in plots without adding an explicit exclusion note.
- Mark any claim as verified without reading the exact JSON field that backs it.
- Reference `pi_gru_pmu_v2.pt` — the actual deployed model file is `src/pi_gru_pmu.pt`.
