# CLAUDE.md — IBR Frequency Estimator Benchmark / SGSMA Camera-Ready
## Scientific Integrity and Execution Protocol

---

## 0. Project identity

**Research domain:** Dynamic frequency estimation in low-inertia / IBR-dominated power systems.
**Audience:** Graduate-level signal processing + power systems researchers. Every decision must be defensible to a domain expert. Assume the reader knows Kalman filters, DFT theory, PLL loop dynamics, IEC 60255-118-1, and Monte Carlo methodology.

**Benchmark:** Twelve estimators across four algorithmic families, evaluated over five stress scenarios with scenario-wise grid-search tuning, dual-rate simulation (1 MHz physics / 10 kHz DSP, RATIO=100), and Monte Carlo (N=30).

**Paper target:** SGSMA 2026 conference. LaTeX source under `Sections/`. Final artifacts drive tables, figures, and all quantitative claims.

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
| `PI-GRU` | Physics-informed GRU (weights: `pi_gru_pmu_v2.pt`) — excluded from MC | Data-driven |

**Monte Carlo runs 11 methods** (PI-GRU excluded — fixed model, no MC benefit).
**LKF permanently excluded.** RMSE 17-21 Hz on all scenarios (degenerate AR(2)). Must not appear in any paper-facing output, MC analysis, or plot.

---

## 2. Five evaluation scenarios

| ID | Code name | Key disturbance | Duration | IEC scope |
|---|---|---|---|---|
| A | `IEEE_Mag_Step` | +10% voltage step at t=0.5s, σ=0.001 | 1.5 s | IEC compliant |
| B | `IEEE_Freq_Ramp` | +5 Hz/s RoCoF from t=0.3–1.0s, σ=0.001 | 1.5 s | IEC compliant |
| C | `IEEE_Modulation` | AM 10%@2Hz + σ=0.001 | 1.5 s | IEC compliant |
| D | `IBR_Nightmare` | +60° phase jump + 5th harmonic + 32.5Hz interharmonic + σ=0.005 | 1.5 s | IBR-specific |
| E | `IBR_MultiEvent` | +40°/+80° jumps + −6 Hz/s RoCoF + 5th+7th harmonics + impulsive noise | 5.0 s | IBR-specific |

---

## 3. Metrics

| Metric | Definition |
|---|---|
| `RMSE` | √(mean(e²)) over evaluation window (Hz) |
| `T_trip` / `TRIP_TIME_0p5` | Cumulative time \|e\|>0.5 Hz in evaluation window (s) |
| `MAX_PEAK` | max(\|e\|) in evaluation window (Hz) |
| `SETTLING` | First time \|e\|<0.2 Hz and remains below (s) |
| `TIME_PER_SAMPLE_US` | CPU µs/sample, median of N=20 reps via `time.process_time()` |

**Evaluation window:** `start_idx = max(1500 samples = 150 ms fixed, structural_samples)`.
The baseline is `int(0.15 * FS_DSP) = 1500 samples = 150 ms` — a **fixed duration**, not 15% of signal length. All disturbance events in all scenarios fall well within the evaluation window. The docstring at `estimators.py:1205` incorrectly says "15% of samples" — this is a documentation error only; the calculation is correct.

---

## 4. Canonical execution

```
Canonical directory:  src/
Entry point:          python main.py
Config:               USE_BAYESIAN_TUNING = False   (estimators.py:25)
EKF/UKF R bound:      p_ekf_r = np.logspace(-5, 0, 12)  (max R = 1.0)
MC:                   N_MC_RUNS = 30, seeds 2000+i
Timing:               N_COST_REPS = 20
RNG:                  SEED = 42 (tuning); MC seeds independent
```

**Run command:**
```bash
cd /c/Users/walla/Documents/Github/paper/src
python main.py 2>&1 | tee ../run_final_$(date +%Y%m%d_%H%M%S).log
```

`code_sim_and_results/` has been deleted. `src/` is the only code directory.

---

## 5. Key output artifacts

| Artifact | Location | Purpose |
|---|---|---|
| `benchmark_results.json` | `src/` | Master results — all claims verification |
| `results_raw/raw_mc.json` | `src/results_raw/` | Per-run MC data |
| `results_raw/<scenario>/*.json` | same | Per-method detailed records |
| `scientific_analysis_extended.json` | `src/` | Q1-Q12 + extended analyses |
| `figures_estimatores_benchmark_test5/` | `src/` | Publication figures (Fig1, Fig2) |
| `Figures/Plots_And_Graphs/` | paper root | LaTeX-referenced figures — copy after run |

---

## 6. Key paper claims to verify from fresh JSON

| Claim | JSON path | Threshold |
|---|---|---|
| 275× trip-risk (EKF2 vs EKF, Scen D) | `IBR_Nightmare.EKF.TRIP / IBR_Nightmare.EKF2.TRIP` | >200× |
| 4.7× RMSE (EKF2 vs EKF, Scen D) | `IBR_Nightmare.EKF.RMSE / IBR_Nightmare.EKF2.RMSE` | >3× |
| 12.6× ramp (PLL vs EKF2, Scen B) | `IEEE_Freq_Ramp.PLL.RMSE / IEEE_Freq_Ramp.EKF2.RMSE` | >8× |
| 1.59× EKF vs EKF2 (Scen B) | `IEEE_Freq_Ramp.EKF.RMSE / IEEE_Freq_Ramp.EKF2.RMSE` | >1.2× |
| 3.3× trip (PLL vs EKF2, Scen E) | `IBR_MultiEvent.PLL.TRIP / IBR_MultiEvent.EKF2.TRIP` | >2× |

**Red flag:** If `EKF.TRIP_TIME_0p5 == 0` for Scen D after the run → R constraint fix failed. Check `p_ekf_r` upper bound in `main.py:575`.

---

## 7. LaTeX hardcoded values to update after run

| File | Current string | JSON source |
|---|---|---|
| `C4_Simulation_Results/main.tex:21` | `0.0113 Hz`, `12.6×` | `IEEE_Freq_Ramp.EKF2.RMSE`, ratio |
| `C4_Simulation_Results/main.tex:77` | `T_trip=0.165 s`, `RMSE=0.325 Hz` (EKF) | `IBR_Nightmare.EKF.*` |
| `C4_Simulation_Results/main.tex:80` | `RMSE=0.0695 Hz`, `T_trip=0.6 ms` (RA-EKF) | `IBR_Nightmare.EKF2.*` |
| `C4_Simulation_Results/main.tex:81` | `4.7×`, `275×` | ratios above |
| `C4_Simulation_Results/main.tex:76` | `0.481 Hz`, `2.91 Hz` (IpDFT) | `IBR_Nightmare.IpDFT.RMSE/.MAX_PEAK` |
| `C5_Conclusions/main.tex:24` | `T_trip=0.471 s` (PLL) | `IBR_MultiEvent.PLL.TRIP_TIME_0p5` |
| `C5_Conclusions/main.tex:33-34` | `275×`, `0.165 s`, `64 µs/sample` | ratios + `EKF2.TIME_PER_SAMPLE_US` |
| Table I `tab:ieee_results` | all cells | `IEEE_Mag_Step`, `IEEE_Freq_Ramp`, `IEEE_Modulation` |
| Table II `tab:risk_metrics` | all cells | `IBR_MultiEvent` |

---

## 8. Non-negotiable rules

1. **Paper follows artifacts.** If fresh results differ from manuscript, update the manuscript — never the reverse.
2. **`src/` is the only code.** `code_sim_and_results/` is deleted.
3. **No silent estimator failure.** If any estimator returns RMSE=60.0 Hz (PI-GRU fallback) or `optimal_params = "PI-GRU FAILED"`, exclude it from all paper-facing outputs.
4. **Latency must be honest.** Koopman and window methods have structural output latency = `window_samples / FS_DSP`. Must be reported or controlled for.
5. **SOGI-FLL is non-standard.** Implements IIR bandpass + FastRMS normalizer — not textbook SOGI-FLL. Label it "bandpass-pre-filtered SOGI-FLL" in paper.
6. **Every number traces to a JSON key.** If you cannot state the exact `benchmark_results.json` path for a manuscript number, it cannot go in the paper.

---

## 9. Session startup checklist

- [ ] Working directory is `src/`
- [ ] `USE_BAYESIAN_TUNING = False` in `estimators.py:25`
- [ ] `p_ekf_r = np.logspace(-5, 0, 12)` in `main.py:575`
- [ ] No `LKF_Estimator` / `tune_lkf` in `main.py` (functional code, not comments)
- [ ] `benchmark_results.json` timestamp — is it from the current run or stale?
- [ ] State the current project phase before touching any file

---

## 10. Ticket roadmap

Status: `[DONE]` `[READY]` `[BLOCKED: needs run]` `[REVIEW — paper edit]`

---

### EPIC A — Code integrity

#### A-001 `[DONE]` Disable Bayesian tuning
`USE_BAYESIAN_TUNING = False` at `src/estimators.py:25`. Verified.
**Root cause fixed:** `popsize=1` differential_evolution was finding degenerate optima — IpDFT Ramp RMSE 42× off, EKF R→100 → T_trip=0.

#### A-002 `[DONE]` Constrain EKF/UKF R search space
`p_ekf_r = np.logspace(-5, 0, 12)` at `src/main.py:575`. Max R=1.0. Verified.
**Root cause fixed:** Unconstrained R→100 made EKF ignore measurements → T_trip=0 → 275× claim undefined.

#### A-003 `[DONE]` Remove LKF from all benchmark-facing code
Removed from `src/main.py`: top-level imports, `tune_lkf` import, MC local import, MC factory dict, `p_lkf_*` dead vars, full benchmark block, print string corrected. Verified: 0 functional LKF refs remain.
**Root cause fixed:** LKF RMSE 17-21 Hz contaminated all MC ensemble statistics.

#### A-004 `[DONE]` Fix IpDFT description in LaTeX
`Sections/C3_Methods/main.tex:24`: "parabolic interpolation" → "Jacobsen complex spectral interpolation". Verified.

#### A-005 `[DONE]` Create requirements.txt
`src/requirements.txt` generated via `pip freeze`.

#### A-006 `[DONE]` Remove LKF from scientific_analysis.py family maps
`FAMILIES["Kalman"]` and `FAMILY_MAP` cleaned in `src/scientific_analysis.py`. Verified: 0 LKF refs remain.

#### A-007 `[DONE]` Fix stale method-count strings in main.py
- Benchmark print: "12 Methods: EKF2, EKF, UKF, PLL, SOGI, IpDFT, TFT, RLS, VFF-RLS, Teager, Koopman, PI-GRU"
- MC print: "11 methods × 5 scenarios, PI-GRU excluded"

---

### EPIC B — Benchmark execution

#### B-001 `[READY]` Full canonical benchmark run
```bash
cd /c/Users/walla/Documents/Github/paper/src
python main.py 2>&1 | tee ../run_final_$(date +%Y%m%d_%H%M%S).log
```
**Expected runtime:** 30–90 min (grid search + 30 MC runs × 5 scenarios).
**Abort conditions:** any RMSE > 10 Hz, crash before all 5 scenarios, `benchmark_results.json` not updated.
**Watch for in log:**
- `[REPRO-1] EKF best params ... R=` → must be ≤ 1.0
- `[REPRO-2] Verifying key paper claims` → must show `✅ All key paper claims verified`
- `Regression assertions: ALL PASS`

#### B-002 `[BLOCKED: needs B-001]` Post-run sanity checks
Minimum pass criteria:
- `EKF IBR_Nightmare TRIP_TIME_0p5 > 0.05 s`
- `IpDFT IEEE_Freq_Ramp RMSE < 0.5 Hz`
- All 5 `paper_claims_numbers` show `verified: true`
- Zero NaN or inf in any RMSE field
- `PI-GRU optimal_params` is NOT `"PI-GRU FAILED"` in any scenario

#### B-003 `[BLOCKED: needs B-001]` Verify scientific analysis artifacts regenerated
- `figures_scientific/scientific_summary.txt` — fresh timestamp
- `figures_scientific/paper_tables_E1_E2_E3.txt` — fresh timestamp
- `scientific_analysis_extended.json` — fresh timestamp
- All Q1-Q12 figures in `figures_scientific/`

#### B-004 `[BLOCKED: needs B-001]` Copy publication figures to LaTeX directory
```bash
cp src/figures_estimatores_benchmark_test5/Fig1_Scenarios_Final.pdf Figures/Plots_And_Graphs/
cp src/figures_estimatores_benchmark_test5/Fig2_Mega_Dashboard.pdf Figures/Plots_And_Graphs/
```
Verify file timestamps after copy.

---

### EPIC C — Manuscript tables

#### C-001 `[BLOCKED: needs B-001]` Update Table I — IEEE standard scenarios
**File:** `Sections/C4_Simulation_Results/main.tex`, `\label{tab:ieee_results}`
**Source:** `benchmark_results.json` → `results.IEEE_Mag_Step / IEEE_Freq_Ramp / IEEE_Modulation`
**Metrics:** RMSE (Hz) and MAX_PEAK (Hz). 8 methods × 3 scenarios. 3 sig figs, bold best per column.

#### C-002 `[BLOCKED: needs B-001]` Update Table II — IBR stress metrics
**File:** `Sections/C4_Simulation_Results/main.tex`, `\label{tab:risk_metrics}`
**Source:** `benchmark_results.json` → `results.IBR_MultiEvent`
**Metrics:** RMSE, T_trip (s), TIME_PER_SAMPLE_US (µs). EKF2 and EKF must be separate rows.

#### C-003 `[BLOCKED: needs B-001]` Update Table III — CPU complexity
**Source:** `TIME_PER_SAMPLE_US` from `IBR_MultiEvent` for all methods.

---

### EPIC D — Manuscript text and claims

#### D-001 `[BLOCKED: needs B-001]` Update §IV-A ramp claims
**File:** `C4_Simulation_Results/main.tex:21`
Replace `0.0113 Hz` with `IEEE_Freq_Ramp.EKF2.RMSE` and recompute `12.6×` and `1.59×` from fresh JSON.

#### D-002 `[BLOCKED: needs B-001]` Update §IV-B Nightmare scenario values
**File:** `C4_Simulation_Results/main.tex:76-81`

| Replace | With |
|---|---|
| `T_trip=0.165 s` (EKF) | `IBR_Nightmare.EKF.TRIP_TIME_0p5` |
| `RMSE=0.325 Hz` (EKF) | `IBR_Nightmare.EKF.RMSE` |
| `RMSE=0.0695 Hz` (RA-EKF) | `IBR_Nightmare.EKF2.RMSE` |
| `T_trip=0.6 ms` (RA-EKF) | `IBR_Nightmare.EKF2.TRIP_TIME_0p5 × 1000` |
| `4.7×` | `EKF.RMSE / EKF2.RMSE` |
| `275×` | `EKF.TRIP / EKF2.TRIP` |
| `0.481 Hz`, `2.91 Hz` (IpDFT) | `IBR_Nightmare.IpDFT.RMSE / MAX_PEAK` |
| `0.506 Hz` (RA-EKF peak, Fig 2g caption) | `IBR_Nightmare.EKF2.MAX_PEAK` |

#### D-003 `[BLOCKED: needs B-001]` Update Conclusions hardcoded values
**File:** `Sections/C5_Conclusions/main.tex:24,33-34`
- `T_trip=0.471 s` (PLL) → `IBR_MultiEvent.PLL.TRIP_TIME_0p5`
- `275×`, `0.165 s` → from D-002
- `64 µs/sample` → `IBR_MultiEvent.EKF2.TIME_PER_SAMPLE_US`

#### D-004 `[REVIEW — paper edit]` Reconcile method count in Introduction
**File:** `Sections/C1_Introduction/main.tex:31`
**Current:** "nine principal estimators... plus two legacy baselines" = 11, but code has 12.
**Fix:** Change to "twelve estimators spanning four algorithmic families" and remove the 9+2 phrasing. Propagate to Conclusions and Methods section headings.

---

### EPIC E — Scientific defensibility

#### E-001 `[REVIEW — paper edit]` Disclose SOGI-FLL non-standard front-end
**File:** `Sections/C3_Methods/main.tex`
Line 20 partially acknowledges bandpass pre-filtering. Add: implementation uses IIR bandpass pre-filter + FastRMS adaptive normalizer, not directly comparable to the textbook SOGI-FLL in [ref].

#### E-002 `[REVIEW — paper edit]` Disclose Koopman structural output latency
**File:** `Sections/C3_Methods/main.tex` or results caption
Koopman `smooth_win = window_samples` → up to 100 ms output delay at 10 kHz. Add one sentence noting this structural latency for the tuned configuration.

#### E-003 `[REVIEW — paper edit]` Fix evaluation window description in paper
**File:** `Sections/C4_Simulation_Setup/main.tex`
If text says "15% warm-up" → change to "150 ms warm-up window (`max(150 ms, structural latency)`)". Also fix `estimators.py:1205` docstring (says "15% of samples", should say "150 ms fixed").

#### E-004 `[REVIEW — paper edit]` Verify IEC 60255-118-1 compliance heatmap
After run: check `scientific_analysis_extended.json` Q12 IEC sufficiency output. Confirm RA-EKF F* annotation (peak≈0.506 Hz for ≈0.6 ms) survives fresh run. Update Fig 2g caption if value changes.

#### E-005 `[REVIEW — pre-run check]` Verify PI-GRU did not silently fail
After run: confirm `benchmark_results.json` shows `optimal_params != "PI-GRU FAILED"` for all 5 scenarios.
**Why critical:** failure fallback outputs 60.0 Hz constant → RMSE≈0 for near-nominal scenarios (A, C, D), making crashed model appear best. PI-GRU loads correctly in dry-run test (verified), but check again in actual run log.

#### E-006 `[REVIEW — paper edit]` Explain SOGI-FLL scenario paradox
**File:** `Sections/C4_Simulation_Results/main.tex` §IV-B discussion
SOGI ranks last on Ramp (RMSE≈2.1 Hz) but competitive on Nightmare. Detected by code at `main.py:1265-1282`. Add 1-2 sentences: "SOGI-FLL's bandpass pre-filtering suppresses the apparent frequency deviation under the phase jump but cannot track sustained linear RoCoF — explaining the divergent ranking across Scenarios B and D."

---

### EPIC F — Final consistency and submission

#### F-000 `[REVIEW — paper edit]` State N_MC=30 in methods section
**File:** `Sections/C4_Simulation_Setup/main.tex`
Add: "Monte Carlo robustness analysis uses N=30 independent runs per estimator per scenario with fixed tuned parameters and noise seeds 2000–2029 (tuning seed: 42)."

#### F-001 `[BLOCKED: needs D-001–D-003]` LaTeX full build — clean compile
```bash
cd /c/Users/walla/Documents/Github/paper
pdflatex index && bibtex index && pdflatex index && pdflatex index
```
Pass criteria: zero `undefined reference`, zero `citation undefined`, no `overfull \hbox` on table rows, within SGSMA page limit.

#### F-002 `[BLOCKED: needs F-001]` Final paper ↔ JSON consistency audit
Every number in abstract, Tables I–II, §IV, and Conclusions must be traceable to a specific JSON key. Read conclusions against Table II. Verify all `\ref{}` labels resolve.

#### F-003 `[BLOCKED: needs F-002]` Git commit, tag, and archive
```bash
git add Sections/ src/benchmark_results.json src/scientific_analysis_extended.json
git commit -m "Camera-ready: fresh run results, updated tables and all claims verified"
git tag v1.0-camera-ready
```

---

## 11. What Claude must never do in this project

- Preserve a hardcoded manuscript number because "it looks about right" — always read the JSON key.
- Treat `benchmark_results.json` as valid if its timestamp predates the most recent code change.
- Edit paper tables before `benchmark_results.json` is regenerated from the fixed code.
- Accept RMSE=60.0 Hz or `"PI-GRU FAILED"` results without flagging and excluding.
- Change scenario parameters, metric definitions, or evaluation windows without a commit message explaining the scientific rationale.
- Silently skip a failed estimator in plots without adding an explicit exclusion note.
- Mark any claim as verified without reading the exact JSON field that backs it.
