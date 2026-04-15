# ROADMAP_CLAUDE.md — Pending Items After Full Textual Revision
# Paper: "Benchmarking Dynamic Frequency Estimators for Low-Inertia IBR Grids"
# Date: 2026-04-01
# Status key: [CODE] requires Python execution | [LATEX] LaTeX-only fix | [MANUAL] human judgment needed

---

## PRIORITY 1 — Blocking (paper cannot be submitted without these)

### P1-A  Run canonical benchmark and regenerate all artifacts  [CODE]
```bash
cd /c/Users/walla/Documents/Github/paper-P03-sgsma-frequency-estimators-benchmark/src
python main.py 2>&1 | tee ../run_final_$(date +%Y%m%d_%H%M%S).log
```
- All hardcoded numeric values in the LaTeX (Tables I–III, §IV text, §V) are from
  a prior run. They must be updated from fresh `benchmark_results.json` after running.
- Watch for in log: `✅ All key paper claims verified` and `Regression assertions: ALL PASS`.
- Abort if: any RMSE > 10 Hz, `PI-GRU FAILED`, EKF RMSE Scenario D == 0.

### P1-B  Update all hardcoded numerical claims from fresh JSON  [LATEX]
After P1-A, update every value in:
- `paper/Sections/C4_Simulation_Results/main.tex`
  - Table `tab:unified_results`: all 64 cells (8 methods × 4 scenarios × 2 metrics)
  - Table `tab:risk_metrics`: RMSE, T_trip, CPU for 7 methods
  - Table `tab:complexity`: CPU for all 9 methods
  - §IV-A ramp claims: `8.23 mHz`, `8.15×`, `1.41×`
  - §IV-B Islanding: `484 mHz`, `2910 mHz`, `45.0 mHz`, `39.3 mHz`, `365 mHz`, `5.7×`
  - §IV-B Multi-Event: `159 mHz`, `0.081 s`, `88.5 µs`, `4.4×`, `4.5×`
  - §IV-D (RLS subsection): `342 mHz`, `0.367 s`, `874 mHz`, `2.20 s`
  - Fig 2g caption: `365 mHz` peak for RA-EKF Islanding
- `paper/Sections/C5_Conclusions/main.tex`
  - `0.426 s` (SRF-PLL Multi-Event T_trip)
  - `0.120 s` (EKF Islanding T_trip)
  - `68.9 µs/sample` (RA-EKF CPU)
  - `145 ms/sample` (PI-GRU CPU)
- Source for every number: `src/figures_estimatores_benchmark_test5/benchmark_results.json`
  → `results.<scenario>.methods.<method>.<metric>`

### P1-C  Copy fresh publication figures to LaTeX directory  [CODE]
```bash
cp src/figures_estimatores_benchmark_test5/Fig1_Scenarios_Final.pdf \
   paper/Figures/Plots_And_Graphs/
cp src/figures_estimatores_benchmark_test5/Fig2_Mega_Dashboard.pdf \
   paper/Figures/Plots_And_Graphs/
ls -lh paper/Figures/Plots_And_Graphs/Fig*.pdf   # verify timestamps
```

### P1-D  Full LaTeX build — verify clean compile  [LATEX]
```bash
cd /c/Users/walla/Documents/Github/paper-P03-sgsma-frequency-estimators-benchmark/paper
pdflatex index && bibtex index && pdflatex index && pdflatex index
```
Pass criteria: zero `undefined reference`, zero `citation undefined`,
no `overfull \hbox` on table rows, within SGSMA page limit.
**Pre-condition:** P1-A through P1-C must be completed first.

---

## PRIORITY 2 — Figure Quality (IEEE submission standard)

### P2-A  Replot Figure 1 (Scenario Suite) with IEEE-compliant fonts  [CODE]
**File:** `src/benchmark_plottings.py` or equivalent plotting script
**Issues:**
- All fonts must be ≥ 8 pt when printed at IEEE column width (3.5 in / 88 mm)
- Legend entries must be legible at single-column width
- Axis labels in Hz for frequency, s for time — match mHz convention added to paper
- Add scenario labels (A)–(E) prominently in each subplot
- Remove any matplotlib default style artifacts (tight_layout warnings, clipped labels)
**After replot:** copy PDF to `paper/Figures/Plots_And_Graphs/Fig1_Scenarios_Final.pdf`

### P2-B  Replot Figure 2 (Dashboard) with improved layout  [CODE]
**File:** `src/benchmark_plottings.py` or `src/scientific_analysis.py`
**Issues:**
- Dashboard has 8 subplots (a)–(h): currently dense at full-page width
- Each subplot needs standalone readable axis labels and legend ≥ 8 pt
- Subplot (e) Pareto: add RMSE as color or bubble size (3D Pareto proxy)
- Subplot (f) Radar: ensure method labels don't overlap at print size
- Subplot (g) Heatmap: add explicit F* annotation with definition box
- Subplot (h): confirm log-scale y-axis label is not clipped
- All units must match mHz convention now in paper text
**After replot:** copy PDF to `paper/Figures/Plots_And_Graphs/Fig2_Mega_Dashboard.pdf`

### P2-C  Add Figure 3 — RLS/VFF-RLS trade-off  [CODE — optional but recommended]
The new §IV-D (RLS family discussion) references no figure.
Consider adding a time-domain comparison of RLS vs VFF-RLS vs EKF in Scenario E
to support the VFF blow-up discussion added to §IV-D.

---

## PRIORITY 3 — Science Completeness (Reviewer 2 risks)

### P3-A  Noise/Harmonic Decomposition Experiment  [CODE — optional]
**Issue:** ROADMAP item 49 — performance degradation in Scenarios D/E cannot be
attributed to harmonic leakage vs. noise amplification without isolation experiments.
**Action:** Run Scenario D variants:
  1. Phase jump only (no harmonics, no noise)
  2. Phase jump + harmonics (no noise)
  3. Phase jump + noise (no harmonics)
  4. Full Scenario D (current)
Report RMSE table for each variant. Add 1–2 sentences to §IV-B.
Even a brief qualitative note from simulations strengthens the claim.

### P3-B  Final paper ↔ JSON consistency audit  [MANUAL]
After P1-B, read every number in abstract, Tables I–III, §IV text, and Conclusions.
Verify each traces to an exact `benchmark_results.json` key.
No number may appear without a traceable source.
Check all `\ref{}` labels resolve, all `\cite{}` keys exist in references.bib.

### P3-C  Verify PI-GRU did not silently fail  [CODE]
After P1-A: confirm `benchmark_results.json` shows
`optimal_params != "PI-GRU FAILED"` for all 5 scenarios.
If it failed: flag in paper, exclude from all metric comparisons,
set RMSE = N/A in tables.

### P3-D  EKF R constraint verification  [CODE]
After P1-A: confirm `IBR_Nightmare.EKF.TRIP_TIME_0p5 > 0.05 s`.
If == 0: the R constraint fix (`p_ekf_r = np.logspace(-5, 0, 12)`) may
have regressed. Check `src/main.py:590`.

---

## PRIORITY 4 — LaTeX Polish

### P4-A  Check for overfull \hbox in tables  [LATEX]
After P1-D: scan compile log for `Overfull \hbox` warnings.
- `tab:unified_results` (wide 8-column table) is most likely to overflow.
- If overflow: reduce `\tabcolsep`, use `\scalebox{0.9}{}`, or split table.

### P4-B  Verify all citation keys are used (no orphans)  [LATEX]
Run: `bibtex index` and check for "unused entry" or "undefined citation" warnings.
References at risk of being orphaned: `TaylorFourier2012`, `NadirDNN_2019`,
`SustInertia2025`, `EnergiesOsc2022`, `ProcessesTransient2023`, `Li2024NadirDL`,
`Dadhich2025AmbientInertia`, `Choi2025ResidualInertia`, `Paramo2022PMUReview`,
`Fiorucci2025DistPMU`, `AlbuquerqueFrazao2015PhD`, `Netto2019KSThesis`.
Remove any truly orphaned entries from references.bib.

### P4-C  Add data availability URL  [LATEX — post submission]
Currently: "simulation code and hyperparameter grids are available from the authors
for reproducibility." Once the repository is public, replace with the actual URL:
`\footnote{\url{https://github.com/...}}`
Location: `paper/Sections/C5_Conclusions/main.tex` (last paragraph).

### P4-D  Section III-A: RA-EKF subsubsection title  [LATEX — consider]
The subsubsection is titled "Implemented RA-EKF Estimator" which still gives it
special status. Since RA-EKF is just one of 12 methods, consider folding its
description into the main "Evaluated Estimators" subsection and removing the
dedicated subsubsection header, OR renaming it to "RA-EKF (Model-Based, Augmented)"
and adding comparable detail subsubsections for IpDFT and SRF-PLL to balance.

### P4-E  Missing method detail parity  [LATEX — consider]
The Methods section gives RA-EKF a dedicated subsubsection with equations (F, H
matrices), SOGI-FLL a detailed justification block, but IpDFT/TFT/RLS/Koopman/PI-GRU
receive only 1–2 sentences. For reviewer fairness, consider brief equation-level
descriptions for IpDFT (Jacobsen formula) and Koopman (EDMD formulation) since they
are cited as capable alternatives.

---

## PRIORITY 5 — SGSMA Submission Checklist

### P5-A  Page limit check
SGSMA 2026 conference limit: verify paper fits within page limit after P1-D.
New sections added (failure mode summary, RLS subsection, latency distinction) may
push length. If over limit: merge §IV-D (RLS) into §IV-B footnotes, shorten Fig 2
caption, or move Table III to appendix.

### P5-B  Author affiliations and ORCiD
Verify `paper/Config/authors.tex` has correct affiliations and ORCiD IDs for all
authors before submission.

### P5-C  IEEE PDF compliance check
After final compile: run IEEE PDF checker (PDF eXpress) to verify:
- Fonts embedded
- Page size correct (letter or A4 per SGSMA requirement)
- No bitmap fonts below 600 dpi
- All figures at ≥ 300 dpi

### P5-D  Git tag camera-ready
After all P1 items pass:
```bash
git add paper/Sections/ \
        src/figures_estimatores_benchmark_test5/benchmark_results.json \
        src/figures_estimatores_benchmark_test5/scientific_analysis_extended.json
git commit -m "Camera-ready: fresh run results, all claims verified, full revision"
git tag v1.0-camera-ready
```

---

## ITEMS CONFIRMED COMPLETE (all textual fixes done in this revision)

| # | Item | Status |
|---|------|--------|
| 1 | Estimator count (twelve, EKF added to Table I) | ✅ |
| 2 | Unit unification (Hz → mHz throughout) | ✅ |
| 3 | Performance ceiling disclaimer | ✅ |
| 4 | PI-GRU fixed-baseline disclosure | ✅ |
| 5 | Ground truth frequency definition (analytical derivative) | ✅ |
| 6 | Reference audit (Zelaya→under review, Pinheiro=journal) | ✅ |
| 7 | Table I Phase Resp. → S/D (single semantic per column) | ✅ |
| 8 | Sig figs: `±0.00` → `±0`; footnote explains negligible MC variance | ✅ |
| 9 | F* heatmap definition added to §III-C and Fig 2g caption | ✅ |
| 11 | IpDFT latency = structural Nc/f0, not stochastic lag | ✅ |
| 12 | LaTeX artifact scan — none found | ✅ |
| 13 | RA-EKF description: single canonical block in §III-A | ✅ |
| 14 | RA-EKF F and H matrices as numbered equations | ✅ |
| 15 | σ̂ν defined as EWMA of innovation magnitudes | ✅ |
| 16 | SOGI-FLL: IIR+FastRMS justified (why added, what effect) | ✅ |
| 17 | SOGI-FLL unit correction: 377× mis-scale explained | ✅ |
| 18 | Scenario A–E consistent; no Scenario F exists | ✅ |
| 19 | Settling time: "remains below 0.2 Hz for ≥20 ms" | ✅ |
| 20 | Grid search: spacing type + number of points in Table II | ✅ |
| 21 | N=30 MC statistical limitation (tail-risk disclaimer) | ✅ |
| 22 | MC seeds 2000–2029, paired across estimators | ✅ |
| 23 | Structural latency vs stochastic lag: formal distinction §II | ✅ |
| 24 | Qualitative scores justified: from benchmark + literature, not certification | ✅ |
| 25 | Failure mode summary subsection §IV | ✅ |
| 26 | UKF 0.5 ms causal delay quantified | ✅ |
| 27 | Computational cost: Python baseline only, 10–100× embedded speedup | ✅ |
| 28 | Pareto 2D projection limitation noted in caption | ✅ |
| 29 | Dual-rate aliasing assumption stated | ✅ |
| 30 | ω=2πf defined at first use in §III | ✅ |
| 31 | Overclaiming: "in this benchmark / under evaluated scenarios" added | ✅ |
| 32 | Fig 2 caption restructured per subplot (standalone readable) | ✅ |
| 33 | Figure readability: PDFs unchanged (require code replot — see P2-A/B) | ⚠️ PDF |
| 34 | Abstract: refocused on benchmark, RA-EKF metrics removed | ✅ |
| 35 | RLS/VFF-RLS/TKEO family subsection §IV-D added | ✅ |
| 36 | Compliance threshold: IEC source cited, visualization-only note | ✅ |
| 37 | Generalization limitation in Conclusions | ✅ |
| 38 | Parameter identifiability limitation in §III-C | ✅ |
| 39 | Event-gating heuristic: not theoretically optimal, γ not optimised | ✅ |
| 41 | Code availability statement in Conclusions | ✅ |
| 42 | HIL validation disclaimer in §IV-C and Conclusions | ✅ |
| 43 | MC seed management: seeds 2000–2029, tuning seed 42 | ✅ |
| 44 | T_trip: non-contiguous sum + multi-excursion protection note | ✅ |
| 45 | 32.5 Hz: cited as IBR converter control interaction (Bernal2024, Zoghby2024) | ✅ |
| 46 | Commercial relay gap: fixed-window DFT+PLL context added to §II | ✅ |
| 47 | Grid search overhead: ~5,000 tuning simulations stated | ✅ |
| 48 | γ=2.0 sensitivity: disclosed as heuristic, not optimised | ✅ |
| 49 | Noise/harmonic interplay: isolation limitation noted in §IV-B | ✅ |
| 50 | Proofreading: "across all N=30" → removed; key grammar issues fixed | ✅ |
| RA-EKF de-emphasis | Bold removed from tables, contributions reframed | ✅ |
