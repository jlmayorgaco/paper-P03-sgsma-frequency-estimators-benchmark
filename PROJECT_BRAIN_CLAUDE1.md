# PROJECT_BRAIN.md
## IBR Frequency Estimator Benchmark — SGSMA 2026 Camera-Ready
### Master Repository Reference Document

**Generated:** 2026-04-01 | **Status:** Pre-canonical-run (B-001 pending)
**Purpose:** Single-file canonical reference for architecture, pipeline, paper linkage, and execution. All facts verified from actual file inspection. Uncertainties explicitly marked.

---

## 1. Executive Summary

This repository benchmarks **12 dynamic frequency estimators** for deployment in low-inertia / Inverter-Based Resource (IBR)-dominated power systems. The paper's central claim is that the **RA-EKF (EKF2)** — a RoCoF-augmented Extended Kalman Filter with innovation-driven covariance scaling and event-gating — delivers the best accuracy-robustness-latency trade-off among all 12 candidates under composite IBR disturbances.

**What the benchmark does:**
- Generates 5 test scenarios (3 IEC standard, 2 IBR-specific) at dual rate: 1 MHz physics / 10 kHz DSP
- Tunes each of 12 estimators per-scenario via grid search (SEED=42)
- Evaluates RMSE, peak error, trip-risk duration, settling time, and CPU cost
- Runs N=30 Monte Carlo robustness trials
- Produces publication figures (Fig1, Fig2) and LaTeX-ready tables

**Paper target:** IEEE SGSMA 2026 conference
**Paper claim strength:** Hinges on EKF2 achieving negligible trip-time (T_trip ≈ 0) in the IBR_Nightmare scenario vs EKF T_trip ≈ 0.165 s (275× ratio) — only valid with EKF R constraint enforced.

---

## 2. Repository Purpose

| Aspect | Detail |
|---|---|
| **Scientific goal** | Comparative evaluation of 12 frequency estimators under realistic IBR grid disturbances |
| **Engineering goal** | Reproducible benchmark with honest metric computation, scenario-wise tuning, and MC robustness |
| **Paper goal** | SGSMA 2026 conference paper — camera-ready submission |
| **Main contribution** | EKF2 (RA-EKF): 4-state augmented Kalman with innovation-driven Q scaling + event gating for phase jumps |
| **Key metric** | TRIP_TIME_0p5 = cumulative time |f_error| > 0.5 Hz (protection relay trip threshold) |

---

## 3. High-Level Architecture

### Logical Layers

```
┌─────────────────────────────────────────────────────────────────────────┐
│  1. SCENARIO GENERATION  (src/estimators.py: get_test_signals)          │
│     5 scenarios × dual-rate: 1 MHz physics → downsample → 10 kHz DSP   │
├─────────────────────────────────────────────────────────────────────────┤
│  2. TUNING / OPTIMIZATION  (src/main.py + estimators.py: tune_*())      │
│     Grid search, RMSE objective, SEED=42, scenario-wise per estimator   │
├─────────────────────────────────────────────────────────────────────────┤
│  3. ESTIMATOR EXECUTION  (src/estimators.py + ekf2.py + pigru_model.py) │
│     12 estimators × 5 scenarios, N_COST_REPS=20 timing runs             │
├─────────────────────────────────────────────────────────────────────────┤
│  4. METRICS  (src/estimators.py: calculate_metrics)                     │
│     RMSE, MAX_PEAK, TRIP_TIME_0p5, SETTLING, TIME_PER_SAMPLE_US         │
│     Evaluation window: max(150 ms, structural_latency)                  │
├─────────────────────────────────────────────────────────────────────────┤
│  5. MONTE CARLO  (src/main.py: run_monte_carlo_all)                     │
│     11 methods × 5 scenarios × 30 runs, seeds 2000–2029                 │
├─────────────────────────────────────────────────────────────────────────┤
│  6. AGGREGATION / EXPORT  (src/main.py: run_benchmark)                  │
│     → src/figures_estimatores_benchmark_test5/benchmark_results.json    │
│     → src/results_raw/<scenario>/<method>.json                          │
├─────────────────────────────────────────────────────────────────────────┤
│  7. STATISTICAL ANALYSIS  (src/statistical_analysis.py)                 │
│     Pareto frontier, hypothesis tests, Kruskal-Wallis, clustering       │
├─────────────────────────────────────────────────────────────────────────┤
│  8. EXTENDED ANALYSIS Q1-Q12  (src/scientific_analysis.py)              │
│     Behavioral taxonomy, IEC sufficiency, complexity bounds, RF selector│
├─────────────────────────────────────────────────────────────────────────┤
│  9. PLOTTING  (src/benchmark_plottings.py + plotting.py)                │
│     Fig1 (scenarios), Fig2 (8-panel dashboard), per-method traces       │
├─────────────────────────────────────────────────────────────────────────┤
│ 10. PAPER  (paper/index.tex + paper/Sections/C*/main.tex)               │
│     LaTeX IEEE format, manual table/claim updates required after run     │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
src/estimators.py              src/ekf2.py         src/pigru_model.py
  get_test_signals()             EKF2 class          build_pigru_estimator()
  tune_*() functions             .step(z) → f_Hz     loads pi_gru_pmu.pt
  calculate_metrics()
       │
       ▼
src/main.py (entry point)
  run_benchmark()
  run_monte_carlo_all()
       │
       ├── src/results_raw/<scenario>/<method>.json   (per-run traces)
       ├── src/figures_estimatores_benchmark_test5/
       │     ├── benchmark_results.json               (MASTER output)
       │     ├── Fig1_Scenarios_Final.pdf/png
       │     ├── Fig2_Mega_Dashboard.pdf/png
       │     └── <method>_<scenario>_*.png
       └── src/figures_scientific/                    (Q1-Q12 outputs)
             ├── scientific_summary.txt
             └── Q*.pdf/png
                                    ▼ (manual copy)
                          paper/Figures/Plots_And_Graphs/
                          paper/index.tex → PDF
```

---

## 4. Full Repository Map

```
paper-P03-sgsma-frequency-estimators-benchmark/
│
├── src/                                # CANONICAL execution path ← only run from here
│   ├── main.py                         # [CORE] Entry point. ~1550+ lines.
│   ├── estimators.py                   # [CORE] All 12 estimator classes + tune_*() + get_test_signals()
│   ├── ekf2.py                         # [CORE] RA-EKF (EKF2) — 4-state Kalman
│   ├── pigru_model.py                  # [CORE] PI-GRU wrapper (loads .pt model)
│   ├── benchmark_plottings.py          # [CORE] Publication figures (Fig1, Fig2)
│   ├── plotting.py                     # [CORE] Per-scenario traces, OUTPUT_DIR definition
│   ├── statistical_analysis.py         # [SUPPORT] Pareto, hypothesis tests, clustering
│   ├── scientific_analysis.py          # [SUPPORT] Extended Q1-Q12 analyses (~5500 lines)
│   ├── pi_gru_pmu.pt                   # [CORE ASSET] Pretrained PI-GRU weights (~900 KB)
│   ├── pi_gru_pmu_config.json          # [CORE ASSET] PI-GRU architecture config
│   ├── requirements.txt               # [SUPPORT] Frozen dependencies (pip freeze)
│   ├── scenarios.py                    # [LEGACY/UNCLEAR] Scenario visualization, possibly superseded
│   ├── run_analysis_offline.py         # [SUPPORT] Offline re-analysis without re-running benchmark
│   ├── plot_chamorro.py                # [UNCLEAR] Purpose unknown — NEEDS VERIFICATION
│   ├── plot_ibr_v2.py                  # [UNCLEAR] IBR plotting v2 — NEEDS VERIFICATION
│   ├── pigru_train.py                  # [LEGACY] PI-GRU training script (model already trained)
│   ├── train_pigru.py                  # [LEGACY] Alternate PI-GRU training script — DUPLICATE?
│   ├── test_smoke.py                   # [TEST] Quick import + instantiation checks
│   ├── test_ipdft_targeted.py          # [TEST] IpDFT-specific tests
│   ├── test_koopman_consistency.py     # [TEST] Koopman consistency tests
│   ├── test_pigru_sanity.py            # [TEST] PI-GRU sanity checks
│   ├── claude.md                       # [STALE] Redirect to root CLAUDE.md — do not use
│   │
│   ├── figures_estimatores_benchmark_test5/   # [GENERATED] Primary output dir
│   │   ├── benchmark_results.json             # MASTER RESULTS — canonical output
│   │   ├── Fig1_Scenarios_Final.pdf/png       # Publication Figure 1
│   │   ├── Fig2_Mega_Dashboard.pdf/png        # Publication Figure 2
│   │   ├── paper_ready_numbers.txt            # Auto-generated claim extraction
│   │   ├── issues_log.txt                     # Run warnings/checklist
│   │   └── <method>_<scenario>_*.png          # Per-method optimized traces
│   │
│   ├── results_raw/                           # [GENERATED] Per-method raw JSON
│   │   ├── IEEE_Mag_Step/<method>.json
│   │   ├── IEEE_Freq_Ramp/<method>.json
│   │   ├── IEEE_Modulation/<method>.json
│   │   ├── IBR_Nightmare/<method>.json
│   │   ├── IBR_MultiEvent_Classic/<method>.json
│   │   └── raw_mc.json
│   │
│   └── figures_scientific/                    # [GENERATED] Q1-Q12 outputs
│       ├── scientific_summary.txt
│       ├── paper_tables_E1_E2_E3.txt
│       └── Q*.pdf/png
│
├── paper/                              # LaTeX paper source
│   ├── index.tex                       # [CORE] Main LaTeX document (IEEEtran conference)
│   ├── index.pdf                       # [GENERATED] Compiled paper
│   ├── Config/
│   │   ├── packages.tex                # LaTeX package imports
│   │   ├── commands.tex                # Custom macros
│   │   ├── title.tex                   # Paper title
│   │   ├── authors.tex                 # Author affiliations
│   │   ├── abstract.tex                # Abstract (has hardcoded numbers!)
│   │   ├── keywords.tex                # IEEE keywords
│   │   └── header.tex                  # Running header
│   ├── Sections/
│   │   ├── C1_Introduction/main.tex    # §I Introduction
│   │   ├── C2_Related_Work/main.tex    # §II Related Work
│   │   ├── C3_Methods/main.tex         # §III Methods + RA-EKF description
│   │   ├── C4_Simulation_Results/main.tex  # §IV Results + Tables III-V
│   │   ├── C5_Conclusions/main.tex     # §V Conclusions
│   │   └── C4_Simulation_Setup/        # ⚠ MISSING — index.tex \inputs this dir (main2.tex)
│   ├── Figures/
│   │   └── Plots_And_Graphs/           # LaTeX figure target (receives Fig1, Fig2 copies)
│   └── bibtex/bib/
│       ├── references.bib              # Primary bibliography
│       └── IEEEabrv.bib / IEEEexample.bib
│
├── results/                            # [STALE] From previous run (not canonical output dir)
│   ├── benchmark_results.json          # Timestamp: 2026-03-30 21:22:50 — STALE COPY
│   ├── figures_estimatores_benchmark_test5/benchmark_results.json  # Another stale copy
│   ├── figures_scientific/             # Stale Q1-Q12 figures
│   └── results_raw/                    # Stale per-method JSON
│
├── CLAUDE.md                           # [CORE] Operational protocol + ticket roadmap
├── README.md                           # [SUPPORT] User-facing documentation
├── requirements.txt                    # [SUPPORT] Root-level dependency file
├── package.json / package-lock.json    # [UNCLEAR] Node.js package — purpose UNKNOWN
└── .claude/                            # Claude Code settings/memory
```

**Critical path confusion:** `results/` at root contains output from a previous run where `main.py` was executed from the project root (not from `src/`). When run correctly (`cd src && python main.py`), output goes to `src/figures_estimatores_benchmark_test5/` and `src/results_raw/`. The `results/benchmark_results.json` is stale and must not be used as paper source.

---

## 5. Real Execution Pipeline

### Step-by-Step Flow

```
STEP 0: Environment setup
  cd /c/Users/walla/Documents/Github/paper-P03-sgsma-frequency-estimators-benchmark/src
  # Verify: USE_BAYESIAN_TUNING = False (estimators.py:25)
  # Verify: p_ekf_r = np.logspace(-5, 0, 12) (main.py:590)

STEP 1: main.py starts
  Imports: estimators.py, ekf2.py, pigru_model.py, plotting.py, benchmark_plottings.py
  Defines: GLOBAL_CFG, N_COST_REPS=20, N_MC_RUNS=30, RESULTS_RAW_DIR="results_raw"
  Output dir: src/figures_estimatores_benchmark_test5/ (from plotting.OUTPUT_DIR)

STEP 2: Signal generation (inside run_benchmark())
  estimators.get_test_signals(seed=42) → dict with 5 scenarios:
    {IEEE_Mag_Step, IEEE_Freq_Ramp, IEEE_Modulation, IBR_Nightmare, IBR_MultiEvent_Classic}
  Each entry: {t_phys, v_phys, t_dsp, v_dsp, f_true_dsp, metadata}

STEP 3: Grid search tuning (per scenario × per method)
  tune_ipdft(), tune_pll(), tune_ekf(), tune_ekf2(), tune_sogi(), tune_rls(),
  tune_teager(), tune_tft(), tune_vff_rls(), tune_ukf(), tune_koopman()
  + build_pigru_estimator() (no tuning — fixed pretrained model)
  Stores: tuned_params_per_scenario[sc_name][method] = {param: value}

STEP 4: Benchmark run (per scenario × per method)
  For each method: run_and_time(factory, signal, n_reps=20)
    → trace (f_hat array), avg_cpu_time_s
  calculate_metrics(trace, f_true, cpu_time, structural_samples)
    → RMSE, MAE, MAX_PEAK, SETTLING, TRIP_TIME_0p5, TIME_PER_SAMPLE_US, PCB, iec_compliance
  Save raw: src/results_raw/<scenario>/<scenario>__<method>.json

STEP 5: Monte Carlo (run_monte_carlo_all)
  For each mc_run in range(30):
    seed = 2000 + mc_run
    Re-generate all 5 signals with this seed
    Evaluate 11 methods (PI-GRU excluded) with FIXED tuned params
    Accumulate: {RMSE, FE_max, RFE_max, TRIP} per (method, scenario)
  Save: src/results_raw/raw_mc.json

STEP 6: Statistical analysis (statistical_analysis.run_full_analysis)
  Pareto frontier, Kruskal-Wallis H1/H2, Wilcoxon, clustering
  → json_export["statistical_analysis"]

STEP 7: Paper claim verification (verify_key_claims)
  Checks EXPECTED values with tolerances — prints ✅ or ❌
  → json_export["paper_claims_verification"]

STEP 8: Regression assertions
  Hard bounds on key metrics (EKF2 Nightmare RMSE, TRIP, etc.)
  Prints: "Regression assertions: ALL PASS" or "FAILURES DETECTED"

STEP 9: Export
  Write: src/figures_estimatores_benchmark_test5/benchmark_results.json
  Write: src/figures_estimatores_benchmark_test5/paper_ready_numbers.txt
  Write: src/figures_estimatores_benchmark_test5/issues_log.txt

STEP 10: Publication figures (benchmark_plottings.generate_publication_figures)
  Reads: json_export (in memory) + results_raw/
  Writes: Fig1_Scenarios_Final.pdf/png, Fig2_Mega_Dashboard.pdf/png
  Also writes: per-method hypermaps, optimized traces

STEP 11: Scientific analysis (src/scientific_analysis.py — called separately or inline)
  Outputs: src/figures_scientific/Q*.pdf, scientific_summary.txt
  Outputs: scientific_analysis_extended.json (location: INFERRED src/ or OUTPUT_DIR)
```

---

## 6. Entry Points and Commands

| Script | Path | What it does | Status |
|---|---|---|---|
| **Main benchmark** | `src/main.py` | Full pipeline: tune + run + MC + plot + export | **CANONICAL** |
| **Offline analysis** | `src/run_analysis_offline.py` | Re-runs scientific analysis from existing JSON | SUPPORT |
| **Plot chamorro** | `src/plot_chamorro.py` | UNKNOWN — inspect before use | UNCLEAR |
| **IBR plot v2** | `src/plot_ibr_v2.py` | UNKNOWN — IBR-specific plotting? | UNCLEAR |
| **Smoke test** | `src/test_smoke.py` | Quick sanity: imports + instantiation | TEST |
| **IpDFT test** | `src/test_ipdft_targeted.py` | IpDFT unit tests | TEST |
| **Koopman test** | `src/test_koopman_consistency.py` | Koopman consistency | TEST |
| **PI-GRU test** | `src/test_pigru_sanity.py` | PI-GRU model load + inference check | TEST |
| **Train PI-GRU** | `src/pigru_train.py` / `src/train_pigru.py` | Model training (ALREADY DONE — model exists) | LEGACY |

**Canonical run command:**
```bash
cd /c/Users/walla/Documents/Github/paper-P03-sgsma-frequency-estimators-benchmark/src
python main.py 2>&1 | tee ../run_final_$(date +%Y%m%d_%H%M%S).log
```

**Expected runtime:** 30–90 min (grid search across 12 estimators × 5 scenarios + 30 MC runs)

**Paper LaTeX build:**
```bash
cd /c/Users/walla/Documents/Github/paper-P03-sgsma-frequency-estimators-benchmark/paper
pdflatex index && bibtex index && pdflatex index && pdflatex index
```
⚠ **Will fail** until `Sections/C4_Simulation_Setup/main2.tex` is created (see §13).

---

## 7. Estimator Inventory

### Complete Set (12 paper-facing)

| Code Label | Class/File | Family | State Vector | Key Notes |
|---|---|---|---|---|
| `EKF2` | `ekf2.EKF2` | Model-based | [θ, ω, A, ω̇] (4D) | **Proposed.** Innovation-driven Q scaling: Q_k = clip(|ν|/σ̂_ν, 0.25, 4)·Q_base. Event gating: if |ν|>2σ̂, soft re-init of θ. Dual-mode covariance. |
| `EKF` | `estimators.ClassicEKF` | Model-based | [θ, ω, A] (3D) | Classical discrete EKF. z = A·sin(θ). Baseline Kalman. |
| `UKF` | `estimators.UKF_Estimator` | Model-based | [θ, ω, A] (3D) | Unscented Kalman. Avoids Jacobian singularities. 10-sample output smoothing. |
| `PLL` | `estimators.StandardPLL` | Loop-based | PI loop state | SRF-PLL with PI controller + moving average filter (MAF). Industrial standard. |
| `SOGI` | `estimators.SOGI_FLL` | Loop-based | 2D internal | **NON-STANDARD.** IIR bandpass pre-filter + FastRMS normalizer. NOT textbook SOGI-FLL. ω adaptation law fixed (rad/s vs Hz correction: factor 120π). |
| `IpDFT` | `estimators.TunableIpDFT` | Window-based | Circular buffer N | Hann window + Jacobsen complex spectral interpolation (NOT parabolic). N_c tunable {2,3,4,6,8,10 cycles}. |
| `TFT` | `estimators.TFT_Estimator` | Window-based | Circular buffer N | 2nd-order Taylor-Fourier transform. |
| `RLS` | `estimators.RLS_Estimator` | Adaptive recursive | AR(2) + covariance | Fixed forgetting factor λ. AR(2) bandpass pre-whitening. Decimated to 200 Hz. |
| `RLS-VFF` | `estimators.RLS_VFF_Estimator` | Adaptive recursive | AR(2) + VFF state | Variable forgetting factor λ(t). Adaptive based on innovation magnitude. |
| `Teager` | `estimators.Teager_Estimator` | Adaptive recursive | 3-sample window | Teager-Kaiser Energy Operator (DESA-2). Minimal state. Very fast. |
| `Koopman-RKDPmu` | `estimators.Koopman_RKDPmu` | Data-driven | Sliding window | Koopman operator, 2D embedding [x_k, x_{k-1}]. Fit linear operator via least squares. Window cap: 1000 samples = 100 ms. Structural latency = window_samples. |
| `PI-GRU` | `pigru_model.build_pigru_estimator` | Data-driven (neural) | GRU cell state | **Pretrained only** (weights: `pi_gru_pmu.pt`). Physics-informed GRU. Python/CPU baseline ~154 ms/sample. **Excluded from MC.** |

**Permanently excluded:**
- `LKF` — Leakage-corrected Kalman. RMSE 17–21 Hz (degenerate AR(2)). Fully removed from all code paths (Epic A-003 complete).

**Monte Carlo set:** 11 methods (all except PI-GRU).

---

## 8. Scenario Inventory

### Five Scenarios

| ID | Code Name | Duration | Disturbance | Noise σ | IEC/IEEE Scope | Paper Name |
|---|---|---|---|---|---|---|
| A | `IEEE_Mag_Step` | 1.5 s | +10% voltage magnitude step at t=0.5 s | 0.001 | IEC 60255-118-1 §5.3.4 | §IV-A Voltage Step |
| B | `IEEE_Freq_Ramp` | 1.5 s | +3 Hz/s RoCoF from t=0.3–0.8 s | 0.001 | IEEE 1547 Cat III | §IV-A Frequency Ramp |
| C | `IEEE_Modulation` | 1.5 s | AM 10% @ 2 Hz + AWGN | 0.001 | IEC 60255-118-1 §5.3.5 | §IV-A Modulation |
| D | `IBR_Nightmare` | 1.5 s | +60° phase jump + 5th harmonic + 32.5 Hz interharmonic | 0.005 | IBR-specific (beyond IEC) | §IV-B Composite Islanding |
| E | `IBR_MultiEvent_Classic` | 5.0 s | Multi-event: ±phase jumps, −3 Hz/s RoCoF, 5th+7th harmonics, impulsive noise | 0.003 + impulsive | IBR-specific | §IV-C Multi-Event |

**Additional scenario in code (not paper-facing):**
- `IBR_PrimaryFrequencyResponse` — primary frequency response simulation; present in `get_test_signals()` output but may not be in main benchmark loop — **NEEDS VERIFICATION**

**Scenario name alias:** CLAUDE.md uses `IBR_MultiEvent` but code JSON key is `IBR_MultiEvent_Classic`. All JSON lookups must use `IBR_MultiEvent_Classic`.

---

## 9. Metrics and Benchmark Logic

### Primary Metrics

| Metric | Key in JSON | Definition | Units |
|---|---|---|---|
| `RMSE` | `RMSE` | √(mean(e²)) over evaluation window | Hz |
| `MAE` | `MAE` | mean(|e|) over eval window | Hz |
| `MAX_PEAK` / `FE_max` | `MAX_PEAK` | max(|e|) in eval window | Hz |
| `SETTLING` | `SETTLING` | First time |e|<0.2 Hz (and remains) | s |
| `TRIP_TIME_0p5` / T_trip | `TRIP_TIME_0p5` | Cumulative time |e|>0.5 Hz in eval window | s |
| `TIME_PER_SAMPLE_US` | `TIME_PER_SAMPLE_US` | Mean CPU µs/sample (N=20 reps via `time.process_time()`) | µs |
| `PCB` | `PCB` | Probabilistic compliance bound = µ + 3σ (MC-derived) | Hz |
| `STRUCTURAL_LATENCY_MS` | `STRUCTURAL_LATENCY_MS` | Estimator-reported inherent latency | ms |

### Evaluation Window

```python
start_idx = max(int(0.15 * FS_DSP), structural_latency_samples)
           = max(1500 samples, structural_samples)
           = max(150 ms, filter_init_latency)
```
**Note:** `estimators.py` docstring at ~line 1205 incorrectly says "15% of samples" — actual calculation is 150 ms fixed (int(0.15 × 10000) = 1500 samples). Documentation error only; calculation correct.

### IEC 60255-118-1 Compliance (for compliance heatmap, Fig 2g)

| Threshold | Value |
|---|---|
| RMSE pass | < 0.05 Hz |
| PEAK pass | < 0.5 Hz |
| T_trip pass | < 0.1 s |

### Trip-Risk Logic

```python
TRIP_TIME_0p5 = np.sum(np.abs(error[start_idx:]) > 0.5) / FS_DSP
```
This is the **primary protection-relevant metric** — cumulative seconds the frequency error exceeds the relay trip threshold.

---

## 10. Tuning / Optimization Logic

### Method

**Grid search, deterministic, SEED=42, scenario-wise** (not global). Each estimator tuned independently per scenario. Objective: minimize RMSE over evaluation window.

**Critical:** `USE_BAYESIAN_TUNING = False` in `estimators.py:25`. Setting this True enables `popsize=1` differential_evolution that finds degenerate optima (fixed in A-001).

### Search Spaces (src/main.py lines 579–629)

| Estimator | Parameters | Search Range |
|---|---|---|
| `IpDFT` | N_c (cycles) | [2, 3, 4, 6, 8, 10] |
| `PLL` | Kp, Ki | Kp: logspace(log10(0.5), log10(100), 24); Ki: linspace(1, 300, 36) |
| `EKF` / `UKF` | Q, R | Q: logspace(-4, 3, 12); **R: logspace(-5, 0, 12) → R_max=1.0** |
| `EKF2` | q_param, r_param, inn_ref, event_thresh, fast_horizon_ms | q: same; r: same; inn_ref: {0.1,0.3}; event_thresh: 2.0 (fixed); fast_horizon_ms: 80.0 |
| `SOGI` | k, gamma, smooth_win | k: [0.5,0.707,1.0,1.414,2.0]; gamma: [5,10,20,30,50,80,100,150]; sw: [None,50,100,167] |
| `RLS` | lambda, win | lambda: [0.90,0.95,0.98,0.99,0.995,0.999,0.9995]; win: [50,100,200,500] |
| `Teager` | win | [10, 20, 30, 40, 50] |
| `TFT` | win_cycles | [2, 3, 4, 6] |
| `RLS-VFF` | lam_min, Ka | lam_min: [0.90,0.95,0.98,0.99,0.999]; Ka: [0.5,1.0,2.0,5.0,10.0] |
| `Koopman` | window_samples | [10, 40, 80, 160, 333, 500, 800, 1000] |
| `PI-GRU` | — | No tuning. Fixed pretrained model. |

**R constraint (A-002):** `p_ekf_r = np.logspace(-5, 0, 12)` → max R=1.0. This prevents degenerate EKF behavior where R→100 causes the filter to ignore measurements, producing T_trip=0 (falsely excellent trip-risk), invalidating the 275× claim.

**Fairness caveat:** Scenario-wise tuning gives ceiling performance — represents best-case for each estimator under each disturbance. This is honest for benchmarking but not representative of a fixed deployment configuration.

---

## 11. Data and Artifact Lineage

| Artifact | Path | Generated by | Consumed by | Canonical? | Stale risk |
|---|---|---|---|---|---|
| `benchmark_results.json` | `src/figures_estimatores_benchmark_test5/benchmark_results.json` | `main.py: run_benchmark()` | Paper tables, claim verification | **YES — canonical** | High if predates code change |
| `benchmark_results.json` | `results/benchmark_results.json` | Previous run (from project root) | — | **NO — stale copy** | **HIGH — do not use** |
| `raw_mc.json` | `src/results_raw/raw_mc.json` | `main.py: run_monte_carlo_all()` | `statistical_analysis.py` | YES | Medium |
| `<scenario>__<method>.json` | `src/results_raw/<scenario>/` | `main.py` per-method run | `benchmark_plottings.py` | YES | Medium |
| `paper_ready_numbers.txt` | `src/figures_estimatores_benchmark_test5/` | `main.py` post-run | Manual paper updates | YES | Regenerated each run |
| `issues_log.txt` | `src/figures_estimatores_benchmark_test5/` | `main.py` ISSUES_RESOLVED checklist | Manual review | YES | Regenerated each run |
| `Fig1_Scenarios_Final.pdf` | `src/figures_estimatores_benchmark_test5/` | `benchmark_plottings.py` | LaTeX (after manual copy) | YES | Stale until re-run |
| `Fig2_Mega_Dashboard.pdf` | `src/figures_estimatores_benchmark_test5/` | `benchmark_plottings.py` | LaTeX (after manual copy) | YES | Stale until re-run |
| `Fig*.pdf` in paper | `paper/Figures/Plots_And_Graphs/` | **Manual copy from above** | `paper/index.tex` | Derived | **HIGH — manual step** |
| `scientific_analysis_extended.json` | INFERRED: `src/` or OUTPUT_DIR | `scientific_analysis.py` | Paper extended analysis | YES | Medium |
| `Q*.pdf/png` | `src/figures_scientific/` | `scientific_analysis.py` | Optional appendix | YES | Medium |
| `pi_gru_pmu.pt` | `src/pi_gru_pmu.pt` | Training (DONE — fixed) | `pigru_model.py` | YES — frozen | Low (binary asset) |

---

## 12. Paper Integration Map

### LaTeX Structure

```
paper/index.tex                      ← Main document (IEEEtran conference)
  \input{Config/packages}
  \input{Config/commands}
  \input{Config/title}               ← Paper title
  \input{Config/authors}             ← Jorge Mayorga + co-authors
  \input{Config/header}
  \input{Config/abstract}            ← ⚠ Contains hardcoded numbers
  \input{Config/keywords}
  \input{Sections/C1_Introduction/main}
  \input{Sections/C2_Related_Work/main}
  \input{Sections/C3_Methods/main}
  \input{Sections/C4_Simulation_Setup/main2}   ← ⚠ FILE DOES NOT EXIST
  \input{Sections/C4_Simulation_Results/main}
  \input{Sections/C5_Conclusions/main}
  \bibliography{bibtex/bib/references}
```

### Hardcoded Values to Update After Fresh Run

| File | What to update | JSON source |
|---|---|---|
| `paper/Config/abstract.tex` | All RMSE ratios and absolute values | `benchmark_results.json` |
| `paper/Sections/C4_Simulation_Results/main.tex` | §IV-A: Ramp EKF2 RMSE, PLL/EKF2 ratio, EKF/EKF2 ratio | `IEEE_Freq_Ramp.{EKF2,PLL,EKF}.RMSE` |
| same | §IV-B: EKF T_trip, EKF2 RMSE, EKF2 T_trip, 4.7× ratio, 275× ratio, IpDFT RMSE/peak | `IBR_Nightmare.*` |
| same | Tables III, IV, V — all cells | `benchmark_results.json` |
| `paper/Sections/C5_Conclusions/main.tex` | PLL T_trip, EKF2 CPU cost, key ratios | `IBR_MultiEvent_Classic.*` |

### Figure Copy Step (Manual — Required)

```bash
# After successful run from src/:
cp src/figures_estimatores_benchmark_test5/Fig1_Scenarios_Final.pdf paper/Figures/Plots_And_Graphs/
cp src/figures_estimatores_benchmark_test5/Fig2_Mega_Dashboard.pdf paper/Figures/Plots_And_Graphs/
# Verify timestamps:
ls -lh paper/Figures/Plots_And_Graphs/Fig*.pdf
```

### Reproducibility Gaps

1. **Manual figure copy** — no automated step
2. **Manual table update** — LaTeX tables are hardcoded (no auto-generation script found)
3. **Missing `C4_Simulation_Setup/main2.tex`** — LaTeX will not compile
4. `paper_ready_numbers.txt` is auto-generated but must be manually transcribed to LaTeX

---

## 13. Mismatch Audit: Code vs Paper

### Known and Verified Mismatches

| Item | Code reality | Paper text | Severity |
|---|---|---|---|
| **Missing LaTeX section file** | `C4_Simulation_Setup/main2.tex` does not exist | `\input{Sections/C4_Simulation_Setup/main2}` in `index.tex:102` | **CRITICAL** — LaTeX won't compile |
| **SOGI-FLL description** | Non-standard: IIR bandpass + FastRMS normalizer | INFERRED: may say "textbook SOGI-FLL" | HIGH — per E-001 |
| **Evaluation window** | 150 ms fixed | Docstring says "15% of samples" | MEDIUM — docstring only |
| **IpDFT interpolation** | Jacobsen complex spectral (FIXED A-004) | Was "parabolic" — verify fix in Sections/C3 | LOW — should be fixed |
| **Monte Carlo comment** | N_MC_RUNS=30, but code comment at line 140 says "10 runs" | — | LOW — stale comment |
| **MC function docstring** | `run_monte_carlo_all` docstring says "13 estimators" | — | LOW — stale docstring |
| **Scenario name** | JSON key: `IBR_MultiEvent_Classic` | CLAUDE.md and some paper text use `IBR_MultiEvent` | MEDIUM — lookup errors |
| **Stale `results/` directory** | `results/benchmark_results.json` timestamp 2026-03-30, possibly from wrong working dir | — | HIGH — do not use for paper |
| **`results/` JSON vs correct output path** | Canonical output: `src/figures_estimatores_benchmark_test5/benchmark_results.json` | Some CLAUDE.md sections say `src/benchmark_results.json` | MEDIUM — path confusion |
| **SOGI paradox** | SOGI achieves lower RMSE than EKF2 in IBR_Nightmare (lines 1312-1329) | INFERRED: paper may claim EKF2 best overall | HIGH if unaddressed |
| **PI-GRU CPU** | ~154 ms/sample (Python/CPU) | Paper should note this is not real-time deployable | MEDIUM |
| **Koopman latency** | Up to 100 ms structural window delay | INFERRED: may be undisclosed | HIGH per E-002 |
| **`src/claude.md`** | Is a stale redirect — says `code_sim_and_results/` is canonical | Conflicts with root CLAUDE.md | MEDIUM — confusing |

### Unverified Paper Claims (Require Fresh Run)

- 275× trip-risk ratio (EKF vs EKF2, Nightmare) — only valid with R constraint
- 4.7× RMSE ratio (EKF vs EKF2, Nightmare)
- 12.6× ramp ratio (PLL vs EKF2)
- 1.59× ramp ratio (EKF vs EKF2)
- 3.3× trip ratio (PLL vs EKF2, Multi-Event)
- All table numeric values

---

## 14. Technical Debt Audit

### Dead / Stale Code

| Item | Location | Issue |
|---|---|---|
| `src/claude.md` | `src/claude.md` | Stale redirect. References old `code_sim_and_results/` path. Misleading. |
| `src/scenarios.py` | `src/scenarios.py` | Possibly superseded by `estimators.get_test_signals()`. Purpose unclear. |
| `src/pigru_train.py` + `src/train_pigru.py` | `src/` | **Duplicate?** Two training scripts for same model. Model already trained. |
| `src/plot_chamorro.py` | `src/` | Unknown purpose. Not imported by main pipeline. |
| `src/plot_ibr_v2.py` | `src/` | Unknown purpose. Not imported by main pipeline. |
| `package.json` at root | `/package.json` | Node.js package file. No JS code visible. Purpose unknown. |
| `results/` at root | `results/` | Stale run output from different working directory. Confusing. |
| Stale comment at `main.py:140` | Line 140 | Says "N_MC = 10 is sufficient" but N_MC_RUNS=30 |
| Stale docstring in `run_monte_carlo_all` | `main.py:~152` | Says "13 estimators" (LKF removal not reflected) |
| `paper/Sections/C4_Simulation_Setup/` | paper/ | **Entire directory missing** — `index.tex:102` will fail |

### Hardcoded Paths / Fragile Wiring

| Issue | Location | Risk |
|---|---|---|
| `OUTPUT_DIR` derived from `plotting.py`'s `BASE_DIR` | `plotting.py:12-13` | If run from wrong directory, outputs land in wrong place |
| `RESULTS_RAW_DIR = "results_raw"` is relative | `main.py:85` | Run from `src/` puts output at `src/results_raw/` ✓. Run from root → `root/results_raw/` |
| `results/` exists as legacy output | root | Will confuse anyone looking for canonical results |
| Manual figure copy step | — | Human step → stale figures in LaTeX if forgotten |
| Manual LaTeX table update | paper/ | No auto-generation script — all tables require human transcription |

### Naming Inconsistencies

| CLAUDE.md name | Code/JSON key | Issue |
|---|---|---|
| `IBR_MultiEvent` | `IBR_MultiEvent_Classic` | JSON lookup will fail if using CLAUDE.md name |
| `benchmark_results.json` at `src/` | Actually at `src/figures_estimatores_benchmark_test5/` | Path confusion in CLAUDE.md §5 |
| `Sections/C4_Simulation_Results/main.tex` lines cited in CLAUDE.md | Actual file exists | Verify line numbers after any edit |
| `RA-EKF` (paper) | `EKF2` (code) | All JSON lookups use `EKF2` |
| `VFF-RLS` (paper) | `RLS-VFF` (code JSON key) | Check JSON keys |

---

## 15. Refactor Recommendations

### P0 — Critical (Paper at risk)

| Issue | Why it matters | Fix | Files |
|---|---|---|---|
| **Missing `C4_Simulation_Setup/main2.tex`** | LaTeX will not compile | Create the file (even if empty or copy from another section) | `paper/Sections/C4_Simulation_Setup/` |
| **`results/` stale JSON** | May be used instead of canonical output | Delete `results/` or add README warning; update CLAUDE.md §5 output paths | `results/`, `CLAUDE.md` |
| **CLAUDE.md run command path** | Says `cd .../paper/src` but repo dir is `paper-P03-...` | Fix canonical run command in CLAUDE.md | `CLAUDE.md` |
| **Scenario name alias** | `IBR_MultiEvent` vs `IBR_MultiEvent_Classic` | Standardize in CLAUDE.md and paper references | `CLAUDE.md` |

### P1 — Important

| Issue | Why it matters | Fix | Files |
|---|---|---|---|
| **SOGI paradox undisclosed** | Reviewer may see SOGI outperforming EKF2 in Nightmare as inconsistency | Add E-006 discussion to paper §IV-B | `C4_Simulation_Results/main.tex` |
| **Koopman latency undisclosed** | Reviewer 2 will notice 100 ms structural delay not mentioned | Add E-002 sentence to methods or results | `C3_Methods/main.tex` |
| **`src/claude.md` stale** | Misleads future maintainers with old paths | Delete or update with redirect warning | `src/claude.md` |
| **Stale comments in main.py** | "13 estimators", "10 runs" | Update comments to reflect actual state | `src/main.py:140,152` |
| **Duplicate PI-GRU training scripts** | Confusion about canonical training process | Keep one, delete or rename the other | `src/pigru_train.py`, `src/train_pigru.py` |
| **Evaluation window docstring error** | "15% of samples" is wrong | Fix docstring at `estimators.py:~1205` | `src/estimators.py` |

### P2 — Cleanup

| Issue | Why it matters | Fix |
|---|---|---|
| `src/scenarios.py` purpose unclear | Dead code or undocumented utility | Add docstring or delete |
| `src/plot_chamorro.py` / `plot_ibr_v2.py` unknown | Could be useful but undiscovered | Add header comment with purpose |
| `package.json` at root | No apparent purpose | Investigate and remove if unused |
| Manual LaTeX table update | Reproducibility gap | Write a Python script to auto-generate LaTeX table rows from benchmark_results.json |

---

## 16. Reproducibility Checklist

### Environment Setup

```bash
# Python 3.13+ (verified)
cd src/
pip install -r requirements.txt
# Key packages: numpy>=2.4, scipy>=1.17, scikit-learn>=1.8, matplotlib>=3.10, torch>=2.10

# Verify pretrained model exists:
ls -lh src/pi_gru_pmu.pt src/pi_gru_pmu_config.json
# Expected: ~900 KB .pt file, small .json

# Quick smoke test (< 5 seconds):
python test_smoke.py
python test_pigru_sanity.py
```

### Pre-Run Config Checks

```python
# estimators.py:25:
assert USE_BAYESIAN_TUNING == False

# main.py:590:
# p_ekf_r = np.logspace(-5, 0, 12)  → max = 10^0 = 1.0
```

### Full Benchmark Run

```bash
cd /c/Users/walla/Documents/Github/paper-P03-sgsma-frequency-estimators-benchmark/src
python main.py 2>&1 | tee ../run_final_$(date +%Y%m%d_%H%M%S).log
```

### Expected Intermediate Outputs (in order)

1. Console: `[REPRO-1] EKF best params ... R=<value ≤ 1.0>`
2. `src/results_raw/<scenario>/<method>.json` — 12 files × 5 scenarios = 60 files
3. `src/results_raw/raw_mc.json` — MC data (large)
4. Console: `[REPRO-2] Verifying key paper claims...` → `✅ All key paper claims verified`
5. Console: `Regression assertions: ALL PASS.`
6. `src/figures_estimatores_benchmark_test5/benchmark_results.json`
7. `src/figures_estimatores_benchmark_test5/paper_ready_numbers.txt`
8. `src/figures_estimatores_benchmark_test5/issues_log.txt`
9. `src/figures_estimatores_benchmark_test5/Fig1_Scenarios_Final.pdf/png`
10. `src/figures_estimatores_benchmark_test5/Fig2_Mega_Dashboard.pdf/png`

### Post-Run Sanity Checks

```python
import json
br = json.load(open("figures_estimatores_benchmark_test5/benchmark_results.json"))

# Check EKF R was constrained:
print(br["results"]["IEEE_Mag_Step"]["methods"]["EKF"]["tuned_params"]["R"])
# Must be ≤ 1.0

# Check key claim preconditions:
ekf_trip = br["results"]["IBR_Nightmare"]["methods"]["EKF"]["TRIP_TIME_0p5"]
ekf2_trip = br["results"]["IBR_Nightmare"]["methods"]["EKF2"]["TRIP_TIME_0p5"]
print(f"EKF trip: {ekf_trip:.4f} s  (must be > 0.05)")
print(f"EKF2 trip: {ekf2_trip:.6f} s  (must be < 0.01)")
print(f"Ratio: {ekf_trip/ekf2_trip:.0f}x  (must be > 100)")

# Check PI-GRU did not fail:
for sc in ["IEEE_Mag_Step", "IEEE_Freq_Ramp", "IEEE_Modulation", "IBR_Nightmare", "IBR_MultiEvent_Classic"]:
    params = br["results"][sc]["methods"]["PI-GRU"]["optimal_params"]
    assert "FAILED" not in str(params), f"PI-GRU failed in {sc}: {params}"
```

### Paper Regeneration Steps

```bash
# 1. After verified run, copy figures:
cp src/figures_estimatores_benchmark_test5/Fig1_Scenarios_Final.pdf paper/Figures/Plots_And_Graphs/
cp src/figures_estimatores_benchmark_test5/Fig2_Mega_Dashboard.pdf paper/Figures/Plots_And_Graphs/

# 2. Open paper_ready_numbers.txt and update LaTeX manually (or write auto-gen script)
cat src/figures_estimatores_benchmark_test5/paper_ready_numbers.txt

# 3. Compile paper (requires fixing missing C4_Simulation_Setup/main2.tex first):
cd paper/
pdflatex index && bibtex index && pdflatex index && pdflatex index
```

### Abort Conditions

- Any `RMSE > 10 Hz` in any method/scenario
- `PI-GRU optimal_params == "PI-GRU FAILED"` in any scenario
- `EKF.TRIP_TIME_0p5 == 0.0` in `IBR_Nightmare` (R constraint failed)
- `benchmark_results.json` not updated (crash before end)
- Regression assertions FAIL printed

---

## 17. Glossary / Naming Crosswalk

| Paper name | Code label | JSON key | Notes |
|---|---|---|---|
| RA-EKF / RoCoF-Augmented EKF | `EKF2` | `"EKF2"` | Proposed method |
| EKF | `EKF` | `"EKF"` | Classical 3-state |
| UKF | `UKF` | `"UKF"` | |
| SRF-PLL | `PLL` | `"PLL"` | |
| bandpass-pre-filtered SOGI-FLL | `SOGI` | `"SOGI"` | Non-standard implementation |
| IpDFT | `IpDFT` | `"IpDFT"` | |
| TFT | `TFT` | `"TFT"` | |
| RLS | `RLS` | `"RLS"` | |
| VFF-RLS | `RLS-VFF` | `"RLS-VFF"` | Note: paper uses VFF-RLS, code key is RLS-VFF |
| TKEO / Teager | `Teager` | `"Teager"` | |
| Koopman-RKDPmu | `Koopman-RKDPmu` | `"Koopman-RKDPmu"` | |
| PI-GRU | `PI-GRU` | `"PI-GRU"` | |
| IEEE Voltage Step | Scenario A | `"IEEE_Mag_Step"` | |
| IEEE Frequency Ramp | Scenario B | `"IEEE_Freq_Ramp"` | |
| IEEE Modulation | Scenario C | `"IEEE_Modulation"` | |
| Composite Islanding / Nightmare | Scenario D | `"IBR_Nightmare"` | |
| IBR Multi-Event / Multi-Event Stress | Scenario E | `"IBR_MultiEvent_Classic"` | ⚠ NOT `IBR_MultiEvent` |
| T_trip / trip-risk duration | `TRIP_TIME_0p5` | `"TRIP_TIME_0p5"` | |
| Evaluation window / warm-up | `start_idx` | — | 150 ms fixed = 1500 DSP samples |
| µs/sample | CPU timing | `"TIME_PER_SAMPLE_US"` | Averaged over N=20 reps |
| PCB | Probabilistic compliance bound | `"PCB"` | µ + 3σ from MC |
| Master results | — | `src/figures_estimatores_benchmark_test5/benchmark_results.json` | Canonical output |
| Stale results | — | `results/benchmark_results.json` | Do NOT use |

---

## 18. Open Questions / Unknowns

| Question | Status | Notes |
|---|---|---|
| What is `src/plot_chamorro.py`? | UNKNOWN | Not imported by main pipeline. May be a standalone diagnostic. |
| What is `src/plot_ibr_v2.py`? | UNKNOWN | Same situation. |
| Are `pigru_train.py` and `train_pigru.py` duplicates? | UNKNOWN | One may be an older version. |
| What is `package.json` / `package-lock.json` for? | UNKNOWN | No JS code visible. Possibly an artifact from a template or CI setup. |
| Does `IBR_PrimaryFrequencyResponse` appear in main benchmark loop? | INFERRED: no | `get_test_signals()` returns 6 scenarios but main loop may only process 5. NEEDS VERIFICATION. |
| Where exactly does `scientific_analysis_extended.json` get written? | INFERRED: OUTPUT_DIR | Verify after run. |
| What does `src/run_analysis_offline.py` do exactly? | INFERRED: re-runs scientific_analysis.py from JSON | Read file to verify. |
| Were the `results/` files produced from a different version of the code? | INFERRED: yes (wrong working dir) | Timestamp 2026-03-30 — likely from same-day run but different cwd. |
| Does the SOGI-FLL truly outperform EKF2 in IBR_Nightmare? | INFERRED: yes (code detects and logs this) | `main.py:1312-1329` checks for this. If true, paper discussion of §IV-B "EKF2 best in Nightmare" claim needs qualification. |
| Is `paper/Sections/C4_Simulation_Setup/` intentionally removed or accidentally deleted? | UNKNOWN | `index.tex:102` has `\input{Sections/C4_Simulation_Setup/main2}`. LaTeX will error without it. |
| What is the exact `scientific_analysis.py` trigger in `main.py`? | INFERRED: called inline after benchmark | Need to verify exact call site in main.py after line 1500. |

---

## REPO_TRIAGE_SUMMARY

### 10 Most Important Findings

1. **Missing LaTeX file blocks compilation:** `paper/Sections/C4_Simulation_Setup/main2.tex` is referenced by `index.tex:102` but does not exist. Paper cannot be compiled until this is fixed.
2. **Stale `results/` directory at root:** Contains `benchmark_results.json` from a run with wrong working directory (2026-03-30). This is NOT the canonical output. Canonical output is `src/figures_estimatores_benchmark_test5/benchmark_results.json`.
3. **SOGI paradox is real:** Code at `main.py:1312-1329` detects and logs that SOGI achieves lower RMSE than EKF2 in IBR_Nightmare. This complicates the paper's narrative of "EKF2 best overall" if unaddressed.
4. **All key paper claims are unverified from a canonical run.** Epic B-001 (canonical run) has not been executed. The 275× trip ratio, 4.7× RMSE ratio, 12.6× ramp ratio are all contingent on the R constraint being active.
5. **EKF R constraint is confirmed in code** (`main.py:590`). The critical fix (A-002) is in place. Without this, EKF.TRIP_TIME_0p5=0 in IBR_Nightmare and the 275× claim is undefined.
6. **`src/claude.md` is stale and misleading.** It references `code_sim_and_results/` as canonical path — contradicts root CLAUDE.md and reality.
7. **CLAUDE.md §5 output path is wrong.** Claims `benchmark_results.json` is at `src/` — actual path is `src/figures_estimatores_benchmark_test5/benchmark_results.json`.
8. **Scenario name inconsistency:** CLAUDE.md and some paper sections use `IBR_MultiEvent` but the JSON key is `IBR_MultiEvent_Classic`. Any code/script using the CLAUDE.md name will silently fail JSON lookups.
9. **Manual steps break reproducibility:** Figure copy and table updates both require human intervention with no automated validation. Risk of stale figures in LaTeX.
10. **Two PI-GRU training scripts exist** (`pigru_train.py` + `train_pigru.py`). One is likely dead code. Unclear which generated the deployed model.

### 10 Highest-Risk Issues

1. **LaTeX compile failure** (missing `C4_Simulation_Setup/main2.tex`) — blocks submission
2. **Using stale `results/benchmark_results.json`** for paper updates — would introduce wrong numbers
3. **Unverified paper claims** — all ratios contingent on canonical run completing successfully
4. **SOGI outperforming EKF2 in Nightmare** — if left unaddressed in paper discussion, Reviewer 2 will catch it
5. **PI-GRU failure mode** — if `pi_gru_pmu.pt` is corrupted or missing, benchmark completes but PI-GRU shows RMSE=60 Hz which looks like "best" on near-nominal scenarios
6. **Figure staleness in LaTeX** — if run completes but copy step is skipped, paper has old figures
7. **EKF R constraint** — only in `main.py:590`; if someone edits this or runs a different script, constraint disappears
8. **Koopman 100 ms structural latency undisclosed** — methodological reviewer objection
9. **SOGI-FLL non-standard implementation undisclosed** — reviewer objection to "SOGI-FLL" label
10. **Evaluation window docstring error** — minor but if a reviewer reads the code, "15% of samples" vs "150 ms fixed" could trigger a reproducibility objection

### 10 Most Important Next Actions

1. **Fix `paper/Sections/C4_Simulation_Setup/`** — create `main2.tex` (even as stub) so LaTeX compiles
2. **Run canonical benchmark** (`cd src && python main.py`) and verify ✅ all claims
3. **Check SOGI/EKF2 result in IBR_Nightmare** — if SOGI truly wins, add 1-2 sentences to paper §IV-B
4. **Update CLAUDE.md** — fix canonical output path from `src/benchmark_results.json` to `src/figures_estimatores_benchmark_test5/benchmark_results.json`
5. **Update or delete `src/claude.md`** — remove stale `code_sim_and_results/` reference
6. **Standardize `IBR_MultiEvent_Classic` name** throughout CLAUDE.md and paper references
7. **Copy figures to LaTeX** after successful run and verify timestamps
8. **Update all LaTeX hardcoded values** from `paper_ready_numbers.txt` after run
9. **Add SOGI and Koopman disclosures** to `C3_Methods/main.tex` (E-001, E-002)
10. **Fix evaluation window docstring** at `estimators.py:~1205`
