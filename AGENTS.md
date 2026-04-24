# AGENTS.md
## Current-Code Execution Protocol

This file documents the repository as it exists now.

If `AGENTS.md`, the paper text, old review notes, or stale artifacts conflict
with the code under `src/`, the code wins.

---

## 0. Current project phase

The repository is in a consolidation phase:

- the active benchmark logic has been moved into `src/pipelines/`
- plotting logic is being modularized under `src/plotting/benchmark/`
- LaTeX still exists under `paper/`
- legacy wrappers and legacy outputs still exist under `tests/montecarlo/`

This is not the old single-entry camera-ready workflow anymore.

---

## 1. Source of truth

Use these paths as the current authority:

- Benchmark entry point:
  - [full_mc_benchmark.py](C:/Users/walla/Documents/Github/paper-P03-sgsma-frequency-estimators-benchmark/src/pipelines/full_mc_benchmark.py)
- Artifact path constants:
  - [paths.py](C:/Users/walla/Documents/Github/paper-P03-sgsma-frequency-estimators-benchmark/src/pipelines/paths.py)
- Artifact sync to LaTeX:
  - [sync_paper_artifacts.py](C:/Users/walla/Documents/Github/paper-P03-sgsma-frequency-estimators-benchmark/src/pipelines/sync_paper_artifacts.py)
- Monte Carlo engine:
  - [monte_carlo_engine.py](C:/Users/walla/Documents/Github/paper-P03-sgsma-frequency-estimators-benchmark/src/analysis/monte_carlo_engine.py)
- Metrics:
  - [metrics.py](C:/Users/walla/Documents/Github/paper-P03-sgsma-frequency-estimators-benchmark/src/analysis/metrics.py)
- Plot orchestration:
  - [generate_mega_dashboards.py](C:/Users/walla/Documents/Github/paper-P03-sgsma-frequency-estimators-benchmark/src/plotting/benchmark/generate_mega_dashboards.py)

Legacy wrappers:

- [test_dedicated_smoke_test.py](C:/Users/walla/Documents/Github/paper-P03-sgsma-frequency-estimators-benchmark/tests/montecarlo/test_dedicated_smoke_test.py)
- [generate_mega_dashboards.py](C:/Users/walla/Documents/Github/paper-P03-sgsma-frequency-estimators-benchmark/tests/montecarlo/generate_mega_dashboards.py)

Those wrappers are compatibility shims only. They are not the primary code.

---

## 2. Canonical execution

Preferred commands:

```bash
cd src
python -m pipelines.full_mc_benchmark
```

or from repo root:

```bash
python src/pipelines/full_mc_benchmark.py
```

To sync the two paper-facing dashboard figures into LaTeX:

```bash
cd src
python -m pipelines.sync_paper_artifacts
```

or:

```bash
python src/pipelines/sync_paper_artifacts.py
```

---

## 3. Current output universe

Current canonical output root:

- `artifacts/full_mc_benchmark/`

Defined in:

- [paths.py](C:/Users/walla/Documents/Github/paper-P03-sgsma-frequency-estimators-benchmark/src/pipelines/paths.py)

Current canonical top-level artifacts:

- `global_metrics_report.csv`
- `benchmark_full_report.json`
- `dashboard_global_tradeoff.png`
- `Fig1_Scenarios_Final.png`
- `Fig1_Scenarios_Final.pdf`
- `Fig2_Mega_Dashboard.png`
- `Fig2_Mega_Dashboard.pdf`

Per-scenario structure:

- `artifacts/full_mc_benchmark/<scenario>/`
- `.../<scenario>/<estimator>/`
- per-estimator `*_summary.csv`
- per-estimator `*_signals.csv`
- per-estimator `run_spec.json`
- scenario-level `summary_stats.csv`

Do not treat `tests/montecarlo/outputs/` as canonical for new work.
It is a legacy output universe and currently coexists with the new one.

---

## 4. Current benchmark configuration

Read from:

- [full_mc_benchmark.py](C:/Users/walla/Documents/Github/paper-P03-sgsma-frequency-estimators-benchmark/src/pipelines/full_mc_benchmark.py)
- [monte_carlo_engine.py](C:/Users/walla/Documents/Github/paper-P03-sgsma-frequency-estimators-benchmark/src/analysis/monte_carlo_engine.py)

Current active constants:

- `N_TRIALS_TUNING = 500`
- `N_MC_RUNS = 100`
- `MonteCarloEngine.n_cost_reps = 20`
- Monte Carlo base seed:
  - `base_seed = 12345`
- MC run seeds:
  - `12345 + run_idx`

Current timing protocol:

- CPU cost is measured with repeated `time.process_time()` runs
- one full estimator pass per repetition
- metric exported as `m13_cpu_time_us`

Current metric contract:

- `m1_rmse_hz`
- `m2_mae_hz`
- `m3_max_peak_hz`
- `m4_std_error_hz`
- `m5_trip_risk_s`
- `m5_trip_risk_resolution_s`
- `m6_max_contig_trip_s`
- `m7_pcb_hz`
- `m8_settling_time_s`
- `m9_rfe_max_hz_s`
- `m10_rfe_rms_hz_s`
- `m11_rnaf_db`
- `m12_isi_pu`
- `m13_cpu_time_us`
- `m14_struct_latency_ms`
- `m15_pcb_compliant`
- `m16_heatmap_pass`
- `m17_hw_class`

Warm-up/evaluation window logic currently lives in:

- [metrics.py](C:/Users/walla/Documents/Github/paper-P03-sgsma-frequency-estimators-benchmark/src/analysis/metrics.py)

The baseline warm-up is currently implemented as:

- `baseline_samples = int(0.15 * fs_dsp)`

At 10 kHz this is `1500` samples = `150 ms`.

---

## 5. Current scenario catalog

The active modular pipeline currently runs 32 scenarios.

Exact exported names:

1. `IBR_Multi_Event`
2. `IBR_Power_Imbalance_Ringdown`
3. `IBR_Harmonics_Small`
4. `IBR_Harmonics_Medium`
5. `IBR_Harmonics_Large`
6. `IEEE_Freq_Step`
7. `IEEE_Modulation`
8. `IEEE_Modulation_AM`
9. `IEEE_Modulation_FM`
10. `IEEE_OOB_Interference`
11. `IEEE_Phase_Jump_20`
12. `IEEE_Phase_Jump_60`
13. `NERC_Phase_Jump_60`
14. `IEEE_Single_SinWave`
15. `IEEE_Mag_Step_1pct`
16. `IEEE_Mag_Step_5pct`
17. `IEEE_Mag_Step_10pct`
18. `IEEE_Mag_Step_15pct`
19. `IEEE_Mag_Step_25pct`
20. `IEEE_Mag_Step_50pct`
21. `IEEE_Freq_Ramp_0.25Hzs`
22. `IEEE_Freq_Ramp_0.5Hzs`
23. `IEEE_Freq_Ramp_1Hzs`
24. `IEEE_Freq_Ramp_2Hzs`
25. `IEEE_Freq_Ramp_5Hzs`
26. `IEEE_Freq_Ramp_10Hzs`
27. `IEEE_Freq_Ramp_15Hzs`
28. `IEEE_Freq_Ramp_20Hzs`
29. `IBR_Power_Imbalance_Ringdown_Low_Noise`
30. `IBR_Power_Imbalance_Ringdown_Normal_Noise`
31. `IBR_Power_Imbalance_Ringdown_Medium_Noise`
32. `IBR_Power_Imbalance_Ringdown_Severe_Noise`

Do not assume the old 5-scenario paper subset unless you explicitly switch back
to that workflow in code.

---

## 6. Current estimator registry

The active estimator registry is explicit and policy-driven.

The canonical definition lives in:

- [benchmark_definition.py](C:/Users/walla/Documents/Github/paper-P03-sgsma-frequency-estimators-benchmark/src/pipelines/benchmark_definition.py)

Current loaded estimator labels in the modular pipeline:

1. `ZCD`
2. `IPDFT`
3. `TFT`
4. `RLS`
5. `PLL`
6. `SOGI-PLL`
7. `SOGI-FLL`
8. `Type-3 SOGI-PLL`
9. `LKF`
10. `LKF2`
11. `EKF`
12. `UKF`
13. `RA-EKF`
14. `TKEO`
15. `Prony`
16. `ESPRIT`
17. `Koopman (RK-DPMU)`
18. `PI-GRU`

Important current facts:

- `LKF` and `LKF2` are both active and intentional in the canonical registry.
- `PI-GRU` is active in the canonical registry.
- `PI-GRU` requires `torch` in the runtime environment.
- The current default PI-GRU checkpoint is `src/estimators/pi_gru_weights.pt`.
- `src/estimators/pi_gru_pmu.pt` still exists, but the current code does not use
  it by default because the active checkpoint behaves better in local validation.
- The active benchmark decision is explicit: the paper must adapt to the active
  pipeline, not the other way around.

If you need the exact loader behavior, read:

- [benchmark_definition.py](C:/Users/walla/Documents/Github/paper-P03-sgsma-frequency-estimators-benchmark/src/pipelines/benchmark_definition.py)
- [full_mc_benchmark.py](C:/Users/walla/Documents/Github/paper-P03-sgsma-frequency-estimators-benchmark/src/pipelines/full_mc_benchmark.py)

---

## 7. Plotting structure

Current plotting contract:

- Figure 1 is produced by:
  - [mega_dashboard1.py](C:/Users/walla/Documents/Github/paper-P03-sgsma-frequency-estimators-benchmark/src/plotting/benchmark/mega_dashboard1.py)
- Figure 2 is orchestrated by:
  - [generate_mega_dashboards.py](C:/Users/walla/Documents/Github/paper-P03-sgsma-frequency-estimators-benchmark/src/plotting/benchmark/generate_mega_dashboards.py)
- Figure 2 subplots are split across:
  - `mega_dashboard2_p00.py`
  - `mega_dashboard2_p01.py`
  - `mega_dashboard2_p10.py`
  - `mega_dashboard2_p11.py`
  - `mega_dashboard2_p20.py`
  - `mega_dashboard2_p21.py`
  - `mega_dashboard2_p30.py`
  - `mega_dashboard2_p31.py`

This subplot split is intentional and should be preserved unless there is a
clear reason to merge modules.

---

## 8. LaTeX and paper integration

Current paper figure sync target:

- `paper/Figures/Plots_And_Graphs/`

Current paper-facing figure names:

- `Fig1_Scenarios_Final`
- `Fig2_Mega_Dashboard`

Current paper build has already been normalized toward standard `IEEEtran`.
Do not reintroduce:

- `titlesec`
- custom `caption` package hacks
- manual float/equation spacing compression blocks
- `\IEEEPARstart` in conference mode

If page count becomes a problem, reduce content instead of forcing the template.

---

## 9. Reproducibility requirements

These rules are active and code-backed:

1. `step()` and `step_vectorized()` must remain algorithmically equivalent.
2. `ESPRIT` must remain deterministic.
3. `Prony` must not silently replace numerical failure with the last valid output.
4. `Scenario.run()` must return decimated 10 kHz data.
5. `MonteCarloEngine.run_once()` asserts `dt_actual == 1e-4`.
6. CPU timing must remain on repeated `process_time()` measurements.

Validation files:

- [test_scalar_vs_vector.py](C:/Users/walla/Documents/Github/paper-P03-sgsma-frequency-estimators-benchmark/src/tests/test_scalar_vs_vector.py)
- [test_esprit.py](C:/Users/walla/Documents/Github/paper-P03-sgsma-frequency-estimators-benchmark/tests/estimators/esprit/test_esprit.py)
- [test_prony.py](C:/Users/walla/Documents/Github/paper-P03-sgsma-frequency-estimators-benchmark/tests/estimators/prony/test_prony.py)

---

## 10. Session startup checklist

- [ ] State the current project phase before editing files
- [ ] Decide whether the task targets the modular pipeline or the legacy paper workflow
- [ ] If the task is benchmark-related, default to `src/pipelines/`
- [ ] If the task is figure-related, default to `src/plotting/benchmark/`
- [ ] If the task is paper-figure sync, use `src/pipelines/sync_paper_artifacts.py`
- [ ] If the task involves PI-GRU, check whether `torch` is available first
- [ ] If the task involves benchmark claims, read current artifacts from `artifacts/full_mc_benchmark/`, not old notes

---

## 11. Non-negotiable rules

1. Do not force the code to match stale paper text.
2. Do not treat historical review guidance as runtime truth if the code differs.
3. Do not use `tests/montecarlo/outputs/` as the source for new benchmark claims.
4. Do not assume the old 5-scenario / 12-estimator camera-ready subset is still the active benchmark.
5. Do not assume `PI-GRU` can run without `torch` and a valid checkpoint.
6. Do not silently drop `LKF`, `LKF2`, or `PI-GRU` from the active registry.
7. Do not reintroduce silent numerical masking in estimators.
8. Do not allow `step()` and `step_vectorized()` to diverge again.
9. Do not add new paper numbers that cannot be traced to current generated artifacts.

---

## 12. Practical guidance for future edits

If you are organizing code:

- move execution logic into `src/pipelines/`
- move plotting logic into `src/plotting/benchmark/`
- keep `tests/montecarlo/` as wrappers or regression harnesses only

If you are working on paper integration:

- generate figures from the modular pipeline
- copy them with `sync_paper_artifacts.py`
- then rebuild the paper

If you are reviewing scientific claims:

- prefer `benchmark_full_report.json`
- cross-check `global_metrics_report.csv`
- treat hardcoded paper numbers as suspect until regenerated
