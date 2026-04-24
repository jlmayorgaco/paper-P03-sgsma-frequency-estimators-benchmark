# Submission Traceability Supplement

This note records the exact benchmark artifact used by the paper and the
difference between that archived snapshot and the current modular pipeline.

## 1. Quantitative source used in the paper

- Artifact file:
  `tests/montecarlo/outputs/benchmark_full_report.json`
- SHA-256:
  `cf40dc9f5be304849d73d1ebde37de97f1276f4f80c2303536a0e441dbfb2050`
- Archive timestamp:
  `2026-04-20T14:29:01.630212Z`
- Paper policy:
  every reported number in the manuscript comes from this archived JSON bundle
  or from figures derived from the same archived bundle.

## 2. Archived submission snapshot

- Loaded estimators: `16`
- Loaded estimator labels:
  `EKF`, `ESPRIT`, `IPDFT`, `Koopman (RK-DPMU)`, `LKF2`, `PLL`, `Prony`,
  `RA-EKF`, `RLS`, `SOGI-FLL`, `SOGI-PLL`, `TFT`, `TKEO`,
  `Type-3 SOGI-PLL`, `UKF`, `ZCD`
- Configured scenarios: `32`
- Scenario-level aggregates in export: `34`
- Retained extra aggregate names:
  `IEEE_Freq_Ramp`, `IEEE_Mag_Step`
- Tuning trials per estimator-scenario pair: `100`
- Monte Carlo runs per estimator-scenario pair: `60`
- Raw run records: `32,640`
- Aggregated estimator-scenario rows: `544`
- Metrics used in the paper:
  `m1_rmse_hz`, `m2_mae_hz`, `m3_max_peak_hz`, `m5_trip_risk_s`,
  `m8_settling_time_s`, `m13_cpu_time_us`

## 3. Active modular pipeline state on 2026-04-23

Source of truth in code:

- `src/pipelines/full_mc_benchmark.py`
- `src/pipelines/benchmark_definition.py`
- `src/pipelines/paths.py`

Current active configuration from code:

- Loaded estimators: `18`
- Active estimator labels:
  `ZCD`, `IPDFT`, `TFT`, `RLS`, `PLL`, `SOGI-PLL`, `SOGI-FLL`,
  `Type-3 SOGI-PLL`, `LKF`, `LKF2`, `EKF`, `UKF`, `RA-EKF`, `TKEO`,
  `Prony`, `ESPRIT`, `Koopman (RK-DPMU)`, `PI-GRU`
- Active scenarios: `32`
- Tuning trials per estimator-scenario pair: `500`
- Monte Carlo runs per estimator-scenario pair: `100`
- Canonical output root:
  `artifacts/full_mc_benchmark/`

At the manuscript freeze, that canonical output root did not contain a complete
`benchmark_full_report.json` export. The paper therefore reports the archived
snapshot above rather than mixing partial active artifacts with legacy outputs.

## 4. Scenario list in the archived submission snapshot

Configured scenarios (`32`):

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

Aggregate scenario names stored in the archived report (`34`):

`IBR_Harmonics_Large`, `IBR_Harmonics_Medium`, `IBR_Harmonics_Small`,
`IBR_Multi_Event`, `IBR_Power_Imbalance_Ringdown`,
`IBR_Power_Imbalance_Ringdown_Low_Noise`,
`IBR_Power_Imbalance_Ringdown_Medium_Noise`,
`IBR_Power_Imbalance_Ringdown_Normal_Noise`,
`IBR_Power_Imbalance_Ringdown_Severe_Noise`, `IEEE_Freq_Ramp`,
`IEEE_Freq_Ramp_0.25Hzs`, `IEEE_Freq_Ramp_0.5Hzs`,
`IEEE_Freq_Ramp_10Hzs`, `IEEE_Freq_Ramp_15Hzs`, `IEEE_Freq_Ramp_1Hzs`,
`IEEE_Freq_Ramp_20Hzs`, `IEEE_Freq_Ramp_2Hzs`, `IEEE_Freq_Ramp_5Hzs`,
`IEEE_Freq_Step`, `IEEE_Mag_Step`, `IEEE_Mag_Step_10pct`,
`IEEE_Mag_Step_15pct`, `IEEE_Mag_Step_1pct`, `IEEE_Mag_Step_25pct`,
`IEEE_Mag_Step_50pct`, `IEEE_Mag_Step_5pct`, `IEEE_Modulation`,
`IEEE_Modulation_AM`, `IEEE_Modulation_FM`, `IEEE_OOB_Interference`,
`IEEE_Phase_Jump_20`, `IEEE_Phase_Jump_60`, `IEEE_Single_SinWave`,
`NERC_Phase_Jump_60`

## 5. Paper-facing figures

The manuscript uses the archived paper-facing figures:

- `paper/Figures/Plots_And_Graphs/Fig1_Scenarios_Final.pdf`
- `paper/Figures/Plots_And_Graphs/Fig2_Mega_Dashboard.pdf`

These figures are treated as part of the archived submission bundle, not as
fresh outputs from the active modular pipeline.
