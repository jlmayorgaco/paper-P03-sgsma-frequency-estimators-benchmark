# Methods

This document defines the scientific method for the initial single-phase
OpenFreqBench profile.

## Scope

The active profile is `canonical-single-phase-v1`. It evaluates single-phase
voltage frequency estimators on synthetic and IBR-oriented dynamic scenarios.
Three-phase and WAMS cases are roadmap items and must use a new versioned
profile.

## Signal Contract

Scenarios generate high-resolution physical waveforms and return benchmark data
through `Scenario.run()`. The benchmark-facing signal is decimated to 10 kHz:

```text
dt = 1e-4 s
fs = 10,000 Hz
```

Every estimator receives the same sample sequence `v[k]` and optional time stamp
`t[k]`. The ground-truth trajectory is `f_true[k]`.

## Monte Carlo Contract

Each Monte Carlo run uses:

```text
seed_i = base_seed + i
```

Scenario parameters are sampled only from each scenario's declared Monte Carlo
space. The same scenario seed is shared across estimators so paired comparisons
use matched conditions.

## Estimator Contract

An estimator must expose one of:

```python
step(z, t_s=None, memory=None) -> float
step_vectorized(v) -> np.ndarray
```

The standardized runner favors the scalar `step(...)` wrapper when available,
because it can measure runtime jitter, invalid outputs, and memory-store usage
under a common interface.

## Metric Profile

Metric formulas are platform-owned and implemented in `analysis.metrics`.
Researchers can select metric ids in YAML, but cannot define formulas in YAML.

### Block 1: Classical Accuracy (IEC/IEEE Baseline)

- `m1_rmse_hz`: sqrt(mean((f_hat - f_true)^2)) — Root Mean Square Error [Hz]
- `m2_mae_hz`: mean(abs(f_hat - f_true)) — Mean Absolute Error [Hz]
- `m3_max_peak_hz`: max(abs(f_hat - f_true)) — Maximum peak error [Hz]
- `m4_std_error_hz`: std(abs(f_hat - f_true)) — Standard deviation of absolute error [Hz]
- `m34_p95_error_hz`: 95th percentile of absolute error [Hz]
- `m35_p99_error_hz`: 99th percentile of absolute error [Hz]

### Block 2: Protection Risk

- `m5_trip_risk_s`: cumulative time where abs(error) > 0.5 Hz [s]
- `m6_max_contig_trip_s`: longest continuous excursion above 0.5 Hz [s]
- `m7_pcb_hz`: mean(abs(error)) + 3 * std(abs(error)) — Probabilistic Compliance Bound [Hz]
- `m8_settling_time_s`: time from window start until error stays within 0.2 Hz [s]

### Block 3: RoCoF and IBR-Centric Metrics

- `m9_rfe_max_hz_s`: 99.5th percentile of RoCoF error magnitude [Hz/s]
- `m10_rfe_rms_hz_s`: RMS RoCoF error [Hz/s]
- `m11_rnaf_db`: RoCoF Noise Amplification Factor [dB]. 10*log10(var(RFE)/var(noise))
- `m12_isi_pu`: Interharmonic Susceptibility Index [pu]. Spectral leakage amplitude at target interharmonic frequency

### Block 4: Hardware Viability and Latency

- `m13_cpu_time_us`: (exec_time_s / n_samples) * 1e6 — CPU time per sample [us]
- `m14_struct_latency_ms`: (structural_samples / fs_dsp) * 1000 — Declared algorithmic latency [ms]
- `m15_pcb_compliant`: True if PCB <= 0.05 Hz — IEEE 60255 compliance check

### Block 5: Paper-Specific (Figures and Tables)

- `m16_heatmap_pass`: True if RMSE < 0.05 AND max_peak < 0.5 AND trip_risk < 0.1
- `m17_hw_class`: Deployment classification (P1: < 20 us, P2: <= 40 us, M1: > 40 us)

### Validity and Diagnostic Metrics

- `m18_valid_run`: True if no post-startup invalid outputs and all metrics finite
- `m19_invalid_output_count`: Number of NaN/Inf outputs from estimator
- `m20_total_samples_processed`: Total samples fed to estimator
- `m21_startup_valid_samples`: Samples before first finite output
- `m22_invalid_output_rate`: invalid_output_count / total_samples_processed
- `m23_memory_usage_bytes`: Estimated memory footprint [bytes]
- `m36_post_startup_invalid_rate`: Invalid-output fraction after first finite output

### Event-Specific Metrics (m24-m30)

Computed only when scenario declares an event_time_s:

- `m24_pre_event_rmse_hz`: RMSE in 100 ms window before event [Hz]
- `m25_post_1cy_rmse_hz`: RMSE in 1 cycle (~16.7 ms) after event [Hz]
- `m26_post_3cy_rmse_hz`: RMSE in 3 cycles (~50 ms) after event [Hz]
- `m27_post_100ms_rmse_hz`: RMSE in 100 ms window after event [Hz]
- `m28_post_event_peak_hz`: Max peak error in 100 ms after event [Hz]
- `m29_late_event_rmse_hz`: RMSE in [150ms, 500ms] window after event [Hz]
- `m30_event_settling_time_s`: Time until error stays within threshold after event [s]

The baseline warm-up/evaluation trim is handled in `analysis.metrics` and is
part of the profile. Changing it requires a new metric profile.

Windowed estimators may return `NaN` before their declared structural latency.
Those startup invalids are counted in `m22_invalid_output_rate`, but paper
failure interpretation must use `m36_post_startup_invalid_rate` together with
`m21_startup_valid_samples`. See `docs/PHASE2D_WINDOWED_ESTIMATOR_POLICY.md`.

## Parameter Policies

OpenFreqBench supports three parameter policies:

- `default`: use estimator `default_params()` plus explicit YAML overrides.
- `explicit`: use only YAML parameters.
- `artifact_tuned`: load per-scenario/per-estimator `run_spec.json` from a
  tuned artifact directory, then apply explicit YAML overrides.

`artifact_tuned` is the correct mode for reproducing final tuned benchmark
claims from canonical artifact folders.

## Statistical Hypotheses

Hypotheses are YAML records with metric, group selectors, test, alpha, correction,
and preregistration mode. Exploratory hypotheses are blocked unless explicitly
enabled.

The initial supported test is Mann-Whitney U with Holm or Benjamini-Hochberg
correction. Future tests must be added in code, documented here, and tested.

## Interpretation Rules

- Claims should reference `benchmark_report.json`, not screenshots.
- Claims should report sample size, parameter policy, seed, metric profile, and
  artifact hashes.
- Differences below timing or sampling resolution should not be overinterpreted.
- PI-GRU claims must include checkpoint checksum and torch version.
