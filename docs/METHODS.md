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

Core metrics:

- `m1_rmse_hz`: sqrt(mean((f_hat - f_true)^2))
- `m2_mae_hz`: mean(abs(f_hat - f_true))
- `m3_max_peak_hz`: max(abs(f_hat - f_true))
- `m5_trip_risk_s`: cumulative time where abs(error) exceeds relay deadband
- `m7_pcb_hz`: mean(abs(error)) + 3 std(abs(error))
- `m9_rfe_max_hz_s`: robust high-percentile RoCoF error magnitude
- `m10_rfe_rms_hz_s`: RMS RoCoF error
- `m13_cpu_time_us`: repeated process-time CPU cost per sample
- `m14_struct_latency_ms`: declared structural latency in milliseconds
- `m18` to `m23`: standardized runtime and memory proxy metrics

The baseline warm-up/evaluation trim is handled in `analysis.metrics` and is
part of the profile. Changing it requires a new metric profile.

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

