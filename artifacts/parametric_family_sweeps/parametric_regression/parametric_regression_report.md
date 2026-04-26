# Parametric Sweep Regression Report

## Interpretation
- `pct_change_per_report_step` comes from a log-linear fit of the metric versus the sweep variable.
- `absolute_change_per_report_step` is the additive slope over the same reporting step.
- `end_to_baseline_ratio` compares the last sweep point against the first.
- Use `R2` and `spearman_rho` to judge whether the fitted statement is trustworthy.

## Phase Jump Magnitude

### RMSE [Hz]

- EKF: RMSE [Hz] changes 7.709e-19 per 10 deg in Phase Jump Magnitude (trend=flat, R2=1.000, rho=0.000, baseline=0.1203, x1.00 end/base).

### Peak Error [Hz]

- EKF: Peak Error [Hz] changes -1.542e-18 per 10 deg in Phase Jump Magnitude (trend=flat, R2=1.000, rho=0.000, baseline=0.1752, x1.00 end/base).

### Trip-Risk [s]

- EKF: Trip-Risk [s] changes 0 per 10 deg in Phase Jump Magnitude (trend=flat, R2=1.000, rho=0.000, baseline=0, xnan end/base).

### Settling Time [s]

- EKF: Settling Time [s] changes 0 per 10 deg in Phase Jump Magnitude (trend=flat, R2=1.000, rho=0.000, baseline=0, xnan end/base).
