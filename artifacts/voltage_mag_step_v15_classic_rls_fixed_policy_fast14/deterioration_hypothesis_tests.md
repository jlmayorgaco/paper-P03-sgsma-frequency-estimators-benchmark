# Deterioration Hypothesis Tests

- Artifact: `C:\Users\walla\Documents\Github\paper-P03-sgsma-frequency-estimators-benchmark\artifacts\voltage_mag_step_v15_classic_rls_fixed_policy_fast14`
- Method version: `amplitude_step_v14_fixed_policy_oracle_audit_2026_05_17`
- Metric: `m1_rmse_hz`
- Max amplitude step included: `1000%`

These are automatic screening tests. They classify the observed deterioration regime; they do not force monotonicity or smooth the curves.

## Regime Counts

- `Power-law-like`: 5
- `Monotone growth`: 3
- `Saturation/plateau`: 3
- `Insensitive/flat`: 2
- `SNR-improving`: 1

## Per-Estimator Classification

- `PLL`: `Insensitive/flat`; b=0.0343, R2_loglog=0.643, bound_hit_max=0, status=`claim_ready_fixed_policy`, switches=0.
- `TKEO`: `Insensitive/flat`; b=0.0968, R2_loglog=0.825, bound_hit_max=0, status=`claim_ready_fixed_policy`, switches=0.
- `EKF`: `Monotone growth`; b=0.781, R2_loglog=0.773, bound_hit_max=0, status=`claim_ready_fixed_policy`, switches=0.
- `Type-3 SOGI-PLL`: `Monotone growth`; b=0.245, R2_loglog=0.755, bound_hit_max=0, status=`claim_ready_fixed_policy`, switches=0.
- `UKF`: `Monotone growth`; b=1.15, R2_loglog=0.844, bound_hit_max=0, status=`claim_ready_fixed_policy`, switches=0.
- `IPDFT`: `Power-law-like`; b=1.67, R2_loglog=0.989, bound_hit_max=0, status=`claim_ready_fixed_policy`, switches=0.
- `LKF2`: `Power-law-like`; b=1.11, R2_loglog=0.868, bound_hit_max=0, status=`claim_ready_fixed_policy`, switches=0.
- `RA-EKF`: `Power-law-like`; b=0.689, R2_loglog=0.854, bound_hit_max=0, status=`claim_ready_fixed_policy`, switches=0.
- `SOGI-FLL`: `Power-law-like`; b=0.72, R2_loglog=0.904, bound_hit_max=0, status=`claim_ready_fixed_policy`, switches=0.
- `TFT`: `Power-law-like`; b=1.72, R2_loglog=0.955, bound_hit_max=0, status=`claim_ready_fixed_policy`, switches=0.
- `ZCD`: `SNR-improving`; b=-0.322, R2_loglog=0.93, bound_hit_max=0, status=`claim_ready_fixed_policy`, switches=0.
- `LKF`: `Saturation/plateau`; b=0.294, R2_loglog=0.84, bound_hit_max=0, status=`claim_ready_fixed_policy`, switches=0.
- `RLS`: `Saturation/plateau`; b=0.809, R2_loglog=0.956, bound_hit_max=0, status=`claim_ready_fixed_policy`, switches=0.
- `SOGI-PLL`: `Saturation/plateau`; b=0.331, R2_loglog=0.86, bound_hit_max=0, status=`claim_ready_fixed_policy`, switches=0.
