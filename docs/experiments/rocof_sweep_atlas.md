# RoCoF Sweep Atlas

Status: draft experiment scaffold.

Canonical entry point:

```powershell
python src\pipelines\rocof_sweep_fixed_policy.py
```

Default artifact directory:

```text
artifacts/freq_ramp_rocof_v1_fixed_policy_fast14/
```

## Scientific Scope

This experiment sweeps the true linear frequency ramp rate and measures how each
frequency estimator degrades under increasing dynamic tracking stress.

The core question is:

> At what ramp rate does an estimator stop behaving like a frequency tracker and
> start behaving like a lagged, saturated, or sign-biased dynamic filter?

The sweep includes positive and negative ramps. Plots use `|RoCoF|` on a
logarithmic x-axis and separate ramp sign by line style.

## Default Protocol

- RoCoF levels: `0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 7.5, 10, 15, 20, 30, 40, 50 Hz/s`.
- Directions: positive and negative ramps.
- Ramp duration: `0.4 s`.
- Frequency cap: `60 + rocof_hz_s * ramp_duration_s`.
- Tuning policy: fixed policy, one parameter set per estimator reused across all RoCoF levels and signs.
- Default estimators: fast 14-estimator set used by the magnitude-step atlas.

## Main Outputs

- `metrics_dashboard_multipage.pdf`
- `rmse_deterioration_by_family.pdf`
- `rocof_method_map.pdf`
- `rocof_sign_asymmetry.pdf`
- `rocof_hypothesis_tests.csv`
- `tuning_parameter_continuity.csv`

## Quick Smoke Run

```powershell
$env:FREQRAMP_OUTPUT_SUBDIR='freq_ramp_rocof_v1_smoke'
$env:FREQRAMP_SWEEP_INCLUDE_ESTIMATORS='ZCD,IPDFT,RLS'
$env:FREQRAMP_LEVELS_HZ_S='0.5,3,10'
$env:FREQRAMP_SWEEP_DIRECTIONS='pos,neg'
$env:FREQRAMP_SWEEP_N_MC_RUNS='2'
$env:FREQRAMP_SWEEP_TUNE_TRIALS='2'
$env:FREQRAMP_FIXED_POLICY_EVAL_RUNS_PER_LEVEL='1'
$env:FREQRAMP_SWEEP_N_COST_REPS='1'
python src\pipelines\rocof_sweep_fixed_policy.py
```

## Interpretation

For ROCOF, non-monotonicity can be real if a method has a resonance, lock-range
transition, or sign-dependent dynamic model. It can also be an artifact from
tuning switches, too few MC runs, or insufficient ramp-window metrics. This
pipeline therefore emits fixed-policy continuity checks and a positive/negative
asymmetry page by default.
