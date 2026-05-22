# Magnitude Step Atlas

Status: closed as the fixed-policy fast-estimator atlas.

Final artifact directory:

```text
artifacts/voltage_mag_step_v15_classic_rls_fixed_policy_fast14/
```

Primary PDF:

```text
artifacts/voltage_mag_step_v15_classic_rls_fixed_policy_fast14/metrics_dashboard_multipage.pdf
```

## Scientific Scope

This experiment is an amplitude-only stress atlas. The true frequency remains
nominal, so frequency and ROCOF errors measure magnitude-step cross-sensitivity,
numerical robustness, and estimator tracking stability. They are not formal
IEEE/IEC magnitude-step compliance numbers.

Noise is fixed in absolute voltage units. This is intentional for the final
atlas. Under that protocol, ZCD can improve with step size because the zero
crossing slope increases with amplitude while the noise amplitude stays fixed.

## Final Protocol

- Step levels: `1, 2, 3, 5, 7.5, 10, 15, 20, 25, 35, 50, 75, 100, 150, 200, 250, 300, 350, 400, 500, 750, 1000%`.
- Fast estimators: `ZCD`, `IPDFT`, `TFT`, `RLS`, `PLL`, `SOGI-PLL`, `SOGI-FLL`, `Type-3 SOGI-PLL`, `LKF`, `LKF2`, `EKF`, `UKF`, `RA-EKF`, `TKEO`.
- Monte Carlo runs per estimator-step pair: `30`.
- CPU timing reps per pair: `3`.
- Fixed-policy tuning: one parameter set per estimator reused across all step sizes.
- Fixed-policy training steps: `1, 5, 25, 100, 300, 1000%`.
- RLS policy: classic fixed-lambda RLS. No VFF, no robust clipping, no transient gate.

## Canonical Commands

From the repository root:

```powershell
python src\pipelines\magnitude_step_sweep_fixed_policy.py
```

From `src/`:

```powershell
python -m pipelines.magnitude_step_sweep_fixed_policy
```

To rebuild reports without rerunning simulations:

```powershell
python src\pipelines\rebuild_amplitude_step_outputs.py `
  --output-subdir voltage_mag_step_v15_classic_rls_fixed_policy_fast14 `
  --tuning-policy fixed_policy
```

## Interpretation Summary

The final hypothesis screening classifies the estimator curves as:

- `Power-law-like`: `IPDFT`, `TFT`, `SOGI-FLL`, `LKF2`, `RA-EKF`.
- `Monotone growth`: `EKF`, `UKF`, `Type-3 SOGI-PLL`.
- `Saturation/plateau`: `RLS`, `LKF`, `SOGI-PLL`.
- `Insensitive/flat`: `PLL`, `TKEO`.
- `SNR-improving`: `ZCD`.

No final fixed-policy curve is classified as `Erratic`. The tuning audit reports
`unique_parameter_sets = 1`, `parameter_switches = 0`, and `fallback_count = 0`
for every included estimator.

## Lessons For The ROCOF Sweep

Copy the experiment structure, not the amplitude-specific physics:

- Keep fixed-policy and per-step-oracle modes separate.
- Keep a rebuild script that reconstructs aggregate plots from per-estimator summaries.
- Keep a tuning-continuity audit in every PDF.
- Keep automatic hypothesis labels, but include a monotonicity override so smooth curves are not mislabeled as erratic.
- Add one scenario-specific diagnostic page. For this atlas it is phase dispersion; for ROCOF it should likely be ramp-rate sensitivity and post-ramp/late-window error.
- Do not upload per-run simulation folders to GitHub unless explicitly needed. Upload the report PDFs, aggregate CSVs, manifests, and code.
