# Parametric Estimator Atlas Template

Use this checklist when cloning the magnitude-step workflow for a new scenario
family such as ROCOF/frequency-ramp, harmonics, interharmonics, OOB, or
multi-event stress.

## Required Pieces

1. Scenario builder
   - Defines the swept parameter grid.
   - Keeps nuisance variables stratified by `run_idx`.
   - Writes the physical meaning of each sweep range into the manifest.

2. Estimator registry
   - Uses the active estimator names from `src/pipelines/benchmark_definition.py`.
   - Separates fast and slow methods.
   - Does not silently drop `LKF`, `LKF2`, or `PI-GRU` from the canonical registry.

3. Tuning policy
   - `fixed_policy`: one parameter set per estimator reused across the sweep.
   - `per_step_oracle`: lower-bound envelope only, not a physical deployment curve.
   - Always emit `tuning_parameter_continuity.csv`.

4. Metrics
   - Include global RMSE/MAE/peak metrics.
   - Include event-window metrics tailored to the scenario.
   - Preserve full numeric precision in CSVs; round only in plots/tables.

5. Plots
   - Family RMSE deterioration plot with p10-p90 Monte Carlo bands.
   - Multipage dashboard with all key metrics.
   - Method stress map.
   - Scenario-specific diagnostic page.
   - Hypothesis classification report.

6. Artifact policy
   - Keep generated simulation folders out of Git unless explicitly needed.
   - Commit code, documentation, aggregate CSVs/JSON/MD, and final PDFs/PNGs.
   - Keep temporary rerun folders ignored.

## ROCOF Sweep Notes

For a ROCOF/frequency-ramp atlas, the copied workflow should replace amplitude
regions with ROCOF regions, and phase dispersion with ramp diagnostics:

- Ramp-rate grid in Hz/s.
- Sign of ramp: positive, negative, or both.
- Event windows: early ramp, full ramp, post-ramp settling, and late error.
- Hypothesis classes: flat, power-law-like, saturation/rail, monotone growth,
  sign-asymmetric, and erratic.
- The physical explanation should separate frequency-tracking lag from ROCOF
  estimation error.
