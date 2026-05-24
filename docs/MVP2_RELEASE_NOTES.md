# MVP 2.0.0 Release Notes

OpenFreqBench MVP 2.0.0 is the first public-ready single-phase package cut.

## Scope

- Single-phase frequency-estimator benchmarking.
- 32 canonical scenarios.
- Canonical estimator registry, including LKF, LKF2, and PI-GRU.
- Locked metric profile: `canonical-single-phase-v1`.
- YAML-driven quick tests, estimator comparisons, Monte Carlo runs, tuned-artifact
  replay, reports, plots, and hypothesis testing.
- Paper-scale replay config: `configs/journal-paper-replay.yaml`.

## Public Contract

- Researchers may modify estimators, scenarios, run matrices, and hypotheses.
- Researchers may select canonical metrics.
- Researchers may not redefine metric formulas in YAML.
- Every run writes a reproducibility manifest with dependency, source, git, and
  checkpoint hashes.

## Packaging Decision

The wheel includes only the active PI-GRU default checkpoint:

- `pi_gru_weights_hybrid.pt`

Historical and experimental checkpoints remain in the source tree but are not
part of the wheel.

## Remaining Scientific Work

MVP 2.0.0 is package-ready. Award, paper, or leaderboard claims still require a
Level 3 or Level 4 artifact set as defined in `docs/SCIENTIFIC_READINESS.md`.
The run plan for that step is in `docs/JOURNAL_RESULTS_PROTOCOL.md`.
