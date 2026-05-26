# ATLAS pipeline

ATLAS is the canonical sweep runner for OpenFreqBench 2.0.0 stress atlases.
Magnitude Step, RoCoF, Frequency Step, Phase Jump, AM Modulation, FM
Modulation, Harmonics, Interharmonics, and White-Noise/SNR studies use the same
code path.

## Command

```bash
python -m pipelines.atlas_sweep \
  --sweeps all \
  --policy fixed_policy \
  --n-runs 100 \
  --n-cost-reps 3 \
  --tune-trials 80 \
  --output-subdir atlas-paper-v1
```

Useful sweep aliases:

- `core`: magnitude-step, RoCoF, and frequency-step.
- `p0`: phase-jump, AM, FM, harmonics, interharmonics, and white-noise/SNR.
- `all`: every ATLAS sweep currently implemented.

Policies:

- `default`: estimator default parameters, no tuning.
- `fixed_policy`: one tuned parameter set per estimator and sweep, reused across
  all severities and both signs.
- `oracle`: per-scenario tuning. Use as a lower-bound diagnostic, not as a
  deployable estimator policy.

## Dense phase-jump sweep

For high-resolution phase-jump curves, level variables accept inclusive range
syntax:

```powershell
$env:ATLAS_PHASE_JUMP_LEVELS_DEG = "0:180:1"
$env:ATLAS_PHASE_JUMP_DIRECTIONS = "pos"
python -m pipelines.atlas_sweep `
  --sweeps phase_jump_sweep `
  --policy default `
  --n-runs 1 `
  --n-cost-reps 1 `
  --tune-trials 0 `
  --output-subdir atlas-phase-jump-0-180deg-1deg-all18
```

This produces 181 phase-jump scenarios and 3258 scenario-estimator pairs for
the 18 canonical estimators. The 0 degree baseline is included once even if both
signs are requested. Use `ATLAS_PHASE_JUMP_DIRECTIONS=pos,neg` only when signed
asymmetry is part of the question; that doubles the nonzero phase-jump workload.

## P0 methodological guardrails

The P0 sweeps are intentionally isolated:

- `harmonics` scales integer-harmonic THD with frequency held fixed. It disables
  interharmonics, subharmonics, impulses, frequency events, and added white
  noise by default.
- `interharmonics` scales a 75 Hz off-bin component with integer harmonics and
  RoCoF disabled. Added white noise is also disabled by default.
- `noise_snr` uses a constant-frequency single tone and varies only additive
  white noise.
- `phase_jump_sweep` changes voltage phase instantaneously while true frequency
  remains nominal.
- `modulation_am_sweep` varies AM modulation frequency at fixed AM depth, so
  frequency error measures AM-to-FM cross-coupling.
- `modulation_fm_sweep` varies FM modulation frequency at fixed peak frequency
  deviation, so curves expose tracking bandwidth and latency effects.

These sweeps should not be interpreted as complete IBR event models. They are
causal stress tests that reveal which disturbance variable changes the ranking.
Use later mixed-event IBR sweeps to test deployment realism.

The active methodological audit is in `docs/ATLAS_METHOD_AUDIT.md`.

## Outputs

Each ATLAS run writes:

- `global_metrics_report.csv`
- `rmse_by_estimator.csv`
- `rmse_by_family.csv`
- `timing_profile.csv`
- `hypothesis_results.csv`
- `metrics_dashboard_multipage.pdf`
- `rmse_deterioration_by_family.pdf`
- `rmse_all_estimators_small_multiples.pdf`
- `atlas_method_map.pdf`
- `atlas_sign_asymmetry.pdf`
- `atlas_accuracy_latency_cpu_pareto.pdf`
- `atlas_readiness_report.json`
- `atlas_readiness_report.md`
- `benchmark_report.json`
- `manifest.json`
- `environment_report.json`
- `artifact_index.csv`
- `paper_traceability.csv`
- `evidence_manifest.json`

## Readiness gate

Every ATLAS run writes `atlas_readiness_report.json` and
`atlas_readiness_report.md`. Treat that report as the first file to read:

- `diagnostic`: useful for debugging plots, estimator behavior, or a subset.
  Do not move numbers into the paper.
- `paper_grade`: full ATLAS, full canonical estimator set, fixed policy, at
  least 30 Monte Carlo runs, and enough levels/sign coverage for trend claims.
- `journal_grade`: same checks as `paper_grade`, but with at least 100 Monte
  Carlo runs.

The gate is intentionally strict. It blocks paper claims when sweeps are
missing, the canonical estimator set is incomplete, policies are mixed,
`oracle` is used as if it were deployable, `default` parameters are used for
publication claims, Monte Carlo support is too small, or a directional sweep is
missing one sign.

Runs with `n_runs < 30` are diagnostic. Use `n_runs >= 100` before making
journal claims, and archive the output directory with hashes.
