# ATLAS pipeline

ATLAS is the canonical sweep runner for OpenFreqBench 2.0.0 stress atlases.
Magnitude Step, RoCoF, Frequency Step, Harmonics, Interharmonics, and
White-Noise/SNR studies use the same code path.

## Command

```bash
python -m pipelines.atlas_sweep \
  --sweeps all \
  --policy fixed_policy \
  --n-runs 100 \
  --tune-trials 80 \
  --output-subdir atlas-paper-v1
```

Useful sweep aliases:

- `core`: magnitude-step, RoCoF, and frequency-step.
- `p0`: harmonics, interharmonics, and white-noise/SNR.
- `all`: every ATLAS sweep currently implemented.

Policies:

- `default`: estimator default parameters, no tuning.
- `fixed_policy`: one tuned parameter set per estimator and sweep, reused across
  all severities and both signs.
- `oracle`: per-scenario tuning. Use as a lower-bound diagnostic, not as a
  deployable estimator policy.

## P0 methodological guardrails

The P0 sweeps are intentionally isolated:

- `harmonics` scales integer-harmonic THD with frequency held fixed. It disables
  interharmonics, subharmonics, impulses, frequency events, and added white
  noise by default.
- `interharmonics` scales a 75 Hz off-bin component with integer harmonics and
  RoCoF disabled. Added white noise is also disabled by default.
- `noise_snr` uses a constant-frequency single tone and varies only additive
  white noise.

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
- `atlas_method_map.pdf`
- `atlas_sign_asymmetry.pdf`
- `atlas_accuracy_latency_cpu_pareto.pdf`
- `benchmark_report.json`
- `manifest.json`
- `environment_report.json`
- `artifact_index.csv`
- `paper_traceability.csv`
- `evidence_manifest.json`

Runs with `n_runs < 30` are diagnostic. Use `n_runs >= 100` before making paper
claims, and archive the output directory with hashes.
