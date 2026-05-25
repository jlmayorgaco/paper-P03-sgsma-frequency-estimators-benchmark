# ATLAS pipeline

ATLAS is the canonical sweep runner for OpenFreqBench 2.0.0 stress atlases.
Magnitude Step, RoCoF, and Frequency Step use the same code path.

## Command

```bash
python -m pipelines.atlas_sweep \
  --sweeps magnitude_step,rocof,frequency_step \
  --policy fixed_policy \
  --n-runs 100 \
  --tune-trials 80 \
  --output-subdir atlas-paper-v1
```

Policies:

- `default`: estimator default parameters, no tuning.
- `fixed_policy`: one tuned parameter set per estimator and sweep, reused across
  all severities and both signs.
- `oracle`: per-scenario tuning. Use as a lower-bound diagnostic, not as a
  deployable estimator policy.

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
