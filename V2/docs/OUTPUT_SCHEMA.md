# Output Schema

OpenFreqBench treats run artifacts as a public API. The first stable API is for
single-phase benchmark replay under `canonical-single-phase-v1`.

## Core files

- `benchmark_report.json`: machine-readable source for report generation.
- `raw_run_records.csv`: one Monte Carlo run per row.
- `aggregated_metrics.csv`: grouped estimator/scenario summaries.
- `metric_confidence_intervals.csv`: bootstrap CIs by scenario family, scenario,
  estimator, family, and metric.
- `failure_analysis.csv`: collapse and failure rates by scenario and estimator.
- `ranking_sensitivity.csv`: estimator rank under each primary metric.
- `pareto_recommendations.csv`: recommended estimators by operating profile.
- `ibr_robustness.csv`: degradation under IBR events relative to nominal and
  frequency-step baselines.
- `pi_gru_generalization.csv`: PI-GRU rank and delta to best estimator.
- `classical_competitiveness.csv`: regimes where classical estimators remain
  competitive.
- `paper_traceability.csv`: claim-to-artifact table for manuscript numbers.
- `artifact_index.csv`: file hashes and sizes.
- `evidence_manifest.json`: run-level evidence index.

## Schema commands

```bash
openfreqbench schema --name benchmark-report
openfreqbench schema --name manifest
openfreqbench schema --name run-config
```

Lightweight validation:

```bash
openfreqbench schema --name benchmark-report \
  --validate artifacts/openfreqbench/<run_id>/benchmark_report.json
```

The repository also keeps static schema copies under `schemas/`.
