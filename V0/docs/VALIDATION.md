# Validation

OpenFreqBench V2 uses layered validation.

## Current Checks

- CLI parse tests.
- YAML guardrail tests that reject metric formula definitions.
- Scenario and estimator registry dry-run validation.
- Smoke execution of single-scenario/single-estimator runs.
- Report generation from a V2 `benchmark_report.json`.
- JSON output is written with strict `allow_nan=False`.

## Quality Gate

Run:

```bash
openfreqbench quality-gate
openfreqbench quality-gate --release
```

The gate checks:

- required package files and documentation exist,
- YAML configs parse and dry-run,
- scenario and metric counts match the expected profile scale,
- pytest passes,
- reproducibility metadata can be generated.

The release gate additionally requires a clean git working tree and release
metadata suitable for public archival.

## Required Before a Public Release

- Run a full tuned replay with `parameter_policy: artifact_tuned`.
- Archive the resulting `benchmark_report.json`.
- Run generated hypotheses on the archived report.
- Store report plots and the reproducibility manifest.
- Verify wheel installation in a fresh environment.

## Residual Risks

- CPU timing is machine-dependent; compare CPU metrics only within the same
  hardware/runtime class unless normalized.
- Custom estimators execute local Python code; do not run untrusted submissions.
- `artifact_tuned` depends on the integrity of the source tuned artifact folder.
