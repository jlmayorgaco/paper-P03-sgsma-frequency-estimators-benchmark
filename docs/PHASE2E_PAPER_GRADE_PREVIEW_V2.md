# Phase 2-E paper-grade preview v2

Date: 2026-06-10

Phase 2-E reruns the Phase 2-C preview after the Phase 2-D startup/invalid-output
policy. This is still diagnostic evidence, not final paper-grade evidence, because
it uses 3 Monte Carlo runs per scenario-estimator pair.

## Configuration

Config:

```text
configs/phase2-paper-grade-preview-v2.yaml
```

Run contract:

- 14 scenarios
- 17 canonical non-neural estimators
- 3 runs per pair
- 238 scenario-estimator pairs
- 714 raw records
- `capture_signals=false`
- `BENCHMARK_N_COST_REPS=1`
- metrics include `m21_startup_valid_samples`, `m22_invalid_output_rate`, and
  `m36_post_startup_invalid_rate`

Log:

```text
artifacts/openfreqbench/logs/phase2-paper-grade-preview-v2.log
```

Bundle:

```text
artifacts/openfreqbench/phase2-paper-grade-preview-v2/
```

## Verification

The run completed without wrapper timeout and produced:

- `benchmark_report.json`
- `raw_run_records.csv`
- `aggregated_metrics.csv`
- `environment_report.json`
- `artifact_index.csv`
- `paper_traceability.csv`
- `evidence_manifest.json`

Checks:

- benchmark-report schema validation: pass
- raw records: 714
- aggregate rows: 238
- scenarios: 14
- estimators: 17
- runs per pair: min 3, max 3
- generated signal CSV files: 0
- `m36_post_startup_invalid_rate` present in raw and aggregate outputs

## Windowed estimator classification

`m36_post_startup_invalid_rate` separates structural startup invalids from real
post-startup invalid outputs.

Summary by windowed/data-driven estimator:

| estimator | pairs | pairs with `m36 > 0` | max `m36` | RMSE-NaN pairs |
| --- | ---: | ---: | ---: | ---: |
| ESPRIT | 14 | 0 | 0.000000 | 0 |
| Koopman (RK-DPMU) | 14 | 0 | 0.000000 | 0 |
| Prony | 14 | 3 | 0.075311 | 1 |

Prony pairs requiring diagnostic-appendix treatment:

| scenario | runs | valid RMSE runs | `m22` mean | `m36` mean | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| IBR_Multi_Event | 3 | 0 | 0.081580 | 0.075311 | appendix; non-finite accuracy |
| IEEE_Phase_Jump_60 | 3 | 1 | 0.037267 | 0.015006 | appendix; post-startup invalids |
| NERC_Phase_Jump_60 | 3 | 1 | 0.037267 | 0.015006 | appendix; post-startup invalids |

ESPRIT and Koopman remain eligible for the main comparison under the invalid-output
policy. Their severe-IBR errors should be interpreted as performance limitations,
not runtime-invalid failures:

- ESPRIT on `IBR_Multi_Event`: RMSE mean 2.966 Hz, peak mean 14.701 Hz, `m36=0`.
- Koopman on `IBR_Multi_Event`: RMSE mean 2.387 Hz, peak mean 12.377 Hz, `m36=0`.

## High-risk performance findings

Largest valid RMSE means in the v2 preview:

| scenario | estimator | RMSE mean | peak mean | `m36` mean |
| --- | --- | ---: | ---: | ---: |
| IBR_Multi_Event | ZCD | 226.303 | 3251.566 | 0.000000 |
| IBR_Multi_Event | RLS | 20.282 | 22.853 | 0.000000 |
| IBR_Harmonics_Medium | RLS | 18.239 | 20.556 | 0.000000 |
| NERC_Phase_Jump_60 | RLS | 15.563 | 19.411 | 0.000000 |
| IEEE_Phase_Jump_20 | RLS | 4.636 | 20.000 | 0.000000 |

These are estimator stress-performance results, not artifact-integrity failures.

## Decision

Phase 2-E closes the preview rerun gate. The next paper workflow must use the v2
interpretation:

1. Exclude Prony pairs with `m36 > 0` or non-finite accuracy from main aggregate
   rankings.
2. Keep those Prony results as diagnostic failure-mode evidence.
3. Keep ESPRIT and Koopman in the main comparison only where `m36 == 0`; annotate
   high severe-IBR error as performance weakness.
4. Move to a 30-seed paper-grade run only after the report/table code can apply
   this main-vs-appendix classification deterministically.
