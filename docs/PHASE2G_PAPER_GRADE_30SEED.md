# Phase 2-G paper-grade 30-seed evidence

Date: 2026-06-10

Phase 2-G promotes the Phase 2-E/2-F contract from diagnostic preview to a
30-seed paper-grade evidence bundle. This run is the first clean candidate for
manuscript tables after the SGSMA presentation. It must not be mixed with older
SGSMA artifacts or preview-only outputs.

## Config

```text
configs/phase2-paper-grade.yaml
```

Run contract:

- `run.id`: `phase2-paper-grade`
- `n_runs`: 30
- `base_seed`: 20260609
- `capture_signals`: false
- scenarios: 14
- estimators: 17
- scenario-estimator pairs: 238
- expected raw records: 7140
- metric profile: `canonical-single-phase-v1`
- invalid-output policy metric: `m36_post_startup_invalid_rate`

The run used:

```powershell
$env:BENCHMARK_N_COST_REPS='1'
python -m openfreqbench run --config configs\phase2-paper-grade.yaml
```

The CPU timing values in this bundle are useful for relative diagnostic context,
but not yet journal-grade CPU claims. Publication-quality CPU comparisons should
use a dedicated timing run with `n_cost_reps >= 3`.

## Artifacts

```text
artifacts/openfreqbench/phase2-paper-grade/
artifacts/openfreqbench/logs/phase2-paper-grade.log
```

Primary outputs:

- `benchmark_report.json`
- `raw_run_records.csv`
- `aggregated_metrics.csv`
- `artifact_index.csv`
- `paper_traceability.csv`
- `evidence_manifest.json`
- `environment_report.json`

The run emitted a pandas `FutureWarning` during final concatenation. It did not
abort the run and does not change the Phase 2-G evidence status, but it should
be cleaned before a public software release.

## Base audit

| item | value |
| --- | ---: |
| schema validation | pass |
| raw records | 7140 |
| aggregated rows | 238 |
| scenario-estimator pairs | 238 |
| scenarios | 14 |
| estimators | 17 |
| min runs per pair | 30 |
| max runs per pair | 30 |
| signal CSV files | 0 |
| `m36_post_startup_invalid_rate` present | yes |

## Phase 2-F report

Command:

```powershell
python -m openfreqbench report build `
  --input-json artifacts\openfreqbench\phase2-paper-grade\benchmark_report.json `
  --output-dir artifacts\openfreqbench\phase2-paper-grade\report-phase2f
```

Report result:

| item | count |
| --- | ---: |
| scope rows | 238 |
| main comparison pairs | 234 |
| diagnostic appendix pairs | 4 |
| plots | 8 |

Diagnostic appendix pairs:

| scenario | estimator | reason | n valid RMSE | mean post-startup invalid |
| --- | --- | --- | ---: | ---: |
| IBR_Multi_Event | Prony | post_startup_invalid;nonfinite_accuracy | 0 | 0.1857507333333333 |
| IEEE_Phase_Jump_20 | Prony | post_startup_invalid;nonfinite_accuracy | 28 | 0.0015006000000000002 |
| IEEE_Phase_Jump_60 | Prony | post_startup_invalid;nonfinite_accuracy | 6 | 0.01698406666666667 |
| NERC_Phase_Jump_60 | Prony | post_startup_invalid;nonfinite_accuracy | 5 | 0.016938599999999995 |

The 30-seed run adds `IEEE_Phase_Jump_20 / Prony` to the diagnostic appendix
relative to the 3-seed preview. This is expected: the larger seed count exposed
rare post-startup invalid outputs that were not stable enough to appear in the
preview.

## Manuscript-use policy

Allowed from this bundle:

- main comparison tables filtered through `paper_scope_classification.csv`;
- failure-mode discussion using `m36_post_startup_invalid_rate`;
- 30-seed accuracy and robustness comparisons for the declared 14-scenario,
  17-estimator matrix;
- diagnostic appendix discussion of Prony's post-startup invalid outputs.

Not allowed yet:

- journal-grade CPU ranking claims;
- n=100 journal-grade uncertainty claims;
- PI-GRU claims, because PI-GRU is not included in this Phase 2-G matrix;
- ATLAS severity-sweep claims, because this is an OpenFreqBench matrix run, not
  an ATLAS sweep.

## Verification

Schema validation:

```powershell
python -m openfreqbench schema --name benchmark-report `
  --validate artifacts\openfreqbench\phase2-paper-grade\benchmark_report.json
```

Result:

```text
status: pass
```

Focused report tests:

```powershell
python -m pytest tests\test_report_journal_tables.py -q -p no:cacheprovider `
  --basetemp artifacts\pytest-tmp-phase2g-report-elevated
```

Result:

```text
2 passed
```

Full quality gate:

```powershell
python -m openfreqbench quality-gate
```

Result:

```text
status: pass
pytest: 359 passed, 9 skipped
```

## Next gate

The manuscript claim ledger is documented in
`docs/PHASE2H_MANUSCRIPT_CLAIM_LEDGER.md`. Draft Results from that ledger only:
each estimator-performance claim should cite one claim ID before it enters the
manuscript.
