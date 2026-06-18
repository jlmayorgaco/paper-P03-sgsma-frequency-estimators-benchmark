# Phase 2-F report scope classification

Date: 2026-06-10

Phase 2-F implements the Phase 2-E main-vs-appendix decision in report code.
The benchmark artifacts remain complete; filtering is applied only to report
tables, rankings, and plots intended for the main comparison.

## Implementation

New report tables:

- `paper_scope_classification.csv`: one row per scenario-estimator pair, with
  `main_table_eligible`, `diagnostic_appendix`, and `classification_reason`.
- `diagnostic_appendix.csv`: subset of pairs excluded from main comparison.

Filtering rule:

```text
diagnostic appendix if m36_post_startup_invalid_rate > 0
diagnostic appendix if RMSE is non-finite in any run for the pair
otherwise main comparison
```

`failure_analysis.csv` now uses `m36_post_startup_invalid_rate` as the invalid
output failure metric when available. It still reports total invalid output
separately as `total_invalid_output_rate_mean`, so startup invalids remain
visible without contaminating the failure flag.

The following main-report outputs are generated from main-eligible rows:

- metric confidence intervals
- pareto recommendations
- IBR robustness
- PI-GRU generalization
- classical competitiveness
- ranking sensitivity
- RMSE/CPU plots, Pareto plot, heatmap, and family boxplot
- RMSE winners in `analysis_summary.json` and `analysis_summary.md`

The original copied `raw_run_records.csv` and `aggregated_metrics.csv` in the
report folder are kept complete for traceability.

## Preview v2 validation

Command:

```powershell
python -m openfreqbench report build `
  --input-json artifacts\openfreqbench\phase2-paper-grade-preview-v2\benchmark_report.json `
  --output-dir artifacts\openfreqbench\phase2-paper-grade-preview-v2\report-phase2f
```

Result:

| item | count |
| --- | ---: |
| scope rows | 238 |
| main comparison pairs | 235 |
| diagnostic appendix pairs | 3 |

Diagnostic appendix pairs:

| scenario | estimator | reason |
| --- | --- | --- |
| IBR_Multi_Event | Prony | post_startup_invalid;nonfinite_accuracy |
| IEEE_Phase_Jump_60 | Prony | post_startup_invalid;nonfinite_accuracy |
| NERC_Phase_Jump_60 | Prony | post_startup_invalid;nonfinite_accuracy |

ESPRIT and Koopman have zero post-startup invalid-output failures in the v2
preview and remain in the main comparison for all 14 scenarios.

## Verification

Focused report tests:

```powershell
python -m pytest tests\test_report_journal_tables.py -q -p no:cacheprovider --basetemp artifacts\pytest-tmp-phase2f-report
```

Result:

```text
2 passed
```

Full gate:

```powershell
python -m openfreqbench quality-gate
```

Result:

```text
status: pass
pytest: 358 passed, 9 skipped
```

## Phase 2-G follow-through

The 30-seed follow-through is documented in
`docs/PHASE2G_PAPER_GRADE_30SEED.md`. The final Phase 2-G report keeps 234
scenario-estimator pairs in the main comparison and places 4 Prony pairs in the
diagnostic appendix. Use that 30-seed report, not the preview v2 report, for any
manuscript table.
