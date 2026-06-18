# Phase 2-H manuscript claim ledger

Date: 2026-06-10

Phase 2-H converts the Phase 2-G paper-grade report into a claim ledger. The
purpose is to prevent manuscript prose from outrunning the evidence. Every
claim below must be traceable to the 30-seed run:

```text
artifacts/openfreqbench/phase2-paper-grade/report-phase2f/
```

## Evidence boundary

Use only the Phase 2-G report for new post-SGSMA numerical claims.

Evidence scope:

- 14 scenarios.
- 17 estimators.
- 238 scenario-estimator pairs.
- 30 Monte Carlo runs per pair.
- 7140 raw run records.
- `capture_signals=false`.
- Main comparison after Phase 2-F filtering: 234 pairs.
- Diagnostic appendix after Phase 2-F filtering: 4 pairs.
- CPU timing context uses `BENCHMARK_N_COST_REPS=1`.

Claims must not use older SGSMA outputs, preview v2 tables, partial temporary
artifacts, or ATLAS readiness files as support for new manuscript results.

## Frozen table set

Main-table evidence:

| artifact | manuscript role |
| --- | --- |
| `analysis_summary.md` / `analysis_summary.json` | high-level counts, winners, and report index |
| `paper_scope_classification.csv` | mandatory inclusion/exclusion gate |
| `metric_confidence_intervals.csv` | scenario-estimator metric estimates and confidence intervals |
| `estimator_rmse_ci.csv` | global RMSE summaries across the Phase 2-G matrix |
| `ranking_sensitivity.csv` | ranking checks across canonical metrics |
| `classical_competitiveness.csv` | competitiveness of classical estimators |
| `ibr_robustness.csv` | IBR degradation analysis |
| `pareto_recommendations.csv` | balanced accuracy/cost recommendations, with CPU caveat |
| `plots/rmse_by_estimator_ci.png` | global RMSE visual |
| `plots/scenario_rmse_heatmap.png` | scenario-estimator accuracy visual |
| `plots/family_rmse_boxplot.png` | estimator-family accuracy visual |
| `plots/ibr_robustness_delta.png` | IBR robustness visual |
| `plots/ranking_sensitivity_top5.png` | ranking sensitivity visual |
| `plots/pareto_rmse_cpu.png` | Pareto visual, with CPU caveat |

Diagnostic or appendix evidence:

| artifact | manuscript role |
| --- | --- |
| `diagnostic_appendix.csv` | excluded pairs and exclusion reasons |
| `failure_analysis.csv` | invalid-output and failure-mode discussion |
| `estimator_cpu_ci.csv` | diagnostic timing context only |
| `plots/cpu_by_estimator_ci.png` | diagnostic timing context only |
| `plots/failure_rate_by_estimator.png` | diagnostic failure-mode visual |

Out of scope for Phase 2-G claims:

| artifact | reason |
| --- | --- |
| `pi_gru_generalization.csv` | empty in this report; PI-GRU is not part of the Phase 2-G matrix |
| `tuning_policy_summary.csv` | empty/default-policy run; no tuning-policy comparison is supported |
| ATLAS sweep outputs | Phase 2-G is an OpenFreqBench matrix run, not an ATLAS severity sweep |
| SGSMA presented-paper tables | historical pilot evidence only, not support for new post-SGSMA claims |

## Claim ledger

| ID | Status | Allowed wording | Evidence | Do not say |
| --- | --- | --- | --- | --- |
| C01 | main | The post-SGSMA evidence layer is a clean 30-seed OpenFreqBench matrix with 14 scenarios, 17 estimators, 238 pairs, and 7140 raw records. | `analysis_summary.md`; `benchmark_report.json`; `PHASE2G_PAPER_GRADE_30SEED.md` | The results are journal-grade across the full scenario registry. |
| C02 | main | Main manuscript tables must use the Phase 2-F scope filter: 234 pairs are main-comparison eligible and 4 pairs are diagnostic appendix pairs. | `paper_scope_classification.csv`; `diagnostic_appendix.csv` | All 238 pairs are directly comparable in the main tables. |
| C03 | main | No single estimator dominates all regimes; RMSE winners vary by scenario. | `analysis_summary.md`; `metric_confidence_intervals.csv` | One estimator is universally best. |
| C04 | main | Model-based estimators win the most scenarios in this matrix: EKF wins 6 scenarios, LKF wins 2, and UKF wins 1, for 9 model-based wins out of 14. | `analysis_summary.md`; `metric_confidence_intervals.csv` | Model-based estimators are always best or always stable. |
| C05 | main | ESPRIT is the RMSE winner in 5 scenarios, concentrated in sinusoidal/frequency-dynamic cases: single sine, frequency step, two frequency ramps, and FM modulation. | `analysis_summary.md`; `paper_scope_classification.csv` | ESPRIT is the best estimator overall or the most robust under IBR stress. |
| C06 | main | EKF is strongest in the magnitude/phase-jump and AM subset of this matrix: it wins both magnitude steps, AM modulation, both IEEE phase jumps, and the NERC phase jump. | `analysis_summary.md`; `metric_confidence_intervals.csv` | EKF solves every dynamic regime. |
| C07 | main | IBR stress separates estimator behavior: LKF wins `IBR_Harmonics_Medium`, UKF wins `IBR_Multi_Event`, and several otherwise accurate estimators suffer large IBR RMSE increases. | `analysis_summary.md`; `ibr_robustness.csv`; `plots/ibr_robustness_delta.png` | The IBR conclusions generalize to every inverter model or every grid condition. |
| C08 | diagnostic appendix | Prony has a reproducible post-startup invalid-output failure mode in 4 pairs: `IBR_Multi_Event`, `IEEE_Phase_Jump_20`, `IEEE_Phase_Jump_60`, and `NERC_Phase_Jump_60`. | `diagnostic_appendix.csv`; `failure_analysis.csv`; `paper_scope_classification.csv` | Prony should be removed entirely from the benchmark or is unusable in all scenarios. |
| C09 | diagnostic appendix | `IBR_Multi_Event / Prony` is not a main-table accuracy result: it has 0 valid RMSE runs and mean post-startup invalid rate 0.1857507333333333. | `diagnostic_appendix.csv` | Report a Prony RMSE winner or rank for IBR multi-event in the main table. |
| C10 | main | ESPRIT and Koopman remain main-comparison eligible in all 14 scenarios under the Phase 2-F post-startup invalid-output gate. | `paper_scope_classification.csv` | ESPRIT and Koopman are accuracy winners in all scenarios. |
| C11 | main | Classical/windowed estimators remain relevant in selected regimes: ESPRIT is competitive in 5 scenarios, TFT and PLL in 2 each, and Prony in 1 after filtering. | `classical_competitiveness.csv` | Classical methods are obsolete, or all classical methods remain competitive everywhere. |
| C12 | conditional | The Pareto table suggests UKF, EKF, RA-EKF, SOGI-FLL, and LKF2 as balanced candidates under the current score definitions. | `pareto_recommendations.csv`; `plots/pareto_rmse_cpu.png` | The Pareto order is a definitive journal-grade CPU ranking. |
| C13 | diagnostic only | ZCD is the fastest estimator in this timing context, but the run used `BENCHMARK_N_COST_REPS=1`, so timing claims are diagnostic. | `estimator_cpu_ci.csv`; `plots/cpu_by_estimator_ci.png`; `PHASE2G_PAPER_GRADE_30SEED.md` | ZCD is definitively fastest under publication-quality timing conditions. |
| C14 | out of scope | PI-GRU generalization cannot be claimed from Phase 2-G because PI-GRU is not included in the 17-estimator matrix and `pi_gru_generalization.csv` is empty. | `pi_gru_generalization.csv`; `configs/phase2-paper-grade.yaml` | PI-GRU generalizes better or worse than classical estimators in this run. |
| C15 | out of scope | ATLAS severity-sweep claims remain outside this evidence bundle. Phase 2-G is a fixed matrix run, not a severity sweep. | `PHASE2G_PAPER_GRADE_30SEED.md`; `POST_SGSMA_ROUTE.md` | ATLAS readiness or severity-taxonomy claims are proven by Phase 2-G. |
| C16 | out of scope | A transaction/journal final submission still needs stronger timing and/or journal-grade uncertainty evidence if CPU ranking or n=100 claims are central. | `PHASE2G_PAPER_GRADE_30SEED.md`; `estimator_cpu_ci.csv` | Phase 2-G alone is final journal-grade evidence for every claim. |
| C17 | main | The global RMSE confidence-interval summary ranks model-based estimators in the first five positions: LKF, UKF, RA-EKF, LKF2, and EKF. | `estimator_rmse_ci.csv`; `ranking_sensitivity.csv` | This aggregate order is a universal deployment rule for every scenario. |

## Results-section skeleton

Use this order when drafting Results:

1. Evidence contract and filtering policy: state the 30-seed matrix and the
   234-main / 4-diagnostic split before any winners.
2. Scenario-level accuracy: present RMSE winners and emphasize regime
   dependence rather than a universal champion.
3. Family-level behavior: compare model-based, window-based, loop-based,
   adaptive, and data-driven families.
4. IBR stress: discuss harmonics and multi-event behavior separately.
5. Diagnostic appendix: isolate Prony invalid-output behavior and avoid mixing
   those pairs into main rankings.
6. Pareto and timing context: present balanced candidates while clearly stating
   that CPU evidence is diagnostic because `n_cost_reps=1`.
7. Limitations and next experiments: CPU timing, PI-GRU, ATLAS, and n=100
   journal-grade runs.

## Route decision

The Phase 2-G/2-H state supports a serious results draft, but not a final
transaction submission if the paper needs strong CPU-ranking or n=100
uncertainty claims. The pragmatic route is:

1. Draft a results/theory manuscript from the Phase 2-G main claims.
2. Keep CPU and Prony failure behavior as bounded, explicitly qualified claims.
3. In parallel, prepare OpenFreqBench as research software, because the clean
   benchmark contract is already a defensible software contribution.
4. Decide on journal escalation after a dedicated timing run or a journal-grade
   n=100 run, depending on which claims the manuscript wants to center.

## Next gate

The first Results draft is documented in `docs/PHASE2I_RESULTS_DRAFT.md`. Before
moving it into a LaTeX manuscript, choose the target route: a results/theory
paper with a fuller Results section, or a software-first paper with a shorter
benchmark case study.
