# Phase 2-I Results draft

Date: 2026-06-10

This file is a first manuscript-oriented Results draft from the Phase 2-H claim
ledger. It is not yet the final paper text. The hard rule for this draft is
that every estimator-performance sentence must cite at least one Phase 2-H claim
ID.

## Mini-outline

1. Establish the evidence contract and the main-vs-diagnostic filtering rule.
2. Report scenario-level RMSE winners without implying a universal best
   estimator.
3. Interpret family-level behavior as regime-dependent specialization.
4. Separate IBR stress behavior from standard IEEE-style events.
5. Isolate Prony invalid-output failures in the diagnostic appendix.
6. Present Pareto/timing evidence as decision support, not journal-grade CPU
   ranking.
7. Close with the limits that must be resolved before stronger journal claims.

## Draft Results

### 1. Evidence Contract and Scope Filter

[Opening, C01-C02] The post-SGSMA benchmark produced a clean Phase 2-G evidence
set with 14 scenarios, 17 estimators, 238 scenario-estimator pairs, and 7140 raw
Monte Carlo records. Each pair was evaluated with 30 runs, with signal capture
disabled and with the canonical single-phase metric profile. Before ranking
estimators, the report applied the Phase 2-F paper-scope filter: 234 pairs
remain eligible for main comparison, while 4 pairs are assigned to the
diagnostic appendix. This separation is important because the benchmark is not
only measuring accuracy; it is also preventing invalid post-startup outputs from
being presented as ordinary RMSE values.

[Transition, C02] All accuracy tables below therefore use the main-comparison
subset rather than the full 238-pair artifact. The raw records and aggregate
files remain complete for traceability, but manuscript-level rankings should
come from the filtered report tables. This rule avoids a common failure in
benchmark papers: mixing valid operating regimes with cases where an estimator
does not produce a defensible post-startup output.

### 2. Scenario-Level Accuracy

[Evidence, C03-C06] The scenario-level RMSE results do not support a universal
winner. Instead, the best estimator changes with the disturbance class. ESPRIT
wins five sinusoidal or frequency-dynamic scenarios, including the single-sine,
frequency-step, two frequency-ramp, and FM-modulation cases. EKF wins six
magnitude, phase-jump, AM-modulation, and NERC phase-jump cases. LKF wins two
cases, including harmonic interference and out-of-band interference, while UKF
wins the IBR multi-event scenario. Thus, the main empirical result is not that a
single estimator dominates the benchmark, but that estimator choice is strongly
regime-dependent.

Candidate Table 1. Scenario-level RMSE winners after Phase 2-F filtering.

| Scenario | RMSE winner | Family | RMSE Hz |
| --- | --- | --- | ---: |
| `IBR_Harmonics_Medium` | LKF | Model-based | 0.0591543 |
| `IBR_Multi_Event` | UKF | Model-based | 1.05892 |
| `IEEE_Freq_Ramp_10Hzs` | ESPRIT | Window-based | 0.0294006 |
| `IEEE_Freq_Ramp_1Hzs` | ESPRIT | Window-based | 0.00964082 |
| `IEEE_Freq_Step` | ESPRIT | Window-based | 0.08119 |
| `IEEE_Mag_Step_25pct` | EKF | Model-based | 0.00195077 |
| `IEEE_Mag_Step_5pct` | EKF | Model-based | 0.000239157 |
| `IEEE_Modulation_AM` | EKF | Model-based | 0.00016879 |
| `IEEE_Modulation_FM` | ESPRIT | Window-based | 0.0358334 |
| `IEEE_OOB_Interference` | LKF | Model-based | 0.0405268 |
| `IEEE_Phase_Jump_20` | EKF | Model-based | 0.0431866 |
| `IEEE_Phase_Jump_60` | EKF | Model-based | 0.116916 |
| `IEEE_Single_SinWave` | ESPRIT | Window-based | 1.92602e-12 |
| `NERC_Phase_Jump_60` | EKF | Model-based | 0.112267 |

### 3. Family-Level Behavior

[Evidence, C04-C05, C11] The filtered scenario winners show a clear but not
absolute model-based advantage. Model-based estimators win 9 of 14 scenarios:
EKF wins 6, LKF wins 2, and UKF wins 1. Window-based estimators remain important
because ESPRIT wins 5 scenarios and is competitive in 5 scenarios under the
classical-competitiveness table. TFT and PLL are competitive in 2 scenarios
each, and Prony is competitive in 1 scenario after the diagnostic exclusions are
removed. These results support a more precise conclusion than "newer is better"
or "classical is obsolete": model-based estimators are the strongest family
overall in this matrix, while selected classical/windowed estimators remain
competitive in specific signal regimes.

[Evidence, C03-C05, C17] The global RMSE confidence-interval table is consistent
with that interpretation. The lowest mean-RMSE group is model-based: LKF, UKF,
RA-EKF, LKF2, and EKF occupy the first five global RMSE ranks. However, the
scenario table explains why this global ranking should not be read as a
universal deployment rule. ESPRIT is globally lower ranked because it degrades
in IBR stress, but it is still the best estimator in several clean
frequency-dynamic events.

Candidate Table 2. Global RMSE summary across the Phase 2-G matrix.

| Rank | Estimator | Family | Mean RMSE Hz | 95% CI |
| ---: | --- | --- | ---: | --- |
| 1 | LKF | Model-based | 0.195301 | [0.169086, 0.226381] |
| 2 | UKF | Model-based | 0.206015 | [0.178663, 0.237714] |
| 3 | RA-EKF | Model-based | 0.225761 | [0.197643, 0.258592] |
| 4 | LKF2 | Model-based | 0.259098 | [0.228308, 0.293635] |
| 5 | EKF | Model-based | 0.265027 | [0.229803, 0.305129] |
| 6 | SOGI-FLL | Loop-based | 0.292897 | [0.254574, 0.336523] |
| 7 | TFT | Window-based | 0.407134 | [0.357663, 0.459843] |
| 8 | PLL | Loop-based | 0.412624 | [0.382271, 0.447098] |

### 4. IBR Stress Behavior

[Evidence, C07] The IBR scenarios reveal a stress regime that is not visible
from the standard IEEE-style events alone. LKF is the RMSE winner for
`IBR_Harmonics_Medium`, while UKF is the winner for `IBR_Multi_Event`. The
robustness table also shows that several estimators with strong baseline
accuracy suffer large RMSE increases under IBR stress. For example, ESPRIT has a
low baseline RMSE but increases by 2.854 Hz under IBR aggregation, and ZCD
increases by 181.667 Hz. This pattern supports a stress-test interpretation:
IBR events separate estimators that appear similar in simpler regimes.

Candidate Table 3. IBR robustness spread.

| Estimator | Baseline RMSE Hz | IBR RMSE Hz | Delta Hz |
| --- | ---: | ---: | ---: |
| PLL | 0.413382 | 0.616581 | 0.203199 |
| LKF | 0.124761 | 0.568929 | 0.444168 |
| EKF | 0.268742 | 0.743544 | 0.474802 |
| UKF | 0.0826767 | 0.56098 | 0.478303 |
| RA-EKF | 0.0826912 | 0.660675 | 0.577983 |
| Koopman (RK-DPMU) | 0.0488875 | 1.85883 | 1.80994 |
| Prony | 0.0686137 | 2.86759 | 2.79897 |
| ESPRIT | 0.040595 | 2.89464 | 2.85404 |
| RLS | 0.182258 | 19.6903 | 19.508 |
| ZCD | 0.0574288 | 181.724 | 181.667 |

### 5. Diagnostic Appendix and Invalid Outputs

[Limitation, C08-C10] The Phase 2-F filter moves 4 Prony pairs to the diagnostic
appendix because they show post-startup invalid outputs or non-finite accuracy
records. The affected pairs are `IBR_Multi_Event`, `IEEE_Phase_Jump_20`,
`IEEE_Phase_Jump_60`, and `NERC_Phase_Jump_60`. The strongest case is
`IBR_Multi_Event / Prony`, which has 0 valid RMSE runs and a mean post-startup
invalid rate of 0.185751. The phase-jump cases show smaller mean invalid rates,
but the valid-RMSE counts remain insufficient for ordinary main-table ranking in
two of the three jump cases. This is not evidence that Prony should be removed
from the benchmark; it is evidence that Prony has a bounded failure mode that
must be reported separately.

Candidate Table 4. Diagnostic appendix pairs.

| Scenario | Estimator | Valid RMSE runs | Mean post-startup invalid | Reason |
| --- | --- | ---: | ---: | --- |
| `IBR_Multi_Event` | Prony | 0 | 0.185751 | post_startup_invalid;nonfinite_accuracy |
| `IEEE_Phase_Jump_20` | Prony | 28 | 0.0015006 | post_startup_invalid;nonfinite_accuracy |
| `IEEE_Phase_Jump_60` | Prony | 6 | 0.0169841 | post_startup_invalid;nonfinite_accuracy |
| `NERC_Phase_Jump_60` | Prony | 5 | 0.0169386 | post_startup_invalid;nonfinite_accuracy |

[Contrast, C10] ESPRIT and Koopman pass the same post-startup invalid-output
gate in all 14 scenarios. This validity result should be kept separate from
accuracy dominance. Passing the validity gate means that their outputs can enter
the main comparison; it does not mean they win every scenario.

### 6. Pareto and Timing Context

[Decision support, C12-C13] The Pareto recommendations identify UKF, EKF,
RA-EKF, SOGI-FLL, and LKF2 as balanced candidates under the current scoring
profiles. UKF ranks first in the protection, monitoring, and offline-analysis
profiles, while SOGI-FLL ranks first in the low-cost profile. This ranking is
useful for estimator selection, but it must be presented with a timing caveat:
the Phase 2-G run used `BENCHMARK_N_COST_REPS=1`, so CPU values are diagnostic
context rather than publication-grade timing evidence. For the same reason, ZCD
can be described as the fastest estimator in this timing context, but not as a
definitive fastest estimator under a journal-grade timing protocol.

Candidate Table 5. Top Pareto recommendations by profile.

| Profile | Rank | Estimator | RMSE Hz | CPU us | Latency ms |
| --- | ---: | --- | ---: | ---: | ---: |
| protection | 1 | UKF | 0.206015 | 10.881 | 0.0 |
| protection | 2 | EKF | 0.265027 | 9.210 | 0.0 |
| protection | 3 | SOGI-FLL | 0.292897 | 8.002 | 0.0 |
| monitoring | 1 | UKF | 0.206015 | 10.881 | 0.0 |
| monitoring | 2 | EKF | 0.265027 | 9.210 | 0.0 |
| monitoring | 3 | RA-EKF | 0.225761 | 12.702 | 0.0 |
| low_cost | 1 | SOGI-FLL | 0.292897 | 8.002 | 0.0 |
| low_cost | 2 | EKF | 0.265027 | 9.210 | 0.0 |
| low_cost | 3 | UKF | 0.206015 | 10.881 | 0.0 |
| offline_analysis | 1 | UKF | 0.206015 | 10.881 | 0.0 |
| offline_analysis | 2 | RA-EKF | 0.225761 | 12.702 | 0.0 |
| offline_analysis | 3 | EKF | 0.265027 | 9.210 | 0.0 |

### 7. Scope Limits

[Limitation, C14-C16] Three boundaries should be stated before any Discussion
section expands the claims. First, Phase 2-G does not support PI-GRU
generalization claims because PI-GRU is not included in the 17-estimator matrix
and the corresponding report table is empty. Second, Phase 2-G does not support
ATLAS severity-sweep claims because it is a fixed OpenFreqBench matrix run.
Third, CPU ranking remains diagnostic until a dedicated timing run with a
stronger `n_cost_reps` protocol is completed. These limits do not weaken the
Phase 2-G accuracy and failure-mode results; they define which claims should be
reserved for the next evidence layer.

## Candidate Figure Plan

| Manuscript figure | Source artifact | Message |
| --- | --- | --- |
| Fig. 1 | `plots/scenario_rmse_heatmap.png` | Accuracy is scenario-dependent, not universal. |
| Fig. 2 | `plots/rmse_by_estimator_ci.png` and/or `plots/family_rmse_boxplot.png` | Model-based estimators lead globally, but family behavior varies. |
| Fig. 3 | `plots/ibr_robustness_delta.png` | IBR stress changes estimator ordering and exposes large degradation. |
| Fig. 4 | `plots/failure_rate_by_estimator.png` | Prony diagnostic failures are isolated by the Phase 2-F gate. |
| Fig. 5 | `plots/pareto_rmse_cpu.png` | Pareto recommendations are useful but CPU must be caveated. |

## Claim-Evidence Map

| Draft section | Claim IDs used | Evidence |
| --- | --- | --- |
| Evidence contract and scope filter | C01, C02 | `analysis_summary.md`, `paper_scope_classification.csv`, `diagnostic_appendix.csv` |
| Scenario-level accuracy | C03, C04, C05, C06 | `analysis_summary.md`, `metric_confidence_intervals.csv` |
| Family-level behavior | C04, C05, C11, C17 | `estimator_rmse_ci.csv`, `classical_competitiveness.csv` |
| IBR stress behavior | C07 | `ibr_robustness.csv`, `plots/ibr_robustness_delta.png` |
| Diagnostic appendix | C08, C09, C10 | `diagnostic_appendix.csv`, `failure_analysis.csv`, `paper_scope_classification.csv` |
| Pareto and timing context | C12, C13 | `pareto_recommendations.csv`, `estimator_cpu_ci.csv` |
| Scope limits | C14, C15, C16 | `configs/phase2-paper-grade.yaml`, `pi_gru_generalization.csv`, Phase 2-G notes |

## Self-Review

Contribution: The draft turns the benchmark from a list of scores into a
taxonomy of estimator behavior by regime. The contribution is clear, but the
final paper still needs a concise Methods section that defines the scenario
families and the Phase 2-F validity gate.

Writing clarity: Each subsection has one message and starts with the evidence
boundary before the interpretation. The next revision should reduce table count
for a two-column paper; not all candidate tables should enter the main text.

Experimental strength: The 30-seed accuracy evidence is strong enough for a
serious results draft. CPU ranking is not yet strong enough for journal-grade
claims because timing used `n_cost_reps=1`.

Evaluation completeness: PI-GRU and ATLAS are intentionally out of scope. If
the paper claims to cover learning-based or severity-sweep behavior, new
evidence must be generated first.

Method design soundness: The main-vs-diagnostic separation is defensible and
should be explained before results. The strongest reviewer risk is that readers
may object to excluding Prony pairs; the response is to keep those pairs in the
diagnostic appendix with explicit invalid-output statistics.

## Next Gate

The matching Methods/Experimental Setup draft is documented in
`docs/PHASE2J_METHODS_EXPERIMENTAL_SETUP_DRAFT.md`. Move both drafts into the
manuscript only after deciding the target format. For a results/theory journal
paper, convert the prose to LaTeX and keep Tables 1, 3, 4, and one
Pareto/timing table as candidates. For a software-first paper, shorten the
estimator-specific discussion and emphasize the reproducible benchmark contract,
output schema, and failure-mode reporting.
