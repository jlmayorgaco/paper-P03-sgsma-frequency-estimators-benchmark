# Paper Claim Audit for `slide_final.tex`

Primary paper source: `C:/Users/walla/Downloads/2026149998 (1).pdf`

The final compact deck uses the paper-extracted assets already present in
`../slides/figures`. These were visually checked before integration.

## Integrated Paper Diagrams

| Deck slide | Paper source | Integrated claim |
|---|---|---|
| Scenario families from the paper | Table I (`paper_table1_scope.png`) | 34 scenarios across 10 disturbance families. |
| Scenario waveforms from the paper | Fig. 1 (`paper_fig1_scenarios.png`) | Step, ramp, modulation, islanding jump, multi-event, and ringdown are the core waveform references. |
| Paper Fig. 2: event responses | Fig. 2(a-d) (`paper_fig2_panels_ad.png`) | Phase jump, ramp lag, multi-event recovery, and ringdown recovery are trace-level evidence, not only table summaries. |
| Paper Fig. 2: scorecard and cost-risk view | Fig. 2(e-h) (`paper_fig2_panels_eh.png`) | Disturbance scorecard, MC RMSE by family, cost vs. accuracy, and cost vs. trip-risk support metric-dependent ranking. |
| Paper-backed global indicators | Table III extraction (`paper_table3_global.png`) | Mean RMSE, mean-rank RMSE, RMSE wins, RMSE top-three, peak-error top-three, and trip-risk top-three have different leaders. |

## Safe Numerical Claims

| Claim | Status |
|---|---|
| 16 estimators, 34 scenarios, 10 families, 60 MC runs | Paper-backed SGSMA matrix. |
| 34 scenarios across 10 disturbance families | Paper Table I. |
| Mean RMSE leader: RA-EKF, 116 mHz | Paper Table III extraction. |
| Mean-rank RMSE leader: TFT, 4.38 | Paper Table III extraction. |
| RMSE wins leader: EKF, 10/34 | Paper Table III extraction. |
| RMSE top-three leader: EKF, 19/34 | Paper Table III extraction. |
| Peak-error top-three leader: EKF, 20/34 | Paper Table III extraction. |
| Trip-risk top-three leader: IpDFT, 25/34 | Paper Table III extraction. |

## Guardrails Kept in the Final Deck

- No "275x" claim is used.
- No field event is described as preventable by this estimator benchmark.
- CPU is described as a software-cost proxy, not hardware latency.
- ATLAS is described as an extended severity sweep, not as a new standard.
- ATLAS Prony stress-corner failures are explicitly flagged.
- The main conclusion is metric-conditional ranking, not a universal winner.

## Visual Checks

- `paper_fig1_scenarios.png`: complete and legible as a full-slide reference.
- `paper_fig2_panels_ad.png`: complete and legible as trace-level evidence.
- `paper_fig2_panels_eh.png`: complete and legible as scorecard/cost-risk evidence.
- `paper_table1_scope.png`: complete and readable.
- `paper_table2_family_results.png`: visually cropped; not used as a primary slide image.
- `paper_table3_global.png`: cropped but contains the global indicators used in the redesigned table.

