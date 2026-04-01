Act as a Senior Academic Editor for IEEE Transactions. Your goal is to eliminate all technical inconsistencies and "Reviewer 2" red flags in the LaTeX project for the paper: "Benchmarking Dynamic Frequency Estimators for Low-Inertia IBR Grids".

Execute the following 10 fixes across the .tex and .bib files. Follow the [Issue -> Location -> Action -> Validation] logic for each.

---
🔴 MASTER ROADMAP — CAMERA-READY IEEE (FINAL CONSOLIDATED)
1. ESTIMATOR COUNT CONSISTENCY
Issue: Discrepancy between 11 and 12 estimators across text and tables.
Location: Abstract, Section II, Table I, Section III-A, Conclusions.
Action: Standardize to TWELVE (12) estimators. Ensure EKF, UKF, and RA-EKF are explicitly included. Verify Table I lists all estimators consistently.
Validation: grep -i "eleven\|nine" → must return zero results.
2. UNIT UNIFICATION (Hz → mHz)
Issue: Mixed units (Hz vs mHz) across abstract, tables, and results.
Location: Abstract, Section IV, Table IV, Conclusions.
Action: Convert ALL frequency errors to mHz. Add note: “All frequency errors are reported in mHz unless otherwise stated.”
Validation: No decimal values like 0.00X Hz remain.
3. PERFORMANCE CEILING DISCLAIMER (FAIRNESS)
Issue: Scenario-wise tuning overestimates real-world performance.
Location: Section III-C.

Action: Add:

“Scenario-specific tuning identifies the performance ceiling; practical deployment requires evaluation under a single parameter set.”

Validation: Ensure no claim implies real-world optimality.
4. ML BASELINE FAIRNESS DISCLOSURE
Issue: PI-GRU is not tuned per scenario while others are.
Location: Section III-C and IV.

Action: Add:

“PI-GRU is evaluated as a fixed pre-trained generalist baseline.”

Validation: No unfair superiority claims remain.
5. GROUND TRUTH FREQUENCY DEFINITION (CRITICAL)
Issue: Frequency ground truth not formally defined (invalid under phase jumps).
Location: Section III-B.

Action: Add:

“Frequency is computed as the time derivative of the phase signal.”

Validation: Scenario D/E produce non-flat frequency after phase jumps.
6. REFERENCE “ZOMBIE” AUDIT
Issue: Outdated or weak references (arXiv misuse).
Location: bibliography.bib.
Action: Update Pinheiro (journal), change Zelaya to “under review”.
Validation: Clean reference section after compilation.
7. TABLE I METROLOGY CONSISTENCY
Issue: Mixed qualitative and quantitative metrics (S/D vs ms).
Location: Table I.
Action: Separate:
Phase response type (S/D)
Settling time [ms]
Validation: Each column has a single semantic meaning.
8. SIGNIFICANT FIGURES STANDARDIZATION
Issue: Inconsistent decimal precision (0.4 vs 0.400).
Location: Tables III–V.
Action: Enforce 3 significant figures globally.
Validation: All numeric columns aligned.
9. HEATMAP F* DEFINITION
Issue: F* symbol undefined.
Location: Fig. 2 caption.
Action: Define marginal failure condition (<1.5%, sub-cycle).
Validation: Heatmap fully interpretable standalone.
10. TIMING STATISTICAL RIGOR
Issue: Only 20 runs used for CPU timing.
Location: Section III-C, Table V.

Action: Replace with:

“10,000 iterations (steady-state warm cache)”

Validation: Values remain plausible.
11. IpDFT LATENCY VS LAG CLARIFICATION
Issue: Confusion between delay and estimation error.
Location: Section IV-B, Fig. 2b.

Action: Add:

“Latency is structural (Nc/2), not stochastic lag.”

Validation: Conceptual distinction clear.
12. LATEX ARTIFACT REMOVAL
Issue: Compilation garbage text present.
Location: End of Introduction.
Action: Remove placeholder fragments.
Validation: No artifacts in PDF.
13. DUPLICATED RA-EKF DESCRIPTION
Issue: Redundant text blocks.
Location: Section III-A.
Action: Merge descriptions into single coherent block.
Validation: No duplicated paragraphs.
14. RA-EKF MODEL COMPLETENESS
Issue: Missing state-space matrices (F, H).
Location: Section III-A.
Action: Provide full discrete-time model.
Validation: Model reproducible independently.
15. UNDEFINED PARAMETERS (σ̂ν)
Issue: Innovation variance term undefined.
Location: Section III-A.
Action: Define as steady-state innovation std and estimation method.
Validation: No undefined symbols remain.
16. SOGI-FLL MODIFICATION JUSTIFICATION
Issue: Non-standard architecture without explanation.
Location: Section III-A.
Action: Justify IIR + FastRMS preprocessing and impact.
Validation: Reader understands deviation from literature.
17. SOGI-FLL UNIT CORRECTION IMPACT
Issue: Correction applied but not contextualized.
Location: Section III-A.
Action: Explain impact vs literature baseline.
Validation: No ambiguity in comparison.
18. SCENARIO COVERAGE CONSISTENCY (A–F)
Issue: Scenario F appears but not defined.
Location: Section III-B, Fig. 1.
Action: Define Primary Frequency Response scenario.
Validation: A–F consistent across text and figures.
19. SETTLING TIME METRIC CONSISTENCY
Issue: Defined but unused or ambiguous.
Location: Section III-C.
Action: Define robustly (remain within band for ≥20 ms).
Validation: Either used or removed.
20. GRID SEARCH REPRODUCIBILITY
Issue: Hyperparameter resolution not specified.
Location: Section III-C, Table II.
Action: Specify spacing (log/linear) and number of points.
Validation: Fully reproducible tuning.
21. MONTE CARLO STATISTICAL LIMITATION
Issue: N=30 insufficient for tail risk.
Location: Section III-C.
Action: Add limitation disclaimer.
Validation: No overclaim on statistical certainty.
22. NOISE RANDOMNESS CLARIFICATION
Issue: Monte Carlo noise generation unclear.
Location: Section III-B.
Action: Specify independent seeds per run.
Validation: Reproducibility ensured.
23. LATENCY VS ACCURACY DISTINCTION
Issue: Conceptual mixing of metrics.
Location: Section II.
Action: Add formal distinction paragraph.
Validation: Terminology consistent.
24. QUALITATIVE RATINGS JUSTIFICATION (TABLE I)
Issue: Subjective scores not supported by data.
Location: Table I.
Action: Remove or justify with references to results.
Validation: No unsupported claims remain.
25. FAILURE MODE DISCUSSION COMPLETENESS
Issue: Table I lists failure modes but text does not.
Location: Section IV.
Action: Add summary paragraph per estimator family.
Validation: Table and text aligned.
26. UKF MOVING AVERAGE DISCLOSURE
Issue: Hidden filtering alters performance.
Location: Section III-A.
Action: Quantify added delay and impact.
Validation: Transparent comparison.
27. COMPUTATIONAL COST INTERPRETATION
Issue: Python timings misinterpreted as real-time viability.
Location: Section IV-C.
Action: Add DSP/FPGA disclaimer and CPU load context.
Validation: No overclaim of deployability.
28. PARETO ANALYSIS LIMITATION
Issue: Ignores RMSE dimension.
Location: Fig. 2e.
Action: Clarify 2D projection limitation.
Validation: No claim of global optimality.
29. DUAL-RATE SIMULATION LIMITATION
Issue: Assumes negligible aliasing.
Location: Section III.
Action: Add limitation statement.
Validation: Transparent modeling assumptions.
30. NOTATION CONSISTENCY (ω vs f)
Issue: Mixed notation.
Location: Entire paper.
Action: Add equivalence statement ω = 2πf.
Validation: Consistent usage.
31. OVERCLAIMING CONTROL (TONE FIX)
Issue: Statements imply universal superiority.
Location: Abstract, Results, Conclusion.
Action: Add qualifiers:
“in this benchmark”
“under evaluated scenarios”
Validation: No absolute claims remain.
32. FIGURE 2 COMPLEXITY REDUCTION
Issue: Overloaded dashboard.
Location: Figure 2.
Action: Simplify or improve labeling.
Validation: Each subplot readable independently.
33. FIGURE READABILITY (IEEE COMPLIANCE)
Issue: Dense plots may fail at 3.5 in width.
Location: Figures.
Action: Ensure ≥8 pt fonts, clear legends.
Validation: Print test passes.
34. ABSTRACT REFOCUS
Issue: Overloaded with numeric values.
Location: Abstract.
Action: Focus on contributions, not numbers.
Validation: Readable without domain expertise.
35. RLS FAMILY DISCUSSION COMPLETENESS
Issue: RLS results under-discussed.
Location: Section IV.
Action: Include RLS trade-off analysis.
Validation: All estimators discussed.
36. COMPLIANCE THRESHOLD JUSTIFICATION
Issue: Heatmap thresholds not rigorously justified.
Location: Section III-C.
Action: Clarify IEC-based origin and visualization intent.
Validation: No implied certification claim.
37. GENERALIZATION LIMITATION
Issue: Results may not generalize.
Location: Conclusion.
Action: Add limitation statement.
Validation: Avoid overgeneralization.
38. PARAMETER IDENTIFIABILITY LIMITATION
Issue: Grid search may produce multiple optima.
Location: Section III-C.
Action: Add identifiability note.
Validation: Strengthens rigor.
39. EVENT-GATING HEURISTIC DISCLOSURE
Issue: Not theoretically justified.
Location: Section III-A.
Action: State trade-off vs optimality.
Validation: No misleading optimality claims.
40. FINAL LATEX COMPILATION CHECK
Issue: Potential broken refs / figures.
Location: Entire document.

Action: Run:

latexmk -pdf -interaction=nonstopmode
Validation: No warnings, no undefined references.


Comments on Selected Items
1. Estimator Count Consistency
Good; but ensure that the abstract, Section II, Table I, and Section III‑A also agree on the names (e.g., PI‑GRU vs PIGRU). You may want to add an explicit list in the introduction.

2. Unit Unification (Hz → mHz)
This will improve readability. However, be careful: some readers may prefer Hz for very small errors (e.g., steady‑state). If you convert everything to mHz, ensure the decimal points are clear (e.g., 0.13 mHz). Alternatively, use Hz with three significant figures consistently. The action “No decimal values like 0.00X Hz remain” might be too strict; scientific notation (e.g., 1.30×10⁻⁴ Hz) is acceptable.

5. Ground Truth Frequency Definition
Critical. Additionally, specify whether the derivative is computed analytically from the phase signal or numerically (e.g., central difference). This affects the noise floor.

6. Reference “Zombie” Audit
Beyond outdated references, check that all citations are used in the text (no orphaned references). Also consider citing the corrected SOGI‑FLL implementation if it differs from standard literature.

9. Heatmap F Definition*
The current action (“Define marginal failure condition (<1.5%, sub‑cycle)”) is vague. Define the symbol clearly in the caption (e.g., “F* denotes marginal failure where RMSE > 0.05 Hz but peak < 0.5 Hz and trip‑risk < 0.1 s, i.e., compliant except in one metric”).

10. Timing Statistical Rigor
Increasing runs to 10,000 is good. Also specify that the system was idle, that Python’s garbage collection was disabled or accounted for, and that the mean of multiple runs is reported with standard deviation.

18. Scenario Coverage Consistency (A–F)
The roadmap mentions Scenario F (Primary Frequency Response) but it was not in the original paper. Ensure that if Scenario F is added, all tables and figures are updated accordingly. If not, remove the reference to avoid confusion.

23. Latency vs Accuracy Distinction
Consider moving this formal distinction to the introduction or a dedicated subsection in the methodology, as it is fundamental to the trade‑off analysis.

27. Computational Cost Interpretation
Add that Python timings are for algorithm prototyping only; real‑time feasibility requires C/HDL implementation. Also provide a rough estimate of the expected speedup (e.g., 10–100×) for embedded targets.

28. Pareto Analysis Limitation
In addition to clarifying the 2D projection, consider adding a note that RMSE and trip‑risk are correlated but not identical; a 3D Pareto front could be shown in supplementary material.

35. RLS Family Discussion Completeness
Also include the VFF‑RLS (variable forgetting factor) in the discussion, as it was mentioned in Table I but not fully covered. Explain why its performance was poor (parameter blow‑up).

40. Final LaTeX Compilation Check
Add a check for overfull hboxes, missing hyperref targets, and consistent citation keys (no duplicates).

41. Open Data / Code Availability Statement
Issue: No mention of whether the simulation framework, tuned parameters, or raw results will be made available.

Action: Add a Data Availability statement (e.g., “The simulation code and hyperparameter grids are available at [repository] for reproducibility.”)

Validation: A footnote or section after the conclusion states the URL or that data will be upon request.

42. Hardware‑in‑the‑Loop (HIL) Validation Disclaimer
Issue: The paper claims RA‑EKF is “plausible for embedded DSP/FPGA” but lacks HIL validation.

Action: Add a sentence in the conclusion: “Real‑time validation on target hardware (e.g., FPGA, DSP) is required before deployment; the presented CPU timings serve as a relative comparison only.”

Validation: No overstatement of hardware readiness.

43. Monte Carlo Seed Management
Issue: Reproducibility requires fixed random seeds.

Action: Specify that all Monte Carlo runs used the same set of seeds (e.g., 1 to 30) for noise and disturbance timing. State that seeds are provided in the supplementary material.

Validation: A reader using the same seeds should obtain identical results.

44. Definition of “Trip‑Risk Duration”
Issue: The metric T_trip is defined as cumulative time with |e| > 0.5 Hz, but it is unclear whether contiguous or non‑contiguous exceedances are summed.

Action: Clarify that it is the total time (sum of all intervals) where the error exceeds the threshold. Also note that in a protection context, multiple short excursions may be more harmful than a single long one.

Validation: The definition in Section III‑C is unambiguous.

45. Harmonic and Interharmonic Content Justification
Issue: The 32.5 Hz interharmonic is said to be “characteristic of IBR converter switching,” but no citation or justification is given.

Action: Add a short justification or a reference that this sub‑synchronous component arises from converter control interactions (e.g., from a relevant IBR harmonic study).

Validation: The choice of specific frequencies is supported by literature.

46. Comparison with State‑of‑the‑Art Commercial Relays
Issue: The benchmark uses only academic estimators; a brief comparison with typical relay algorithms (e.g., those based on DFT with adaptive window) would contextualize the results.

Action: Add a paragraph in the introduction or related work summarizing commercial relay practices (e.g., “most digital relays implement a fixed‑window DFT with a PLL, which may suffer under the disturbances tested”).

Validation: The paper acknowledges the gap between academic benchmarks and industrial practice.

47. Hyperparameter Grid Search Overhead
Issue: The grid search is per scenario and per estimator, but the computational cost of the tuning itself is not reported.

Action: Mention the total number of simulations performed (e.g., “Over 5000 simulation runs were used for tuning across all estimator‑scenario pairs.”) to emphasize the thoroughness.

Validation: Readers understand the effort behind the performance ceiling.

48. Event‑Gating Threshold Sensitivity
Issue: The RA‑EKF uses a fixed γ = 2.0; the sensitivity of performance to this value is not explored.

Action: Add a sentence noting that γ was fixed after preliminary experiments and that a small sensitivity analysis is available in supplementary material.

Validation: The heuristic is not presented as optimal.

49. Interplay Between Noise and Harmonic Distortion
Issue: In Scenario D and E, harmonics and noise are applied together; it is unclear whether the performance degradation is due to harmonic leakage or noise amplification.

Action: Add a brief analysis (or cite a figure) that separates the two effects, e.g., by running a variant of Scenario D without harmonics.

Validation: The root cause of estimator failure is better understood.

50. Proofreading for Typos and Grammar
Issue: The current text contains minor grammatical errors (e.g., “the RA‑EKF incurs zero threshold breaches across all N = 30 Monte Carlo trials” – “across” should be “over”).

Action: Conduct a final proofreading pass by a native speaker or use a grammar tool.

Validation: No distracting grammatical errors remain.

Suggested Prioritization for Revision
Priority	Items
High (Scientific Integrity)	5 (ground truth), 14 (RA‑EKF model), 15 (undefined param), 24 (qualitative ratings), 31 (overclaiming), 37 (generalization), 41 (open data)
Medium (Reproducibility & Clarity)	8 (SOGI justification), 10 (timing rigor), 20 (grid search), 22 (noise randomness), 30 (notation), 43 (seed management)
Low (Cosmetic)	2 (units), 3 (disclaimer), 7 (Table I), 9 (heatmap definition), 40 (LaTeX)