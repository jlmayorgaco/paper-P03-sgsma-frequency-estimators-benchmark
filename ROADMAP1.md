# ROADMAP.md - Reviewer-Grade Submission Hardening Plan for SGSMA 2026

Date: 2026-04-23
Scope: Code, benchmark protocol, figures, paper, and final submission package
Current phase: Consolidation
Source of truth: `src/` and the active modular pipeline

## 1. Reviewer verdict

Current decision: NO-GO for final submission in the repository state reviewed on 2026-04-23.

This is not because the topic is weak. The topic is relevant and publishable.
The blockers are:

1. The compiled paper is 7 pages, while the SGSMA 2026 public author page states that full papers must be at most 6 pages.
2. The manuscript still describes a frozen 5-scenario, 12-estimator study, while the active codebase and outputs reflect a broader active benchmark universe.
3. The canonical artifact chain is broken in practice:
   - `artifacts/full_mc_benchmark/` is declared canonical but does not exist in the current workspace
   - `python src/pipelines/sync_paper_artifacts.py` does not run from repo root
   - the paper figures in `paper/Figures/Plots_And_Graphs/` are older than later code and output changes
4. The main dashboard figure is not readable enough at conference scale.

## 2. Scorecard

- Topic relevance to SGSMA: 8/10
- Scientific value if claims are made auditable: 7/10
- Novelty: 6/10
- Methodological rigor in current repository state: 5/10
- Reproducibility and provenance: 3/10
- Writing quality: 6/10
- IEEE / SGSMA format readiness: 4/10
- Figure and table readiness: 4/10
- Submission readiness today: 3/10

## 3. What is already strong

- The topic is pertinent to SGSMA. The conference explicitly covers advanced signal processing, waveform component estimation, performance metrics, power system stability/transients, and analytics for modern power systems.
- The paper has a conventional IEEE structure and the first page renders credibly in conference style.
- The abstract is within IEEE's 250-word guidance.
- Core estimator and pipeline tests pass in the current environment, including `ESPRIT`, `Prony`, `PI-GRU`, `src/tests`, and `src/pipelines/full_mc_benchmark.py`.
- The scientific angle is useful: composite IBR disturbance benchmarks can reveal failure modes that isolated IEC/IEEE-style tests do not expose.

## 4. What is weak

- The paper is trying to behave like a frozen camera-ready submission while the repository is still behaving like an evolving benchmark platform.
- The manuscript still reads as if it is describing one exact experiment, but the code and outputs show multiple competing experiment universes.
- The paper over-compresses too much information into a few tables and one oversized dashboard, which harms readability more than it helps completeness.
- The prose is technically competent but still shows human-light compression patterns: repeated stock claims, dense contrast formulas, and conclusion paragraphs that partially restate the abstract instead of adding judgment.

## 5. Execution rules

1. Do not add new claims before the current claims are traceable.
2. Do not invent new figures before the current figure stack is made readable.
3. Do not widen the benchmark scope during submission hardening.
4. Do not keep both "paper subset" and "active pipeline" narratives alive at the same time.
5. Any final manuscript number must map to one canonical artifact path and one frozen commit.

## 6. Critical path

1. Close all P0 tickets.
2. Rebuild figures and tables from the frozen run.
3. Reduce to 6 pages without layout hacks.
4. Run the final compliance checks and only then submit.

## 7. Ticket board

### P0-001 - Enforce the 6-page SGSMA limit
Priority: P0
Problem:
The current compiled PDF is 7 pages. SGSMA 2026 publicly states that full papers must be at most 6 pages.
Actions:
1. Reduce the paper to 6 pages total, including references unless the chairs explicitly confirm otherwise in writing.
2. Do this by reducing content density, not by reintroducing template hacks.
3. Remove one low-value table or replace the multi-panel dashboard with fewer stronger panels.
Acceptance:
- `pdfinfo paper/index.pdf` reports `Pages: 6`.
- No non-standard layout compression packages or manual spacing hacks are reintroduced.
Deliverables:
- Updated `paper/index.pdf`
- short `paper/SUBMISSION_NOTES.md` note recording the final page count

### P0-002 - Choose one experiment universe and kill the other narrative
Priority: P0
Problem:
The paper describes 5 scenarios and 12 estimators, while the active pipeline defines a 32-scenario, 18-estimator benchmark.
Actions:
1. Decide between:
   - Option A: freeze an explicit paper subset in code and publish only that subset
   - Option B: rewrite the paper to match the active pipeline
2. Record the choice in one file that governs both code and paper.
3. Remove contradictory wording from `README.md`, `paper`, and pipeline metadata.
Acceptance:
- The paper, README, and JSON manifest describe the same scenario set and estimator set.
- There is no remaining text that simultaneously implies both a frozen 5-scenario study and an active 32-scenario benchmark.
Deliverables:
- updated `src/pipelines/benchmark_definition.py`
- updated manuscript wording
- one machine-readable benchmark manifest

### P0-003 - Freeze the canonical benchmark configuration
Priority: P0
Problem:
Paper claims are not currently anchored to one immutable run configuration.
Actions:
1. Store the exact benchmark configuration in the exported JSON:
   - scenario list
   - estimator registry
   - tuning method
   - `N_TRIALS_TUNING`
   - `N_MC_RUNS`
   - seed policy
   - timing protocol
2. Add git SHA and environment metadata.
3. Make sure the final paper references this frozen benchmark identity.
Acceptance:
- Final JSON includes commit, config, and environment metadata.
- A reviewer can answer "which exact run produced Table/Fig X?" from repository artifacts alone.
Deliverables:
- updated benchmark export manifest
- `claim_map.csv` or equivalent traceability file

### P0-004 - Restore the canonical artifact chain
Priority: P0
Problem:
`src/pipelines/paths.py` declares `artifacts/full_mc_benchmark/` as canonical, but that directory does not exist in the reviewed workspace.
Actions:
1. Produce the canonical artifact directory from the frozen benchmark.
2. Ensure the expected files exist:
   - `benchmark_full_report.json`
   - `global_metrics_report.csv`
   - `Fig1_Scenarios_Final.{pdf,png}`
   - `Fig2_Mega_Dashboard.{pdf,png}`
3. Stop using legacy outputs as paper truth after this point.
Acceptance:
- `artifacts/full_mc_benchmark/` exists and is populated.
- Paper-facing figures can be synced from that directory only.
Deliverables:
- canonical artifact directory

### P0-005 - Fix the paper sync workflow
Priority: P0
Problem:
The documented root-level command `python src/pipelines/sync_paper_artifacts.py` fails because the script lacks path bootstrap. The module form also fails if canonical artifacts are missing.
Actions:
1. Make both documented commands work:
   - `python src/pipelines/sync_paper_artifacts.py`
   - `cd src && python -m pipelines.sync_paper_artifacts`
2. Add a helpful error message if artifacts are missing.
3. Add a smoke test for the sync path.
Acceptance:
- Both documented invocation styles run successfully when artifacts exist.
- The failure mode for missing artifacts is explicit and actionable.
Deliverables:
- updated `src/pipelines/sync_paper_artifacts.py`
- one sync smoke test

### P0-006 - Build claim traceability for every hard number
Priority: P0
Problem:
The paper still contains hardcoded values that are not demonstrably tied to current canonical outputs.
Actions:
1. Map every hard number in abstract, results, conclusions, captions, and tables to:
   - source artifact
   - key or row
   - transformation rule
   - tolerance
2. Add a verification script that fails on mismatch.
3. Regenerate all numbers from the canonical run.
Acceptance:
- 100 percent of hard numbers in the paper are traceable.
- Verification script fails when a paper number drifts.
Deliverables:
- `paper/CLAIM_TRACEABILITY.md` or `artifacts/full_mc_benchmark/claim_map.csv`
- validation script

### P0-007 - Remove future-work scenario leakage from the final paper
Priority: P0
Problem:
The Methods section and Figure 1 include Scenario F as a planned extension, which weakens experimental discipline in a final conference paper.
Actions:
1. Remove Scenario F from the scenario figure and scenario list.
2. If it matters strategically, keep it as one sentence in future work only.
Acceptance:
- No figure, table, or method description presents non-evaluated scenarios as part of the benchmark.
Deliverables:
- updated `paper/Sections/C3_Methods/main.tex`
- regenerated Figure 1

### P0-008 - Replace the current mega-dashboard with readable paper figures
Priority: P0
Problem:
The current 8-panel dashboard is not readable at final conference scale.
Actions:
1. Replace the current Figure 2 with 2-4 panels maximum.
2. Keep only the panels that advance the core argument:
   - one disturbance tracking panel
   - one risk/cost tradeoff panel
   - one compliance or error summary panel
3. Move anything else to supplemental repository artifacts, not the main paper.
Acceptance:
- All axis labels and legends are readable at printed conference scale.
- The figure can be interpreted without zooming.
Deliverables:
- redesigned `Fig2_Mega_Dashboard`

### P1-001 - Reconcile tuning method language with the code
Priority: P1
Problem:
The paper says "scenario-wise grid search" while the active pipeline uses Optuna/TPE search.
Actions:
1. If the final benchmark truly uses Optuna, state that exactly.
2. If the final benchmark truly uses an explicit grid, remove or disable Optuna from the canonical path.
3. Do not leave hybrid wording.
Acceptance:
- Paper text, code path, and exported metadata agree on the tuning protocol.
Deliverables:
- updated Methods section
- updated benchmark metadata

### P1-002 - Reconcile Monte Carlo counts and seeds
Priority: P1
Problem:
The paper states `N=30`; active code has `N_MC_RUNS = 100`; reviewed legacy outputs include `n_mc_runs = 60`.
Actions:
1. Decide the final Monte Carlo count.
2. Apply it in code, artifacts, captions, and text.
3. Record the seed policy explicitly.
Acceptance:
- There is exactly one Monte Carlo count in the code-paper-artifact chain.
Deliverables:
- updated `full_mc_benchmark.py`
- updated paper captions/text

### P1-003 - Reconcile estimator registry and exclusions
Priority: P1
Problem:
The active benchmark registry includes `LKF`, `LKF2`, `Prony`, `ESPRIT`, `Koopman (RK-DPMU)`, and `PI-GRU`, while the paper presents a reduced subset without always explaining the exclusion policy clearly.
Actions:
1. Define which estimators are evaluated, which are reported, and why.
2. If some methods are excluded from headline tables, state the rule cleanly.
3. Keep the rule scientific, not cosmetic.
Acceptance:
- No estimator appears "hidden" or selectively omitted without an explicit policy.
Deliverables:
- updated Table I / Methods / captions

### P1-004 - Make the qualitative comparison table defensible
Priority: P1
Problem:
Table I mixes literature judgment and observed performance into categorical scores without a reproducible rubric.
Actions:
1. Either provide a reproducible scoring rubric or downgrade the table to a narrative taxonomy.
2. Avoid pseudo-objective scoring if it cannot be regenerated.
Acceptance:
- Every qualitative score is explainable, or the table is converted into a non-scored comparison.
Deliverables:
- revised Table I

### P1-005 - Tighten the timing claims
Priority: P1
Problem:
Timing statements are useful but vulnerable if not tightly bounded. The paper uses Python baselines while also making deployment class statements.
Actions:
1. Validate `m13_cpu_time_us` on the frozen run.
2. Keep relative-comparison language only.
3. Remove any implication that Python timings alone demonstrate deployability.
Acceptance:
- Timing captions and text contain no overclaim.
- Timing protocol is documented in the artifact bundle.
Deliverables:
- `timing_validation.md`
- revised Methods and Results wording

### P1-006 - Add a hard limitations paragraph
Priority: P1
Problem:
The paper still reads slightly stronger than the evidence base supports.
Actions:
1. Add one compact limitations paragraph covering:
   - simulation-only validation
   - no HIL or field measurements
   - scenario design choices
   - dependence on tuning policy
2. Put it in Results or Conclusion.
Acceptance:
- The paper explicitly states what it does not prove.
Deliverables:
- revised Results or Conclusion section

### P1-007 - Strengthen SGSMA relevance framing
Priority: P1
Problem:
The work is relevant to SGSMA, but the manuscript sometimes sounds like a generic relay-estimator benchmark rather than a measurement-and-analytics paper.
Actions:
1. Add 2-3 sentences connecting the contribution to SGSMA themes:
   - waveform component estimation
   - measurement integrity under transients
   - performance metrics for modern grid measurement chains
2. Keep the local relay angle, but position it as part of the broader measurement analytics stack.
Acceptance:
- The paper reads as naturally belonging to SGSMA, not merely adjacent to it.
Deliverables:
- revised Introduction and Conclusion framing

### P1-008 - Remove conclusion repetition
Priority: P1
Problem:
The conclusion partly restates the abstract and earlier results rather than distilling judgment.
Actions:
1. Keep only the three strongest takeaways.
2. Remove repeated wording already used in the abstract and introduction.
3. End with one precise limitation sentence and one precise forward path.
Acceptance:
- Conclusion is shorter, sharper, and less repetitive.
Deliverables:
- revised `paper/Sections/C5_Conclusions/main.tex`

### P1-009 - Rewrite the abstract in a stricter IEEE style
Priority: P1
Problem:
The abstract is within the word limit but still dense, abbreviation-heavy, and slightly generic in phrasing.
Actions:
1. Keep it below 200 words if possible.
2. Reduce abbreviations and inline notation.
3. State:
   - what was benchmarked
   - what is genuinely new
   - what the strongest finding is
4. Remove stock wording like "no single family dominates" unless directly supported by the final frozen scope.
Acceptance:
- Abstract is one tight paragraph, under 200 words, and reads naturally.
Deliverables:
- revised `paper/Config/abstract.tex`

### P1-010 - Normalize keywords to IEEE guidance
Priority: P1
Problem:
The keyword list currently contains too many terms for a concise IEEE-style keyword block.
Actions:
1. Reduce to 3-5 focused keywords or keyword phrases.
2. Keep them discoverable and field-relevant.
Acceptance:
- Keyword block contains 3-5 high-value phrases.
Deliverables:
- revised `paper/Config/keywords.tex`

### P1-011 - Remove AI-like compression patterns from the prose
Priority: P1
Problem:
The manuscript does not read like low-quality AI text, but it does show AI-assisted compression signals:
repeated contrast pairs, repeated stock claims, stacked noun phrases, and summary sentences with little new content.
Actions:
1. Rewrite the abstract, contribution paragraph, results lead-ins, and conclusion in a more human editorial voice.
2. Remove repeated phrases such as:
   - "stress-oriented benchmark"
   - "performance ceiling"
   - "beyond the standard scope"
   - "the results suggest that"
3. Prefer one direct sentence over two balanced contrast clauses.
Acceptance:
- A technical reader no longer gets the impression of machine-compressed prose.
Deliverables:
- revised Introduction, Methods lead-in, Results lead-ins, and Conclusion

### P1-012 - Do a full grammar and punctuation pass
Priority: P1
Problem:
The English is serviceable, but too many sentences are overloaded and punctuation is inconsistent.
Actions:
1. Shorten long sentences in the Introduction and Results.
2. Standardize hyphenation and capitalization:
   - low-inertia
   - model-based
   - loop-based
   - trip-risk
   - phase jump vs. phase-step
3. Remove unnecessary semicolon chains.
Acceptance:
- No sentence in the abstract or results section tries to carry more than one major claim.
Deliverables:
- full manuscript copyedit

### P1-013 - Replace unreadable tables with decision-oriented tables
Priority: P1
Problem:
The current tables are mathematically dense but not reviewer-friendly at 6-page conference scale.
Actions:
1. Keep one quantitative benchmark table and one deployment/risk table.
2. Remove or simplify the qualitative mega-table if it is not central.
3. Ensure table font remains readable without violating the template.
Acceptance:
- Each table answers one distinct question clearly.
Deliverables:
- revised Table I / Table II / Table III layout

### P1-014 - Fix fonts and graphics export quality
Priority: P1
Problem:
The generated PDF contains Type 3 fonts in the graphics pipeline. This may still pass, but it is a production risk and often hurts print quality.
Actions:
1. Regenerate figures with embedded TrueType or Type 1 fonts where possible.
2. Remove font substitutions that trigger warnings.
3. Verify final `pdffonts` output on the paper PDF.
Acceptance:
- No avoidable Type 3 fonts remain in the final paper PDF.
- Figure text is crisp in print and zoom.
Deliverables:
- regenerated figures
- `pdffonts` verification note

### P1-015 - Reduce caption overload
Priority: P1
Problem:
Figure and table captions are carrying too much narrative burden.
Actions:
1. Move interpretation into main text.
2. Keep captions focused on what is shown, units, and key caveats.
Acceptance:
- Captions are shorter and easier to scan.
Deliverables:
- revised captions across the paper

### P2-001 - Clean the README and remove contradictory execution paths
Priority: P2
Problem:
`README.md` still references `src/main.py` and two code universes, which conflicts with the active modular pipeline and confuses reproduction.
Actions:
1. Rewrite the README around the chosen canonical benchmark path.
2. Remove stale references to paths that do not exist.
3. Add a short submission runbook.
Acceptance:
- A new collaborator can reproduce the paper workflow from the README alone.
Deliverables:
- revised `README.md`

### P2-002 - Normalize source-file encoding
Priority: P2
Problem:
Some source files display mojibake in terminal output, which is a reproducibility and editing risk even if the compiled PDF currently renders acceptably.
Actions:
1. Normalize text files to UTF-8 without hidden encoding drift.
2. Rebuild the paper after normalization.
Acceptance:
- No mojibake appears when reading text files in a normal UTF-8 terminal.
Deliverables:
- normalized `.tex` and `.md` sources

### P2-003 - Add a final submission checker
Priority: P2
Problem:
There is no one-command readiness check that validates the submission package.
Actions:
1. Add a script that checks:
   - page count
   - existence of canonical artifacts
   - claim-map pass
   - key tests pass
   - figure files exist
2. Fail loudly on any missing piece.
Acceptance:
- One command returns pass/fail for submission readiness.
Deliverables:
- `scripts/check_submission_ready.py` or equivalent

### P2-004 - Archive the frozen submission package
Priority: P2
Problem:
Even after fixes, the final package will not be auditable unless it is archived as one frozen bundle.
Actions:
1. Archive:
   - final PDF
   - source `.tex`
   - canonical figures
   - canonical JSON/CSV
   - requirements / environment note
   - git SHA
2. Tag the commit.
Acceptance:
- The exact submitted version can be reconstructed later without guesswork.
Deliverables:
- tagged commit
- `submission_bundle/` or equivalent manifest

### P2-005 - Run IEEE PDF eXpress and record the result
Priority: P2
Problem:
The paper is not fully "IEEE-ready" until the final PDF is actually checked by PDF eXpress.
Actions:
1. Run the final PDF through PDF eXpress after all content changes are complete.
2. Record pass/fail and the timestamp.
Acceptance:
- PDF eXpress pass is documented before submission.
Deliverables:
- short PDF eXpress receipt note in `paper/SUBMISSION_NOTES.md`

## 8. Suggested sequencing

Day 1
1. `P0-001`
2. `P0-002`
3. `P0-003`
4. `P0-005`

Day 2
1. `P0-004`
2. `P0-006`
3. `P0-007`
4. `P1-001`
5. `P1-002`
6. `P1-003`

Day 3
1. `P0-008`
2. `P1-004`
3. `P1-005`
4. `P1-006`
5. `P1-007`

Day 4
1. `P1-008`
2. `P1-009`
3. `P1-010`
4. `P1-011`
5. `P1-012`
6. `P1-013`
7. `P1-014`
8. `P1-015`

Day 5
1. `P2-001`
2. `P2-002`
3. `P2-003`
4. `P2-004`
5. `P2-005`

## 9. Submission gates

Gate A - Scientific truth
- One frozen benchmark definition
- One canonical artifact family
- One verified claim map

Gate B - Manuscript quality
- 6 pages
- readable figures
- consistent methodology description
- no obvious AI-compression prose

Gate C - Production readiness
- sync script works
- README is correct
- PDF eXpress passes
- submission bundle is archived

## 10. Final publish recommendation

Recommendation today: Do not submit from the current repository state.

Recommendation after roadmap completion: Yes, this is worth publishing at SGSMA 2026.

Reason:
The paper has real scientific value as a benchmark contribution on estimator robustness under composite IBR disturbances, and the topic is aligned with SGSMA. The current problem is not lack of value. The problem is that the repository and manuscript are not yet disciplined enough to make the value bulletproof.
