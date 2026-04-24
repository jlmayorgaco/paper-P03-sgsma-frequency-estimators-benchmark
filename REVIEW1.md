# REVIEW - Camera-Ready Audit (Paper, Science, Code, Structure, Outputs)

Date: 2026-04-23
Repository audited: `C:\Users\walla\Documents\Github\paper-P03-sgsma-frequency-estimators-benchmark`
Reviewer mode: strict technical audit (no adulation)

## 1) Executive Verdict

**Current status: NOT submission-ready for a defensible camera-ready package.**

The manuscript quality has improved versus the reviewed version, but the project currently has a **reproducibility and provenance break** between:
- paper claims/tables,
- benchmark code being actively changed,
- and multiple conflicting output sets (`results0/*` vs `tests/montecarlo/outputs/*`).

The paper can likely be made submission-ready quickly, but only after freezing one canonical benchmark path and regenerating artifacts from that frozen code.

---

## 2) Scope and Evidence Reviewed

I reviewed:
- Reviewed PDF from TPC cycle:
  - `C:/Users/walla/Downloads/Paper___Reviewed__Benchmarking_Dynamic_Frequency_Estimators_for_Low_Inertia_IBR_Grids__A_Trade_off_Analysis_of_Latency_vs__Robustness.pdf`
- Current LaTeX and compiled paper:
  - `paper/index.tex`
  - `paper/Sections/C1_Introduction/main.tex`
  - `paper/Sections/C2_Related_Work/main.tex`
  - `paper/Sections/C3_Methods/main.tex`
  - `paper/Sections/C4_Simulation_Results/main.tex`
  - `paper/Sections/C5_Conclusions/main.tex`
  - `paper/index.pdf` (rebuilt during audit)
- Benchmark artifacts:
  - `results0/figures_estimatores_benchmark_test5/benchmark_results.json`
  - `results0/figures_estimatores_benchmark_test5/paper_ready_numbers.txt`
  - `tests/montecarlo/outputs/benchmark_full_report.json`
- Key changed code:
  - `src/estimators/esprit.py`
  - `src/estimators/prony.py`
  - `src/scenarios/ibr_power_imbalance_ringdown.py`
  - `tests/montecarlo/test_dedicated_smoke_test.py`
  - `tests/montecarlo/generate_mega_dashboards.py`
- Build/test checks executed:
  - `pdflatex/bibtex` full cycle on `paper/index.tex`
  - `pytest src/tests -q` (11 passed)
  - `pytest tests/scenarios/ibr_power_imbalance_ringdown/test_ibr_power_imbalance_ringdown_plot.py -q` (1 passed)
  - `pytest tests/estimators -q` (fails because `torch` missing)

---

## 3) Reviewer-Philosophy Alignment (Against Original Critiques)

## 3.1 Clearly improved

1. **IBR motivation and failure modes are better explained** (phase jumps, RoCoF, harmonics):
   - `paper/Sections/C1_Introduction/main.tex`
2. **Tone is less over-claiming than the first reviewed draft**:
   - no explicit "establishes a rigorous framework" claim in current abstract/body
3. **Method count inconsistency largely fixed in manuscript text**:
   - now states twelve estimators in intro/methods/results
4. **DFT vs TFT conceptual distinction is now explicit**:
   - `paper/Sections/C3_Methods/main.tex`
5. **IEC/IEEE 60255-118-1 usage and non-normative framing are clearer**:
   - `paper/Sections/C3_Methods/main.tex`
   - `paper/Sections/C4_Simulation_Results/main.tex` Fig.2(g) caption
6. **Units/definitions are much clearer than before** (Table units, trip-risk definition, etc.).

## 3.2 Still weak or risky

1. **Scientific traceability is currently broken by artifact/code divergence** (details below).
2. **Template compliance is questionable** due non-standard layout compression packages/settings.
3. **Reproducibility is weakened by active benchmark drift and multiple incompatible output universes.**

---

## 4) Critical Findings (Blockers)

### [C1] No single canonical pipeline currently governs paper numbers

Evidence:
- AGENTS protocol expects canonical run path under `src/main.py` and canonical output under `src/figures_estimatores_benchmark_test5/...`.
- This repository currently has no `src/main.py` and no `src/figures_estimatores_benchmark_test5` directory.
- Paper-facing numbers appear tied to `results0/figures_estimatores_benchmark_test5/*` (timestamp 2026-03-30), while active development now produces `tests/montecarlo/outputs/*` (timestamp 2026-04-20+).

Impact:
- You cannot prove that current manuscript reflects current code.
- This is the biggest risk for any post-acceptance scrutiny.

### [C2] Paper values are not consistent with the newest benchmark output set

Evidence:
- Paper states scenario E values in `paper/Sections/C4_Simulation_Results/main.tex:67-100`:
  - RA-EKF RMSE 159 mHz, trip 0.081 s, CPU 88.5 us
  - SRF-PLL trip 0.353 s
- In `tests/montecarlo/outputs/benchmark_full_report.json`:
  - run config says `n_mc_runs = 60`
  - scenario set is 32 scenarios (not 5)
  - scenario key is `IBR_Multi_Event` (not `IBR_MultiEvent_Classic`)
  - contains estimators including `LKF2` (paper excludes LKF)

Impact:
- Even if the paper text is internally coherent, it is disconnected from the currently active benchmark branch.

### [C3] "Canonical" results file used for paper has internal claim-verification failures

Evidence from `results0/figures_estimatores_benchmark_test5/benchmark_results.json`:
- `paper_claims_verification.verified = false`
- `n_failures = 2`
- Failures include mismatches in `IBR_MultiEvent_Classic` expected values.

Impact:
- You have a machine-readable signal saying "paper claims not verified" in the same artifact family used for paper numbers.
- This must be resolved before submission confidence.

### [C4] Benchmark code re-introduces estimators/scenarios that manuscript/execution protocol says to exclude

Evidence in `tests/montecarlo/test_dedicated_smoke_test.py`:
- `N_TRIALS_TUNING = 500` and `N_MC_RUNS = 100` at lines 91-92.
- 32-scenario set built at lines 120-183 (vs paper's five scenarios).
- LKF/LKF2 are in family mapping and search spaces (lines 251-265 and 311+).
- Scenario alias usage includes `IBR_Multi_Event` (line 1284), conflicting with strict `IBR_MultiEvent_Classic` naming rules.

Impact:
- Scientific protocol drift: benchmark no longer matches declared camera-ready experiment design.

---

## 5) High-Severity Findings

### [H1] ESPRIT implementation has deterministic-equivalence and reproducibility issues

Evidence in `src/estimators/esprit.py`:
- Random dither added inside core each call: line 23.
- Decimation trigger uses local `i % stride` (line 76).
- `step()` wraps a 1-sample vector each call, so `i` resets to 0 every call (line 129), meaning scalar path executes core every sample, while vector path executes every `stride` samples.

Impact:
- `step` and `step_vectorized` are not algorithmically equivalent.
- Monte Carlo repeatability is weakened by internal random perturbation not externally controlled.
- Violates strict scientific reproducibility expectations.

### [H2] Prony silently masks critical numerical failures

Evidence in `src/estimators/prony.py`:
- Global suppression of `NumbaWarning` (lines 6-10).
- Broad `except Exception` around core numerical steps (lines 64-75, 198+), fallback returns last known frequency.

Impact:
- Failure modes can be hidden and look "stable" in outputs.
- This is dangerous in benchmarking; hidden numerical collapses should be surfaced and counted.

### [H3] Manuscript uses heavy layout-compression hacks likely non-compliant with IEEE template spirit

Evidence in `paper/index.tex`:
- Explicit "extreme space optimization" block (line 31 onward).
- Manual spacing changes, `titlesec`, and `caption` package usage (`\usepackage[compact]{titlesec}` line 46; `\usepackage[font=small,skip=2pt]{caption}` line 56).
- Build warning confirms `caption` unknown class defaults and `titlesec` non-standard warning.

Impact:
- Camera-ready desk checks can reject or request changes if layout manipulation is considered aggressive/non-template.

### [H4] Paper build emits template and structure warnings that should be cleaned pre-submission

From `paper/index.log` after rebuild:
- `Package titlesec Warning: Non standard sectioning command \subparagraph`
- `Package caption Warning: Unknown document class (or package)`
- `** WARNING: \IEEEPARstart is locked out when in conference mode`
- Output: 7 pages

Impact:
- Not fatal compile errors, but indicates avoidable template risk and final-polish gaps.

---

## 6) Medium Findings

### [M1] Mixed artifact generations and stale figure risk

Evidence:
- Figure PDFs in paper folder were last regenerated on 2026-04-01.
- Core code edits exist through 2026-04-18.
- Benchmark outputs in `tests/montecarlo/outputs` updated 2026-04-20/23.

Impact:
- Visuals may not represent latest code state.

### [M2] CPU-time claims may not be comparable across output families

Evidence:
- Paper reports CPU in tens/hundreds of microseconds.
- New `benchmark_full_report.json` aggregated metrics show sub-microsecond values for several estimators in some scenarios, while heavy estimators show milliseconds.

Impact:
- Suggests timing methodology changed or is inconsistent; this requires a single harmonized timing protocol before claiming computational feasibility.

### [M3] Test suite partially blocked by missing hard dependency (`torch`)

Evidence:
- `pytest tests/estimators -q` fails collecting PI-GRU tests due `ModuleNotFoundError: torch`.

Impact:
- Reproducibility for full estimator set is environment-sensitive without clear dependency gating.

### [M4] Repository documentation and encoding hygiene are weak

Evidence:
- `README.md` displays mojibake and outdated tree (`src/benchmark`, `latex/`, `results/`) not matching current repository layout.

Impact:
- Increases onboarding errors and accidental use of wrong execution path.

---

## 7) Positive Progress (What is genuinely better)

1. Manuscript is significantly more technical and reviewer-aware than the reviewed PDF.
2. The distinction between structural latency and stochastic lag is explicitly stated and scientifically useful.
3. Non-normative compliance framing is better and avoids over-certification language.
4. Most previously ambiguous terms now have definitions/units.
5. The paper is closer to expert-reader expectations in methods framing.

---

## 8) Is the Current PDF/LaTeX Better Than the Reviewed Version?

**Yes, materially better in clarity and scientific framing.**

But:
- "Better text" is not enough for camera-ready defensibility.
- Right now, **paper-text quality is ahead of experiment provenance quality**.
- You need one reproducible frozen benchmark path and fresh synchronized artifacts before final submission.

---

## 9) Submission Readiness Decision

### Decision: **NO-GO (for now)**

Reason:
- Critical reproducibility/provenance gaps (C1-C4).

### Fast path to GO

1. Freeze one benchmark pipeline (single entrypoint, single scenario set, single estimator list).
2. Regenerate one authoritative results artifact.
3. Re-link every manuscript hard number to that artifact.
4. Rebuild figures from the same run.
5. Remove or minimize non-standard template compression hacks.
6. Final language pass (grammar + concision + consistency).

---

## 10) Concrete 48-Hour Recovery Plan

1. **Protocol freeze commit**
   - Lock scenario set to the exact five paper scenarios.
   - Lock estimator list to paper-facing set (with explicit inclusion/exclusion policy).
   - Lock MC count and seeds.

2. **Single-run regeneration**
   - Produce one fresh machine-readable artifact (with timestamp and git commit hash).
   - Produce figures from same run only.

3. **Automated manuscript consistency check**
   - Script that checks all hardcoded paper numbers vs JSON keys.
   - Fail build if mismatch > tolerance.

4. **Template compliance cleanup**
   - Remove `titlesec` and custom `caption` package unless strictly allowed by SGSMA instructions.
   - Keep IEEEtran defaults as much as possible.

5. **Language polish**
   - Tighten long sentences, reduce clause density, normalize terminology (trip-risk, structural latency, benchmark scope).

---

## 11) Final Advisor Note

You are close on scientific narrative quality. The real risk is not language; it is **auditability**.
If a reviewer or chair asks "which exact run produced Table/Fig X?", your current repo state cannot answer that cleanly. Fix that, and your camera-ready will be strong.




POST-AUDIT ANALYSIS & REFINED CODEX EXECUTION PROMPT
SGSMA 2026 — Submission Hardening, Phase 2
Date: 2026-04-23
Input documents: Reviewer 1 & 2 comments, Codex Roadmap (ROADMAP.md), Revised paper (index.pdf, 7 pages)

PART A: WHAT THE ROADMAP GOT RIGHT
The Codex audit was thorough and honest. Its core verdict — NO-GO in current state — is correct. The scoring (Submission readiness: 3/10, Reproducibility: 3/10) is harsh but fair. The ticket structure (P0/P1/P2) is well-prioritized. Specifically:

P0-001 (6-page limit): Correctly identified as the single hardest blocker.
P0-002 (experiment universe split): Correctly flags that the paper describes 5 scenarios + 12 estimators while the codebase has 32 scenarios + 18 estimators.
P0-007 (Scenario F leakage): Correctly caught — Scenario F appears in Figure 1 and Section II.B but is described as "planned extension" with no results. This wastes space and weakens discipline.
P1-011 (AI-compression prose): Good catch. The revised paper has improved but still shows patterns.
P0-006 (claim traceability): The most important infrastructure fix.


PART B: WHAT CHANGED BETWEEN V1 AND THE REVISED PAPER
I've compared the original reviewed paper against the revised index.pdf. Here's what improved and what's still open.
B.1 — Issues FIXED (fully or substantially) in the revision
Reviewer IssueStatusEvidence in Revised PaperOverclaimed conclusion ("rigorous benchmarking framework")✅ FIXEDConclusion now says "do not fully characterize" — humble framing per R1DFT vs. TFT conflation✅ FIXEDTFT clearly described as "2nd-order Taylor-Fourier model, Hann-weighted" with distinct properties. Separate from IpDFT in Table I and text.EKF+RoCoF novelty overclaim✅ FIXEDText now says "extends the 3-state EKF" and cites [10],[11] as prior work. Contribution explicitly scoped to Innovation-Driven Covariance Scaling + Event Gating."Precision" vs. "accuracy"✅ FIXEDNo misuse detected in revised paper.IEEE C37.118.1 → IEC/IEEE 60255-118-1✅ FIXEDConsistently uses IEC/IEEE 60255-118-1 throughout.Variable 'e' undefined✅ FIXEDDefined in Section II.C: "Let e(t) = f̂(t) − f_true(t)"Table II units✅ SUBSTANTIALLY FIXEDCaption now includes units for Kp, Ki, R, Nc, Nsmooth. Some gaps remain (see below).Noise σ inconsistency (0.005 vs 0.05)✅ FIXEDClear distinction: σ = 0.005 pu is Gaussian noise for Scenario D; ≤0.05 pu is impulsive spike amplitude for Scenario E."Harmonic rich steady state" definition✅ FIXEDScenario E now specifies: "5th: 4%, 7th: 2%; THD ≈ 4.5%, IEEE 519 compliant"Section II.B last sentence open✅ FIXEDSection now ends with complete sentence about IBR-specific disturbances.Table I "Phase Resp." definition✅ FIXEDCaption defines: "S = Settles without cycle-slip, D = Cycle-slip susceptible"Time-stamp stability column✅ FIXEDColumn removed entirely; no residual references.Monte Carlo added✅ DONEN=30 MC runs, seeds 2000–2029, reported as mean ± std in Table III.IpDFT delay explanation✅ FIXEDBody text + caption both explain structural Nc/f0 latency vs. stochastic lag.Algorithm count consistency✅ FIXEDConsistently says "twelve estimators from five algorithmic families"
B.2 — Issues PARTIALLY FIXED (need more work)
IssueCurrent StateWhat's Still NeededIBR motivation too vague (R1)Improved — mentions harmonics, phase discontinuities, RoCoF.Still lacks concrete real-world incident citations (UK 9-Aug-2019, ERCOT 2021, AEMO). Add 1–2 sentences + refs.Reference list outdated (R2)Only 14 references, several unchanged from V1.Missing key IEEE TIM authors: Pegoraro, Toscani, Macii, Fontanelli, Derviškadić, Paolone (2017+). Add 3–5 critical refs. BUT: 6-page limit constrains this."Hyperparameters" terminologyNot found in revised text (likely fixed).Verify no residual uses. The word "hyperparameter" appears once in the conclusion — confirm removal.Table III completeness (R2)Table III now has 9 methods in main body. VFF-RLS, RLS, TKEO, PI-GRU in Table IV notes/captions.Acceptable if exclusion policy is stated. Currently: TKEO excluded "≥7800 mHz"; PI-GRU excluded with CPU note. Make exclusion policy ONE clean sentence in Methods.Computational cost procedure (R2)Improved: "mean time per sample over 20 timed repetitions, Python 3.13, Intel Core Ultra 5 125H, single-threaded"Adequate for conference. Consider adding warm-up note if space permits.Amplitude step phase angle (R2: only 0° tested)Unchanged. Step test uses zero phase angle only.Add ONE sentence acknowledging this limitation. "Scenario A applies the step at zero phase angle; non-zero incidence angles may introduce additional frequency transients not evaluated here."Scope overload (R2: too many things)Slightly improved — RA-EKF contribution better scoped.The paper still does review + comparison + framework + EKF variant. At 6 pages this is tight. The solution is NOT to cut scope but to ensure each part is crisp and doesn't overpromise.
B.3 — Issues NOT FIXED (still open)
IssueStatusAction RequiredPaper is 7 pages❌ CRITICAL BLOCKERMust cut to 6. See cutting plan below.Scenario F in Figure 1 and text❌ BLOCKERScenario F takes ~15% of Figure 1 and ~4 lines of text. Remove from both — immediate space savings.Figure 2 subfigure ordering (R2)❌ NOT FIXEDSubfigures still appear as (a,b,c,d) top-left to right, then (g,h,e,f) bottom. Not in reading order. Fix layout.Figure 2 readability at conference scale❌ FLAGGED BY CODEX8 panels in one figure. Legends and axis labels are small. Reduce to 4–6 panels maximum.Significant digits in tables⚠️ PARTIALLYTable III uses 3 s.f. with ± std. Good. But check: are "±0" entries meaningful? (They mean std < 0.5 mHz.) Clarify with footnote — already done partially ("±0: std < 0.5 mHz").Figure 2g compliance heatmap includes Scenario F/PFR column❌ BLOCKERIf Scenario F results don't exist, this column must be removed. Currently shows "PFR" column — all F's. Remove.VFF-RLS/RLS filter specification (R2)⚠️ PARTIALText says "AR(2) bandpass pre-whitening" but no filter coefficients or cutoff frequencies. Add one sentence: "AR(2) bandpass pre-whitening centered at f₀ = 50 Hz with 3 dB bandwidth of X Hz" or reference the design.Q_k update details (R2)⚠️ PARTIALInnovation-Driven Covariance Scaling described but σ̂_ν estimation method not specified. Is it a running estimate? Exponential moving average? Window? Add: "σ̂_ν estimated via exponential moving average with forgetting factor α = 0.99" (or whatever the code uses).Prose compression patterns⚠️ STILL PRESENTPhrases like "stress-oriented benchmark", "performance ceiling per estimator–scenario pair", "beyond the standard scope" appear multiple times. Each should appear at most once.Grammar issues⚠️ SOME REMAINSee specific list below.32.5 Hz explained before use✅ NOW OKTable I caption explains: "representative of IBR converter control interactions [3]"

PART C: THE 7→6 PAGE CUT PLAN
This is the most critical action. Here's a concrete cutting strategy:
Immediate space gains (targeting ~1 full page):

Remove Scenario F entirely (~0.25 pages)

Delete Scenario F panel from Figure 1 (saves ~1/6 of figure height)
Delete Scenario F text in Section II.B (~4 lines)
Delete "PFR" column from compliance heatmap (Fig 2g)
Delete Fig 2c which shows Scenario F results


Reduce Figure 2 from 8 to 5 panels (~0.3 pages)

KEEP: (a) Phase Jump Robustness, (b) Ramp Tracking, (e) Cost vs Risk Pareto, (g) Compliance Heatmap (without PFR column), (h) Steady-State Error
REMOVE: (c) Primary Freq Response (redundant with Scenario F removal), (d) Multi-Event unstable (interesting but not essential — describe in text instead), (f) Radar chart (nice but lowest information density)


Compress Table I (~0.15 pages)

Remove "Comp. Order" column (well-known, not used in analysis)
Consider: merge into body text the legacy baselines (RLS, TKEO) since they're excluded from headline results anyway


Tighten prose (~0.15 pages)

Introduction: currently ~1.1 columns. Target: 0.9 columns. Cut repeated motivation clauses.
Conclusion: currently ~0.7 columns. Target: 0.5 columns. Remove restated results — point to tables.
Section II.A estimator descriptions: each currently 2–3 sentences. Tighten to 1–2 where possible.


Reduce Table V (~0.1 pages)

Already partially duplicated in Table IV (CPU column). Consider merging into Table IV or making a compact inline list.



DO NOT:

Use \vspace{-Xpt} hacks
Reduce font below template minimum
Remove the Monte Carlo results (they address a key reviewer concern)
Remove Table III (core quantitative evidence)


PART D: REFINED CODEX EXECUTION PROMPT (PHASE 2)
Feed this to Codex to execute the remaining fixes:

markdown# CODEX PHASE 2: Execute Submission Hardening

## CONTEXT
The first Codex audit produced ROADMAP.md. The authors have revised the paper (index.pdf, 7 pages). This prompt executes the remaining fixes to reach a submit-ready 6-page paper.

## GROUND RULES
1. The paper MUST be exactly 6 pages when compiled. No layout hacks.
2. Scenario F must be completely removed (text, Figure 1 panel, Figure 2 panels, heatmap column).
3. Every numerical claim must trace to the canonical MC run (N=30, seeds 2000–2029).
4. No new estimators, scenarios, or analyses. This is a FREEZE + POLISH operation.
5. IEEE PES two-column format is non-negotiable.

## TASK LIST (execute in order)

### TASK 1: Remove Scenario F completely
Files: paper/Sections/C3_Methods/main.tex, figure generation scripts, Fig 1 script, Fig 2 script

Actions:
- Delete Scenario (F) definition from Section II.B
- Delete the "(F) Primary Frequency Response" paragraph entirely
- Regenerate Figure 1 with only Scenarios A–E (5 rows, not 6)
- Remove "PFR" column from the compliance heatmap (Fig 2g)
- Remove panel (c) "Prim. Freq. Response" from Figure 2
- Search entire .tex for "Scenario F", "PFR", "Primary Frequency Response", "planned extension", "future work" related to Scenario F. Remove or reword.
- In the Conclusion, if there's a sentence about Scenario F as future work, either remove it or replace with a generic "additional disturbance profiles" note.

Acceptance: No mention of Scenario F remains in any figure, table, or text. Figure 1 has exactly 5 scenario rows (A–E).

### TASK 2: Reduce Figure 2 to 5 panels
Files: Figure 2 generation script, paper/Sections/C4_Results/main.tex

Keep panels: (a) Phase Jump Robustness, (b) Ramp Tracking, (e) Cost vs Risk Pareto, (g) Compliance Heatmap (5 scenario columns: Step, Ramp, Mod, Isl, Multi-E), (h) Steady-State Error.

Remove panels: (c) Primary Freq Response, (d) Multi-Event unstable, (f) Radar chart.

For removed panel (d): add one sentence to Section III.B.2: "Legacy baselines (TKEO, RLS) and VFF-RLS diverge under the composite multi-event sequence; TKEO reaches RMSE ≥ 7800 mHz and is excluded from headline tables."

For removed panel (f): the radar chart information is qualitative and already conveyed by Table I. No replacement text needed.

Rearrange remaining 5 panels in a clean 2×2 + 1 or 3+2 layout. Ensure all axis labels are ≥ 8pt font at final print size.

Acceptance: Figure 2 has exactly 5 labeled panels (a)–(e), in reading order. All text is readable at printed conference column width.

### TASK 3: Compress Table I
Files: paper/Sections/C2_RelatedWork/main.tex or wherever Table I lives

Remove the "Comp. Order" column. The computational complexity is well-known for each family and is not referenced in the analysis.

Move legacy baselines (RLS, TKEO) to a footnote: "†Legacy baselines RLS (fixed λ) and TKEO are included for reference; both are excluded from headline results due to divergence under IBR disturbances (TKEO: RMSE ≥ 7800 mHz; RLS: covariance blow-up)."

This saves one column width and two rows.

Acceptance: Table I has 10 estimator rows (not 12), no "Comp. Order" column, and a footnote explaining legacy exclusions.

### TASK 4: Merge or compress Table V
Files: paper/Sections/C4_Results/main.tex

Option A (preferred): Merge Table V CPU data into Table IV by adding a column. Table IV already has a CPU column. Remove Table V entirely and fold "Deployment Class" and "Target HW" into the Table IV caption or a footnote.

Option B: If Table IV would be too wide, convert Table V into an inline sentence: "Per-sample costs (Python baseline): SRF-PLL 16.7 µs, EKF 29.5 µs, SOGI-FLL 41.7 µs, TFT 45.6 µs, RA-EKF 88.5 µs, IpDFT 110 µs, Koopman 120 µs, UKF 215 µs, PI-GRU ≈154 ms."

Acceptance: One fewer table. No loss of information.

### TASK 5: Tighten Introduction prose
Files: paper/Sections/C1_Introduction/main.tex

Target: reduce Introduction from ~1.1 columns to ~0.9 columns.

Specific cuts:
- The sentence "Unlike synchronous machines, Inverter-Based Resources (IBRs) regulate power electronically, decoupling the primary energy source from electrical frequency and increasing the impact of harmonics, phase discontinuities, and nonstationary disturbances on protection and dynamic security assessment [1], [3], [4]." → This is good but repeats the previous sentence's point. Merge into one.
- The three failure modes (phase discontinuities, RoCoF, harmonics) are listed twice: once in paragraph 1, once in Section I.A. Keep only ONE list, in paragraph 1.
- Section I.A ("Algorithmic Landscape and Prior Art") overlaps with Table I and Section II.A. Cut Section I.A to 3 sentences max: one sentence per estimator family + one sentence about the gap. The details belong in Section II.

Add (if space): ONE concrete real-world incident citation. Example: "The 9 August 2019 GB frequency event demonstrated that RoCoF can breach 1 Hz/s within 500 ms following the loss of two large generators [ref], underscoring the need for sub-cycle tracking capability." This addresses Reviewer 1's request for broader motivation.

### TASK 6: Tighten Conclusion
Files: paper/Sections/C5_Conclusions/main.tex

Target: reduce from ~0.7 columns to ~0.5 columns.

Specific fixes:
- Opening paragraph restates what the paper did. Cut to ONE sentence: "Standard IEC/IEEE 60255-118-1 tests do not fully characterize estimator behavior in converter-dominated grids."
- Keep the three numbered findings but shorten each to 2–3 sentences (currently 4–5 each).
- Remove the PI-GRU paragraph if already covered in Section III.
- End with: "This benchmark is simulation-based; hardware-in-the-loop validation on physical relay platforms is an essential next step. Simulation code and tuning grids are available from the authors."

De-duplicate phrases that appear in both abstract and conclusion. If a phrase appears in the abstract, use DIFFERENT wording in the conclusion.

### TASK 7: Fix Figure 2 subfigure ordering
Files: Figure 2 generation script

After reducing to 5 panels, label them (a)–(e) in strict left-to-right, top-to-bottom reading order. Update all body text cross-references (e.g., "Fig. 2g" becomes "Fig. 2d" if the heatmap moves).

### TASK 8: Add missing technical details
Files: paper/Sections/C2_RelatedWork/main.tex, paper/Sections/C3_Methods/main.tex

Add (each is ONE sentence, no more):

1. VFF-RLS filter: "The RLS and VFF-RLS estimators use a 2nd-order autoregressive bandpass pre-whitening filter centered at the nominal frequency (50/60 Hz)."

2. Q_k estimation: In the RA-EKF description, after "inflating Qk during transients": add "σ̂_ν is estimated online via exponential moving average of |ν_k| with forgetting factor α = [value from code]."

3. Amplitude step limitation: After Scenario A description: "The zero-incidence angle avoids phase discontinuities at the step onset; non-zero angles would introduce additional transients not evaluated here."

4. Exclusion policy: Add one sentence to Section II.A or II.C: "Methods with RMSE exceeding 5000 mHz on any scenario (TKEO) or computational cost prohibitive for real-time deployment (PI-GRU: ≈154 ms/sample) are reported in table footnotes rather than headline results."

### TASK 9: Grammar and style pass
Files: all .tex files

Systematic fixes:
- Search and limit to ONE occurrence each: "stress-oriented benchmark", "performance ceiling", "beyond the standard scope", "relay-class"
- Ensure consistent hyphenation: "low-inertia" (always hyphenated as adjective), "model-based" (always), "loop-based" (always), "trip-risk" (always), "phase jump" (never hyphenated — it's a noun phrase), "grid search" (never)
- Check every acronym defined on first use in body AND separately in abstract
- Remove any remaining "hyperparameter" → replace with "tuning parameter"
- Verify no sentence in abstract or results carries more than one major claim
- Fix: caption of Figure 1 still says "(F) Primary Freq. Response" — remove after Task 1

### TASK 10: Reference list improvements
Files: paper/references.bib

Priority additions (if space allows — 6-page limit may constrain):
1. A. Derviškadić, P. Romano, M. Paolone, "Active distribution network state estimation using PMUs and micro-PMUs," IEEE TIM, 2018 — relevant for measurement context
2. D. Macii, D. Petri, "Phase-based DFT synchrophasor estimation," IEEE TIM, 2014 — addresses R2's concern about DFT-based literature
3. The 9-August-2019 GB event report (OFGEM/National Grid ESO) — addresses R1's real-world motivation request

Priority removals (only if needed for space):
- [6] EPRI webinar presentation — weakest reference (not peer-reviewed, not strictly necessary)
- [12] arXiv preprint "under review; no peer-reviewed venue confirmed" — vulnerability flagged by the authors themselves

Do NOT remove [7] Roscoe (phase steps), [9] Platas-García (TFT), [10] Ferrero (EKF), [13] Golestan (PLL) — these are core technical references.

### TASK 11: Code-paper consistency verification
Files: all scripts, all .tex

For each hard number in the paper, verify against canonical output:
- Table III: all RMSE ± std values must match the MC run output CSV
- Table IV: all Ttrip values must match
- Table V: all CPU timings must match
- Figure captions: "TripTime = 0.6 ms" vs body text "Ttrip = 0" — is this inconsistent? (Fig 2a says 0.6 ms but Table III says Ttrip = 0.000 for RA-EKF Scenario D.) Investigate: the 0.6 ms may be from V1 and not updated. If the MC run gives Ttrip = 0, update the figure annotation.
- Caption says "RA-EKF: TripTime = 0.6 ms" but body says "Ttrip = 0; peak 365 mHz < 0.5 Hz" → These are contradictory. RESOLVE: either the figure annotation is wrong (old run) or the body text is wrong. Check the canonical run.

### TASK 12: Final page count verification
After all changes, compile and verify:
- `pdfinfo paper/index.pdf` reports Pages: 6
- No non-standard spacing packages
- All figures and tables fit within column/page boundaries
- No orphan lines or widow paragraphs
- References do not spill onto a 7th page

### TASK 13: Claim traceability file
Create: artifacts/claim_map.csv

Columns: claim_text | location_in_paper | source_artifact | key_or_row | value | tolerance

Example rows:
"RA-EKF RMSE = 39.3 ± 2.40 mHz" | Table III, Scenario D | benchmark_full_report.csv | RA-EKF_ScenD_RMSE_mean | 39.3 | ±0.5 mHz
"SRF-PLL Ttrip = 0.353 s" | Table IV | benchmark_full_report.csv | SRF-PLL_ScenE_Ttrip_mean | 0.353 | ±0.005 s

### TASK 14: Final submission checklist script
Create: scripts/check_submission_ready.py

Checks:
1. PDF page count == 6
2. All canonical artifact files exist
3. claim_map.csv validates against canonical outputs (within tolerance)
4. No "Scenario F" or "PFR" text in any .tex file
5. No "hyperparameter" in any .tex file
6. Figure file count matches expected
7. All .tex files are valid UTF-8

Output: PASS or FAIL with specific failure reasons.

---

## PRIORITY SUMMARY

| Priority | Tasks | Estimated Effort |
|---|---|---|
| BLOCKING (do first) | 1, 2, 3, 4, 12 | ~4 hours |
| QUALITY (do second) | 5, 6, 7, 8, 9, 11 | ~3 hours |
| INFRASTRUCTURE (do third) | 10, 13, 14 | ~2 hours |

Total estimated effort: ~9 hours of focused work.

After completion, the paper should be:
- 6 pages exactly
- All reviewer concerns addressed or explicitly acknowledged as limitations
- All claims traceable to canonical MC outputs
- Clean, professional IEEE PES formatting
- No AI-compression prose patterns
- Ready for PDF eXpress check

PART E: SPECIFIC GRAMMAR ISSUES FOUND IN REVISED PAPER
Line-level issues spotted in index.pdf:

Abstract: "A stress-oriented benchmark evaluates twelve estimators" — passive construction is fine for abstract but the phrase "stress-oriented benchmark" appears 3 more times in the paper. Limit to 1 total.
Section I, para 1: "Frequency deviations and Rate of Change of Frequency (RoCoF) can exceed protection thresholds within milliseconds" — good, but "Rate of Change of Frequency" should not be capitalized (it's not a proper noun). IEEE style: "rate of change of frequency (RoCoF)".
Section I, para 2: "Each estimator family balances latency and robustness differently." — This is a topic sentence followed by a list that largely repeats Table I. If cutting for space, this paragraph is a candidate.
Section I, contributions list: "(1) a benchmark of twelve estimators from five algorithmic families under five stress scenarios" — this partially restates the abstract. Tighten.
Section II.A: "The SOGI-FLL departs from the textbook formulation [13] through two additions" — the phrase "departs from the textbook formulation" is informal for IEEE. Suggest: "The evaluated SOGI-FLL includes two modifications relative to [13]:"
Section II.A: "A 2π-scaling error in the standard adaptation law (gain mis-scaled by ≈377×) was corrected before evaluation" — This is important methodological transparency. Keep. But note: "mis-scaled" should be "miscaled" or better: "erroneously scaled".
Section II.B, Scenario D: "THD ≈ 4.5%, IEEE 519 compliant" — This parenthetical appears identically in Scenarios D and E. Say it once, then reference: "both within IEEE 519 limits".
Section II.B, Scenario F: DELETE ENTIRELY (Task 1).
Section II.C: "establishing the performance ceiling per estimator–scenario pair" — this exact phrase appears in the abstract too. Use it only once.
Section III.A: "SRF-PLL achieves RMSE = 0.130 mHz (Modulation)" — inconsistent: sometimes writes "RMSE = X mHz", sometimes "RMSE X mHz". Standardize to "RMSE = X mHz" throughout.
Section IV (Conclusion): "the RA-EKF (augmented with an explicit ω̇ state, Innovation-Driven Covariance Scaling, and Event Gating) achieves Ttrip = 0 under Composite Islanding" — this sentence is 3 lines long. Split.
Table IV caption: "(TRIP PROXY; NO IEC COMPLIANCE CLAIMED)" — good caveat. Keep.
Table V footnote: "Production deployments with TorchScript/ONNX and batched windowing would reduce per-sample cost by ≥100×" — This is speculation in a results table footnote. Either cite evidence or soften to "may reduce".
Throughout: The em-dash (—) is used correctly but inconsistently with spacing. IEEE style: no spaces around em-dashes. Verify.


PART F: KEY CONTRADICTIONS TO RESOLVE

RA-EKF Ttrip in Scenario D:

Figure 2a annotation: "TripTime = 0.6 ms"
Table III body text: "Ttrip = 0"
Section III.B.1: "Ttrip = 0; peak 365 mHz < 0.5 Hz"
Resolution: If peak = 365 mHz and threshold = 500 mHz, then Ttrip should indeed be 0. The figure annotation "0.6 ms" is from V1 (where peak was 506 mHz). UPDATE THE FIGURE ANNOTATION to match current results.


Monte Carlo count:

Paper says N = 30
Codex roadmap says code has N_MC_RUNS = 100, legacy outputs show 60
Resolution: Decide. If final canonical run is N=30, ensure code is set to 30 and all artifacts match. If you've already run N=100, consider using those (more statistically robust) but update ALL text.


Ramp rate:

V1 paper: +5 Hz/s
Revised paper: +3.0 Hz/s (IEEE 1547-2018 Category III compliant)
Resolution: This is an intentional change (good — better justified). Verify ALL figures and tables reflect +3.0 Hz/s, not +5 Hz/s.


Number of estimators in Table I vs. Table III:

Table I: 12 estimators
Table III: 9 estimators in the main body
Resolution: Acceptable IF the exclusion policy is stated clearly. Currently partially stated in footnotes. Make it explicit in one sentence.




PART G: WHAT TO TELL CODEX NEXT
Copy PART D (the refined execution prompt) into Codex. It contains 14 concrete, ordered tasks with acceptance criteria. After Codex executes, verify:

PDF is 6 pages (pdfinfo)
No "Scenario F" anywhere
Figure 2 has 5 panels in order
RA-EKF TripTime annotation matches body text
claim_map.csv exists and validates
All grammar fixes applied

Then bring the result back here and I'll do a final pre-submission review

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
