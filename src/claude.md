# CLAUDE.md (src/ — REDIRECT)
# Canonical CLAUDE.md is at the repository root: ../CLAUDE.md
# This file is outdated. Do not use it as the operating protocol.

> Canonical operating document for Claude Code while auditing, fixing, rerunning, and reconciling the benchmark and the SGSMA camera-ready paper.
>
> **Primary principle:** if a claim, table, plot, or sentence in the paper disagrees with the code or the fresh artifacts, **the paper must change**.  
> We do **not** protect legacy text. We protect **truth, reproducibility, and scientific validity**.

---

## 0. Mission

This repository exists to produce a **scientifically valid, reproducible benchmark** for dynamic frequency estimators in low-inertia / IBR power systems, and to support a paper submission to SGSMA.

The immediate goal is:

1. make the benchmark execution path canonical and trustworthy,
2. run the benchmark from the correct codebase,
3. regenerate result artifacts,
4. reconcile tables, claims, figures, and manuscript text with the **actual** outputs,
5. submit only what is **true**.

---

## 1. Non-negotiable rules

### 1.1 Truth over consistency with old text
If new runs produce different values than the manuscript, then:
- update the manuscript,
- update tables,
- update claims,
- update figure captions,
- update narrative.

Never “massage” code or outputs to preserve legacy paper numbers.

### 1.2 One canonical execution path
The authoritative execution directory is:

`src/`

Any duplicate, stale, partial, or legacy copy outside the canonical path must be treated as suspect unless explicitly proven necessary.

### 1.3 No silent scientific drift
Any change that alters:
- tuning,
- estimator inclusion,
- metric definitions,
- warm-up / evaluation windows,
- latency accounting,
- scenario generation,
- paper-facing aggregation

must be documented in a short note or commit message.

### 1.4 Paper claims must be artifact-backed
Any numerical statement in the manuscript must be traceable to:
- a fresh result JSON/CSV,
- a script output,
- a reproducible analysis artifact,
- or a derived computation with explicit provenance.

### 1.5 “Looks good” is not a validation criterion
Passing syntax checks, imports resolving, and weights existing do **not** imply scientific validity.

---

## 2. Canonical repository assumptions

Unless proven otherwise, assume:

- Canonical benchmark runner lives under: `code_sim_and_results/`
- Main execution entrypoint: `code_sim_and_results/main.py`
- Fresh benchmark outputs ultimately feed:
  - `benchmark_results.json`
  - scientific analysis JSONs
  - paper tables
  - paper figures
- LaTeX paper source is separate and must be updated **after** fresh results are generated

If multiple similarly named result files exist, do not assume the newest-looking one is the one used by the paper. Verify provenance.

---

## 3. Current known reality from audit

These findings are assumed true unless new evidence disproves them:

1. `USE_BAYESIAN_TUNING` had introduced a paper/code reproducibility mismatch and needed to be disabled.
2. EKF/UKF measurement-noise search space allowed pathological `R` values that could yield degenerate behavior.
3. LKF was broken and contaminating analyses and had to be removed from benchmark-facing outputs.
4. There was ambiguity between multiple code locations (`src/` vs `code_sim_and_results/`), which is scientifically dangerous.
5. Several LaTeX numbers are hardcoded and must be updated **after** the fresh run.
6. Paper text already required correction from “parabolic interpolation” to “Jacobsen complex spectral interpolation”.
7. The paper may contain stale claims, stale tables, stale figure values, or stale narrative.

These are not cosmetic issues. They are scientific integrity issues.

---

## 4. Operating mode for Claude Code

When working in this repo, Claude Code must behave as:

- Principal Research Software Engineer
- Scientific benchmarking auditor
- Reviewer 2 for reproducibility and methodological rigor

### 4.1 Default mode
Work in this order:

1. inspect,
2. verify,
3. patch minimally,
4. run small sanity checks,
5. run canonical benchmark,
6. regenerate artifacts,
7. reconcile paper.

### 4.2 Forbidden behaviors
Do **not**:
- preserve hardcoded paper numbers because they are convenient,
- assume an old JSON is valid,
- edit the paper before fresh artifacts exist,
- change multiple scientific mechanisms at once without documenting them,
- keep duplicate code paths alive “just in case” if they create ambiguity,
- silently skip failed estimators in a way that biases tables.

### 4.3 Preferred behaviors
Do:
- prefer small surgical fixes,
- log exact file paths touched,
- state explicitly when something is not verifiable,
- separate code integrity issues from methodological issues,
- make paper updates only after result provenance is confirmed.

---

## 5. Definition of done for this project phase

This phase is done only when all of the following are true:

1. Benchmark runs from canonical directory only.
2. All estimators included in paper-facing outputs are functioning and intended.
3. Fresh result artifacts exist and are traceable.
4. Tables in the paper come from the fresh artifacts.
5. Key claims in abstract/results/conclusion match the artifacts.
6. Figure PDFs in LaTeX directory are copied from the fresh run outputs.
7. Manuscript terminology matches implementation reality.
8. A future rerun by the same user can reproduce the paper outputs with minimal ambiguity.

---

## 6. Immediate workflow for camera-ready

### Phase A — Stabilize benchmark
- confirm canonical execution path,
- confirm environment,
- confirm required models/assets,
- confirm disabled Bayesian tuning,
- confirm EKF/UKF search bounds,
- confirm LKF exclusion,
- confirm syntax/import health.

### Phase B — Sanity verification
Run small, targeted tests before the full run:
- IEEE_Freq_Ramp
- IBR_Nightmare
- verify absence of LKF from outputs
- verify no degenerate `T_trip = 0` artifacts from pathological tuning
- verify Ramp RMSE scale is plausible again

### Phase C — Full canonical run
Run benchmark from canonical path with logs captured.

### Phase D — Regenerate artifacts
- JSON/CSV outputs
- scientific analysis artifacts
- figures
- figure copy into paper folder
- claim extraction/checks

### Phase E — Reconcile manuscript
Update:
- abstract numeric claims
- results section values
- tables
- figure captions
- estimator count
- method descriptions

### Phase F — Final consistency pass
Check:
- paper ↔ JSON consistency
- figure ↔ JSON consistency
- narrative ↔ numbers consistency
- file provenance consistency

---

## 7. Ticket roadmap

Below are the actual tickets Claude Code should work through.  
These are **not generic**. They are derived from known project risks and the current benchmark/paper state.

---

# EPIC A — Canonical execution and reproducibility baseline

## A-001 — Enforce canonical execution directory
**Priority:** Critical  
**Goal:** Ensure `code_sim_and_results/` is the only benchmark execution path.

**Problem**  
There was ambiguity between multiple code locations, which can cause the benchmark to be run from the wrong codebase.

**Likely locations**
- `code_sim_and_results/`
- `src/`
- any duplicated runner scripts
- wrapper shell/batch scripts
- README instructions

**Symptoms**
- same module name exists twice,
- edits in one location do not affect run outputs,
- two `main.py`/runner paths exist,
- result provenance unclear.

**Probable cause**
- repository evolved with duplicate code roots,
- older experimental code not retired.

**Solution direction**
- declare `code_sim_and_results/` canonical in docs,
- mark legacy copies as non-authoritative,
- remove ambiguity in execution instructions,
- ensure any run logs clearly show canonical path.

**Acceptance criteria**
- there is exactly one documented execution path for paper-facing runs,
- no team member could reasonably run the wrong code by following repo docs,
- final run log clearly originates from canonical directory.

**Impact on paper**
- prevents invalid paper tables from stale code.

---

## A-002 — Freeze environment for reproducibility
**Priority:** Critical  
**Goal:** Ensure reruns are reproducible enough for camera-ready.

**Likely files**
- `requirements.txt`
- `pyproject.toml`
- `environment.yml`
- model-weight loading code
- README or run docs

**Symptoms**
- missing dependency spec,
- package drift,
- results vary by environment.

**Solution direction**
- keep `requirements.txt`,
- verify all imports from clean environment,
- note Python version if necessary,
- document required model file presence.

**Acceptance criteria**
- environment file exists,
- imports resolve,
- model weights required by benchmark are present and loadable.

**Impact on paper**
- without this, even correct numbers are hard to defend.

---

## A-003 — Add minimal provenance note for final run
**Priority:** High  
**Goal:** Record exactly how the final artifacts were produced.

**Likely output**
- `FINAL_RUN_NOTES.md`
- or short section in README / run log template

**Must include**
- canonical path used,
- command used,
- commit hash if available,
- key config toggles,
- date/time,
- output directories.

**Acceptance criteria**
- a future self can identify which run generated the paper artifacts.

---

# EPIC B — Benchmark integrity blockers

## B-001 — Confirm Bayesian tuning is fully disabled
**Priority:** Critical  
**Goal:** Ensure paper-consistent tuning regime.

**Problem**
Bayesian tuning introduced a reproducibility mismatch with the paper’s assumed methodology.

**Likely files**
- config files
- `main.py`
- tuning utilities
- estimator tuning modules
- duplicated configs in both code roots if still present

**Symptoms**
- tuning results do not match legacy grid-search style behavior,
- run metadata says Bayesian,
- paper says grid but code runs Bayesian.

**Solution direction**
- verify `USE_BAYESIAN_TUNING = False`,
- verify no alternate config overrides it,
- verify log output reports the real tuning method.

**Acceptance criteria**
- benchmark runs with Bayesian tuning off,
- logs or config dump show this clearly,
- manuscript says the true tuning method.

**Impact on paper**
- direct paper-code mismatch if unresolved.

---

## B-002 — Constrain EKF/UKF measurement noise search bounds
**Priority:** Critical  
**Goal:** Prevent pathological tuning that makes filters ignore measurements.

**Problem**
Excessive `R` values can create degenerate but deceptively “stable” behavior.

**Likely files**
- EKF tuning spec
- UKF tuning spec
- search-space config
- estimator parameter schema

**Symptoms**
- `T_trip = 0`,
- absurdly good or weirdly inert outputs,
- unstable claim ratios,
- mismatch between expected dynamic response and actual response.

**Solution direction**
- cap `p_ekf_r` and `p_ukf_r` at `<= 1.0`,
- verify search space in all active code paths,
- inspect log output for actual chosen parameters.

**Acceptance criteria**
- tuned `R` values fall within sane bounds,
- sanity run no longer produces the degenerate pathological behavior previously observed.

**Impact on paper**
- affects core claims in nightmare scenario.

---

## B-003 — Purge LKF from benchmark-facing outputs
**Priority:** Critical  
**Goal:** Remove broken estimator from paper-facing comparisons.

**Problem**
LKF produced absurd RMSE and contaminated MC / aggregate analysis.

**Likely files**
- `main.py`
- estimator imports
- factory lists
- Monte Carlo estimator lists
- aggregation scripts
- table/plot generation scripts

**Symptoms**
- LKF appears in raw results,
- LKF appears in MC summaries,
- LKF appears in plots/tables,
- dead variables remain.

**Solution direction**
- remove from imports,
- remove from factory lists,
- remove from aggregation logic,
- verify no analysis artifact still includes LKF.

**Acceptance criteria**
- LKF absent from fresh benchmark outputs,
- LKF absent from paper-facing analysis JSONs,
- estimator counts updated accordingly.

**Impact on paper**
- contaminates comparisons and method counts.

---

# EPIC C — Sanity validation before expensive full run

## C-001 — Sanity run for IEEE_Freq_Ramp
**Priority:** Critical  
**Goal:** Confirm Ramp RMSE scale is scientifically plausible again.

**Problem**
Previous discrepancy suggested a major issue in tuning, execution path, or artifact provenance.

**Likely scenario**
- `IEEE_Freq_Ramp`

**Checks**
- IpDFT RMSE should return to a plausible scale near prior paper order of magnitude,
- EKF2/IpDFT ratio should be believable and derivable from fresh JSON.

**Acceptance criteria**
- no multi-order-of-magnitude mismatch,
- values no longer look like stale/misaligned artifact output.

**Impact on paper**
- affects abstract/result claims and table values.

---

## C-002 — Sanity run for IBR_Nightmare
**Priority:** Critical  
**Goal:** Validate trip-risk and robustness claims with real fresh outputs.

**Checks**
- EKF `T_trip` is not degenerate,
- EKF2 values are plausible,
- ratio-based claims are derivable from fresh artifacts only.

**Acceptance criteria**
- no pathological zero-trip artifact due to tuning,
- nightmare scenario outputs are stable enough to support honest narrative.

**Impact on paper**
- directly affects strongest robustness claims.

---

## C-003 — Absence check for forbidden estimator artifacts
**Priority:** High  
**Goal:** Ensure removed estimators do not survive in downstream outputs.

**Checks**
- LKF absent from:
  - raw benchmark JSON,
  - MC JSON,
  - scientific analysis JSON,
  - plot labels,
  - LaTeX-generated tables.

**Acceptance criteria**
- zero benchmark-facing occurrences.

---

# EPIC D — Metric and methodology audit

## D-001 — Audit metric computation window and warm-up logic
**Priority:** High  
**Goal:** Verify evaluation window is not hiding event response or unfairly favoring some estimators.

**Problem**
Warm-up / `start_idx` logic may skip meaningful transient behavior.

**Likely files**
- metric computation module
- `calculate_metrics(...)`
- scenario evaluation code
- structural latency handling code

**Symptoms**
- long-window methods appear artificially better,
- fast event portions omitted,
- scenario E or multi-event stress partially ignored.

**Solution direction**
- inspect exact evaluation start logic,
- verify interaction with structural delay,
- document whether current choice is defensible for conference paper.

**Acceptance criteria**
- can explain exactly which samples enter metrics and why,
- no hidden skip of the key event portion without being documented.

**Impact on paper**
- affects fairness and reviewer trust.

---

## D-002 — Separate structural latency, detection delay, and reporting delay
**Priority:** High  
**Goal:** Avoid collapsing distinct notions of latency into one vague “delay”.

**Problem**
Window-based, smoothed, and recursive methods may be compared unfairly if latency is not decomposed.

**Likely files**
- latency computation
- estimator metadata
- reporting tables
- dashboard figure scripts

**Symptoms**
- “accurate but late” methods look too favorable,
- hidden smoothing delay not reported,
- Koopman / window methods under-penalized.

**Solution direction**
- document current latency definition,
- compute or expose separate components if feasible,
- at minimum clarify in manuscript what is included.

**Acceptance criteria**
- latency definition used in paper is explicit and true.

**Impact on paper**
- central to the paper’s trade-off thesis.

---

## D-003 — Review tuning objective for fairness
**Priority:** High  
**Goal:** Ensure optimizer is not rewarding “do nothing / ignore disturbance” behavior.

**Problem**
A poorly chosen objective can favor inert or heavily smoothed estimators.

**Likely files**
- tuning objective function
- search wrapper
- estimator scoring code

**Symptoms**
- estimator with weak dynamic tracking gets low RMSE,
- scenario-wise tuning produces suspiciously inert filters,
- best parameters are physically unintuitive.

**Solution direction**
- inspect objective and penalties,
- consider whether trip-risk / event response is represented,
- document limitations if kept as-is for camera-ready.

**Acceptance criteria**
- objective is explainable and not obviously pathological.

**Impact on paper**
- affects scientific defensibility of all tuned methods.

---

## D-004 — Verify estimator descriptions match implementations
**Priority:** High  
**Goal:** Prevent methodological mislabeling in paper.

**Known items**
- IpDFT description must reflect Jacobsen complex spectral interpolation.
- SOGI-FLL may include prefilter/AGC behavior and should not be described as a pure textbook implementation if it is not.
- PI-GRU should only be described as actually implemented and loaded.

**Likely files**
- estimator source
- LaTeX methodology section
- captions / method summaries

**Acceptance criteria**
- paper language matches actual implementation family and modifications.

**Impact on paper**
- avoids reviewer criticism for misleading taxonomy.

---

# EPIC E — Artifact provenance and reporting pipeline

## E-001 — Map every paper number to an artifact source
**Priority:** Critical  
**Goal:** Every manuscript number must be traceable.

**Need mapping for**
- abstract ratios,
- Table I,
- Table II,
- results section values,
- figure caption values,
- nightmare scenario metrics.

**Likely files**
- `main.tex`
- benchmark JSON
- scientific analysis JSON
- table generation scripts
- notebooks or helper scripts

**Acceptance criteria**
- for each number in paper, there is a known source file and derivation path.

**Impact on paper**
- this is the backbone of honesty.

---

## E-002 — Standardize figure generation and copy step
**Priority:** High  
**Goal:** Ensure the figure PDFs in the paper folder are the ones from the fresh run.

**Known copy commands**
- `Fig1_Scenarios_Final.pdf`
- `Fig2_Mega_Dashboard.pdf`

**Problem**
Manual figure copying can leave stale plot PDFs in LaTeX.

**Solution direction**
- use a small script or documented copy step,
- verify timestamps / overwrite behavior,
- avoid manual ambiguity.

**Acceptance criteria**
- paper folder figures are confirmed fresh and sourced from canonical run outputs.

---

## E-003 — Build a paper-claim verification script or checklist
**Priority:** High  
**Goal:** Catch stale hardcoded manuscript numbers before submission.

**Should check**
- estimator count,
- ramp ratio,
- nightmare metrics,
- table dimensions,
- any headline ratio.

**Acceptance criteria**
- script or checklist can flag mismatch between paper and fresh JSON.

---

# EPIC F — Manuscript reconciliation after fresh run

## F-001 — Update hardcoded abstract/result claims
**Priority:** Critical  
**Goal:** Replace legacy numbers with fresh artifact-backed values.

**Known hardcoded targets from current notes**
- abstract ramp RMSE / ratio
- nightmare EKF values
- nightmare RA-EKF values
- ratio claims such as 12.6× / 4.7× / 275× or their replacements

**Likely file**
- `main.tex`

**Acceptance criteria**
- no headline numerical claim remains stale.

---

## F-002 — Regenerate Table I from fresh IEEE scenario outputs
**Priority:** Critical  
**Goal:** Ensure standard scenario comparison table is real.

**Likely source**
- `benchmark_results.json` or derived fresh analysis file

**Acceptance criteria**
- every cell comes from fresh results,
- method count and scenario count are correct,
- removed methods are absent.

---

## F-003 — Regenerate Table II from fresh IBR_MultiEvent outputs
**Priority:** Critical  
**Goal:** Ensure stress-test table is real and current.

**Acceptance criteria**
- table values match fresh JSON,
- no stale methods or stale ratios survive.

---

## F-004 — Reconcile all estimator counts in manuscript
**Priority:** High  
**Goal:** Ensure method counts are consistent everywhere.

**Known risk**
- old text referenced 13 methods incl. LKF; benchmark-facing set is now 12.

**Acceptance criteria**
- abstract, methods, captions, print strings, and tables all agree.

---

## F-005 — Reconcile captions and terminology
**Priority:** High  
**Targets**
- figure captions,
- method names,
- scenario labels,
- parameter names such as `N_c`,
- any latency-related caption language.

**Acceptance criteria**
- no caption contradicts implementation or final data.

---

# EPIC G — Final camera-ready confidence pass

## G-001 — Full consistency audit after rerun
**Priority:** Critical  
**Goal:** Check paper ↔ code ↔ artifact alignment one final time.

**Checklist**
- numbers in manuscript match artifacts,
- figures match artifacts,
- estimator count matches artifacts,
- removed estimators absent,
- methodology text matches implementation,
- no stale JSON is referenced by scripts.

**Acceptance criteria**
- no known contradiction remains.

---

## G-002 — Submission safety review
**Priority:** High  
**Goal:** Ask one final brutal question: “Would Reviewer 2 reject this for inconsistency or unreproducibility?”

**Check**
- can every main claim be defended,
- are limitations honestly stated,
- are strong claims supported by fresh outputs.

**Acceptance criteria**
- any unsupported claim is softened or removed.

---

## G-003 — Archive final run bundle
**Priority:** Medium  
**Goal:** Preserve the exact artifact set used for submission.

**Should include**
- final log,
- result JSONs,
- figure PDFs,
- manuscript PDF,
- short provenance note.

**Acceptance criteria**
- final submission package can be reconstructed later.

---

## 8. Current run command

Use the canonical run command from the canonical directory.

Example:

```bash
cd /c/Users/walla/Documents/Github/paper/code_sim_and_results
python main.py 2>&1 | tee ../run_final_$(date +%Y%m%d_%H%M%S).log
```

Do not run from any ambiguous duplicate code path.

9. Post-run mandatory actions

After a successful fresh run:

verify output JSON freshness,
verify sanity metrics,
regenerate tables,
regenerate / verify figures,
copy figures into LaTeX directory,
update manuscript values,
compile manuscript,
perform final consistency audit.

Known figure copy step:

cp figures_estimatores_benchmark_test5/Fig1_Scenarios_Final.pdf ../Figures/Plots_And_Graphs/
cp figures_estimatores_benchmark_test5/Fig2_Mega_Dashboard.pdf ../Figures/Plots_And_Graphs/

Do not assume copied figures are fresh without checking source timestamps / provenance.

10. What Claude Code should do first on every session

On every new session, before changing code, Claude Code should:

confirm canonical working directory,
inspect current benchmark config,
inspect estimator inclusion list,
inspect tuning bounds,
inspect result file freshness,
inspect any hardcoded paper numbers mentioned in previous notes,
state current risk posture before editing.
11. Success criteria for Claude Code

Claude Code is successful only if it leaves the repo in a state where:

the benchmark is run from the correct place,
the outputs are real,
the paper matches the outputs,
the claims are scientifically honest,
and a future rerun is still understandable.
12. Tone and values

This project does not optimize for preserving prior narrative.
It optimizes for:

correctness,
honesty,
reproducibility,
scientific defensibility,
clean traceability.

If a result gets weaker after the correct rerun, that is acceptable.
A weaker true result is better than a stronger false one.

13. Immediate execution order

Claude Code should usually follow this exact order unless a blocker is discovered:

confirm canonical path,
confirm environment,
confirm tuning + estimator set,
run sanity checks,
inspect sanity outputs,
run full benchmark,
regenerate analysis artifacts,
copy figures,
update manuscript,
final consistency pass.
14. If something goes wrong

If the rerun produces values that strongly disagree with prior expectations:

do not patch the paper first,
do not assume the old paper was right,
inspect provenance,
inspect metric logic,
inspect scenario execution path,
inspect which JSON the paper had used,
document discrepancy,
only then decide whether code or manuscript must change.
15. Final principle

The benchmark is the source of truth only if the benchmark itself is correct, canonical, and reproducible.

The manuscript is never the source of truth for numbers.

The fresh, validated, canonical artifacts are the source of truth.