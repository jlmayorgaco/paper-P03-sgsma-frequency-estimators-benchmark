# Roadmap to an IEEE TIM submission

Date: 2026-06-11

Target venue: IEEE Transactions on Instrumentation and Measurement (IEEE TIM).

Purpose: turn the post-SGSMA work into a credible, reproducible, review-ready
journal manuscript. This roadmap is acceptance-oriented, but no roadmap can
guarantee acceptance or an award. The standard is: every quantitative claim must
be regenerated from clean code, mapped to an artifact, defensible under peer
review, and formatted for IEEE TIM.

## Current baseline

Repository state at the start of this roadmap:

- Branch: `MVP2.0.0`.
- Strongest completed evidence bundle:
  `artifacts/openfreqbench/phase2-paper-grade/`.
- Phase 2-G evidence:
  - 14 scenarios.
  - 17 estimators.
  - 30 Monte Carlo runs.
  - 238 scenario-estimator pairs.
  - 7140 raw records.
  - 234 main-comparison pairs.
  - 4 diagnostic appendix pairs, all Prony:
    `IBR_Multi_Event`, `IEEE_Phase_Jump_20`, `IEEE_Phase_Jump_60`,
    `NERC_Phase_Jump_60`.
- Validated report schema:
  `artifacts/openfreqbench/phase2-paper-grade/benchmark_report.json` passes
  the local `benchmark-report` schema.
- Current full registry:
  - 32 scenarios.
  - 18 canonical estimators.
- Current blocker for full journal replay:
  `configs/journal-paper-replay.yaml` expects 32 x 18 = 576 tuned pairs under
  `artifacts/full_mc_benchmark`, but `validate-artifacts` reports 0 present and
  576 missing.
- Current ATLAS blocker:
  no `atlas_readiness_report.json` was found under `artifacts/` in this
  workspace.
- Current CPU limitation:
  Phase 2-G used `n_cost_reps=1`; CPU/Pareto claims remain diagnostic.
- Current manuscript state:
  `paper/paper.md` is a software-first/JOSS-style skeleton, not an IEEE TIM
  Transactions manuscript.
- Current code state:
  there is active WIP for reproducible IEEE-style tuning-trace plots:
  `src/openfreqbench/tuning_trace_plots.py`, CLI wiring, and tests.

## External venue constraints to respect

IEEE author guidance relevant to this roadmap:

- A paper can be rejected before review if it is outside the journal scope; the
  target journal scope must be checked before submission.
  Source: IEEE Author Center, "The IEEE Article Submission Process".
- IEEE journals require authors to follow each target publication's
  "Information for Authors" or publication details.
  Source: IEEE Author Center, "The IEEE Article Submission Process".
- IEEE provides article templates and recommends using the template selector.
  Source: IEEE Author Center, "IEEE Article Templates".
- IEEE encourages reproducibility through clear methods, data repositories, and
  code repositories.
  Source: IEEE Author Center, "Research Reproducibility".
- IEEE graphics must communicate accurately and should be accessible in
  greyscale/color-vision-deficiency contexts.
  Source: IEEE Author Center, "Create Graphics for Your Article".
- IEEE submission checklist includes reviewing target-publication requirements,
  having all files ready, selecting the corresponding author, and having ORCID.
  Source: IEEE Author Center, "Checklist for Submitting Your Article for Peer
  Review".

Working target framing:

> A reproducible measurement benchmark for single-phase grid frequency
> estimators, with traceable uncertainty, diagnostic validity gates, and
> stress-regime analysis across PMU-style measurement conditions.

This framing is closer to IEEE TIM than a pure power-systems ranking paper
because it emphasizes measurement methodology, instrumentation-grade
uncertainty, traceability, reproducibility, and estimator validity.

## Definition of ready

The paper is ready for IEEE TIM submission only when all required gates below
are complete.

Required gates:

- G0: repository is clean, tested, tagged, and reproducible from a fresh clone.
- G1: full journal replay passes artifact validation with 576/576 tuned pairs,
  or the manuscript explicitly narrows the scope and explains why.
- G2: journal-grade OpenFreqBench run exists with n >= 100 and archived hashes.
- G3: dedicated timing run exists with `n_cost_reps >= 3` and hardware metadata.
- G4: ATLAS run exists with readiness `journal_grade`, or ATLAS is removed from
  main claims and kept as future work.
- G5: every figure/table in the manuscript maps to an artifact path and hash.
- G6: every estimator method has a reference and implementation caveat.
- G7: the manuscript makes a clear measurement contribution, not only a
  leaderboard.
- G8: IEEE template, references, figures, supplementary material, and submission
  files pass final audit.

## Milestones and tickets

Status legend:

- `TODO`: not started.
- `WIP`: started but not complete.
- `BLOCKED`: cannot be completed until dependency is done.
- `DONE`: completed and verified.

Priority legend:

- `P0`: blocks any serious submission.
- `P1`: blocks a strong IEEE TIM submission.
- `P2`: improves acceptance probability, clarity, or award potential.

### M0. Stabilize current workspace

#### TIM-0001: Finish or revert IEEE tuning-trace plot WIP

Status: WIP

Priority: P0

Problem:
The repo is not clean. The current WIP adds reproducible IEEE-style plots for
`full_mc_tuning_matrix`, but tests were interrupted because `pytest` got stuck
around temp/cache permissions.

Files:

- `src/openfreqbench/tuning_trace_plots.py`
- `src/openfreqbench/cli.py`
- `tests/test_cli.py`
- `tests/test_tuning_trace_plots.py`

Tasks:

- Run a direct smoke command for the plot generator without full pytest.
- Ensure it regenerates:
  - `ground_truth_ieee_mhz.{png,pdf,svg}`
  - `traces_<objective>_ieee_mhz.{png,pdf,svg}`
  - `trace_summary_ieee_mhz.csv`
  - `ieee_plot_manifest.json`
- Fix test temp handling so local tests use a writable base temp.
- Run:

```powershell
$env:TEMP='C:\tmp'
$env:TMP='C:\tmp'
python -m pytest -p no:cacheprovider --basetemp artifacts\pytest-tmp-roadmap-plots tests\test_tuning_trace_plots.py tests\test_cli.py
```

Acceptance criteria:

- Plot CLI works on the pilot artifact.
- Focused tests pass.
- Either commit the feature or revert it cleanly.
- `git status --short` shows no accidental generated files.

#### TIM-0002: Clean generated temp/test artifact directories

Status: TODO

Priority: P0

Problem:
Several pytest temp folders are present under `artifacts/`, and some produced
permission errors during recursive searches.

Tasks:

- Inventory generated folders under `artifacts/`.
- Keep scientifically meaningful bundles:
  - `artifacts/openfreqbench/phase2-paper-grade/`
  - `artifacts/full_mc_tuning_matrix/pilot-zcd-ipdft-ekf-v1/`
- Delete or quarantine temporary pytest folders after verifying they are not
  referenced by any report.
- Confirm `.gitignore` excludes temp folders.

Acceptance criteria:

- Artifact tree is readable by `rg` and PowerShell without access-denied noise.
- README/docs reference only retained evidence bundles.

#### TIM-0003: Re-run release quality gate from clean tree

Status: BLOCKED by TIM-0001 and TIM-0002

Priority: P0

Tasks:

- Run:

```powershell
python -m openfreqbench.cli doctor --output artifacts\release-check\environment_report.json
python -m openfreqbench.cli quality-gate
python -m openfreqbench.cli quality-gate --release
```

Acceptance criteria:

- Both quality gates pass.
- The release gate confirms required files, configs, registry, citations,
  remote origin, and clean tree.
- Result is documented in `docs/RELEASE.md` or a new release audit note.

### M1. Lock the IEEE TIM paper thesis

#### TIM-0101: Decide exact target and article type

Status: TODO

Priority: P0

Problem:
The current paper folder targets a software-first paper. IEEE TIM requires a
measurement contribution. The manuscript must not read as "we wrote a package"
only.

Tasks:

- Confirm target:
  - Primary: IEEE Transactions on Instrumentation and Measurement.
  - Backup: IEEE Open Journal of Instrumentation and Measurement, IEEE OAJPE,
    SoftwareX, or JOSS.
- Check the current TIM "Information for Authors" and publication details on
  IEEE Xplore before drafting.
- Decide whether submission will be:
  - Regular paper.
  - Review/survey plus benchmark.
  - Instrumentation/measurement methodology paper.

Acceptance criteria:

- `docs/IEEE_TIM_TARGET_AUDIT.md` exists.
- It states target scope, article type, expected contribution, page/format
  constraints, and why OpenFreqBench fits TIM.

#### TIM-0102: Write the one-sentence contribution

Status: TODO

Priority: P0

Working version:

> We introduce a reproducible measurement benchmark that quantifies accuracy,
> tail error, protection risk, latency, runtime cost, and validity failure modes
> of single-phase grid frequency estimators under traceable IEEE/NERC/IBR stress
> scenarios.

Tasks:

- Stress-test the contribution against likely reviewer objections:
  - "This is only software."
  - "This is only a leaderboard."
  - "The scenarios are synthetic."
  - "The estimator implementations are not exact reproductions."
  - "The timing results depend on hardware."
  - "PI-GRU is undertrained or not comparable."
- Convert the contribution into 3 to 5 specific claims.

Acceptance criteria:

- `docs/IEEE_TIM_CLAIM_LEDGER.md` lists claim ID, statement, evidence file,
  artifact hash, allowed wording, and forbidden overclaim.

#### TIM-0103: Define falsifiable research questions

Status: TODO

Priority: P1

Required questions:

- RQ1: Which estimator families keep frequency error low across clean,
  dynamic, phase-jump, modulation, harmonic, IBR, and noise stress regimes?
- RQ2: Which estimators trade accuracy for latency/runtime in online measurement?
- RQ3: Which methods have low average RMSE but unacceptable tail error or
  invalid-output behavior?
- RQ4: Does objective-specific tuning change estimator rankings relative to
  default parameters?
- RQ5: Do severity sweeps reveal monotonic, saturating, or non-monotonic failure
  regimes?

Acceptance criteria:

- RQs appear in the Introduction.
- Each RQ maps to one or more preregistered hypotheses or artifact tables.
- No RQ depends on unavailable artifacts.

### M2. Complete method and implementation audit

#### TIM-0201: Audit every estimator implementation against its cited paper

Status: TODO

Priority: P0

Current state:
All 18 active estimators have `REFERENCE_KEYS`, but that is method-family
support, not proof of exact reproduction.

Tasks:

- For each active estimator, create an audit row:
  - method label;
  - source module;
  - cited paper(s);
  - exact implemented state/update equations;
  - intentional simplifications;
  - default parameters;
  - known limitations;
  - unit tests;
  - evidence plot.
- Flag "family baseline" versus "paper-exact implementation".

Acceptance criteria:

- `docs/ESTIMATOR_IMPLEMENTATION_AUDIT.md` exists.
- No estimator is described as an exact reproduction unless equations and tests
  support that wording.
- Manuscript Methods uses conservative wording.

#### TIM-0202: Add estimator-level numerical sanity tests

Status: TODO

Priority: P0

Tasks:

- For each estimator, define at least three deterministic tests:
  - clean 60 Hz sinusoid;
  - off-nominal steady frequency;
  - small dynamic event or startup validity behavior.
- Include expected bounds for:
  - steady RMSE;
  - first valid sample;
  - invalid-output rate;
  - structural latency where relevant.

Acceptance criteria:

- Every active estimator has non-plot tests.
- Tests are deterministic and do not depend on generated artifacts.
- `pytest tests/estimators` passes.

#### TIM-0203: Decide PI-GRU status

Status: TODO

Priority: P0

Problem:
PI-GRU is in the canonical registry, but Phase 2-G excluded it. IEEE TIM
reviewers will notice if the manuscript says 18 estimators but the main run uses
17.

Options:

- Include PI-GRU in journal-grade runs with weight hash, training data
  disclosure, and generalization caveats.
- Move PI-GRU to a diagnostic appendix and state that the main comparison uses
  17 non-neural estimators.
- Remove PI-GRU from the TIM paper scope.

Acceptance criteria:

- One decision is recorded in `docs/IEEE_TIM_SCOPE_DECISIONS.md`.
- Manuscript counts match the decision everywhere.

#### TIM-0204: Validate scenario definitions and physical assumptions

Status: TODO

Priority: P0

Tasks:

- Audit all 32 registered scenarios:
  - parameter ranges;
  - ground-truth frequency construction;
  - phase continuity;
  - voltage sampling and decimation;
  - noise/distortion model;
  - standard or field-motivation reference.
- Separate:
  - standards-inspired tests;
  - synthetic stress tests;
  - IBR-motivated scenarios;
  - diagnostic variants.

Acceptance criteria:

- `docs/SCENARIO_IMPLEMENTATION_AUDIT.md` exists.
- Every scenario has a short Methods-ready description.
- No scenario is called IEEE/NERC-standard-compliant unless the implementation
  actually matches the relevant standard requirement.

### M3. Build journal-grade OpenFreqBench evidence

#### TIM-0301: Generate full objective-specific tuned artifacts

Status: TODO

Priority: P0

Problem:
`configs/journal-paper-replay.yaml` expects tuned artifacts under
`artifacts/full_mc_benchmark`, but 0/576 are present.

Tasks:

- Decide tuning objectives:
  - primary accuracy: `m1_rmse_hz`;
  - protection: `m5_trip_risk_s`;
  - runtime: `m13_cpu_time_us`;
  - tail risk: `m34_p95_error_hz` or `m35_p99_error_hz`;
  - event behavior: selected m24-m30 metrics if central.
- Run `full_mc_tuning_matrix` at increasing scale:
  - smoke: 1 scenario x 3 estimators x 3 objectives;
  - integration: 4 scenarios x all estimators x 3 objectives;
  - full: 32 scenarios x 18 estimators x selected objectives.
- Ensure the final artifact layout can satisfy `artifact_tuned` replay.

Acceptance criteria:

- `openfreqbench validate-artifacts --config configs/journal-paper-replay.yaml`
  reports 576 present and 0 missing, or the config is intentionally narrowed
  and renamed.
- `tuning_matrix.csv`, `selected_replay/`, and `benchmark_report.json` are
  archived.
- Tuning seeds, trials, tune runs, eval runs, and objective directions are
  documented.

#### TIM-0302: Run journal-paper replay at n >= 100

Status: BLOCKED by TIM-0301

Priority: P0

Tasks:

- Run:

```powershell
python -m openfreqbench.cli run --config configs\journal-paper-replay.yaml
```

- Build reports:

```powershell
python -m openfreqbench.cli report build --input-json artifacts\openfreqbench\journal-paper-replay-v2\benchmark_report.json
python -m openfreqbench.cli schema --name benchmark-report --validate artifacts\openfreqbench\journal-paper-replay-v2\benchmark_report.json
```

Acceptance criteria:

- `benchmark_report.json` exists for `journal-paper-replay-v2`.
- n_runs >= 100.
- 32 scenarios and 18 estimators are included, unless scope was deliberately
  narrowed.
- Raw records equal expected scenarios x estimators x n_runs.
- Schema validation passes.
- Report outputs include tables, plots, traceability, and evidence manifest.

#### TIM-0303: Run dedicated timing study

Status: TODO

Priority: P0

Problem:
Current CPU timing is diagnostic because `n_cost_reps=1`.

Tasks:

- Create `configs/journal-timing.yaml`.
- Use:
  - fixed hardware;
  - no other heavy workloads;
  - `n_cost_reps >= 3`, preferably 10 for final timing;
  - repeated seeds;
  - environment report.
- Report:
  - median CPU time;
  - confidence intervals;
  - runtime jitter;
  - hardware and OS metadata;
  - dependency versions.

Acceptance criteria:

- CPU claims are either backed by this run or removed from main claims.
- Pareto figure uses journal-grade timing data.
- Manuscript states hardware dependence.

#### TIM-0304: Rebuild report layer for manuscript-grade figures

Status: WIP

Priority: P1

Tasks:

- Finish IEEE-style plot CLI from TIM-0001.
- Add publication-grade outputs for:
  - RMSE by estimator/family;
  - scenario x estimator heatmap;
  - tail-error and invalid-output map;
  - Pareto accuracy/runtime;
  - selected trace plots with mHz scale;
  - diagnostic appendix failures.
- Export PNG, PDF, and SVG.
- Use colorblind-safe palettes, line styles, markers, readable labels, and
  greyscale-compatible encodings.

Acceptance criteria:

- Every manuscript figure has a generating command.
- Every figure path appears in an evidence manifest.
- Figures pass IEEE graphics checklist and visual inspection.

#### TIM-0305: Build artifact hash and claim traceability package

Status: TODO

Priority: P0

Tasks:

- Freeze the final run root:

```powershell
python -m openfreqbench.cli archive --run-root artifacts\openfreqbench\journal-paper-replay-v2 --zip
```

- Ensure outputs include:
  - `artifact_index.csv`;
  - `paper_traceability.csv`;
  - `evidence_manifest.json`;
  - source hashes;
  - environment report;
  - data/code availability text.

Acceptance criteria:

- Every quantitative sentence in the manuscript maps to a claim ledger row.
- Every claim ledger row maps to an artifact path and hash.

### M4. Build journal-grade ATLAS or remove ATLAS from claims

#### TIM-0401: Decide ATLAS role

Status: TODO

Priority: P0

Options:

- Main contribution: ATLAS severity sweeps are central to the paper.
- Secondary evidence: ATLAS confirms stress-regime behavior.
- Future work only: remove ATLAS from main claims.

Acceptance criteria:

- Decision is recorded in `docs/IEEE_TIM_SCOPE_DECISIONS.md`.
- Manuscript outline reflects the decision.

#### TIM-0402: Run ATLAS journal-grade sweep

Status: BLOCKED by TIM-0401 if ATLAS remains in scope

Priority: P0 if ATLAS is in scope; otherwise P2

Command template:

```powershell
python -m pipelines.atlas_sweep --sweeps all --policy fixed_policy --n-runs 100 --base-seed 12345 --n-cost-reps 3 --tune-trials 80 --tune-eval-runs 5 --output-subdir atlas-paper-fixed-v2 --resume
```

Acceptance criteria:

- `atlas_readiness_report.json` exists.
- Readiness is `journal_grade` for full ATLAS claims, or `paper_grade` for
  clearly labeled confirmatory evidence.
- `paper_claims_allowed` is true for any ATLAS number used in the paper.
- `journal_claims_allowed` is true for any journal-grade ATLAS conclusion.

#### TIM-0403: ATLAS method audit and hypothesis classification

Status: BLOCKED by TIM-0402

Priority: P1

Tasks:

- Revisit `docs/ATLAS_METHOD_AUDIT.md`.
- Confirm:
  - severity sweep definitions;
  - fixed-policy versus oracle-policy separation;
  - invalid-output handling;
  - positive/negative event symmetry;
  - figure generation.
- Create hypothesis classifications:
  - preregistered;
  - confirmatory;
  - exploratory.

Acceptance criteria:

- ATLAS claims in the manuscript are marked by claim class.
- Exploratory ATLAS patterns are not written as confirmed conclusions.

### M5. Statistics, uncertainty, and claim discipline

#### TIM-0501: Preregister final hypotheses

Status: TODO

Priority: P1

Tasks:

- Create final `hypotheses.ieee-tim.yaml`.
- Include hypotheses for:
  - family-level accuracy;
  - IBR stress degradation;
  - tail-error failure;
  - invalid-output gate;
  - timing/latency tradeoff;
  - tuning-objective sensitivity.

Acceptance criteria:

- Hypotheses are run against final journal artifact.
- Multiple-comparison correction method is declared.
- Exploratory findings are separated.

#### TIM-0502: Add uncertainty and effect-size tables

Status: TODO

Priority: P1

Tasks:

- For each key claim, report:
  - mean/median;
  - confidence interval;
  - effect size or ratio;
  - number of runs;
  - validity/exclusion status.
- Avoid relying only on rank order.

Acceptance criteria:

- Main text contains uncertainty, not just winners.
- Tables are generated from code.

#### TIM-0503: Stress-test diagnostic appendix policy

Status: TODO

Priority: P1

Problem:
The diagnostic gate is defensible, but reviewers may see it as cherry-picking
unless it is preregistered and sensitivity-tested.

Tasks:

- Run sensitivity analysis:
  - with invalid pairs excluded;
  - with invalid pairs penalized;
  - with invalid pairs listed but not ranked.
- Confirm ranking conclusions do not depend on arbitrary exclusion.

Acceptance criteria:

- Manuscript explains diagnostic appendix policy before results.
- Sensitivity table is included in supplement or appendix.

### M6. Manuscript architecture

#### TIM-0601: Create IEEE TIM manuscript folder

Status: TODO

Priority: P0

Tasks:

- Create:
  - `paper-ieee-tim/main.tex`;
  - `paper-ieee-tim/references.bib`;
  - `paper-ieee-tim/figures/`;
  - `paper-ieee-tim/tables/`;
  - `paper-ieee-tim/supplement/`;
  - `paper-ieee-tim/cover_letter.md`.
- Use IEEE template selected from IEEE Author Center.

Acceptance criteria:

- Manuscript builds locally to PDF.
- IEEE bibliography style is used.
- Figure/table paths are relative and reproducible.

#### TIM-0602: Write final outline

Status: TODO

Priority: P0

Target structure:

1. Introduction
   - measurement problem;
   - why frequency-estimator comparisons are hard to reproduce;
   - contributions.
2. Related Work
   - standards and PMU measurement;
   - estimator families;
   - reproducible benchmarking.
3. Benchmark Methodology
   - scenarios;
   - estimator contract;
   - metrics;
   - validity gate;
   - tuning policy.
4. Journal-Grade Evidence Protocol
   - Monte Carlo design;
   - seeds;
   - artifact traceability;
   - statistical method.
5. Results
   - accuracy and tail error;
   - stress-regime behavior;
   - invalid-output failures;
   - timing/latency tradeoff;
   - tuning sensitivity.
6. Discussion
   - measurement implications;
   - estimator selection guidance;
   - limitations.
7. Reproducibility and Data Availability
8. Conclusion

Acceptance criteria:

- Each section maps to available or planned artifacts.
- No section depends on SGSMA numbers.

#### TIM-0603: Write Methods from code, not memory

Status: TODO

Priority: P0

Tasks:

- Derive Methods from:
  - `src/scenarios/`;
  - `src/estimators/`;
  - `src/analysis/metrics.py`;
  - `src/openfreqbench/runner.py`;
  - `src/openfreqbench/scientific.py`;
  - final configs.

Acceptance criteria:

- Every equation/metric in Methods is traceable to code.
- Any simplification is stated explicitly.

#### TIM-0604: Replace software-paper language with TIM language

Status: TODO

Priority: P1

Problem:
Current `paper/paper.md` says the contribution is software. IEEE TIM needs a
measurement science contribution.

Tasks:

- Rewrite wording from:
  - "software platform";
  to:
  - "measurement benchmark protocol";
  - "traceable estimator validity and uncertainty assessment";
  - "instrumentation-grade reproducibility layer".
- Keep software as enabling contribution, not the only contribution.

Acceptance criteria:

- Abstract and Introduction pass the "not just software" test.

#### TIM-0605: Prepare cover letter and editor-facing positioning

Status: TODO

Priority: P2

Tasks:

- State why the paper fits IEEE TIM.
- State novelty relative to PMU standards and estimator papers.
- State reproducibility package availability.
- Avoid inflated claims such as "first ever" unless proven.

Acceptance criteria:

- Cover letter is concise, technical, and aligned with scope.

### M7. Figures and tables for an award-level paper

#### TIM-0701: Design figure set

Status: TODO

Priority: P1

Required figures:

- Fig. 1: Benchmark architecture and artifact chain.
- Fig. 2: Scenario taxonomy and stress-regime map.
- Fig. 3: Accuracy heatmap across scenario-estimator pairs.
- Fig. 4: Tail error and invalid-output diagnostic map.
- Fig. 5: Representative waveform/frequency/error traces in mHz.
- Fig. 6: Accuracy-runtime-latency Pareto plot from journal-grade timing.
- Fig. 7: ATLAS severity sweep, if ATLAS remains in scope.

Acceptance criteria:

- No figure is copied manually from exploratory output.
- Each figure has a generation command and artifact hash.
- Figures remain legible in two-column IEEE format and greyscale.

#### TIM-0702: Design table set

Status: TODO

Priority: P1

Required tables:

- Table I: Scenario taxonomy.
- Table II: Estimator families and implementation caveats.
- Table III: Metric definitions.
- Table IV: Main performance summary with uncertainty.
- Table V: Diagnostic appendix summary.
- Table VI: Timing/latency summary.
- Supplementary tables: full 32 x 18 matrix, hypothesis results, artifact
  traceability.

Acceptance criteria:

- Main paper tables are compact.
- Full matrix goes to supplement or repository.

### M8. Reproducibility package and public release

#### TIM-0801: Create public release candidate

Status: TODO

Priority: P0

Tasks:

- Confirm license.
- Confirm README install path.
- Confirm examples run on fresh clone.
- Tag release candidate.
- Create GitHub release.
- Archive with Zenodo or another DOI provider.

Acceptance criteria:

- DOI exists before final submission or is documented as pending.
- `CITATION.cff` includes DOI and author metadata.

#### TIM-0802: Build Code/Data availability statement

Status: TODO

Priority: P0

Tasks:

- State:
  - repository URL;
  - release tag;
  - DOI;
  - artifact archive;
  - commands to reproduce main tables;
  - hardware limitations for timing.

Acceptance criteria:

- Statement appears in manuscript.
- It is consistent with IEEE reproducibility guidance.

#### TIM-0803: Optional Code Ocean capsule

Status: TODO

Priority: P2

Rationale:
IEEE supports linked code/capsules. This is not strictly required, but it can
increase reviewer confidence.

Tasks:

- Prepare minimal reproducible subset:
  - smoke run;
  - report build;
  - one figure regeneration.
- Document runtime and dependencies.

Acceptance criteria:

- Capsule or equivalent reproducible subset runs without private data.

### M9. Pre-submission review and risk audit

#### TIM-0901: Internal red-team review

Status: TODO

Priority: P1

Reviewer personas:

- TIM measurement expert.
- PMU/power systems expert.
- Signal processing estimator expert.
- Reproducibility/software reviewer.
- Skeptical statistics reviewer.

Acceptance criteria:

- `docs/IEEE_TIM_INTERNAL_REVIEW.md` contains findings, severity, and fixes.
- No P0/P1 critique remains unresolved.

#### TIM-0902: Citation audit

Status: TODO

Priority: P0

Tasks:

- Verify every reference:
  - authors;
  - title;
  - venue;
  - year;
  - DOI;
  - claim supported.
- Remove citations that do not support the stated claim.

Acceptance criteria:

- `docs/IEEE_TIM_CITATION_AUDIT.md` exists.
- All bibliography entries are real and contextually correct.

#### TIM-0903: Overclaim audit

Status: TODO

Priority: P0

Forbidden until evidence exists:

- "journal-grade across all scenarios" if only Phase 2-G is used.
- "fastest estimator" from `n_cost_reps=1`.
- "PI-GRU generalizes" without PI-GRU journal run.
- "ATLAS confirms" without readiness report.
- "IEEE/NERC compliant" unless scenario audit proves it.
- "exact implementation of paper X" unless implementation audit proves it.

Acceptance criteria:

- `docs/IEEE_TIM_OVERCLAIM_AUDIT.md` lists each risky phrase and approved
  replacement.
- Abstract, Introduction, Results, and Conclusion pass this audit.

#### TIM-0904: Final IEEE submission checklist

Status: TODO

Priority: P0

Tasks:

- Confirm target publication requirements.
- Confirm corresponding author and ORCID.
- Confirm all files:
  - main manuscript PDF/source;
  - figures;
  - supplementary material;
  - graphical abstract if required;
  - cover letter;
  - data/code availability statement;
  - conflict/funding/AI disclosure if required.
- Confirm single-submission rule.
- Submit through IEEE Publishing Portal or journal submission system.

Acceptance criteria:

- `docs/IEEE_TIM_SUBMISSION_CHECKLIST.md` is complete.
- Final PDF has no missing refs, bad figures, overfull tables, or stale counts.

## Execution order

Recommended order:

1. TIM-0001, TIM-0002, TIM-0003.
2. TIM-0101, TIM-0102, TIM-0103.
3. TIM-0201, TIM-0202, TIM-0203, TIM-0204.
4. TIM-0301, TIM-0302, TIM-0303, TIM-0304, TIM-0305.
5. TIM-0401. If ATLAS remains in scope, TIM-0402 and TIM-0403.
6. TIM-0501, TIM-0502, TIM-0503.
7. TIM-0601 through TIM-0605.
8. TIM-0701 and TIM-0702.
9. TIM-0801 through TIM-0803.
10. TIM-0901 through TIM-0904.

## Stop/go decisions

### Decision A: software-first versus IEEE TIM

Make this decision after TIM-0101 and TIM-0102.

Go IEEE TIM if:

- contribution is clearly measurement methodology;
- journal replay and timing can be completed;
- ATLAS role is clear;
- paper has more than software packaging.

Go software-first if:

- journal replay remains too expensive;
- ATLAS remains incomplete;
- contribution remains primarily CLI/package/reproducibility.

### Decision B: full 32 x 18 versus narrowed scope

Make this decision before TIM-0301 full run.

Use full 32 x 18 if:

- compute budget permits;
- PI-GRU can be fairly included or explicitly scoped.

Use narrowed scope if:

- the paper becomes a methodology paper with selected representative scenarios;
- claims are limited to that declared scope;
- title and abstract do not imply full registry coverage.

### Decision C: ATLAS main claim versus appendix

Make this decision before TIM-0402.

Use ATLAS as main claim only if readiness reaches journal-grade.
Otherwise keep ATLAS as confirmatory appendix or future work.

## Minimum viable IEEE TIM submission

The smallest defensible IEEE TIM submission is not the full dream paper. It is:

- 17 or 18 estimators, explicitly declared.
- A justified subset or full set of scenarios.
- n >= 100 Monte Carlo runs for main claims.
- Dedicated timing run.
- Validity/diagnostic appendix policy.
- Reproducible artifact archive.
- Measurement-focused thesis.
- No ATLAS main claim unless ATLAS readiness passes.

## Award-level stretch goals

These are not required for submission, but they improve distinctiveness:

- Full 32 x 18 tuned replay.
- ATLAS journal-grade severity sweeps.
- Public DOI archive plus reproducibility capsule.
- Interactive artifact browser or static dashboard.
- Reviewer-friendly supplement with one-command reproduction.
- Theory section linking estimator families to observed failure modes:
  window length, phase discontinuity, harmonic contamination, nonlinear startup,
  state-space model mismatch, and runtime/latency constraints.
- Clean visual identity: IEEE-ready figures, consistent symbols, greyscale-safe
  plots, and trace examples in mHz.

## Immediate next tickets

Start here:

1. TIM-0001: finish the IEEE tuning-trace plot CLI and tests.
2. TIM-0101: write `docs/IEEE_TIM_TARGET_AUDIT.md`.
3. TIM-0203: decide PI-GRU status.
4. TIM-0301: design the scalable full tuning artifact run.
5. TIM-0401: decide whether ATLAS is main claim, appendix, or future work.

