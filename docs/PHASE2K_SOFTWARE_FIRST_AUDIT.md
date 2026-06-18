# Phase 2-K Software-First Audit

Date: 2026-06-10

Purpose: audit whether OpenFreqBench is ready to become the primary post-SGSMA
publication object: an open research software package with a reproducible
benchmark use case.

## Executive Verdict

OpenFreqBench is a credible software-first route. The repository now has a real
Python package, CLI, locked metric profile, scenario and estimator registries,
public output schemas, reproducibility manifests, release scripts, tests,
contribution files, citation metadata, and a clean Phase 2-G benchmark evidence
bundle.

It is not submission-ready today for JOSS. The blockers are open-source process
and publication packaging, not the core benchmark implementation:

- no `.github/workflows` CI is present;
- no local release tags are present;
- no JOSS `paper.md` or SoftwareX manuscript package is present;
- `.zenodo.json` exists, but there is no confirmed software DOI/release archive;
- AI-assisted development/documentation has not yet been disclosed in a formal
  manuscript statement;
- local history starts on 2026-04-01, which is less than six months before this
  audit date if the local mirror reflects the public repository history.

Best route: keep `2026149998.pdf` as the presented SGSMA record, then prepare an
OpenFreqBench software paper. Phase 2-G should be framed as a reproducible
reference use case, not as the main claim of a results-first journal paper.

## Audit Inputs

Local checks run in this audit:

- `python -m openfreqbench doctor`
- `python -m openfreqbench manifest`
- `python -m openfreqbench run --config configs\quick.yaml --dry-run`
- `python -m openfreqbench quality-gate --skip-tests`
- `.github` existence check
- `git tag --list`
- `git remote -v`
- local commit-history inspection
- local test-file inventory

External venue criteria checked on 2026-06-10:

- JOSS submission requirements:
  https://joss.readthedocs.io/en/latest/submitting.html
- JOSS review criteria:
  https://joss.readthedocs.io/en/latest/review_criteria.html
- JOSS paper format:
  https://joss.readthedocs.io/en/latest/paper.html
- SoftwareX guide for authors:
  https://www.sciencedirect.com/journal/softwarex/publish/guide-for-authors

## Current Software Surface

| Area | Current state | Evidence |
| --- | --- | --- |
| Package metadata | Python package `openfreqbench`, version `2.0.0`, MIT license, Python `>=3.10` | `pyproject.toml`; `LICENSE` |
| CLI | `doctor`, `manifest`, `list`, `init`, `run`, `quick-test`, `compare`, `report`, `archive`, `schema`, `quality-gate`, `hypotheses` | `src/openfreqbench/cli.py` |
| Registry | 32 scenarios, 18 canonical estimators, 37 metrics under `canonical-single-phase-v1` | `openfreqbench doctor`; `src/openfreqbench/registry.py` |
| Runner | YAML benchmark execution, raw records, aggregate metrics, report JSON, environment report, artifact index, traceability, evidence manifest | `src/openfreqbench/runner.py` |
| Scientific reporting | Scope filter, diagnostic appendix, bootstrap CIs, failure analysis, Pareto recommendations, ranking sensitivity, IBR robustness | `src/openfreqbench/scientific.py`; `src/openfreqbench/reports.py` |
| User docs | README, user guide, tutorials, architecture, methods, validation, output schema, release checklist | `README.md`; `docs/` |
| Contribution path | Contributing, code of conduct, security, citation metadata, Zenodo metadata | `CONTRIBUTING.md`; `CODE_OF_CONDUCT.md`; `SECURITY.md`; `CITATION.cff`; `.zenodo.json` |
| Tests | 107 test files detected; prior full gate recorded `359 passed, 9 skipped` | `tests/`; `docs/PHASE2G_PAPER_GRADE_30SEED.md` |
| Evidence bundle | Phase 2-G: 14 scenarios, 17 estimators, 238 pairs, 7140 raw records, 234 main pairs, 4 diagnostic appendix pairs | `docs/PHASE2G_PAPER_GRADE_30SEED.md`; `docs/PHASE2H_MANUSCRIPT_CLAIM_LEDGER.md` |

## Readiness Matrix

| Dimension | Status | Assessment |
| --- | --- | --- |
| Research-software scope | Strong | The package solves a real reproducible benchmarking problem for power-system frequency-estimator research. |
| Packaging | Good | `pyproject.toml` uses modern Python packaging and exposes `openfreqbench` and `ofb` console scripts. Needs clean-wheel evidence in the public release. |
| CLI usability | Strong | The CLI covers discovery, smoke runs, YAML runs, reporting, schema validation, archiving, and quality checks. |
| Extensibility | Good | Custom estimators and scenario/estimator contracts exist. API reference is still thin. |
| Reproducibility | Strong locally | Runner writes manifests, hashes, environment reports, and traceability files. Needs DOI-backed release/archive. |
| Tests | Strong locally | Broad estimator/scenario/contract tests exist. Missing visible CI is a publication blocker. |
| Open-source process | Partial | License, contributing, conduct, security, citation files exist. Missing CI, tags, releases, issue templates, and public release evidence. |
| Software paper material | Partial | Methods and results drafts exist, but no JOSS `paper.md` or SoftwareX article package exists. |
| AI disclosure | Missing | Required for JOSS and expected for Elsevier/SoftwareX when generative AI assisted code/docs/paper drafting. |
| Results-journal evidence | Not final | Phase 2-G supports a bounded software demonstration and some accuracy/failure claims, but not n=100 uncertainty, PI-GRU generalization, ATLAS severity, or journal-grade CPU ranking. |

## Findings

### Blockers

1. Missing CI.

   `.github` is absent. JOSS review criteria explicitly look for automated tests,
   CI or documented verification, and tagged releases/formal release process. The
   local verification scripts are useful, but a software paper should have GitHub
   Actions or an equivalent public CI signal.

2. Missing release tag and DOI.

   `git tag --list` returned no tags. `.zenodo.json` is present, but there is no
   confirmed Zenodo release DOI. A citable software paper needs a frozen software
   version and a persistent identifier.

3. Missing software-paper package.

   There is no `paper/paper.md` for JOSS and no SoftwareX manuscript bundle.
   JOSS requires a short software paper with Summary, Statement of need, State of
   the field, Software design, Research impact statement, and AI usage
   disclosure.

4. JOSS public-history risk.

   The local all-branch history starts on 2026-04-01. As of 2026-06-10 that is
   about 70 days, not six months. If this reflects the public GitHub history,
   JOSS should wait until the public development record is mature enough. The
   earliest simple six-month checkpoint from 2026-04-01 is 2026-10-01, assuming
   the repository was public from that date and active development continues.

5. Missing formal AI disclosure.

   This repository has used AI assistance in software cleanup, auditing, and
   manuscript drafting. The manuscript must disclose tool use, scope of
   assistance, and human review/validation.

### Major Risks

1. The release gate is useful but not journal-complete.

   `run_quality_gate(..., release=True)` checks required files, configs, registry
   counts, pytest, clean git state, remote origin, and citation presence. It does
   not currently enforce CI presence, release tags, DOI, JOSS paper files, issue
   templates, or AI disclosure.

2. README scope can confuse software scope and paper evidence scope.

   README correctly advertises the full software registry as 32 scenarios and 18
   canonical estimators, while Phase 2-G uses a 14-scenario, 17-estimator
   paper-grade matrix. The distinction should be stated explicitly near the
   quick-start and publication sections.

3. API documentation is below ideal software-paper level.

   Custom estimator contracts are documented, but a reviewer would benefit from a
   compact API reference covering estimator classes, scenario classes,
   configuration schema, report schema, and extension points.

4. Runner/report modules are still large.

   This is not a blocker for a software paper if the design tradeoff is
   documented, but `docs/ARCHITECTURE.md` already identifies runner/report
   refactoring as debt. A reviewer may ask why artifact writing, execution, and
   aggregation are not separated more aggressively.

5. Phase 2-G artifacts are local and ignored.

   This is good for Git hygiene, but a release needs a citable archive. The
   software paper should either archive a compact evidence bundle or provide a
   deterministic script that regenerates the demonstration outputs.

6. Python-version evidence should be public.

   Local audit ran on Python 3.13.9, while `pyproject.toml` supports `>=3.10`.
   CI should test at least Python 3.10, 3.11, and 3.12. Python 3.13 can be
   included if dependency stability permits.

## Strengths to Preserve

- The software has a clear scientific contract: locked metric profile,
  single-phase scope, paired Monte Carlo seeds, parameter-policy tracking, and
  explicit validity metrics.
- The runner writes machine-readable artifacts rather than relying on manual
  spreadsheets.
- The report layer separates main-comparison pairs from diagnostic appendix
  pairs without deleting raw data.
- The repository has a real user workflow: install, `doctor`, quick test, YAML
  run, report build, schema validation, archive.
- Phase 2-H is a good claim ledger because it prevents post-SGSMA numerical
  drift.

## Venue Assessment

### JOSS

Fit: strong long-term fit.

Submission today: no-go.

Reason: the software story matches JOSS, but JOSS currently gates on public
development history, open-source practices, tests/CI, releases, documentation,
paper files, and AI disclosure. The code and docs are close; the open-project
evidence is not yet complete.

JOSS framing:

> OpenFreqBench is a reusable open benchmark platform for single-phase grid
> frequency-estimator evaluation. Its contribution is the auditable benchmark
> contract: canonical metrics, matched Monte Carlo scenarios, extensible
> estimator interfaces, validity-aware reporting, and reproducible artifact
> manifests.

Phase 2-G should be one demonstration of the platform, not the center of the
paper.

### SoftwareX

Fit: good near-term fit.

Submission today: conditional no-go.

Reason: SoftwareX can support a "software plus research use case" article more
comfortably than JOSS if public-history timing is a problem. It still needs a
formal manuscript, citable software version, data/software availability
statements, software citation metadata, and AI declaration.

SoftwareX framing:

> OpenFreqBench provides an installable benchmark engine and artifact contract
> for comparing frequency estimators under reproducible grid-event scenarios.
> A Phase 2-G benchmark demonstrates how the software produces claim-traceable
> tables and diagnostic failure classifications.

### Results/Theory Journal

Fit: later route.

Submission today: no-go.

Reason: Phase 2-G supports bounded accuracy and failure-mode claims, but a
results-first transaction/journal article should wait for dedicated timing,
n=100 uncertainty if central, PI-GRU inclusion if discussed, and ATLAS severity
evidence if claimed.

## Recommended Next Phase: Phase 2-L

Phase 2-L should convert this audit into an open-software release candidate.

Required work:

1. Add CI.
   - GitHub Actions for Python 3.10, 3.11, 3.12.
   - Run pytest, quick dry-run, schema validation, and wheel build.

2. Extend the release gate.
   - Check `.github/workflows`.
   - Check at least one release tag exists for release mode.
   - Check `paper/paper.md` or `docs/SOFTWAREX_MANUSCRIPT_PLAN.md`.
   - Check AI disclosure file/section exists.
   - Check DOI fields after Zenodo release.

3. Create the software-paper package.
   - `paper/paper.md` and `paper/paper.bib` for JOSS.
   - Or `manuscripts/softwarex/` with abstract, highlights, data/software
     availability, AI declaration, and references.

4. Make the release citable.
   - Tag `v2.0.0` or `v2.1.0`.
   - Create GitHub release notes.
   - Archive the release in Zenodo.
   - Update `CITATION.cff` with DOI after archive creation.

5. Clarify software scope versus evidence scope.
   - Software registry: 32 scenarios, 18 canonical estimators.
   - Phase 2-G demo matrix: 14 scenarios, 17 estimators.
   - Full journal replay remains future work unless regenerated and archived.

6. Add API documentation.
   - Estimator contract.
   - Scenario contract.
   - YAML config schema.
   - Output schema.
   - Custom estimator example with expected output.

7. Archive or regenerate the demonstration bundle.
   - Either archive the Phase 2-G report bundle with hashes, or provide a
     small deterministic demo that reviewers can rerun quickly.

## Decision

Proceed software-first.

Do not submit JOSS immediately. Use the next phase to make OpenFreqBench an
auditable open-source release. SoftwareX can be prepared sooner than JOSS if the
six-month public-history gate is the limiting factor, but it still needs the
same release discipline: CI, tag, DOI, manuscript package, and AI disclosure.
