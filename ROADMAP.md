# Roadmap to v1.0.0 (Submit-Ready OSS + Academic Platform)

## Objective
Ship a reproducible, modular, and publication-grade open-source benchmark platform with clear quality gates.

## Phase 1 - Release Branch Hygiene (Day 0-1)
- Create a clean release branch focused on canonical pipeline and OSS files.
- Keep `tests/montecarlo` as legacy compatibility harness.
- Ensure heavy generated outputs are ignored by git.
- Acceptance:
  - `openfreqbench quality-gate --profile canonical` passes.
  - Repo tree is free of local paper/debug artifacts in release branch.

## Phase 2 - Scientific Reproducibility Lock (Day 1-2)
- Run full canonical benchmark with official settings.
- Generate complete canonical artifacts under `artifacts/full_mc_benchmark/`.
- Validate artifact contract with `openfreqbench benchmark validate`.
- Sync canonical figures to paper path.
- Acceptance:
  - Artifact validator passes with 32 scenarios and full estimator coverage.
  - Paper-facing figures come from canonical artifacts only.

## Phase 3 - Engineering Hardening (Day 2-3)
- Keep canonical and legacy gates green in CI.
- Add/maintain regression tests for:
  - `step()` vs `step_vectorized()`
  - ESPRIT determinism
  - Prony no silent masking
  - CPU timing contract via `process_time()`
  - PI-GRU fail-fast dependency behavior
- Acceptance:
  - Canonical CI is stable across repeated runs.

## Phase 4 - OSS Completeness (Day 3-4)
- Finalize maintainership docs:
  - `CONTRIBUTING.md`
  - `CODE_OF_CONDUCT.md`
  - `SECURITY.md`
  - `CITATION.cff`
  - `CHANGELOG.md`
- Add issue and PR templates.
- Acceptance:
  - External collaborator can install, run canonical checks, and open a PR with clear guidance.

## Phase 5 - Release & Archival (Day 4-5)
- Cut `v1.0.0` tag and release notes.
- Attach reproducibility notes:
  - commit hash
  - Python version
  - benchmark settings
  - artifact validator status
- Prepare Zenodo metadata and DOI workflow.
- Acceptance:
  - Public release is reproducible from docs only.

## Daily Execution Checklist
- [ ] Canonical gate passes locally.
- [ ] Legacy gate passes locally.
- [ ] No unintended generated outputs staged.
- [ ] Docs updated when behavior changes.
- [ ] CI green for current branch.
