# Refactor Plan

## Goal

Consolidate the actively used benchmark code into a modular structure that is
easier to validate, debug, and connect to the paper workflow.

## Canonical modules after this refactor

- Pipeline entry point: `src/pipelines/full_mc_benchmark.py`
- Plot orchestration: `src/plotting/benchmark/generate_mega_dashboards.py`
- Plot modules:
  - `src/plotting/benchmark/mega_dashboard1.py`
  - `src/plotting/benchmark/mega_dashboard2_p*.py`
- Paper artifact sync: `src/pipelines/sync_paper_artifacts.py`

Legacy wrappers remain temporarily in `tests/montecarlo/` to avoid breaking old
commands while the migration is still in progress.

## Phase 1: completed

- Copied the working benchmark logic out of `tests/montecarlo/` into
  `src/pipelines/`.
- Moved dashboard generation into `src/plotting/benchmark/`.
- Standardized paper-facing figure names:
  - `Fig1_Scenarios_Final`
  - `Fig2_Mega_Dashboard`
- Added a sync script to copy generated dashboard PDFs/PNGs into the LaTeX
  figure directory.
- Fixed deterministic-equivalence issues in `src/estimators/esprit.py`.
- Removed silent numerical failure masking in `src/estimators/prony.py`.
- Harmonized Monte Carlo timing to repeated `process_time()` measurements.
- Removed aggressive IEEE template compression from `paper/index.tex`.
- Gated PI-GRU tests cleanly when `torch` is not installed.
- Rewrote `README.md` to reflect the actual repository layout.

## Phase 2: next code tasks

1. Unify estimator registry and scenario registry.
   - Remove duplicated registry logic between `tests/montecarlo/` and
     `src/pipelines/full_mc_benchmark.py`.
   - Keep a single source of truth for estimator labels and family mapping.

2. Align benchmark configuration with the camera-ready protocol.
   - Reconcile `N_MC_RUNS`, scenario set, and estimator inclusion/exclusion with
     `AGENTS.md`.
   - Ensure the reorganized pipeline either matches the paper benchmark exactly
     or is clearly labeled as exploratory.

3. Collapse output universes.
   - Stop writing new benchmark artifacts under `tests/montecarlo/outputs/`.
   - Use one canonical output root only.
   - Mark any legacy output directory as stale if it remains in the repo.

4. Trace manuscript numbers to generated artifacts.
   - After a fresh canonical run, update paper tables and hardcoded claims from
     the generated JSON/CSV outputs only.

5. Clean residual LaTeX warnings.
   - Resolve remaining typography warnings (`Underfull` boxes and font
     substitution warnings) only if they can be fixed without template hacks.

## Phase 3: submission lock

1. Run the full benchmark from the canonical entry point.
2. Regenerate figures.
3. Sync figures into `paper/Figures/Plots_And_Graphs/`.
4. Rebuild the paper.
5. Perform a paper-versus-artifact consistency audit.

## Non-negotiable rule

If a manuscript number cannot be traced to a generated artifact path, it does
not belong in the camera-ready version.
