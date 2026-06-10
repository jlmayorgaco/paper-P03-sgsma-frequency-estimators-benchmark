# Post-SGSMA route

Date: 2026-06-08

This repo now treats the SGSMA submission as a frozen presented-paper artifact. The file
`2026149998.pdf` is the record of what was presented. New work should not silently revise
those claims. Any follow-up must start from a clean OpenFreqBench/ATLAS evidence contract.

## Position

Do not go straight to a transaction/journal results paper yet. The current strongest path is:

0. Complete Phase 0 stabilization: clean artifacts, deterministic test collection, and a
   manifest-backed smoke run.
1. Stabilize OpenFreqBench as research software.
2. Produce one clean reproducibility release with small and full benchmark runs.
3. Use that release to choose between a software paper and a results/theory paper.

This avoids wasting the SGSMA work while preventing mixed old artifacts from contaminating
new claims.

## Recommended path: software-first, then journal

### Phase 0: stabilization

Goal: reset the repository after the SGSMA presentation without making new claims.

Contract: see `docs/PHASE0_STABILIZATION.md`.

### Phase 1: Clean software release

Goal: OpenFreqBench v2.1 as a usable benchmark package.

Current contract: see `docs/PHASE1_QUALITY_BASELINE.md`.

Required gates:

- `openfreqbench doctor` passes on a fresh clone.
- Pytest collection is clean; no top-level test scripts execute during import.
- Temporary and generated folders stay out of Git: `artifacts/`, `output/`, `outputs/`,
  `scripts/out/`, build folders, pytest caches, and LaTeX auxiliaries.
- A smoke benchmark runs with `capture_signals=false` and writes a manifest.
- Public configs distinguish `smoke`, `paper-replay`, and `journal-full`.
- README explains install, one-command smoke run, output schema, and how to cite.

Likely publication route after this phase:

- JOSS if the repo is public, documented, tested, installable, and has enough open-source
  development history. JOSS is best when the paper is about the software, not new benchmark
  results.
- SoftwareX if the story is "software plus research use case" and a short software paper is
  enough.

### Phase 2: Clean benchmark evidence

Goal: one reproducible OpenFreqBench evidence layer, separate from SGSMA.

Required gates:

- No legacy tuned artifact mismatch.
- No partial estimator matrix unless explicitly declared.
- `artifact_index.csv`, `paper_traceability.csv`, `benchmark_report.json`, and
  `evidence_manifest.json` are produced by the runner.
- Canonical run uses a fixed estimator/scenario matrix and fixed seeds.
- Signal CSV capture is disabled by default; only summaries and selected traces are retained.

Minimum run ladder:

1. `smoke`: 2-3 scenarios, 3-5 estimators, 2-5 seeds.
2. `integration`: full estimator list, 2-3 scenarios, 10 seeds.
3. `paper-grade-preview`: medium matrix, 3-5 seeds, diagnostic only.
4. `paper-grade`: full matrix, 30 seeds, no mixed artifacts.
5. `journal-grade`: full matrix, 100 seeds, archived manifest and hashes.

Current paper-grade contract: see `docs/PHASE2G_PAPER_GRADE_30SEED.md`.

### Phase 3: Results/theory paper

Goal: a new manuscript whose claims come only from Phase 2 evidence.

Best fit:

- IEEE Transactions on Smart Grid if the paper is framed around low-inertia smart-grid
  measurement/control implications, IBR stress regimes, and estimator behavior under smart-grid
  operating conditions.
- IEEE Open Access Journal of Power and Energy if the paper is broader: measurement,
  monitoring, power-system operation, reproducible benchmarking, and practical estimator
  selection.

Do not submit this route until:

- Full benchmark results are regenerated from clean configs.
- ATLAS readiness passes under the current code, not an old readiness JSON.
- The theory is stated as a falsifiable benchmark taxonomy: dynamic events, spectral
  contamination, nonlinear relay risk, cost/latency, and estimator failure modes.
- Slides/paper/results all use the same counts: estimators, scenarios, seeds, and families.

## Optional route: ATLAS-focused methods paper

This is separate from OpenFreqBench release.

Use it only if ATLAS becomes a real contribution:

- Severity sweeps have a stable metric definition.
- Invalid-output handling is defensible.
- Fixed-policy and oracle-policy results are separated.
- PI-GRU/MUSIC/canonical estimator sets are declared consistently.
- n=100 journal-grade runs are complete.

## What SGSMA becomes

SGSMA should be cited internally as:

- a presented pilot study,
- motivation for the benchmark platform,
- evidence that the problem is worth formalizing,
- not the source of new post-SGSMA numerical claims.

## Immediate work queue

Completed:

- artifact cleanup and ignore-policy change;
- deterministic test collection and optional Chamorro playback handling;
- Phase 0 smoke run;
- Phase 1 integration run;
- IPDFT/LKF/LKF2 numerical-debt closure;
- Phase 2-C paper-grade preview bundle.
- Phase 2-D invalid-output/startup policy for Prony, ESPRIT, and Koopman.
- Phase 2-E paper-grade preview v2 with `m36_post_startup_invalid_rate`.
- Phase 2-F deterministic report classification for main comparison versus diagnostic appendix.
- Phase 2-G 30-seed paper-grade run and Phase 2-F report over that run.
- Phase 2-H manuscript claim ledger from the Phase 2-G report.
- Phase 2-I first Results draft from the Phase 2-H claim ledger.
- Phase 2-J Methods/Experimental Setup draft for the Phase 2-G evidence contract.

Next:

1. Decide the manuscript route: results/theory paper or software-first paper.
2. Move `docs/PHASE2J_METHODS_EXPERIMENTAL_SETUP_DRAFT.md` and
   `docs/PHASE2I_RESULTS_DRAFT.md` into the chosen manuscript format.
3. Keep CPU ranking, PI-GRU, ATLAS, and n=100 journal-grade claims out of the main
   results unless new evidence is generated.
4. Draft the Introduction around the same evidence boundary: regime-dependent
   estimator behavior, post-startup validity, and reproducible benchmarking.

## Venue notes checked on 2026-06-08

- JOSS submission docs: https://joss.readthedocs.io/en/latest/submitting.html
- JOSS paper format: https://joss.readthedocs.io/en/latest/paper.html
- SoftwareX aims/scope: https://www.sciencedirect.com/journal/softwarex
- IEEE Transactions on Smart Grid scope: https://ieee-pes.org/publications/transactions-on-smart-grid/
- IEEE Open Access Journal of Power and Energy scope:
  https://ieee-pes.org/publications/open-access-journal-of-power-and-energy/
