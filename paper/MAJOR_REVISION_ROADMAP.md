# Major Revision Roadmap

This roadmap converts the reviewer findings into small tickets that can be
executed one by one. Tickets marked `done` were addressed in the current
revision pass. Tickets marked `pending` still need code, artifact, or
manuscript work.

## Ticket MR-01
Status: `done`
Title: Freeze benchmark provenance for the submission
Goal: Eliminate ambiguity about which benchmark release the manuscript reports.
Files:
- `paper/Config/abstract.tex`
- `paper/Sections/C3_Methods/main.tex`
- `paper/Sections/C5_Conclusions/main.tex`
Acceptance:
- The paper names one frozen benchmark snapshot.
- The paper explains why `34` scenario-level aggregates appear in the export.
- All tables and figures are stated to come from that frozen report.

## Ticket MR-02
Status: `done`
Title: Normalize small frequency errors to mHz
Goal: Improve readability in abstract, tables, and results discussion.
Files:
- `paper/Config/abstract.tex`
- `paper/Sections/C3_Methods/main.tex`
- `paper/Sections/C4_Simulation_Results/main.tex`
Acceptance:
- Frequency errors in tables and prose use `mHz` when reported as benchmark
  values.
- No more awkward values such as `0.000388 Hz` in the narrative.

## Ticket MR-03
Status: `done`
Title: Add benchmark scope table
Goal: Make the protocol legible at submission time without opening the code.
Files:
- `paper/Sections/C3_Methods/main.tex`
Acceptance:
- The manuscript includes scenario-family counts.
- The manuscript explains what each family is intended to test.

## Ticket MR-04
Status: `done`
Title: Replace winner-only summary with uncertainty-aware family table
Goal: Show margins, runner-up separation, and within-scenario Monte Carlo spread.
Files:
- `paper/Sections/C4_Simulation_Results/main.tex`
Acceptance:
- The main results table includes family counts, winners, uncertainty, and
  margin to runner-up.
- The table no longer looks like a selective leaderboard.

## Ticket MR-05
Status: `done`
Title: Add statistical robustness language
Goal: Back the manuscript with archived significance evidence instead of
point-estimate prose alone.
Files:
- `paper/Sections/C4_Simulation_Results/main.tex`
Acceptance:
- The paper states what inferential tests were archived in the report.
- The paper labels them as secondary evidence, not as the main story.

## Ticket MR-06
Status: `done`
Title: Strengthen the novelty and prior-work gap
Goal: Make it explicit that the contribution is a benchmark protocol, not a new
estimator.
Files:
- `paper/Sections/C2_Related_Work/main.tex`
Acceptance:
- Related work states what prior benchmarks do.
- Related work states the exact gap this paper addresses.

## Ticket MR-07
Status: `done`
Title: Tighten figure provenance and captioning
Goal: Prevent reviewer confusion about which panels are scenario exemplars and
which are global summaries.
Files:
- `paper/Sections/C4_Simulation_Results/main.tex`
Acceptance:
- The figure caption distinguishes single-scenario panels from aggregate panels.
- The caption states the Monte Carlo basis used for the bundle.

## Ticket MR-08
Status: `done`
Title: Reconcile the manuscript with a fresh active-pipeline export
Goal: Remove the remaining risk created by the evolving modular benchmark.
Files:
- `src/pipelines/full_mc_benchmark.py`
- `artifacts/full_mc_benchmark/`
- `paper/Sections/C3_Methods/main.tex`
- `paper/Sections/C4_Simulation_Results/main.tex`
Acceptance:
- A complete active-pipeline export exists under `artifacts/full_mc_benchmark/`.
- The paper either migrates to that export or explicitly justifies staying on
  the frozen archival snapshot.
- Estimator counts, scenario counts, and Monte Carlo counts match the chosen
  source exactly.
Resolution:
- The manuscript now documents the active modular pipeline state from code and
  explicitly justifies staying on the archived submission snapshot because the
  canonical artifact root did not contain a complete JSON export at freeze
  time.
- The paper includes a source-alignment table so the active code authority and
  archived quantitative source are not conflated.

## Ticket MR-09
Status: `done`
Title: Add a rebuttal-grade reproducibility appendix or supplement
Goal: Prepare for reviewer requests about artifact traceability without bloating
the six-page manuscript.
Files:
- `paper/`
- optional supplement file
Acceptance:
- One compact supplement lists estimator registry, scenario list, metrics, and
  archive hash or filename.
- The main paper can point to that supplement in a rebuttal or camera-ready
  cycle.
Resolution:
- Added `paper/SUPPLEMENT_TRACEABILITY.md` with archival path, SHA-256,
  estimator list, scenario list, metric contract, and active-vs-archived
  provenance notes.

## Recommended execution order
1. `MR-01` through `MR-09`: complete

The remaining work is no longer roadmap cleanup. It is a scientific extension:
generate a fresh full export from the active modular pipeline and decide
whether a later submission should migrate the paper fully to that benchmark.
