# Journal results protocol

This file defines the path from MVP 2.0.0 to paper-grade results.

## Goal

The Journal result set must answer engineering questions, not just rank methods.
The useful questions are:

- Which estimators keep frequency error low across IEEE, NERC, IBR, harmonic,
  magnitude-step, phase-jump, and ramp cases?
- Which estimators are fast enough for online use?
- Which methods have low RMSE but poor RoCoF behavior?
- Which estimators fail under high RoCoF, severe noise, or IBR ringdown?
- Does PI-GRU generalize outside the conditions that favor it?
- Which estimator family is the best default for protection, monitoring, and
  offline analysis?

## Required runs

Use three layers of evidence.

1. `openfreqbench run --config configs/journal-paper-replay.yaml`

   This replays the paper-style tuned artifacts through the V2 report contract.
   It is the bridge from the existing paper workflow to OpenFreqBench.
   Install `openfreqbench[benchmark-full]` first because the matrix includes
   PI-GRU.

2. `python -m pipelines.full_mc_benchmark`

   This keeps the original full Monte Carlo workflow alive. Use it when the
   paper figures or legacy artifact structure need to be regenerated.

3. `python -m pipelines.rocof_sweep_fixed_policy`

   This produces the RoCoF atlas. It answers whether tracking error grows
   smoothly with ramp stress, whether methods saturate, and whether positive and
   negative ramps behave differently.

Do not replace these runs with a smoke test. Smoke tests check software. They do
not support paper claims.

## Artifact standard

Every claim must point to files, not memory or old notes. Keep these files:

- `benchmark_report.json`
- `raw_run_records.csv`
- `aggregated_metrics.csv`
- `analysis_summary.json`
- `analysis_summary.md`
- hypothesis outputs
- plots used in the paper
- reproducibility manifest with source and checkpoint hashes
- sweep manifests and aggregate CSVs for RoCoF, voltage, and magnitude-step runs

Generated per-run folders can stay out of git if the aggregate files and
manifests are archived.

## Statistical standard

Report:

- Monte Carlo run count and base seed
- parameter policy (`default`, `explicit`, or `artifact_tuned`)
- estimator and scenario sets
- confidence intervals for estimator means
- preregistered hypothesis results with correction method
- failure counts or invalid-output rates where relevant
- CPU timing hardware and dependency versions

Use exploratory plots to find patterns. Use preregistered hypotheses for claims.

## Paper preservation

Keep the original paper results until V2 reproduces them. When a number changes,
write down why:

- new metric profile
- corrected estimator behavior
- changed checkpoint
- changed tuning policy
- changed scenario list
- changed Monte Carlo count or seed

If the reason is unknown, do not update the paper number.

## Acceptance criteria

The result set is Journal-ready when:

- V2 can regenerate the main tables and plots from a public branch.
- The report includes source hashes and checkpoint hashes.
- The RoCoF atlas has aggregate CSV, manifest, PDF/PNG figures, and hypothesis
  classifications.
- A fresh clone or wheel install can reproduce at least one confirmatory subset.
- The paper cites artifact paths or hashes for every quantitative claim.
