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

   This replays the paper-style tuned artifacts through the 2.0.0 report contract.
   It is the bridge from the existing paper workflow to OpenFreqBench.
   Install `openfreqbench[benchmark-full]` first because the matrix includes
   PI-GRU.

2. `python -m pipelines.full_mc_benchmark`

   This keeps the original full Monte Carlo workflow alive. Use it when the
   paper figures or legacy artifact structure need to be regenerated.

3. `scripts/run_atlas_paper_grade.ps1` on Windows, or
   `scripts/run_atlas_paper_grade.sh` on Linux/macOS.

   The canonical command is:

   `python -m pipelines.atlas_sweep --sweeps all --policy fixed_policy --n-runs 100 --base-seed 12345 --n-cost-reps 3 --tune-trials 80 --tune-eval-runs 5 --output-subdir atlas-paper-fixed-v2 --resume`

   This produces the unified ATLAS sweep set. Magnitude Step, RoCoF, Frequency
   Step, Phase Jump, AM Modulation, FM Modulation, Harmonics, Interharmonics,
   and White-Noise/SNR use the same scenario factory, estimator selection,
   Monte Carlo aggregation, tuning/oracle policy, manifests, traceability
   tables, and PDF plotting code. It answers whether tracking error grows
   smoothly with stress, whether methods saturate, whether positive and
   negative events behave differently, and whether ranking changes under
   isolated phase, modulation, distortion, or noise mechanisms.

   For a smaller confirmatory slice, use `--sweeps p0` for phase-jump, AM, FM,
   harmonics, interharmonics, and noise, or `--sweeps core` for magnitude-step,
   RoCoF, and frequency-step.

Do not replace these runs with a smoke test. Smoke tests check software. They do
not support paper claims.

Before moving ATLAS numbers into the paper, check
`docs/ATLAS_METHOD_AUDIT.md`, confirm that the run was generated with the
current ATLAS method version, and open `atlas_readiness_report.json`.
ATLAS numbers can enter the paper only when `paper_claims_allowed` is `true`.
Journal claims require `journal_claims_allowed=true`.

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
- ATLAS manifests and aggregate CSVs for magnitude-step, RoCoF, frequency-step,
  phase-jump, AM modulation, FM modulation, harmonics, interharmonics, and
  noise/SNR runs
- `atlas_readiness_report.json` and `atlas_readiness_report.md`

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

Keep the original paper results until OpenFreqBench 2.0.0 reproduces them. When a number changes,
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

- OpenFreqBench 2.0.0 can regenerate the main tables and plots from a public branch.
- The report includes source hashes and checkpoint hashes.
- The ATLAS run has aggregate CSV, manifest, artifact index, traceability table,
  PDF/PNG figures, readiness report, and hypothesis classifications.
- `atlas_readiness_report.json` reports `journal_grade` for the full ATLAS
  evidence set, or `paper_grade` for a clearly labeled smaller confirmatory
  result.
- A fresh clone or wheel install can reproduce at least one confirmatory subset.
- The paper cites artifact paths or hashes for every quantitative claim.
