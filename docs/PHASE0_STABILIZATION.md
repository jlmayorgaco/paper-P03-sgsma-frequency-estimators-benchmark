# Phase 0 stabilization

Date: 2026-06-08

Phase 0 is the reset point after the SGSMA presentation. It is not a results phase.
It exists to make sure the repository can support clean future runs without mixing old
artifacts, generated slide audits, or partial Monte Carlo outputs.

## Scope

Phase 0 includes:

- keeping `2026149998.pdf` as the frozen SGSMA presented-paper artifact;
- removing generated artifacts and tracked temporary outputs;
- keeping generated outputs out of Git;
- making pytest collection deterministic;
- separating optional external-data tests from core tests;
- adding a small smoke benchmark that proves the OpenFreqBench runner can write a complete
  evidence bundle.

Phase 0 does not include:

- new scientific claims;
- journal-grade runs;
- retuning estimator parameters;
- fixing estimator numerical behavior;
- regenerating paper tables or slide figures.

## Acceptance gates

Run:

```powershell
.\scripts\verify_phase0.ps1
```

The gate checks:

1. `python -m openfreqbench doctor`
2. `python -m pytest --collect-only tests -q -p no:cacheprovider`
3. focused contract/config tests
4. `configs/phase0-smoke.yaml` dry run
5. actual `phase0-smoke` benchmark with `capture_signals=false`

The smoke run writes to:

```text
artifacts/openfreqbench/phase0-smoke/
```

That directory is intentionally ignored by Git. It should contain:

- `benchmark_report.json`
- `raw_run_records.csv`
- `aggregated_metrics.csv`
- `environment_report.json`
- `artifact_index.csv`
- `paper_traceability.csv`
- `evidence_manifest.json`

## Known exclusions

`tests/montecarlo/temp` contains historical scripts and is excluded from pytest collection.

Chamorro playback tests are skipped unless `src/scenarios/data/chamorro_data.csv` is present.
That CSV is an optional external dataset and should not block the core package gate.

## Exit criteria

Phase 0 is complete when:

- the verification script passes;
- no generated benchmark outputs are tracked by Git;
- the SGSMA paper PDF remains intact;
- `configs/phase0-smoke.yaml` produces a manifest-backed run;
- remaining full-suite failures are listed as Phase 1 engineering work rather than hidden.

## Phase 1 handoff

Phase 1 should focus on package quality:

1. repair or quarantine failing estimator behavior tests;
2. decide the canonical CPU estimator set versus optional GPU estimators;
3. create a clean `phase1-integration.yaml`;
4. make `openfreqbench quality-gate` meaningful again;
5. prepare an installable OpenFreqBench release candidate.
