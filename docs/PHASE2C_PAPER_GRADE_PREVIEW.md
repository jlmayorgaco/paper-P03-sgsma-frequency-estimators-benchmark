# Phase 2-C paper-grade preview

Date: 2026-06-09

Phase 2-C is a medium-size preview after numerical-debt closure. It is not final paper evidence.
Its purpose is to expose estimator-family instability, invalid-output policy gaps, and runtime
cost before widening to the full journal/transaction-grade matrix.

## Configuration

Config:

```text
configs/phase2-paper-grade-preview.yaml
```

Execution contract:

- 14 scenarios
- 17 canonical non-neural estimators
- 3 runs per pair
- 238 scenario-estimator pairs
- 714 raw records
- `capture_signals=false`
- `BENCHMARK_N_COST_REPS=1`
- PI-GRU excluded until neural dependency/weights policy is explicit

The run writes ignored artifacts to:

```text
artifacts/openfreqbench/phase2-paper-grade-preview/
```

## Verification

Dry run passed.

The full benchmark command exceeded the shell wrapper timeout after producing the complete
artifact bundle. The bundle was audited directly:

- `benchmark_report.json`: present
- `raw_run_records.csv`: present, 714 records
- `aggregated_metrics.csv`: present, 238 rows
- benchmark-report schema validation: pass
- generated `signals.csv` files: 0
- `openfreqbench quality-gate`: pass
- full pytest through quality gate: 352 passed, 9 skipped, 0 xfailed

Because the shell wrapper returned a timeout status, this preview should be treated as a valid
diagnostic bundle, not as archival final evidence.

## Findings

No active unit-test numerical debt remains after Phase 2-B, and LKF/LKF2 stayed executable in
the preview without invalid outputs.

Observed preview risks:

- Invalid-output rates are concentrated in window/exotic estimators, especially Prony and
  ESPRIT, with Koopman showing a smaller startup-invalid footprint.
- Prony reached the highest invalid rate in `IBR_Multi_Event` (`m22_invalid_output_rate_mean`
  about 0.0816).
- ZCD and RLS show large RMSE under severe IBR or harmonic stress. This is expected baseline
  weakness, not a software failure, but it must be presented as such.
- LKF/LKF2 are no longer unit-test debt. In the preview their worst RMSE is under severe
  `IBR_Multi_Event`, which should be interpreted as scenario stress sensitivity rather than a
  broken nominal contract.
- IPDFT remains valid, but large phase-jump peak errors should be treated as structural
  transient behavior and reported separately from steady-state error.

## Phase 2-D resolution

Phase 2-D is now defined in `docs/PHASE2D_WINDOWED_ESTIMATOR_POLICY.md`.

The core decision is:

1. `m22_invalid_output_rate` includes expected startup invalids and is not sufficient by
   itself to disqualify a windowed estimator;
2. `m36_post_startup_invalid_rate` is the invalid-output failure metric;
3. Prony scenario-estimator pairs with `m36 > 0` or non-finite accuracy metrics move to a
   diagnostic appendix and must be excluded from main aggregate rankings;
4. ESPRIT and Koopman may stay in the main comparison when `m36 == 0`, with high severe-IBR
   RMSE interpreted as performance weakness rather than invalid-output failure.

Before final paper-grade runs, rerun this preview with log capture and a longer execution
timeout so the artifact bundle includes `m36`.
