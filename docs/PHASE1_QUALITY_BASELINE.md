# Phase 1 quality baseline

Date: 2026-06-08

Phase 1 turns the cleaned repository into a usable software baseline. It is still not a
scientific results phase. Its job is to make failures explicit, keep the OpenFreqBench runner
healthy, and prepare a small integration run that exercises multiple estimator families without
creating heavy artifacts.

## Acceptance gates

Run:

```powershell
.\scripts\verify_phase1.ps1
```

The gate checks:

1. `python -m openfreqbench doctor`
2. full pytest with generated outputs ignored by Git
3. `openfreqbench quality-gate`
4. `configs/phase1-integration.yaml` dry run
5. actual `phase1-integration` run with `capture_signals=false`
6. schema validation for the generated `benchmark_report.json`

The run writes to:

```text
artifacts/openfreqbench/phase1-integration/
```

That directory is ignored by Git.

## Canonical estimator policy for Phase 1

Phase 1 uses a CPU-only integration set:

- ZCD
- IPDFT
- TFT
- PLL
- SOGI-PLL
- EKF
- UKF
- RA-EKF

PI-GRU stays optional until the GPU/dependency path is made explicit. LKF/LKF2 remain in the
package and tests, but their strict behavior assertions are tracked as numerical debt before
they are used as release-critical evidence.

## Known numerical debt

These tests are marked `known_numerical_debt` and `xfail(strict=True)`. They still run; if the
behavior is fixed, pytest will report an unexpected pass and force removal from this list.

- `tests/estimators/ipdft/test_ipdft.py::test_ipdft_off_nominal_interpolation`
- `tests/estimators/ipdft/test_ipdft.py::test_ipdft_structural_latency`
- `tests/estimators/lkf/test_lkf.py::test_lkf_nominal_pure_sine`
- `tests/estimators/lkf/test_lkf.py::test_lkf_robustness_to_noise`
- `tests/estimators/lkf2/test_lkf2.py::test_lkf2_step_tracking`

## Exit criteria

Phase 1 is complete when:

- pytest passes with only documented skips/xfails;
- the quality gate passes;
- `phase1-integration` produces a schema-valid `benchmark_report.json`;
- no `signals.csv` files are produced by Phase 1 configs;
- estimator numerical debt is visible and assigned to Phase 2 repair.

## Phase 2 handoff

Phase 2 should repair estimator behavior rather than widen the benchmark:

1. resolve IPDFT off-nominal interpolation and latency-unit contract;
2. retune or redefine LKF/LKF2 acceptance thresholds;
3. decide whether LKF-family estimators are canonical or diagnostic;
4. add a `phase2-paper-grade-preview.yaml` only after the numerical debt is closed.
