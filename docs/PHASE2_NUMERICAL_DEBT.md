# Phase 2 numerical debt

Date: 2026-06-09

Phase 2 repairs estimator-level behavior before any wider paper-grade benchmark run. It should
change estimator contracts or tuning first, then remove `xfail(strict=True)` only when the
corresponding test passes as a normal assertion.

## IPDFT status

Closed.

The IPDFT issue was a sample-rate contract bug in direct `step_vectorized(...)` usage. The
estimator already tracked the 52.5 Hz off-nominal case through `estimate(t, v)`, because
`estimate(...)` synchronized `dt` from the input time vector. Direct vectorized use with
`decim=100` kept the default DSP `dt`, which made the effective DSP rate 100 Hz instead of
10 kHz.

The contract is now explicit:

- `dt` is the input-sample period;
- `decim` is the input-sample decimation factor;
- the internal DSP sample rate is `1 / (dt * decim)`;
- `structural_latency_samples()` reports latency in input-sample units.

The IPDFT tests no longer carry `known_numerical_debt`.

Verification on 2026-06-09:

- `tests/estimators/ipdft/test_ipdft.py`: 3 passed
- `tests/scenarios/nerc_phase_jump_60/test_nerc_phase_jump_60.py`: 12 passed
- full pytest: 349 passed, 9 skipped, 3 xfailed
- `openfreqbench quality-gate`: pass

## Remaining debt

- `tests/estimators/lkf/test_lkf.py::test_lkf_nominal_pure_sine`
- `tests/estimators/lkf/test_lkf.py::test_lkf_robustness_to_noise`
- `tests/estimators/lkf2/test_lkf2.py::test_lkf2_step_tracking`

## Next gate

Run the focused estimator checks before moving to LKF/LKF2:

```powershell
python -m pytest tests/estimators/ipdft/test_ipdft.py -q -p no:cacheprovider
```

After LKF/LKF2 are resolved, run:

```powershell
python -m pytest tests -q -p no:cacheprovider --basetemp artifacts/pytest-tmp
python -m openfreqbench quality-gate
```
