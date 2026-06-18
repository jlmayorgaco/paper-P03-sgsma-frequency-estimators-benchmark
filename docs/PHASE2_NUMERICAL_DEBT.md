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

None after Phase 2-B.

## LKF status

Closed in Phase 2-B.

The LKF two-state model was tracking the mean frequency correctly, but its default tuning
allowed a large periodic phase-difference ripple in pure nominal and noisy nominal cases. The
release default now uses a lower process-noise to measurement-noise ratio and lighter output
smoothing:

- `q=3e-8`
- `r=1e-2`
- `output_smoothing=0.005`

This keeps the nominal/noisy ripple within the strict tests while preserving the existing
step-tracking contract.

## LKF2 status

Closed in Phase 2-B.

The LKF2 frequency loop used `omega_leak=0.995`, which created a steady-state frequency bias
after a 50 -> 52 Hz step. The default is now `omega_leak=1.0`, removing the post-step
under-tracking without breaking nominal or noisy cases.

## Next gate

Verification after Phase 2-B on 2026-06-09:

- focused IPDFT/LKF/LKF2 checks: 9 passed
- full pytest: 352 passed, 9 skipped, 0 xfailed
- `openfreqbench quality-gate`: pass

Run the focused estimator checks before creating a paper-grade preview:

```powershell
python -m pytest tests/estimators/ipdft/test_ipdft.py tests/estimators/lkf/test_lkf.py tests/estimators/lkf2/test_lkf2.py -q -p no:cacheprovider
```

Then run:

```powershell
python -m pytest tests -q -p no:cacheprovider --basetemp artifacts/pytest-tmp
python -m openfreqbench quality-gate
```
