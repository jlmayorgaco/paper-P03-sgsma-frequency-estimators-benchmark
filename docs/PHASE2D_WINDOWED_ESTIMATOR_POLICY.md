# Phase 2-D windowed estimator startup policy

Date: 2026-06-09

Phase 2-D closes the invalid-output ambiguity exposed by the Phase 2-C preview.
The issue was not one bug but a missing interpretation rule: windowed estimators
must return `NaN` during their unavoidable cold start, but those startup `NaN`
values must not be conflated with post-startup numerical failures.

## Code contract

For Prony, ESPRIT, and Koopman:

- `m21_startup_valid_samples` is the first finite estimator output in input-sample
  units.
- `m14_struct_latency_ms` is derived from the same declared latency.
- `m22_invalid_output_rate` is the total invalid-output fraction and includes
  expected startup `NaN` samples.
- `m36_post_startup_invalid_rate` is the failure metric for invalid outputs after
  the first finite output.

The implementation now enforces this contract:

- Prony latency is `ceil(window_size / execution_stride) * execution_stride`.
- ESPRIT latency is `ceil(N / execution_stride) * execution_stride`.
- Koopman exposes `execution_stride`, waits for a full window before EDMD updates,
  and uses the same stride policy in scalar and vectorized paths.

## Interpretation policy

Startup invalids are acceptable only when:

1. `m21_startup_valid_samples == structural_latency_samples()`;
2. outputs before that sample are invalid by construction;
3. outputs from that sample onward are finite in nominal conditions;
4. `m36_post_startup_invalid_rate == 0` for the tested scenario-estimator pair.

Total invalid output rate (`m22`) must therefore be reported together with `m21`
and `m36`. A nonzero `m22` alone does not disqualify a windowed estimator. A
nonzero `m36` is a numerical failure signal.

## Phase 2-D checks

Focused unit checks:

```powershell
python -m pytest tests\estimators\test_windowed_startup_policy.py tests\estimators\prony\test_prony.py tests\estimators\esprit\test_esprit.py tests\estimators\koopman\test_koopman.py -q -p no:cacheprovider --basetemp artifacts\pytest-tmp-phase2d-focused
```

Result:

```text
23 passed
```

CLI smoke config:

```text
configs/phase2-windowed-policy-smoke.yaml
```

End-to-end metric check:

```powershell
$env:BENCHMARK_N_COST_REPS='1'
python -m openfreqbench quick-test --scenario IEEE_Single_SinWave --estimator Prony --n-runs 1 --id phase2d-m36-cli-check --output-dir artifacts\openfreqbench --no-capture-signals
```

Observed nominal Prony result:

| scenario | estimator | m21 | m22 | m36 |
| --- | --- | ---: | ---: | ---: |
| IEEE_Single_SinWave | Prony | 340 | 0.02260 | 0.000000 |

Targeted IBR checks:

| scenario | estimator | runs | m21 | m22 mean | m36 mean | interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| IBR_Multi_Event | Prony | 2 | 340 | 0.51889 | 0.515605 | post-startup numerical failure |
| IBR_Multi_Event | ESPRIT | 1 | 170 | 0.00338 | 0.000000 | valid outputs after startup, high error |
| IBR_Multi_Event | Koopman (RK-DPMU) | 1 | 250 | 0.00498 | 0.000000 | valid outputs after startup, high error |

## Paper classification

For final paper-grade runs:

- Keep ESPRIT and Koopman in the main comparison when `m36 == 0`; interpret large
  RMSE/peak error under IBR stress as performance limitation, not invalid-output
  failure.
- Keep Prony in nominal/classical reference comparisons where `m36 == 0`.
- Move Prony scenario-estimator pairs to diagnostic appendix, and exclude them
  from main aggregate rankings, whenever `m36 > 0` or accuracy metrics become
  non-finite.

This preserves the scientific distinction between structural startup latency,
poor tracking performance, and true post-startup numerical failure.

## Next gate

The preview rerun is complete in `docs/PHASE2E_PAPER_GRADE_PREVIEW_V2.md`.
Before widening to 30 seeds, report/table generation must apply the Phase 2-E
main-vs-appendix classification deterministically.
