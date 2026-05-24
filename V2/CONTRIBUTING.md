# Contributing to OpenFreqBench

OpenFreqBench welcomes estimator, scenario, documentation, and reproducibility
contributions.

## Estimators

Add an estimator class with either:

- `step(z, t_s=None, memory=None) -> float`
- `step_vectorized(v) -> np.ndarray`

The class must expose a stable `name`. Constructor parameters should have
defaults and be serializable in YAML.

## Scenarios

Scenario contributions must return validated single-phase 10 kHz data through
the scenario contract. Do not bypass `Scenario.run()`.

## Metrics

Metric formulas are part of the versioned platform profile. Do not add metric
formulas in YAML. New metrics need code, tests, docs, and a profile-version
decision.

## Before opening a pull request

```bash
python -m pytest tests -q
openfreqbench run --config configs/quick.yaml --dry-run
openfreqbench quick-test --scenario IEEE_Single_SinWave --estimator ZCD --n-runs 1
```

