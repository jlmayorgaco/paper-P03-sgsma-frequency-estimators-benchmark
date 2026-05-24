# Researcher Contract

OpenFreqBench is meant to make benchmark participation easy without weakening
the benchmark itself.

## Allowed

- Add estimator code with `step(...)` or `step_vectorized(...)`.
- Add scenario code that returns a valid scenario container.
- Choose canonical scenarios, estimators, run counts, seeds, and output paths in YAML.
- Add preregistered or exploratory hypotheses in YAML.

## Blocked

- Redefining metric formulas in YAML.
- Replacing the canonical sample-rate contract.
- Silently masking numerical failures in an estimator.
- Dropping canonical output fields from public reports.

## Metric rule

Metric formulas belong to the platform profile. A YAML block like this is
rejected:

```yaml
metrics:
  profile: canonical-single-phase-v1
  formulas:
    m1_rmse_hz: custom_python_code
```

Researchers can request new metrics by opening an issue or adding a new versioned
profile in code with tests and documentation.

