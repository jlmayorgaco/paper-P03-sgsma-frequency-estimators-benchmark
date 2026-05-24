# Examples

## Quick Benchmark

```bash
openfreqbench run --config configs/quick.yaml
```

## Compare Two Estimators

```bash
openfreqbench run --config configs/compare.yaml
openfreqbench report build --input-json artifacts/openfreqbench/compare-zcd-ipdft/benchmark_report.json
```

## Add a Custom Estimator

See `examples/custom_estimators/my_estimator.py` and `configs/custom-estimator.yaml`.

```bash
openfreqbench run --config configs/custom-estimator.yaml
```

## Generate a Hypothesis Bank

```bash
openfreqbench hypotheses generate --scope canonical --output hypotheses.generated.yaml
```

