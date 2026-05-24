# OpenFreqBench Tutorials

These tutorials use the public OpenFreqBench contract. They do not modify canonical metric
formulas.

## First estimator in 10 minutes

Install the package from this checkout, then copy the starter config:

```bash
python -m pip install -e ".[dev]"
openfreqbench init --template custom-estimator --output my-estimator.yaml
```

Create a class with `step(...)` or `step_vectorized(...)`. The included example
is:

```bash
examples/custom_estimators/my_estimator.py
```

Run the smoke benchmark:

```bash
openfreqbench run --config my-estimator.yaml
```

Expected output: `benchmark_report.json`, `raw_run_records.csv`,
`aggregated_metrics.csv`, per-pair `run_spec.json`, and a reproducibility
manifest.

## Compare two estimators

```bash
openfreqbench compare \
  --scenario IEEE_Freq_Step \
  --estimator ZCD \
  --estimator IPDFT \
  --n-runs 3 \
  --id compare-zcd-ipdft
```

Build plots and journal tables:

```bash
openfreqbench report build \
  --input-json artifacts/openfreqbench/compare-zcd-ipdft/benchmark_report.json
```

## Monte Carlo benchmark

```bash
openfreqbench init --template montecarlo --output montecarlo.yaml
openfreqbench run --config montecarlo.yaml
```

For paper replay use:

```bash
openfreqbench validate-artifacts --config configs/journal-paper-replay.yaml
openfreqbench run --config configs/journal-paper-replay.yaml
```

That config uses `parameter_policy: artifact_tuned` and expects tuned
`run_spec.json` files under `artifacts/full_mc_benchmark/`.

## Custom hypothesis

Hypotheses live in YAML and run against `benchmark_report.json`.

```bash
openfreqbench hypotheses generate \
  --scope canonical \
  --output hypotheses.generated.yaml

openfreqbench hypotheses run \
  --hypotheses configs/hypotheses_preregistered.yaml \
  --schema hypotheses_schema.yaml \
  --input-json artifacts/openfreqbench/compare-zcd-ipdft/benchmark_report.json \
  --output-dir artifacts/openfreqbench/compare-zcd-ipdft/stats
```

The runner writes both `statistical_tests_report.csv` and
`hypothesis_results.csv`.

## Freeze a run for a paper

```bash
openfreqbench report build \
  --input-json artifacts/openfreqbench/journal-paper-replay-v2/benchmark_report.json

openfreqbench archive \
  --run-root artifacts/openfreqbench/journal-paper-replay-v2 \
  --config configs/journal-paper-replay.yaml
```

Use `paper_traceability.csv` for manuscript numbers. If a number is not in that
file or another archived CSV with a hash, it should not enter the paper.
