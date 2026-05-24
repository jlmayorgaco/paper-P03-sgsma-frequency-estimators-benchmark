# User Guide

## Quick test

```bash
openfreqbench quick-test --scenario IEEE_Single_SinWave --estimator ZCD --n-runs 1
```

## Compare estimators

```bash
openfreqbench compare --scenario IEEE_Freq_Step --estimator ZCD --estimator IPDFT --n-runs 3
```

## YAML run

```bash
openfreqbench init --template compare --output my_compare.yaml
openfreqbench run --config my_compare.yaml --dry-run
openfreqbench run --config my_compare.yaml
```

## Reproduce tuned artifacts

Use this when you want to replay parameter choices from a final tuned benchmark
artifact directory:

```yaml
benchmark:
  parameter_policy: artifact_tuned
  tuned_artifacts_dir: artifacts/full_mc_benchmark
```

Then run:

```bash
openfreqbench run --config configs/tuned-artifacts.yaml
```

## Add an estimator

Create a Python file with a class:

```python
class MyEstimator:
    name = "MyEstimator"

    def step(self, z, t_s=None, memory=None):
        return 60.0
```

Reference it in YAML:

```yaml
benchmark:
  scenarios:
    - IEEE_Single_SinWave
  custom_estimators:
    - name: MyEstimator
      path: examples/custom_estimators/my_estimator.py
      class: MyEstimator
```

## Run hypotheses

```bash
openfreqbench hypotheses generate --scope canonical --output hypotheses.generated.yaml
openfreqbench hypotheses run \
  --hypotheses hypotheses.generated.yaml \
  --schema hypotheses_schema.yaml \
  --input-json artifacts/openfreqbench/compare-zcd-ipdft/benchmark_report.json \
  --output-dir artifacts/openfreqbench/compare-zcd-ipdft/stats
```

## Generate plots and analysis

```bash
openfreqbench report build \
  --input-json artifacts/openfreqbench/compare-zcd-ipdft/benchmark_report.json
```

The report builder writes:

- `analysis_summary.md`
- `analysis_summary.json`
- `raw_run_records.csv`
- `aggregated_metrics.csv`
- `plots/rmse_by_estimator.png`
- `plots/cpu_by_estimator.png`
- `plots/pareto_rmse_cpu.png`
- `plots/scenario_rmse_heatmap.png`
- `plots/family_rmse_boxplot.png`
- `plots/trace_<scenario>__<estimator>.png` when signal CSVs exist

## Quality gate

```bash
openfreqbench quality-gate
```

Use this before publishing results or tagging a release.
