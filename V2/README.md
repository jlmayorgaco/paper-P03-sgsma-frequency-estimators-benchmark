# OpenFreqBench

OpenFreqBench is an open source benchmark platform for grid frequency estimators.
MVP 2.0.0 starts with the single-phase benchmark from the SGSMA paper codebase
and wraps it in a public CLI, YAML configs, fixed metric guardrails,
reproducible manifests, reports, and plots.

The design goal is simple: researchers can change estimators and scenarios, but
the benchmark method and metric formulas stay locked by the platform profile
`canonical-single-phase-v1`.

## Install

From a local checkout:

```bash
python -m pip install -e ".[dev]"
```

From the public branch:

```bash
python -m pip install "openfreqbench @ git+https://github.com/jlmayorgaco/paper-P03-sgsma-frequency-estimators-benchmark.git@MVP2.0.0#subdirectory=V2"
```

For PI-GRU:

```bash
python -m pip install -e ".[benchmark-full]"
```

For future WAMS/dynamic-system extensions:

```bash
python -m pip install -e ".[andes,opendss]"
```

## First commands

```bash
openfreqbench doctor
openfreqbench list scenarios
openfreqbench list estimators
openfreqbench list metrics
```

Run one estimator in one scenario:

```bash
openfreqbench quick-test --scenario IEEE_Single_SinWave --estimator ZCD --n-runs 1
```

Compare two estimators:

```bash
openfreqbench compare --scenario IEEE_Freq_Step --estimator ZCD --estimator IPDFT --n-runs 3
```

Run from YAML:

```bash
openfreqbench run --config configs/quick.yaml
openfreqbench run --config configs/compare.yaml
openfreqbench run --config configs/montecarlo.yaml
openfreqbench run --config configs/tuned-artifacts.yaml
```

Replay the paper-style tuned matrix through the V2 artifact contract:

```bash
openfreqbench run --config configs/journal-paper-replay.yaml
```

Install `.[benchmark-full]` first if the run includes PI-GRU.

Validate without running:

```bash
openfreqbench run --config configs/montecarlo.yaml --dry-run
openfreqbench quality-gate
```

## Output contract

Each run writes to `artifacts/openfreqbench/<run_id>/` by default:

- `benchmark_report.json`: machine-readable report with raw run records.
- `raw_run_records.csv`: one row per Monte Carlo run.
- `aggregated_metrics.csv`: grouped estimator/scenario summaries.
- per-scenario/per-estimator summary, signal, and run-spec files.

Hypotheses run against `benchmark_report.json`:

```bash
openfreqbench hypotheses generate --scope canonical --output hypotheses.generated.yaml
openfreqbench hypotheses run \
  --hypotheses hypotheses.generated.yaml \
  --schema hypotheses_schema.yaml \
  --input-json artifacts/openfreqbench/compare-zcd-ipdft/benchmark_report.json \
  --output-dir artifacts/openfreqbench/compare-zcd-ipdft/stats
```

Generate analysis tables and plots from any V2 run:

```bash
openfreqbench report build \
  --input-json artifacts/openfreqbench/compare-zcd-ipdft/benchmark_report.json
```

This creates `analysis_summary.md`, `analysis_summary.json`, CSV tables, and
PNG plots such as RMSE/CPU bars with bootstrap confidence intervals,
RMSE-vs-CPU Pareto, scenario heatmaps, family RMSE boxplots, and time traces
when signal CSVs are available.

## Researcher contract

OpenFreqBench V2 intentionally separates user-modifiable code from benchmark
method code:

- Estimators: researchers may add a class with `step(...)` or `step_vectorized(...)`.
- Scenarios: researchers may add scenario classes that return validated 10 kHz data.
- Metrics: researchers may select canonical metrics, but may not define formulas in YAML.
- Hypotheses: researchers may add preregistered or exploratory YAML hypotheses.

If a YAML file tries to define metric formulas, the CLI rejects it.

## Roadmap

1. Single-phase public release: current MVP 2.0.0 scope.
2. Three-phase extension: phasor/vector scenarios and per-phase metric adapters.
3. WAMS extension: OpenDSS/ANDES dynamic cases, multi-bus events, and network-aware
   estimator stress tests.

See `docs/ARCHITECTURE.md` and `docs/RESEARCHER_CONTRACT.md`.
For scientific use, also read `docs/METHODS.md`,
`docs/VALIDATION.md`, `docs/SCIENTIFIC_READINESS.md`,
`docs/ARCHITECTURE_REVIEW.md`, `docs/JOURNAL_RESULTS_PROTOCOL.md`, and
`docs/MVP2_RELEASE_NOTES.md`.
