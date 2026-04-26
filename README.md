# Benchmarking Dynamic Frequency Estimators for Low-Inertia IBR Grids

This repository contains the simulation code, benchmarking pipeline, plotting
modules, and LaTeX source for the SGSMA 2026 paper:

`Benchmarking Dynamic Frequency Estimators for Low-Inertia IBR Grids: A Latency-Robustness Trade-off Analysis`

## Scope

The project evaluates dynamic frequency estimators for relay-class and local
measurement use in low-inertia, inverter-based-resource (IBR) grids. The code
base includes:

- estimator implementations under `src/estimators/`
- disturbance scenarios under `src/scenarios/`
- metric and Monte Carlo logic under `src/analysis/`
- the reorganized benchmark pipeline under `src/pipelines/`
- modular dashboard plotting under `src/plotting/benchmark/`
- the paper source under `paper/`

## Repository layout

```text
.
|-- src/
|   |-- analysis/
|   |-- estimators/
|   |-- pipelines/
|   |-- plotting/
|   |   `-- benchmark/
|   |-- scenarios/
|   `-- tests/
|-- tests/
|   |-- estimators/
|   `-- montecarlo/
|-- paper/
|   |-- Config/
|   |-- Figures/
|   `-- Sections/
|-- artifacts/
|   `-- full_mc_benchmark/
`-- REVIEW.md
```

## Canonical execution paths

There are currently two code universes in the repository:

- `src/main.py`
  Used by the camera-ready paper workflow described in `AGENTS.md`.
- `src/pipelines/full_mc_benchmark.py`
  The reorganized benchmarking pipeline derived from the previous
  `tests/montecarlo/test_dedicated_smoke_test.py` workflow.

If you are working on the new modular benchmark structure, use:

```bash
cd src
python -m pipelines.full_mc_benchmark
```

Generated dashboard artifacts are written to:

- `artifacts/full_mc_benchmark/`

To copy the canonical paper figures into the LaTeX figure directory with the
expected names, run:

```bash
cd src
python -m pipelines.sync_paper_artifacts
```

## Plotting structure

Dashboard generation is split into modular subplot builders under
`src/plotting/benchmark/`. The intent is to keep each subplot independently
debuggable while preserving a single orchestration entry point:

- `src/plotting/benchmark/generate_mega_dashboards.py`
- `src/plotting/benchmark/mega_dashboard1.py`
- `src/plotting/benchmark/mega_dashboard2_p*.py`

## Python environment

Install the project dependencies in your active environment before running the
benchmark or the paper build support scripts.

Typical scientific stack required by the repository:

- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `numba`
- `optuna`
- `pytest`
- `tqdm`

Optional dependency:

- `torch`
  Required only for the PI-GRU estimator and its dedicated tests. If `torch` is
  not installed, PI-GRU-specific tests are skipped.

## Testing

Useful focused checks:

```bash
pytest src/tests/test_scalar_vs_vector.py -v
pytest tests/estimators/esprit/test_esprit.py -v
pytest tests/estimators/prony/test_prony.py -v
pytest tests/estimators/pi_gru -q
```

Layered quality gates:

```bash
# Canonical (submit gate)
cd src && python -m pipelines.run_quality_gate --profile canonical

# Legacy compatibility (kept, but outside submit gate)
cd src && python -m pipelines.run_quality_gate --profile legacy

# Expensive full run for manual/nightly
cd src && python -m pipelines.run_quality_gate --profile manual-nightly
```

Canonical artifact contract check:

```bash
cd src && python -m pipelines.validate_canonical_artifacts
```

The canonical dashboard path is strict by default. `generate_mega_dashboards`
fails if canonical reports are missing. Legacy fallback is only enabled in
legacy mode (`data_mode="legacy"` or `BENCHMARK_DASHBOARD_MODE=legacy`).

## Advanced analysis output

The benchmark JSON now includes a `robust_statistics` block with:
- bootstrap confidence intervals for RMSE and CPU metrics
- Friedman omnibus tests across estimators
- pairwise Wilcoxon tests versus the current best estimator
- pairwise scenario win-rate matrices
- multi-metric dominance score ranking

Optional environment control:
- `BENCHMARK_ADV_BOOTSTRAP_ITERS` (default: `2000`)

## Hypotheses and auto-reports

Preregistered hypotheses are defined in `hypotheses.yaml`. Generate inferential
outputs and deterministic report artifacts with:

```bash
openfreqbench stats run
openfreqbench report build
```

Outputs include:
- `artifacts/full_mc_benchmark/statistical_tests_report.json`
- `artifacts/full_mc_benchmark/statistical_tests_report.csv`
- `artifacts/full_mc_benchmark/statistical_tests_report.md`
- `reports/benchmark_report.md`
- `reports/benchmark_report.html`
- `reports/benchmark_appendix_tables.csv`
- `reports/benchmark_run_manifest.json`

## Paper build

Build from `paper/`:

```bash
pdflatex index
bibtex index
pdflatex index
pdflatex index
```

The LaTeX source now targets the IEEEtran conference template more directly.
If the manuscript exceeds the page limit after removing layout compression
hacks, content must be reduced instead of forcing the template.

## Open-source packaging

The repository now includes a `pyproject.toml` and MIT license.

## Upgrade to v1.0.0

The CLI namespace has moved from `sgsma-*` to `openfreqbench`.

- `sgsma-full-benchmark` -> `openfreqbench benchmark run`
- `sgsma-sync-paper-artifacts` -> `openfreqbench benchmark sync-paper`
- `sgsma-validate-artifacts` -> `openfreqbench benchmark validate`
- `sgsma-quality-gate --profile <name>` -> `openfreqbench quality-gate --profile <name>`

After editable install:

```bash
pip install -e .
openfreqbench benchmark run
openfreqbench benchmark sync-paper
openfreqbench benchmark validate
openfreqbench quality-gate --profile canonical
openfreqbench stats run
openfreqbench report build
openfreqbench andes run-ieee39
```

## Docker

CPU baseline image:

```bash
docker build -t openfreqbench:cpu -f Dockerfile .
docker run --rm -v ${PWD}/artifacts:/workspace/artifacts openfreqbench:cpu quality-gate --profile canonical
```

Extended image (torch + andes):

```bash
docker build -t openfreqbench:full -f Dockerfile.full .
docker run --rm -v ${PWD}/artifacts:/workspace/artifacts openfreqbench:full andes run-ieee39
```

## Project governance

- Roadmap: `ROADMAP.md`
- Changelog: `CHANGELOG.md`
- Contribution guide: `CONTRIBUTING.md`
- Code of conduct: `CODE_OF_CONDUCT.md`
- Security policy: `SECURITY.md`
- Citation metadata: `CITATION.cff`

## Artifact policy

Heavy generated outputs under `artifacts/full_mc_benchmark/` and
`tests/montecarlo/outputs/` are intentionally not versioned. Regenerate them
from the canonical pipeline and validate with
`python -m pipelines.validate_canonical_artifacts`.

## Current status

The repository is under active consolidation. The main technical priorities are:

- keep `step()` and `step_vectorized()` behavior equivalent
- keep benchmark outputs traceable to one canonical artifact set
- keep plotting logic modular and paper-facing filenames stable
- keep paper numbers derived from fresh generated artifacts, not hardcoded edits
