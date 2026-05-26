# OpenFreqBench

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-111827">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-16a34a">
  <img alt="Metric profile" src="https://img.shields.io/badge/metrics-locked-b91c1c">
  <img alt="Release" src="https://img.shields.io/badge/MVP-2.0.0-7c3aed">
</p>

OpenFreqBench is a benchmark CLI for grid-frequency estimators. It runs
single-phase scenarios, compares estimators, writes reproducible artifacts, and
keeps the canonical metric formulas locked.

You can change the estimator, scenario, run config, and hypotheses. You cannot
rewrite the benchmark metrics from YAML. That is the point.

| Layer | Status | What changes |
| --- | --- | --- |
| Single-phase benchmark | Active | 32 scenarios, 18 canonical estimators |
| Metric profile | Locked | `canonical-single-phase-v1` |
| Research workflow | Active | Monte Carlo, reports, plots, hypotheses, manifests |
| Three-phase / WAMS | Planned | future profiles: `three-phase-v1`, `wams-v1` |

## Install

Use Python 3.10 or newer.

```bash
git clone https://github.com/jlmayorgaco/paper-P03-sgsma-frequency-estimators-benchmark.git
cd paper-P03-sgsma-frequency-estimators-benchmark
git checkout MVP2.0.0
python -m pip install -e ".[dev]"
```

Install directly from the branch:

```bash
python -m pip install "openfreqbench @ git+https://github.com/jlmayorgaco/paper-P03-sgsma-frequency-estimators-benchmark.git@MVP2.0.0"
```

PI-GRU needs PyTorch:

```bash
python -m pip install -e ".[benchmark-full]"
```

Optional future backends:

```bash
python -m pip install -e ".[andes,opendss]"
```

## First run

Check the environment and see what is registered:

```bash
openfreqbench doctor
openfreqbench list scenarios
openfreqbench list estimators
openfreqbench list metrics
```

Run the smallest smoke test:

```bash
openfreqbench quick-test \
  --scenario IEEE_Single_SinWave \
  --estimator ZCD \
  --n-runs 1
```

Compare two estimators:

```bash
openfreqbench compare \
  --scenario IEEE_Freq_Step \
  --estimator ZCD \
  --estimator IPDFT \
  --n-runs 3
```

Build the plots and tables:

```bash
openfreqbench report build \
  --input-json artifacts/openfreqbench/compare-zcd-ipdft/benchmark_report.json
```

## ATLAS sweeps

Use ATLAS when you want the same Monte Carlo, tuning policy, manifests, and PDFs
for magnitude-step, RoCoF, frequency-step, phase-jump, AM modulation, FM
modulation, harmonics, interharmonics, and white-noise/SNR studies.

```bash
python -m pipelines.atlas_sweep \
  --sweeps all \
  --policy fixed_policy \
  --n-runs 100 \
  --n-cost-reps 3 \
  --tune-trials 80 \
  --output-subdir atlas-paper-v1
```

Use `--sweeps core` for only magnitude-step/RoCoF/frequency-step, or
`--sweeps p0` for isolated phase-jump, AM, FM, harmonics, interharmonics, and
noise.

For a quick diagnostic run:

```bash
ATLAS_INCLUDE_ESTIMATORS=ZCD,IPDFT \
ATLAS_FREQSTEP_LEVELS_HZ=0.1 \
python -m pipelines.atlas_sweep \
  --sweeps frequency_step \
  --policy default \
  --n-runs 1 \
  --output-subdir atlas-smoke
```

ATLAS writes `metrics_dashboard_multipage.pdf`,
`rmse_deterioration_by_family.pdf`,
`rmse_all_estimators_small_multiples.pdf`, `atlas_method_map.pdf`,
`atlas_sign_asymmetry.pdf`, `benchmark_report.json`, `manifest.json`,
`artifact_index.csv`, `paper_traceability.csv`, `evidence_manifest.json`, and
`atlas_readiness_report.json/md`.

Dense phase-jump scans can use inclusive range syntax:

```powershell
$env:ATLAS_PHASE_JUMP_LEVELS_DEG = "0:180:1"
$env:ATLAS_PHASE_JUMP_DIRECTIONS = "pos"
python -m pipelines.atlas_sweep `
  --sweeps phase_jump_sweep `
  --policy default `
  --n-runs 1 `
  --output-subdir atlas-phase-jump-0-180deg-1deg-all18
```

Read `atlas_readiness_report.md` before using results in the paper. Preview runs
are marked `diagnostic`; full publication evidence must pass the readiness gate
with all ATLAS sweeps, the canonical estimator set, fixed policy, and enough
Monte Carlo support.

## YAML workflows

Most runs should live in YAML. Start from a template:

```bash
openfreqbench init --template quick --output my-quick.yaml
openfreqbench run --config my-quick.yaml --dry-run
openfreqbench run --config my-quick.yaml
```

Ready-to-run configs:

| Config | Use |
| --- | --- |
| `configs/quick.yaml` | one estimator, one scenario |
| `configs/compare.yaml` | two estimators in one scenario |
| `configs/montecarlo.yaml` | small Monte Carlo benchmark |
| `configs/custom-estimator.yaml` | custom estimator smoke test |
| `configs/tuned-artifacts.yaml` | replay tuned estimator parameters |
| `configs/journal-paper-replay.yaml` | 32-scenario paper replay contract |

Journal replay uses `parameter_policy: artifact_tuned`. It expects tuned
`run_spec.json` files under `artifacts/full_mc_benchmark/`.

```bash
openfreqbench validate-artifacts --config configs/journal-paper-replay.yaml
openfreqbench run --config configs/journal-paper-replay.yaml
```

## Output files

By default, runs write to:

```text
artifacts/openfreqbench/<run_id>/
```

Key files:

| File | Purpose |
| --- | --- |
| `benchmark_report.json` | full machine-readable run report |
| `raw_run_records.csv` | one row per Monte Carlo run |
| `aggregated_metrics.csv` | grouped metric summary |
| `analysis_summary.md` | readable report from `report build` |
| `hypothesis_results.csv` | preregistered hypothesis results |
| `manifest.json` | hashes, runtime info, config hash |
| `artifact_index.csv` | file index for archive and paper tracing |
| `paper_traceability.csv` | claim-to-artifact map |

Public schemas are included:

```bash
openfreqbench schema --name benchmark-report
openfreqbench schema --name manifest --output schemas/manifest.generated.schema.json
openfreqbench schema --name benchmark-report \
  --validate artifacts/openfreqbench/<run_id>/benchmark_report.json
```

## Hypotheses

Generate starter hypotheses:

```bash
openfreqbench hypotheses generate \
  --scope canonical \
  --output hypotheses.generated.yaml
```

Run them against a benchmark report:

```bash
openfreqbench hypotheses run \
  --hypotheses hypotheses.generated.yaml \
  --schema hypotheses_schema.yaml \
  --input-json artifacts/openfreqbench/<run_id>/benchmark_report.json \
  --output-dir artifacts/openfreqbench/<run_id>/stats
```

## Freeze a paper run

Use this sequence when a result will enter a paper, supplement, release, or DOI
archive:

```bash
openfreqbench doctor \
  --output artifacts/openfreqbench/journal-paper-replay-v2/environment_report.json

openfreqbench run --config configs/journal-paper-replay.yaml

openfreqbench report build \
  --input-json artifacts/openfreqbench/journal-paper-replay-v2/benchmark_report.json

openfreqbench hypotheses run \
  --hypotheses configs/hypotheses_preregistered.yaml \
  --schema hypotheses_schema.yaml \
  --input-json artifacts/openfreqbench/journal-paper-replay-v2/benchmark_report.json \
  --output-dir artifacts/openfreqbench/journal-paper-replay-v2/stats

openfreqbench archive \
  --run-root artifacts/openfreqbench/journal-paper-replay-v2 \
  --config configs/journal-paper-replay.yaml
```

No paper number should be copied by hand. Point each claim to an artifact path,
hash, command, and commit.

## Researcher contract

| You may change | You may not change from YAML |
| --- | --- |
| estimator code | canonical metric formulas |
| scenario code | metric windows inside a locked profile |
| run matrix | metric names outside the registered profile |
| hypotheses | artifact hashes after archive |

The CLI rejects config files that try to redefine canonical metrics.

## Docs

Start here:

- [Tutorials](docs/TUTORIALS.md): first estimator, two-estimator comparison, Monte Carlo, custom hypotheses.
- [User guide](docs/USER_GUIDE.md): CLI and YAML usage.
- [Methods](docs/METHODS.md): scenarios, metrics, seeds, timing, tuning policy.
- [Output schema](docs/OUTPUT_SCHEMA.md): stable JSON/CSV contract.
- [Researcher contract](docs/RESEARCHER_CONTRACT.md): what stays locked.
- [Journal protocol](docs/JOURNAL_RESULTS_PROTOCOL.md): artifact rules for paper-grade runs.
- [Weights](docs/WEIGHTS.md): PI-GRU checkpoint notes.

## Project layout

```text
configs/        YAML runs and preregistered hypotheses
docs/           public method, tutorial, release, and schema notes
examples/       custom estimator examples
schemas/        JSON schemas for public artifacts
scripts/        local and release verification scripts
src/            package source, estimators, scenarios, pipelines
tests/          CLI, schema, contract, and reproducibility tests
```

## Current scope

MVP 2.0.0 is public-package cleanup plus the single-phase benchmark platform.
The full journal replay still needs the tuned artifacts in
`artifacts/full_mc_benchmark/` before the complete paper matrix can be rerun
from a fresh clone.
