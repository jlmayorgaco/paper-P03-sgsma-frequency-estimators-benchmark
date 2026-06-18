#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src
export BENCHMARK_N_COST_REPS=1
mkdir -p artifacts

python -m openfreqbench doctor
python -m pytest tests -q -p no:cacheprovider --basetemp artifacts/pytest-tmp
python -m openfreqbench quality-gate
python -m openfreqbench run --config configs/phase1-integration.yaml --dry-run
python -m openfreqbench run --config configs/phase1-integration.yaml
python -m openfreqbench schema --name benchmark-report --validate artifacts/openfreqbench/phase1-integration/benchmark_report.json
