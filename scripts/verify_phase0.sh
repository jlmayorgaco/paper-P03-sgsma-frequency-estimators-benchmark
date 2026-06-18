#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src
export BENCHMARK_N_COST_REPS=1
mkdir -p artifacts

python -m openfreqbench doctor
python -m pytest --collect-only tests -q -p no:cacheprovider --basetemp artifacts/pytest-tmp
python -m pytest tests/test_config_guardrails.py tests/test_contracts_schemas_artifacts.py -q -p no:cacheprovider --basetemp artifacts/pytest-tmp
python -m openfreqbench run --config configs/phase0-smoke.yaml --dry-run
python -m openfreqbench run --config configs/phase0-smoke.yaml
