#!/usr/bin/env bash
set -euo pipefail

python -m pytest tests -q -p no:cacheprovider
openfreqbench doctor
openfreqbench run --config configs/quick.yaml --dry-run
openfreqbench quick-test --scenario IEEE_Single_SinWave --estimator ZCD --n-runs 1 --max-workers 1 --id local-smoke --output-dir artifacts/openfreqbench --no-capture-signals
openfreqbench report build --input-json artifacts/openfreqbench/local-smoke/benchmark_report.json
openfreqbench quality-gate

