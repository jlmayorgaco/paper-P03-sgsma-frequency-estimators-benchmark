#!/usr/bin/env bash
# OpenFreqBench smoke test - verifies install and basic functionality
# Run from repository root
set -euo pipefail

echo "=== OpenFreqBench Smoke Test ==="

echo "[1/6] Checking installation..."
openfreqbench --version

echo "[2/6] Running doctor..."
openfreqbench doctor

echo "[3/6] Listing registries..."
openfreqbench list scenarios | head -3
openfreqbench list estimators | head -3
openfreqbench list metrics | head -3

echo "[4/6] Running quick-test..."
openfreqbench quick-test --scenario IEEE_Single_SinWave --estimator ZCD --n-runs 1 --output-dir artifacts/smoke-test

echo "[5/6] Running compare..."
openfreqbench compare --scenario IEEE_Freq_Step --estimator ZCD --estimator IPDFT --n-runs 3 --output-dir artifacts/smoke-test --id smoke-compare

echo "[6/6] Building report..."
openfreqbench report build --input-json artifacts/smoke-test/smoke-compare/benchmark_report.json --output-dir artifacts/smoke-test/smoke-compare/report

echo "=== Smoke test PASSED ==="
echo "Artifacts: artifacts/smoke-test/"
