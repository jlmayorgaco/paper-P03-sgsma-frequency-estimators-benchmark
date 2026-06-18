#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

for target in .venv-release-test build dist src/openfreqbench.egg-info; do
  if [ -e "$target" ]; then
    resolved="$(python -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$target")"
    case "$resolved" in
      "$ROOT"/*) rm -rf "$resolved" ;;
      *) echo "Refusing to remove path outside package root: $resolved" >&2; exit 1 ;;
    esac
  fi
done

python -m pip install --upgrade pip build
python -m pytest tests -q -p no:cacheprovider
python -m build --wheel

python -m venv .venv-release-test
.venv-release-test/bin/python -m pip install --upgrade pip
wheel="$(find dist -name 'openfreqbench-*.whl' -type f | sort | tail -n 1)"
.venv-release-test/bin/python -m pip install "$wheel"

.venv-release-test/bin/openfreqbench doctor
.venv-release-test/bin/openfreqbench quick-test --scenario IEEE_Single_SinWave --estimator ZCD --n-runs 1 --max-workers 1 --id release-smoke --output-dir artifacts/openfreqbench --no-capture-signals
.venv-release-test/bin/openfreqbench report build --input-json artifacts/openfreqbench/release-smoke/benchmark_report.json

PYTHONPATH=src python -m openfreqbench quality-gate --release
