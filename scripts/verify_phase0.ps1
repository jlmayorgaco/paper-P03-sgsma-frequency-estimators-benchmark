$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $Root
try {
    $env:PYTHONPATH = "src"
    $env:BENCHMARK_N_COST_REPS = "1"
    New-Item -ItemType Directory -Force -Path "artifacts" | Out-Null

    python -m openfreqbench doctor
    python -m pytest --collect-only tests -q -p no:cacheprovider --basetemp artifacts/pytest-tmp
    python -m pytest tests/test_config_guardrails.py tests/test_contracts_schemas_artifacts.py -q -p no:cacheprovider --basetemp artifacts/pytest-tmp
    python -m openfreqbench run --config configs/phase0-smoke.yaml --dry-run
    python -m openfreqbench run --config configs/phase0-smoke.yaml
}
finally {
    Pop-Location
}
