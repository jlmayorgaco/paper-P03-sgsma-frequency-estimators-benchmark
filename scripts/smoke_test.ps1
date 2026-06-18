# OpenFreqBench smoke test - verifies install and basic functionality
# Run from repository root
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot\..

Write-Host "=== OpenFreqBench Smoke Test ==="

Write-Host "[1/6] Checking installation..."
openfreqbench --version
if ($LASTEXITCODE -ne 0) { throw "openfreqbench not found" }

Write-Host "[2/6] Running doctor..."
openfreqbench doctor
if ($LASTEXITCODE -ne 0) { throw "doctor failed" }

Write-Host "[3/6] Listing registries..."
openfreqbench list scenarios | Select-Object -First 3
openfreqbench list estimators | Select-Object -First 3
openfreqbench list metrics | Select-Object -First 3

Write-Host "[4/6] Running quick-test..."
openfreqbench quick-test --scenario IEEE_Single_SinWave --estimator ZCD --n-runs 1 --output-dir artifacts\smoke-test
if ($LASTEXITCODE -ne 0) { throw "quick-test failed" }

Write-Host "[5/6] Running compare..."
openfreqbench compare --scenario IEEE_Freq_Step --estimator ZCD --estimator IPDFT --n-runs 3 --output-dir artifacts\smoke-test --id smoke-compare
if ($LASTEXITCODE -ne 0) { throw "compare failed" }

Write-Host "[6/6] Building report..."
openfreqbench report build --input-json artifacts\smoke-test\smoke-compare\benchmark_report.json --output-dir artifacts\smoke-test\smoke-compare\report
if ($LASTEXITCODE -ne 0) { throw "report build failed" }

Write-Host "=== Smoke test PASSED ==="
Write-Host "Artifacts: artifacts\smoke-test\"
