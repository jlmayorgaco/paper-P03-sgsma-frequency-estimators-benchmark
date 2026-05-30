# Runs the AM-modulation ATLAS sweep at paper-grade floor (fixed_policy, n=30).
# 6 levels (0.5,1,2,5,10,20 Hz) x 17 estimators (PI-GRU excluded) = 102 cells (~75-80 min).
# Safe to interrupt: --resume continues from the last finished cell.
#
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\run_atlas_modulation_am.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)   # repo root

$env:PYTHONPATH = "src"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"          # avoid OpenMP duplicate-lib crash with torch
$env:BENCHMARK_INCLUDE_EXPERIMENTAL = "0"   # clean canonical estimators
$env:ATLAS_EXCLUDE_ESTIMATORS = "PI-GRU"    # PI-GRU is GPU-bound on CPU -> 17 estimators here

Write-Output "Starting ATLAS modulation_am_sweep at n=30 (fixed_policy)..."

python -u -m pipelines.atlas_sweep `
  --sweeps "modulation_am_sweep" `
  --policy fixed_policy `
  --n-runs 30 `
  --base-seed 42 `
  --n-cost-reps 3 `
  --output-subdir atlas-modulation-am-paper-v1 `
  --resume *>&1 | Tee-Object -FilePath modulation_am_run.log

Write-Output "ATLAS modulation_am run finished. Results in artifacts\atlas-modulation-am-paper-v1\ ; log in modulation_am_run.log"
