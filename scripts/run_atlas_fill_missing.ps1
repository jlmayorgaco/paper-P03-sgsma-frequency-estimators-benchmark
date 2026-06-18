# Fills the missing ATLAS sweeps at paper-grade-floor (fixed_policy, n=30).
# Covers the families that never had a full 18-estimator run:
#   frequency_step, modulation_am_sweep, modulation_fm_sweep, harmonics
# (magnitude_step is run separately -> artifacts/atlas-magnitude-step-paper-v1)
#
# Safe to interrupt: --resume continues from the last finished cell.
# Run this once magnitude_step has finished (or overlap it; just slower).
#
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\run_atlas_fill_missing.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)   # repo root

$env:PYTHONPATH = "src"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"          # avoid OpenMP duplicate-lib crash with torch
$env:BENCHMARK_INCLUDE_EXPERIMENTAL = "0"   # clean canonical estimators
$env:ATLAS_EXCLUDE_ESTIMATORS = "PI-GRU"    # PI-GRU is GPU-bound (~minutes/cell on CPU); benchmark it separately -> 17 estimators here

Write-Output "Starting ATLAS fill run (frequency_step, modulation_am, modulation_fm, harmonics) at n=30..."

python -u -m pipelines.atlas_sweep `
  --sweeps "frequency_step,modulation_am_sweep,modulation_fm_sweep,harmonics" `
  --policy fixed_policy `
  --n-runs 30 `
  --base-seed 42 `
  --n-cost-reps 3 `
  --output-subdir atlas-fill-missing-v1 `
  --resume *>&1 | Tee-Object -FilePath atlas_fill.log

Write-Output "ATLAS fill run finished. Results in artifacts\atlas-fill-missing-v1\ ; log in atlas_fill.log"
