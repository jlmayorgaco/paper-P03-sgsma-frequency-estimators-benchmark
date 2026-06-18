# Paper-grade run of the 6 missing required ATLAS sweeps (fixed_policy, n=30).
# Required set = magnitude_step, rocof, frequency_step, phase_jump_sweep,
#   modulation_am_sweep, modulation_fm_sweep, harmonics, interharmonics, noise_snr.
# Already done at n=30: magnitude_step, frequency_step, modulation_am_sweep.
# This script runs the remaining 6 into ONE resumable output dir.
#
# RESUMABLE: --resume continues from the last finished cell. If the PC hangs,
# is closed, or the run is killed, just re-launch this same script and it picks
# up where it left off (skips completed cells in the output dir).
#
# Re-launch:  powershell -ExecutionPolicy Bypass -File scripts\run_atlas_papergrade_missing.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)   # repo root

$env:PYTHONPATH = "src"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"          # avoid OpenMP duplicate-lib crash (torch)
$env:BENCHMARK_INCLUDE_EXPERIMENTAL = "0"   # clean canonical estimators
$env:ATLAS_EXCLUDE_ESTIMATORS = "PI-GRU"    # PI-GRU is GPU-bound -> 17 estimators here

Write-Output "Starting paper-grade fill: rocof, phase_jump, modulation_fm, harmonics, interharmonics, noise_snr (n=30, --resume)..."

python -u -m pipelines.atlas_sweep `
  --sweeps "rocof,phase_jump_sweep,modulation_fm_sweep,harmonics,interharmonics,noise_snr" `
  --policy fixed_policy `
  --n-runs 30 `
  --base-seed 42 `
  --n-cost-reps 3 `
  --output-subdir atlas-papergrade-missing-v1 `
  --resume *>&1 | Tee-Object -FilePath papergrade_missing_run.log

Write-Output "Done. Results in artifacts\atlas-papergrade-missing-v1\ ; log in papergrade_missing_run.log"
