# Full ATLAS (all 9 required sweeps) restricted to the SLOW estimators only:
# MUSIC, ESPRIT (subspace/spectral) and PI-GRU (GPU-bound, excluded from the
# fast bulk runs). Run as a SEPARATE long campaign after the fast paper-grade
# fill (run_atlas_papergrade_missing.ps1) has finished, so it does not compete
# for CPU with the bulk run.
#
# RESUMABLE: --resume continues from the last finished cell. Re-launch this same
# script after any crash/shutdown and it picks up where it left off.
#
# Launch:  powershell -ExecutionPolicy Bypass -File scripts\run_atlas_slow_estimators.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)   # repo root

$env:PYTHONPATH = "src"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:BENCHMARK_INCLUDE_EXPERIMENTAL = "1"            # allow PI-GRU (experimental)
$env:ATLAS_INCLUDE_ESTIMATORS = "MUSIC,ESPRIT,PI-GRU" # ONLY these three
Remove-Item Env:\ATLAS_EXCLUDE_ESTIMATORS -ErrorAction SilentlyContinue

Write-Output "Starting full ATLAS for MUSIC, ESPRIT, PI-GRU only (all sweeps, n=30, --resume)..."

python -u -m pipelines.atlas_sweep `
  --sweeps all `
  --policy fixed_policy `
  --n-runs 30 `
  --base-seed 42 `
  --n-cost-reps 3 `
  --output-subdir atlas-slow-estimators-v1 `
  --resume *>&1 | Tee-Object -FilePath atlas_slow_run.log

Write-Output "Done. Results in artifacts\atlas-slow-estimators-v1\ ; log in atlas_slow_run.log"
