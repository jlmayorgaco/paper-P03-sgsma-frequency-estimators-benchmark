# Full ATLAS (all sweeps, n=30) for PI-GRU only -- DEFERRED until a GPU is available.
#
# Measured on this CPU-only box: PI-GRU is ~115 min/cell (vs ESPRIT ~3 min,
# MUSIC ~6 min), i.e. ~17 days for the full sweep -- impractical. ESPRIT is
# already in the paper-grade canonical set; MUSIC runs via
# run_atlas_music_only.ps1. PI-GRU is the only one left, and it needs a GPU
# (torch.cuda) to be feasible. Run this on a GPU machine.
#
# RESUMABLE: --resume continues from the last finished cell. Re-launch this same
# script after any crash/shutdown and it picks up where it left off.
#
# Launch (on a GPU box):  powershell -ExecutionPolicy Bypass -File scripts\run_atlas_slow_estimators.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)   # repo root

$env:PYTHONPATH = "src"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:BENCHMARK_INCLUDE_EXPERIMENTAL = "1"     # allow PI-GRU (experimental)
$env:ATLAS_INCLUDE_ESTIMATORS = "PI-GRU"      # ONLY PI-GRU (GPU-bound)
Remove-Item Env:\ATLAS_EXCLUDE_ESTIMATORS -ErrorAction SilentlyContinue

Write-Output "Starting full ATLAS for PI-GRU only (all sweeps, n=30, --resume) -- GPU recommended..."

python -u -m pipelines.atlas_sweep `
  --sweeps all `
  --policy fixed_policy `
  --n-runs 30 `
  --base-seed 42 `
  --n-cost-reps 3 `
  --output-subdir atlas-slow-estimators-v1 `
  --resume *>&1 | Tee-Object -FilePath atlas_slow_run.log

Write-Output "Done. Results in artifacts\atlas-slow-estimators-v1\ ; log in atlas_slow_run.log"
