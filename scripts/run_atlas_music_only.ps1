# Full ATLAS (all sweeps, n=30) restricted to MUSIC only.
#
# Why MUSIC alone: ESPRIT is already in the paper-grade canonical set
# (atlas-papergrade-missing-v1, full coverage), and PI-GRU is ~115 min/cell on
# CPU (no GPU) -> ~weeks, so it is deferred to a GPU box. MUSIC (~6 min/cell)
# is the only genuinely-missing subspace method; this run finishes in ~1 day
# and lifts the canonical set to 18 estimators.
#
# RESUMABLE: --resume continues from the last finished cell (the 3 MUSIC cells
# already done under the earlier combined launch are skipped). Re-launch this
# same script after any crash/shutdown.
#
# Launch:  powershell -ExecutionPolicy Bypass -File scripts\run_atlas_music_only.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)   # repo root

$env:PYTHONPATH = "src"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:BENCHMARK_INCLUDE_EXPERIMENTAL = "1"     # consistent with the cells already on disk
$env:ATLAS_INCLUDE_ESTIMATORS = "MUSIC"       # ONLY MUSIC
Remove-Item Env:\ATLAS_EXCLUDE_ESTIMATORS -ErrorAction SilentlyContinue

Write-Output "Starting full ATLAS for MUSIC only (all sweeps, n=30, --resume)..."

python -u -m pipelines.atlas_sweep `
  --sweeps all `
  --policy fixed_policy `
  --n-runs 30 `
  --base-seed 42 `
  --n-cost-reps 3 `
  --output-subdir atlas-slow-estimators-v1 `
  --resume *>&1 | Tee-Object -FilePath atlas_music_run.log

Write-Output "Done. Results in artifacts\atlas-slow-estimators-v1\ ; log in atlas_music_run.log"
