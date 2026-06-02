$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)   # repo root

$env:PYTHONPATH = "src"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:BENCHMARK_INCLUDE_EXPERIMENTAL = "0"

python -m pipelines.atlas_sweep `
  --sweeps paper_required `
  --policy fixed_policy `
  --n-runs 100 `
  --base-seed 12345 `
  --n-cost-reps 3 `
  --tune-trials 80 `
  --tune-eval-runs 5 `
  --output-subdir atlas-paper-fixed-v2 `
  --resume
