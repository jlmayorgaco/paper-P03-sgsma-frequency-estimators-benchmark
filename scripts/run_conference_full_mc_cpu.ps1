$ErrorActionPreference = "Stop"

$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo

$logDir = Join-Path $repo "output\conference_run"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logPath = Join-Path $logDir "full_mc_benchmark_conference_cpu30_v1.transcript.log"
if (Test-Path $logPath) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    Rename-Item -LiteralPath $logPath -NewName "full_mc_benchmark_conference_cpu30_v1_$stamp.transcript.log"
}
Start-Transcript -Path $logPath -Force | Out-Null

$env:PYTHONPATH = "src"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$env:BENCHMARK_OUTPUT_DIR = "artifacts\full_mc_benchmark_conference_cpu30_v1"
$env:BENCHMARK_EXCLUDE_ESTIMATORS = "PI-GRU,MUSIC"
$env:BENCHMARK_N_MC_RUNS = "30"
$env:BENCHMARK_N_TRIALS_TUNING = "40"
$env:BENCHMARK_N_COST_REPS = "3"
$env:BENCHMARK_MC_MAX_WORKERS = "4"
$env:BENCHMARK_CAPTURE_SIGNALS = "0"

Write-Output "Starting conference CPU benchmark run"
Write-Output "Repository: $repo"
Write-Output "BENCHMARK_OUTPUT_DIR=$env:BENCHMARK_OUTPUT_DIR"
Write-Output "BENCHMARK_EXCLUDE_ESTIMATORS=$env:BENCHMARK_EXCLUDE_ESTIMATORS"
Write-Output "BENCHMARK_N_MC_RUNS=$env:BENCHMARK_N_MC_RUNS"
Write-Output "BENCHMARK_N_TRIALS_TUNING=$env:BENCHMARK_N_TRIALS_TUNING"
Write-Output "BENCHMARK_N_COST_REPS=$env:BENCHMARK_N_COST_REPS"
Write-Output "BENCHMARK_MC_MAX_WORKERS=$env:BENCHMARK_MC_MAX_WORKERS"
Write-Output "BENCHMARK_CAPTURE_SIGNALS=$env:BENCHMARK_CAPTURE_SIGNALS"

try {
    python -m pipelines.full_mc_benchmark
}
finally {
    Stop-Transcript | Out-Null
}
