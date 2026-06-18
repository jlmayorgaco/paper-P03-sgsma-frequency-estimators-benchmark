$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $Root
try {
    $generated = @(".venv-release-test", "build", "dist", "src/openfreqbench.egg-info")
    foreach ($target in $generated) {
        $path = Join-Path $Root $target
        if (Test-Path $path) {
            $resolved = Resolve-Path $path
            if (-not $resolved.Path.StartsWith($Root.Path)) {
                throw "Refusing to remove path outside package root: $($resolved.Path)"
            }
            Remove-Item -LiteralPath $resolved.Path -Recurse -Force
        }
    }

    python -m pip install --upgrade pip build
    python -m pytest tests -q -p no:cacheprovider
    python -m build --wheel

    python -m venv .venv-release-test
    $VenvPython = Join-Path $Root ".venv-release-test\Scripts\python.exe"
    $VenvCli = Join-Path $Root ".venv-release-test\Scripts\openfreqbench.exe"
    & $VenvPython -m pip install --upgrade pip
    $Wheel = Get-ChildItem -Path (Join-Path $Root "dist") -Filter "openfreqbench-*.whl" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    & $VenvPython -m pip install $Wheel.FullName

    & $VenvCli doctor
    & $VenvCli quick-test --scenario IEEE_Single_SinWave --estimator ZCD --n-runs 1 --max-workers 1 --id release-smoke --output-dir artifacts/openfreqbench --no-capture-signals
    & $VenvCli report build --input-json artifacts/openfreqbench/release-smoke/benchmark_report.json

    $env:PYTHONPATH = "src"
    python -m openfreqbench quality-gate --release
}
finally {
    Pop-Location
}
