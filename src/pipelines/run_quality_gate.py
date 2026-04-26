from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


def _run(cmd: list[str]) -> int:
    print(f"[RUN] {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.run(cmd, cwd=ROOT, check=False, env=env).returncode


def run_canonical() -> int:
    commands = [
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "src/tests/test_scalar_vs_vector.py",
            "tests/estimators/esprit/test_esprit.py",
            "tests/estimators/prony/test_prony.py",
            "src/tests/test_monte_carlo_timing_contract.py",
            "src/tests/test_pi_gru_runtime_contract.py",
            "src/tests/test_estimator_base_memory_contract.py",
            "src/tests/test_advanced_benchmark_analysis.py",
            "src/pipelines/full_mc_benchmark.py",
        ],
    ]
    for cmd in commands:
        code = _run(cmd)
        if code != 0:
            return code
    return 0


def run_legacy() -> int:
    return _run([sys.executable, "-m", "pytest", "-q", "tests/montecarlo", "-k", "not temp"])


def run_manual_nightly() -> int:
    allow_partial = os.getenv("BENCHMARK_ALLOW_PARTIAL_CANONICAL", "0").strip().lower() in {"1", "true", "yes", "on"}
    validate_cmd = [sys.executable, "-m", "pipelines.validate_canonical_artifacts"]
    if allow_partial:
        validate_cmd.append("--allow-partial")
    commands = [
        [sys.executable, "-m", "pipelines.run_quality_gate", "--profile", "canonical"],
        [sys.executable, "-m", "pytest", "-q", "tests/estimators/experimental"],
        [sys.executable, "-m", "pipelines.full_mc_benchmark"],
        validate_cmd,
    ]
    for cmd in commands:
        code = _run(cmd)
        if code != 0:
            return code
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run layered quality gates.")
    parser.add_argument(
        "--profile",
        choices=["canonical", "legacy", "manual-nightly"],
        required=True,
        help="Quality gate profile to run.",
    )
    args = parser.parse_args()

    if args.profile == "canonical":
        code = run_canonical()
    elif args.profile == "legacy":
        code = run_legacy()
    else:
        code = run_manual_nightly()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
