from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from .commands.andes import run_ieee39
from .commands.env_doctor import env_doctor
from .commands.report import build_report
from .commands.stats import run_stats

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


def _run_python_module(module_name: str, args: list[str] | None = None) -> int:
    cmd = [sys.executable, "-m", module_name]
    if args:
        cmd.extend(args)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.run(cmd, cwd=ROOT, check=False, env=env).returncode


def _cmd_benchmark_run(_: argparse.Namespace) -> int:
    return _run_python_module("pipelines.full_mc_benchmark")


def _cmd_benchmark_validate(_: argparse.Namespace) -> int:
    return _run_python_module("pipelines.validate_canonical_artifacts")


def _cmd_benchmark_sync_paper(_: argparse.Namespace) -> int:
    return _run_python_module("pipelines.sync_paper_artifacts")


def _cmd_quality_gate(args: argparse.Namespace) -> int:
    return _run_python_module("pipelines.run_quality_gate", ["--profile", args.profile])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openfreqbench",
        description="OpenFreqBench command-line interface for benchmark execution, statistics, and reporting.",
    )
    sub = parser.add_subparsers(dest="domain", required=True)

    benchmark = sub.add_parser("benchmark", help="Run and validate benchmark flows.")
    benchmark_sub = benchmark.add_subparsers(dest="benchmark_cmd", required=True)

    bench_run = benchmark_sub.add_parser("run", help="Run canonical full benchmark.")
    bench_run.set_defaults(handler=_cmd_benchmark_run)

    bench_validate = benchmark_sub.add_parser("validate", help="Validate canonical artifacts.")
    bench_validate.set_defaults(handler=_cmd_benchmark_validate)

    bench_sync = benchmark_sub.add_parser("sync-paper", help="Copy canonical paper figures into LaTeX folder.")
    bench_sync.set_defaults(handler=_cmd_benchmark_sync_paper)

    andes = sub.add_parser("andes", help="Run ANDES-based IEEE39 benchmark scenarios.")
    andes_sub = andes.add_subparsers(dest="andes_cmd", required=True)
    andes_run = andes_sub.add_parser("run-ieee39", help="Generate IEEE39 severe-event traces and run selected estimators.")
    andes_run.add_argument("--seed", type=int, default=12345, help="Random seed for event generation.")
    andes_run.add_argument("--output", type=str, default=None, help="Optional output directory override.")
    andes_run.set_defaults(handler=run_ieee39)

    stats = sub.add_parser("stats", help="Run preregistered statistical hypothesis tests.")
    stats_sub = stats.add_subparsers(dest="stats_cmd", required=True)
    stats_run = stats_sub.add_parser("run", help="Execute hypotheses defined in hypotheses.yaml.")
    stats_run.add_argument("--hypotheses", type=str, default="hypotheses.yaml", help="Path to hypotheses YAML file.")
    stats_run.add_argument(
        "--input-json",
        type=str,
        default=str(ROOT / "artifacts" / "full_mc_benchmark" / "benchmark_full_report.json"),
        help="Path to canonical benchmark JSON report.",
    )
    stats_run.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "artifacts" / "full_mc_benchmark"),
        help="Directory for statistical test outputs.",
    )
    stats_run.set_defaults(handler=run_stats)

    report = sub.add_parser("report", help="Build deterministic benchmark reports.")
    report_sub = report.add_subparsers(dest="report_cmd", required=True)
    report_build = report_sub.add_parser("build", help="Build markdown/html/csv report outputs.")
    report_build.add_argument(
        "--input-json",
        type=str,
        default=str(ROOT / "artifacts" / "full_mc_benchmark" / "benchmark_full_report.json"),
        help="Path to canonical benchmark JSON report.",
    )
    report_build.add_argument(
        "--stats-json",
        type=str,
        default=str(ROOT / "artifacts" / "full_mc_benchmark" / "statistical_tests_report.json"),
        help="Path to statistical tests JSON report (optional).",
    )
    report_build.add_argument("--output-dir", type=str, default=str(ROOT / "reports"), help="Report output directory.")
    report_build.set_defaults(handler=build_report)

    env = sub.add_parser("env", help="Runtime environment checks.")
    env_sub = env.add_subparsers(dest="env_cmd", required=True)
    env_doctor_cmd = env_sub.add_parser("doctor", help="Check runtime dependencies and filesystem expectations.")
    env_doctor_cmd.set_defaults(handler=env_doctor)

    qg = sub.add_parser("quality-gate", help="Run layered quality gate profiles.")
    qg.add_argument("--profile", choices=["canonical", "legacy", "manual-nightly"], required=True)
    qg.set_defaults(handler=_cmd_quality_gate)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        raise SystemExit(2)
    code = handler(args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
