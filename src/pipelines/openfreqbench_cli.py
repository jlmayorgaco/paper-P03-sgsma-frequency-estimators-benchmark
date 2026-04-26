from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from pipelines.andes_ieee39 import run_andes_ieee39
from pipelines.report_builder import build_benchmark_report
from pipelines.stats_hypotheses import run_hypotheses

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


def _csv_items(raw_values: list[str] | None) -> list[str]:
    items: list[str] = []
    for raw in raw_values or []:
        for part in str(raw).split(","):
            val = part.strip()
            if val:
                items.append(val)
    deduped = list(dict.fromkeys(items))
    return deduped


def _run_python_module(
    module_name: str,
    args: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
) -> int:
    cmd = [sys.executable, "-m", module_name]
    if args:
        cmd.extend(args)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if extra_env:
        env.update(extra_env)
    return subprocess.run(cmd, cwd=ROOT, check=False, env=env).returncode


def _run_benchmark_with_filters(
    scenarios: list[str] | None,
    estimators: list[str] | None,
    dry_run: bool,
) -> int:
    scenario_items = _csv_items(scenarios)
    estimator_items = _csv_items(estimators)
    extra_env: dict[str, str] = {}

    if scenario_items:
        extra_env["BENCHMARK_INCLUDE_SCENARIOS"] = ",".join(scenario_items)
    if estimator_items:
        extra_env["BENCHMARK_INCLUDE_ESTIMATORS"] = ",".join(estimator_items)

    if dry_run:
        if scenario_items:
            print(f"BENCHMARK_INCLUDE_SCENARIOS={extra_env['BENCHMARK_INCLUDE_SCENARIOS']}")
        else:
            print("BENCHMARK_INCLUDE_SCENARIOS=<all>")
        if estimator_items:
            print(f"BENCHMARK_INCLUDE_ESTIMATORS={extra_env['BENCHMARK_INCLUDE_ESTIMATORS']}")
        else:
            print("BENCHMARK_INCLUDE_ESTIMATORS=<all>")
        return 0

    return _run_python_module(
        "pipelines.full_mc_benchmark",
        extra_env=extra_env if extra_env else None,
    )


def _cmd_benchmark_run(args: argparse.Namespace) -> int:
    return _run_benchmark_with_filters(
        scenarios=getattr(args, "scenario", None),
        estimators=getattr(args, "estimator", None),
        dry_run=bool(getattr(args, "dry_run", False)),
    )


def _cmd_benchmark_run_scenario(args: argparse.Namespace) -> int:
    return _run_benchmark_with_filters(
        scenarios=[str(v) for v in getattr(args, "name", [])],
        estimators=None,
        dry_run=bool(getattr(args, "dry_run", False)),
    )


def _cmd_benchmark_run_estimator(args: argparse.Namespace) -> int:
    return _run_benchmark_with_filters(
        scenarios=None,
        estimators=[str(v) for v in getattr(args, "name", [])],
        dry_run=bool(getattr(args, "dry_run", False)),
    )


def _cmd_benchmark_run_matrix(args: argparse.Namespace) -> int:
    return _run_benchmark_with_filters(
        scenarios=getattr(args, "scenario", None),
        estimators=getattr(args, "estimator", None),
        dry_run=bool(getattr(args, "dry_run", False)),
    )


def _cmd_benchmark_validate(_: argparse.Namespace) -> int:
    return _run_python_module("pipelines.validate_canonical_artifacts")


def _cmd_benchmark_sync_paper(_: argparse.Namespace) -> int:
    return _run_python_module("pipelines.sync_paper_artifacts")


def _cmd_quality_gate(args: argparse.Namespace) -> int:
    return _run_python_module("pipelines.run_quality_gate", ["--profile", args.profile])


def _cmd_andes_run_ieee39(args: argparse.Namespace) -> int:
    out = Path(args.output) if args.output else None
    result = run_andes_ieee39(seed=int(args.seed), output_dir=out)
    print(result["manifest_path"])
    return 0


def _cmd_stats_run(args: argparse.Namespace) -> int:
    input_json = Path(args.input_json)
    if not input_json.exists():
        print(f"[ERROR] Benchmark report not found: {input_json}")
        print("Run `openfreqbench benchmark run` first.")
        return 2
    result = run_hypotheses(
        hypotheses_path=Path(args.hypotheses),
        schema_path=Path(args.hypotheses_schema),
        input_json_path=input_json,
        output_dir=Path(args.output_dir),
        allow_exploratory=bool(args.allow_exploratory),
        require_canonical_input=not bool(args.allow_noncanonical_input),
    )
    print(result["json_path"])
    if args.fail_on_blocked and int(result.get("n_blocked", 0)) > 0:
        print("[ERROR] Blocked hypotheses detected under guardrails.")
        return 3
    return 0


def _cmd_report_build(args: argparse.Namespace) -> int:
    input_json = Path(args.input_json)
    if not input_json.exists():
        print(f"[ERROR] Benchmark report not found: {input_json}")
        print("Run `openfreqbench benchmark run` first.")
        return 2
    canonical = ROOT / "artifacts" / "full_mc_benchmark" / "benchmark_full_report.json"
    if not args.allow_noncanonical_input and input_json.resolve() != canonical.resolve():
        print(f"[ERROR] Report build requires canonical input: {canonical}")
        return 2
    stats_json = Path(args.stats_json)
    result = build_benchmark_report(
        input_json=input_json,
        output_dir=Path(args.output_dir),
        stats_json=stats_json if stats_json.exists() else None,
    )
    print(result["markdown"])
    return 0


def _check_module(name: str) -> tuple[bool, str]:
    try:
        __import__(name)
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def _check_module_subprocess(name: str) -> tuple[bool, str]:
    probe = subprocess.run(
        [sys.executable, "-c", f"import {name}; print('ok')"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode == 0:
        return True, "ok"
    return False, (probe.stderr or probe.stdout or "").strip()


def _cmd_env_doctor(_: argparse.Namespace) -> int:
    required = ["numpy", "scipy", "pandas", "matplotlib", "optuna", "numba", "sklearn", "yaml"]
    optional = ["torch", "andes"]
    missing = []
    for mod in required:
        ok, msg = _check_module(mod)
        if not ok:
            missing.append((mod, msg))
    for mod, msg in missing:
        print(f"[MISSING] {mod}: {msg}")
    for mod in optional:
        if mod == "andes":
            ok, msg = _check_module_subprocess(mod)
        else:
            ok, msg = _check_module(mod)
        status = "OK" if ok else "MISSING"
        print(f"[{status}] optional {mod}: {msg}")
    return 1 if missing else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openfreqbench",
        description="OpenFreqBench command-line interface for benchmark execution, statistics, and reporting.",
    )
    sub = parser.add_subparsers(dest="domain", required=True)

    benchmark = sub.add_parser("benchmark", help="Run and validate benchmark flows.")
    benchmark_sub = benchmark.add_subparsers(dest="benchmark_cmd", required=True)

    bench_run = benchmark_sub.add_parser(
        "run",
        help="Run benchmark over the selected scenario/estimator slice (defaults to full matrix).",
    )
    bench_run.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Scenario name(s) to include. Repeat flag or pass comma-separated values.",
    )
    bench_run.add_argument(
        "--estimator",
        action="append",
        default=[],
        help="Estimator label(s) to include. Repeat flag or pass comma-separated values.",
    )
    bench_run.add_argument("--dry-run", action="store_true", help="Print resolved filters and exit without running.")
    bench_run.set_defaults(handler=_cmd_benchmark_run)

    bench_run_scenario = benchmark_sub.add_parser(
        "run-scenario",
        help="Run one or more scenarios against all estimators.",
    )
    bench_run_scenario.add_argument(
        "--name",
        required=True,
        action="append",
        help="Scenario name. Repeat flag or pass comma-separated names.",
    )
    bench_run_scenario.add_argument("--dry-run", action="store_true", help="Print resolved filters and exit.")
    bench_run_scenario.set_defaults(handler=_cmd_benchmark_run_scenario)

    bench_run_estimator = benchmark_sub.add_parser(
        "run-estimator",
        help="Run one or more estimators against all scenarios.",
    )
    bench_run_estimator.add_argument(
        "--name",
        required=True,
        action="append",
        help="Estimator label. Repeat flag or pass comma-separated labels.",
    )
    bench_run_estimator.add_argument("--dry-run", action="store_true", help="Print resolved filters and exit.")
    bench_run_estimator.set_defaults(handler=_cmd_benchmark_run_estimator)

    bench_run_matrix = benchmark_sub.add_parser(
        "run-matrix",
        help="Run a subset x subset matrix selection.",
    )
    bench_run_matrix.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Scenario name(s) to include. Repeat flag or pass comma-separated values.",
    )
    bench_run_matrix.add_argument(
        "--estimator",
        action="append",
        default=[],
        help="Estimator label(s) to include. Repeat flag or pass comma-separated values.",
    )
    bench_run_matrix.add_argument("--dry-run", action="store_true", help="Print resolved filters and exit.")
    bench_run_matrix.set_defaults(handler=_cmd_benchmark_run_matrix)

    bench_validate = benchmark_sub.add_parser("validate", help="Validate canonical artifacts.")
    bench_validate.set_defaults(handler=_cmd_benchmark_validate)

    bench_sync = benchmark_sub.add_parser("sync-paper", help="Copy canonical paper figures into LaTeX folder.")
    bench_sync.set_defaults(handler=_cmd_benchmark_sync_paper)

    andes = sub.add_parser("andes", help="Run ANDES-based IEEE39 benchmark scenarios.")
    andes_sub = andes.add_subparsers(dest="andes_cmd", required=True)
    andes_run = andes_sub.add_parser("run-ieee39", help="Generate IEEE39 severe-event traces and run selected estimators.")
    andes_run.add_argument("--seed", type=int, default=12345, help="Random seed for event generation.")
    andes_run.add_argument("--output", type=str, default=None, help="Optional output directory override.")
    andes_run.set_defaults(handler=_cmd_andes_run_ieee39)

    stats = sub.add_parser("stats", help="Run preregistered statistical hypothesis tests.")
    stats_sub = stats.add_subparsers(dest="stats_cmd", required=True)
    stats_run = stats_sub.add_parser("run", help="Execute hypotheses defined in hypotheses.yaml.")
    stats_run.add_argument("--hypotheses", type=str, default="hypotheses.yaml", help="Path to hypotheses YAML file.")
    stats_run.add_argument("--hypotheses-schema", type=str, default="hypotheses_schema.yaml", help="Path to hypotheses schema YAML file.")
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
    stats_run.add_argument("--allow-exploratory", action="store_true", help="Allow exploratory hypotheses (guardrail off).")
    stats_run.add_argument("--allow-noncanonical-input", action="store_true", help="Allow stats run on non-canonical report path.")
    stats_run.add_argument("--fail-on-blocked", action="store_true", help="Return non-zero exit code if any hypothesis is blocked by guardrails.")
    stats_run.set_defaults(handler=_cmd_stats_run)

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
    report_build.add_argument("--allow-noncanonical-input", action="store_true", help="Allow report build from non-canonical report path.")
    report_build.set_defaults(handler=_cmd_report_build)

    env = sub.add_parser("env", help="Runtime environment checks.")
    env_sub = env.add_subparsers(dest="env_cmd", required=True)
    env_doctor_cmd = env_sub.add_parser("doctor", help="Check runtime dependencies and filesystem expectations.")
    env_doctor_cmd.set_defaults(handler=_cmd_env_doctor)

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
