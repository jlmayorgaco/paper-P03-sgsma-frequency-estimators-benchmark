from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from pipelines.stats_hypotheses import run_hypotheses

from .artifacts import freeze_artifacts, validate_tuned_artifacts, write_environment_report
from .config import load_config, parse_config
from .hypotheses import write_hypothesis_bank
from .paths import PACKAGE_ROOT, PROJECT_ROOT, SOURCE_ROOT
from .quality import run_quality_gate
from .reports import build_report_outputs
from .registry import (
    CANONICAL_METRIC_IDS,
    CANONICAL_METRIC_PROFILE,
    estimator_specs,
    metric_registry,
    platform_manifest,
    scenario_registry,
)
from .runner import run_benchmark_config
from .schemas import get_schema, schema_names, validate_payload

ROOT = PROJECT_ROOT
PACKAGE_TEMPLATE_DIR = PACKAGE_ROOT / "templates"
CHECKOUT_TEMPLATE_DIR = ROOT / "configs"


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _cmd_doctor(args: argparse.Namespace) -> int:
    required = ["numpy", "scipy", "pandas", "matplotlib", "yaml"]
    optional = ["optuna", "numba", "sklearn", "torch", "andes"]
    rows = []
    for name in required + optional:
        try:
            importlib.import_module(name)
            rows.append({"module": name, "status": "ok", "required": name in required})
        except Exception as exc:
            rows.append(
                {
                    "module": name,
                    "status": "missing",
                    "required": name in required,
                    "detail": str(exc),
                }
            )
    payload = {
        "python": sys.version,
        "root": str(ROOT),
        "metric_profile": CANONICAL_METRIC_PROFILE,
        "scenarios": len(scenario_registry()),
        "canonical_estimators": len(estimator_specs()),
        "checks": rows,
    }
    if getattr(args, "output", None):
        output = Path(args.output)
        write_environment_report(ROOT, output, source_root=SOURCE_ROOT)
        payload["environment_report"] = str(output)
    _print_json(payload)
    return 1 if any(row["required"] and row["status"] != "ok" for row in rows) else 0


def _cmd_manifest(_: argparse.Namespace) -> int:
    _print_json(platform_manifest())
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    if args.kind == "scenarios":
        for name in sorted(scenario_registry()):
            print(name)
        return 0
    if args.kind == "estimators":
        for spec in estimator_specs(include_experimental=bool(args.include_experimental)):
            marker = "canonical" if spec.status == "active" else spec.status
            print(f"{spec.label}\t{spec.family}\t{marker}")
        return 0
    if args.kind == "metrics":
        for spec in metric_registry().values():
            print(f"{spec.metric_id}\t{spec.unit}\t{spec.group}\t{spec.label}")
        return 0
    raise ValueError(f"Unknown list kind: {args.kind}")


def _template_path(name: str) -> Path:
    package_template = PACKAGE_TEMPLATE_DIR / f"{name}.yaml"
    if package_template.exists():
        return package_template
    return CHECKOUT_TEMPLATE_DIR / f"{name}.yaml"


def _cmd_init(args: argparse.Namespace) -> int:
    template = str(args.template)
    src = _template_path(template)
    if not src.exists():
        known = sorted(
            {
                path.stem
                for template_dir in (PACKAGE_TEMPLATE_DIR, CHECKOUT_TEMPLATE_DIR)
                for path in template_dir.glob("*.yaml")
            }
        )
        print(f"[ERROR] Unknown template {template!r}. Known: {known}")
        return 2
    dest = Path(args.output)
    if dest.exists() and not args.force:
        print(f"[ERROR] Refusing to overwrite existing file: {dest}")
        return 2
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    print(dest)
    return 0


def _load_and_run(config_path: Path, dry_run: bool) -> int:
    config = load_config(config_path)
    result = run_benchmark_config(config, dry_run=dry_run)
    _print_json(result)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    return _load_and_run(Path(args.config), bool(args.dry_run))


def _cmd_quick_test(args: argparse.Namespace) -> int:
    cfg = parse_config(
        {
            "run": {
                "id": args.id,
                "mode": "single",
                "output_dir": args.output_dir,
                "n_runs": args.n_runs,
                "base_seed": args.base_seed,
                "capture_signals": args.capture_signals,
                "max_workers": args.max_workers,
            },
            "benchmark": {
                "scenarios": [args.scenario],
                "estimators": [args.estimator],
            },
            "metrics": {"profile": CANONICAL_METRIC_PROFILE},
        }
    )
    result = run_benchmark_config(cfg, dry_run=bool(args.dry_run))
    _print_json(result)
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    cfg = parse_config(
        {
            "run": {
                "id": args.id,
                "mode": "compare",
                "output_dir": args.output_dir,
                "n_runs": args.n_runs,
                "base_seed": args.base_seed,
                "capture_signals": args.capture_signals,
                "max_workers": args.max_workers,
            },
            "benchmark": {
                "scenarios": [args.scenario],
                "estimators": list(args.estimator),
            },
            "metrics": {"profile": CANONICAL_METRIC_PROFILE},
        }
    )
    result = run_benchmark_config(cfg, dry_run=bool(args.dry_run))
    _print_json(result)
    return 0


def _cmd_hypotheses_run(args: argparse.Namespace) -> int:
    result = run_hypotheses(
        hypotheses_path=Path(args.hypotheses),
        schema_path=Path(args.schema),
        input_json_path=Path(args.input_json),
        output_dir=Path(args.output_dir),
        allow_exploratory=bool(args.allow_exploratory),
        require_canonical_input=False,
    )
    _print_json(result)
    return 0


def _cmd_hypotheses_generate(args: argparse.Namespace) -> int:
    path = write_hypothesis_bank(Path(args.output), scope=str(args.scope))
    print(path)
    return 0


def _cmd_report_build(args: argparse.Namespace) -> int:
    result = build_report_outputs(
        input_json=Path(args.input_json),
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    _print_json(result)
    return 0


def _cmd_validate_artifacts(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    result = validate_tuned_artifacts(config)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    display = result
    if not bool(args.full):
        display = dict(result)
        display["present_preview"] = result["present"][:10]
        display["missing_preview"] = result["missing"][:10]
        display.pop("present", None)
        display.pop("missing", None)
        if result["n_missing_pairs"]:
            display["hint"] = (
                "Run the canonical tuning pipeline first or pass a tuned_artifacts_dir "
                "that contains <scenario>/<estimator>/run_spec.json files. "
                "Use --full or --output for the complete pair list."
            )
    _print_json(display)
    return 0 if result["status"] == "pass" else 1


def _cmd_archive(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve() if args.config else None
    result = freeze_artifacts(
        Path(args.run_root),
        package_root=ROOT,
        source_root=SOURCE_ROOT,
        config_path=config_path,
        make_zip=bool(args.zip),
    )
    _print_json(result)
    return 0


def _cmd_schema(args: argparse.Namespace) -> int:
    payload = get_schema(str(args.name))
    if args.validate:
        target = Path(args.validate)
        data = json.loads(target.read_text(encoding="utf-8"))
        errors = validate_payload(str(args.name), data)
        result = {"schema": args.name, "target": str(target), "status": "pass" if not errors else "fail", "errors": errors}
        _print_json(result)
        return 0 if not errors else 1
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(output)
    else:
        _print_json(payload)
    return 0


def _cmd_canonical_full(args: argparse.Namespace) -> int:
    cmd = [sys.executable, "-m", "pipelines.full_mc_benchmark"]
    env = None
    if args.scenario or args.estimator:
        env_dict = dict(os.environ)
        if args.scenario:
            env_dict["BENCHMARK_INCLUDE_SCENARIOS"] = ",".join(args.scenario)
        if args.estimator:
            env_dict["BENCHMARK_INCLUDE_ESTIMATORS"] = ",".join(args.estimator)
        env = env_dict
    code = subprocess.run(cmd, cwd=ROOT, env=env, check=False).returncode
    return int(code)


def _cmd_quality_gate(args: argparse.Namespace) -> int:
    result = run_quality_gate(
        ROOT,
        run_tests=not bool(args.skip_tests),
        release=bool(args.release),
    )
    _print_json(result)
    return 0 if result["status"] == "pass" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openfreqbench",
        description="OpenFreqBench 2.0.0: reproducible frequency-estimator benchmarks from YAML.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Check local runtime and platform registry.")
    doctor.add_argument("--output", default=None, help="Write environment_report.json to this path.")
    doctor.set_defaults(handler=_cmd_doctor)

    manifest = sub.add_parser("manifest", help="Print the platform manifest.")
    manifest.set_defaults(handler=_cmd_manifest)

    list_cmd = sub.add_parser("list", help="List canonical scenarios, estimators, or metrics.")
    list_cmd.add_argument("kind", choices=["scenarios", "estimators", "metrics"])
    list_cmd.add_argument("--include-experimental", action="store_true", help="Show experimental estimator candidates too.")
    list_cmd.set_defaults(handler=_cmd_list)

    list_scenarios = sub.add_parser("list-scenarios", help="List canonical scenarios.")
    list_scenarios.set_defaults(kind="scenarios", include_experimental=False, handler=_cmd_list)
    list_estimators = sub.add_parser("list-estimators", help="List canonical estimators.")
    list_estimators.add_argument("--include-experimental", action="store_true")
    list_estimators.set_defaults(kind="estimators", handler=_cmd_list)

    init = sub.add_parser("init", help="Create a starter YAML config.")
    init.add_argument(
        "--template",
        choices=["quick", "compare", "montecarlo", "custom-estimator", "tuned-artifacts", "hypotheses"],
        default="quick",
    )
    init.add_argument("--output", default="openfreqbench.yaml")
    init.add_argument("--force", action="store_true")
    init.set_defaults(handler=_cmd_init)

    run = sub.add_parser("run", help="Run a YAML-defined benchmark matrix.")
    run.add_argument("--config", required=True, help="Path to an OpenFreqBench YAML config.")
    run.add_argument("--dry-run", action="store_true", help="Validate and print resolved run plan without executing.")
    run.set_defaults(handler=_cmd_run)

    quick = sub.add_parser("quick-test", help="Run one estimator in one scenario.")
    quick.add_argument("--scenario", default="IEEE_Single_SinWave")
    quick.add_argument("--estimator", default="ZCD")
    quick.add_argument("--n-runs", type=int, default=1)
    quick.add_argument("--base-seed", type=int, default=12345)
    quick.add_argument("--id", default="quick-test")
    quick.add_argument("--output-dir", default="artifacts/openfreqbench")
    quick.add_argument("--max-workers", type=int, default=1)
    quick.add_argument("--capture-signals", action=argparse.BooleanOptionalAction, default=True)
    quick.add_argument("--dry-run", action="store_true")
    quick.set_defaults(handler=_cmd_quick_test)

    compare = sub.add_parser("compare", help="Compare two or more estimators in one scenario.")
    compare.add_argument("--scenario", default="IEEE_Freq_Step")
    compare.add_argument("--estimator", action="append", required=True)
    compare.add_argument("--n-runs", type=int, default=3)
    compare.add_argument("--base-seed", type=int, default=12345)
    compare.add_argument("--id", default="estimator-compare")
    compare.add_argument("--output-dir", default="artifacts/openfreqbench")
    compare.add_argument("--max-workers", type=int, default=1)
    compare.add_argument("--capture-signals", action=argparse.BooleanOptionalAction, default=True)
    compare.add_argument("--dry-run", action="store_true")
    compare.set_defaults(handler=_cmd_compare)

    benchmark = sub.add_parser("benchmark", help="Compatibility namespace for benchmark operations.")
    benchmark_sub = benchmark.add_subparsers(dest="benchmark_cmd", required=True)
    bench_run = benchmark_sub.add_parser("run", help="Run a YAML-defined benchmark matrix.")
    bench_run.add_argument("--config", required=True)
    bench_run.add_argument("--dry-run", action="store_true")
    bench_run.set_defaults(handler=_cmd_run)
    bench_full = benchmark_sub.add_parser("full", help="Run the current full canonical pipeline with tuning.")
    bench_full.add_argument("--scenario", action="append", default=[])
    bench_full.add_argument("--estimator", action="append", default=[])
    bench_full.set_defaults(handler=_cmd_canonical_full)

    report = sub.add_parser("report", help="Build analysis tables and plots from an OpenFreqBench benchmark report.")
    report_sub = report.add_subparsers(dest="report_cmd", required=True)
    report_build = report_sub.add_parser("build", help="Generate analysis summary, CSV tables, and PNG plots.")
    report_build.add_argument("--input-json", required=True)
    report_build.add_argument("--output-dir", default=None)
    report_build.set_defaults(handler=_cmd_report_build)

    validate_artifacts = sub.add_parser(
        "validate-artifacts",
        help="Validate an artifact_tuned config against its tuned run_spec.json sources.",
    )
    validate_artifacts.add_argument("--config", required=True)
    validate_artifacts.add_argument("--output", default=None)
    validate_artifacts.add_argument("--full", action="store_true", help="Print every present/missing pair.")
    validate_artifacts.set_defaults(handler=_cmd_validate_artifacts)

    archive = sub.add_parser("archive", help="Freeze a run directory with hashes and paper traceability.")
    archive.add_argument("--run-root", required=True)
    archive.add_argument("--config", default=None)
    archive.add_argument("--zip", action="store_true", help="Also create a zip archive beside the run directory.")
    archive.set_defaults(handler=_cmd_archive)

    schema_cmd = sub.add_parser("schema", help="Print, write, or validate public JSON schemas.")
    schema_cmd.add_argument("--name", choices=schema_names(), default="benchmark-report")
    schema_cmd.add_argument("--output", default=None)
    schema_cmd.add_argument("--validate", default=None, help="Validate a JSON file using lightweight built-in checks.")
    schema_cmd.set_defaults(handler=_cmd_schema)

    plots = sub.add_parser("plots", help="Generate plots from an OpenFreqBench benchmark report.")
    plots_sub = plots.add_subparsers(dest="plots_cmd", required=True)
    plots_build = plots_sub.add_parser("build", help="Alias for `report build` focused on plot artifacts.")
    plots_build.add_argument("--input-json", required=True)
    plots_build.add_argument("--output-dir", default=None)
    plots_build.set_defaults(handler=_cmd_report_build)

    quality = sub.add_parser("quality-gate", help="Run package, science, and reproducibility readiness checks.")
    quality.add_argument("--skip-tests", action="store_true", help="Skip pytest during the quality gate.")
    quality.add_argument("--release", action="store_true", help="Require a clean git tree and release-level metadata.")
    quality.set_defaults(handler=_cmd_quality_gate)

    hypotheses = sub.add_parser("hypotheses", help="Generate or run preregistered hypotheses.")
    hypotheses_sub = hypotheses.add_subparsers(dest="hypotheses_cmd", required=True)
    hyp_generate = hypotheses_sub.add_parser("generate", help="Generate an organized hypothesis YAML bank.")
    hyp_generate.add_argument(
        "--scope",
        choices=["starter", "canonical", "mvp2_music_no_pigru"],
        default="starter",
    )
    hyp_generate.add_argument("--output", default="hypotheses.generated.yaml")
    hyp_generate.set_defaults(handler=_cmd_hypotheses_generate)
    hyp_run = hypotheses_sub.add_parser("run", help="Run hypotheses against a benchmark_report.json file.")
    hyp_run.add_argument("--hypotheses", default=str(ROOT / "hypotheses.yaml"))
    hyp_run.add_argument("--schema", default=str(ROOT / "hypotheses_schema.yaml"))
    hyp_run.add_argument("--input-json", required=True)
    hyp_run.add_argument("--output-dir", default="artifacts/openfreqbench/stats")
    hyp_run.add_argument("--allow-exploratory", action="store_true")
    hyp_run.set_defaults(handler=_cmd_hypotheses_run)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        raise SystemExit(2)
    raise SystemExit(int(handler(args)))


if __name__ == "__main__":
    main()
