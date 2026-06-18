from __future__ import annotations

from openfreqbench import __version__
from openfreqbench.cli import build_parser
from openfreqbench.registry import platform_manifest


def test_run_command_parses() -> None:
    parser = build_parser()
    args = parser.parse_args(["run", "--config", "configs/quick.yaml", "--dry-run"])
    assert args.command == "run"
    assert args.config == "configs/quick.yaml"
    assert args.dry_run is True


def test_quick_test_command_parses() -> None:
    parser = build_parser()
    args = parser.parse_args(["quick-test", "--scenario", "IEEE_Single_SinWave", "--estimator", "ZCD"])
    assert args.command == "quick-test"
    assert args.scenario == "IEEE_Single_SinWave"
    assert args.estimator == "ZCD"


def test_compare_command_parses_estimators() -> None:
    parser = build_parser()
    args = parser.parse_args(["compare", "--estimator", "ZCD", "--estimator", "IPDFT"])
    assert args.command == "compare"
    assert args.estimator == ["ZCD", "IPDFT"]


def test_hypotheses_generate_parses() -> None:
    parser = build_parser()
    args = parser.parse_args(["hypotheses", "generate", "--scope", "canonical", "--output", "h.yaml"])
    assert args.command == "hypotheses"
    assert args.hypotheses_cmd == "generate"
    assert args.scope == "canonical"


def test_report_build_parses() -> None:
    parser = build_parser()
    args = parser.parse_args(["report", "build", "--input-json", "benchmark_report.json"])
    assert args.command == "report"
    assert args.report_cmd == "build"
    assert args.input_json == "benchmark_report.json"


def test_plots_tuning_traces_parses() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "plots",
            "tuning-traces",
            "--run-root",
            "artifacts/full_mc_tuning_matrix/pilot",
            "--objective",
            "m1_rmse_hz",
            "--format",
            "png",
        ]
    )
    assert args.command == "plots"
    assert args.plots_cmd == "tuning-traces"
    assert args.run_root == "artifacts/full_mc_tuning_matrix/pilot"
    assert args.objective == ["m1_rmse_hz"]
    assert args.format == ["png"]


def test_quality_gate_parses() -> None:
    parser = build_parser()
    args = parser.parse_args(["quality-gate", "--skip-tests", "--release"])
    assert args.command == "quality-gate"
    assert args.skip_tests is True
    assert args.release is True


def test_validate_artifacts_command_parses() -> None:
    parser = build_parser()
    args = parser.parse_args(["validate-artifacts", "--config", "configs/journal-paper-replay.yaml"])
    assert args.command == "validate-artifacts"
    assert args.config == "configs/journal-paper-replay.yaml"


def test_archive_command_parses() -> None:
    parser = build_parser()
    args = parser.parse_args(["archive", "--run-root", "artifacts/openfreqbench/run", "--zip"])
    assert args.command == "archive"
    assert args.run_root == "artifacts/openfreqbench/run"
    assert args.zip is True


def test_benchmark_tune_matrix_command_parses() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "benchmark",
            "tune-matrix",
            "--scenario",
            "IEEE_Single_SinWave",
            "--estimator",
            "ZCD",
            "--objective",
            "m1_rmse_hz",
            "--n-trials",
            "1",
        ]
    )
    assert args.command == "benchmark"
    assert args.benchmark_cmd == "tune-matrix"
    assert args.scenario == ["IEEE_Single_SinWave"]
    assert args.estimator == ["ZCD"]
    assert args.objective == ["m1_rmse_hz"]
    assert args.n_trials == 1


def test_schema_command_parses() -> None:
    parser = build_parser()
    args = parser.parse_args(["schema", "--name", "benchmark-report"])
    assert args.command == "schema"
    assert args.name == "benchmark-report"


def test_platform_manifest_uses_package_version() -> None:
    assert platform_manifest()["version"] == __version__
