from __future__ import annotations

from pipelines.openfreqbench_cli import build_parser


def test_cli_benchmark_run_command_parses() -> None:
    parser = build_parser()
    args = parser.parse_args(["benchmark", "run"])
    assert args.domain == "benchmark"
    assert args.benchmark_cmd == "run"
    assert callable(args.handler)


def test_cli_stats_run_parses_optional_paths() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "stats",
            "run",
            "--hypotheses",
            "hypotheses.yaml",
            "--input-json",
            "artifacts/full_mc_benchmark/benchmark_full_report.json",
            "--output-dir",
            "artifacts/full_mc_benchmark",
        ]
    )
    assert args.domain == "stats"
    assert args.stats_cmd == "run"
    assert args.hypotheses.endswith(".yaml")


def test_cli_run_scenario_parses_names() -> None:
    parser = build_parser()
    args = parser.parse_args(["benchmark", "run-scenario", "--name", "IEEE_Freq_Step", "--name", "IEEE_Modulation"])
    assert args.domain == "benchmark"
    assert args.benchmark_cmd == "run-scenario"
    assert args.name == ["IEEE_Freq_Step", "IEEE_Modulation"]


def test_cli_run_estimator_parses_names() -> None:
    parser = build_parser()
    args = parser.parse_args(["benchmark", "run-estimator", "--name", "EKF,UKF"])
    assert args.domain == "benchmark"
    assert args.benchmark_cmd == "run-estimator"
    assert args.name == ["EKF,UKF"]


def test_cli_run_matrix_parses_filters() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "benchmark",
            "run-matrix",
            "--scenario",
            "IEEE_Freq_Step,IEEE_Modulation",
            "--estimator",
            "EKF",
            "--estimator",
            "UKF",
            "--dry-run",
        ]
    )
    assert args.domain == "benchmark"
    assert args.benchmark_cmd == "run-matrix"
    assert args.dry_run is True
    assert args.scenario == ["IEEE_Freq_Step,IEEE_Modulation"]
    assert args.estimator == ["EKF", "UKF"]
