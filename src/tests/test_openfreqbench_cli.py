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
