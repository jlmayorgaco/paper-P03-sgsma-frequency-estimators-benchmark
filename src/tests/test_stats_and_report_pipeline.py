from __future__ import annotations

import json
from pathlib import Path

from pipelines.report_builder import build_benchmark_report
from pipelines.stats_hypotheses import run_hypotheses


def _fake_report(path: Path) -> None:
    payload = {
        "run_configuration": {
            "benchmark_identity": "test",
            "scenarios": ["S1", "S2"],
            "n_mc_runs": 2,
            "estimator_families": {"EKF": "Model-based", "PLL": "Loop-based"},
        },
        "estimators_loaded": ["EKF", "PLL"],
        "raw_run_records": [
            {"scenario": "S1", "estimator": "EKF", "m1_rmse_hz": 0.10, "m13_cpu_time_us": 100.0, "m2_mae_hz": 0.06},
            {"scenario": "S2", "estimator": "EKF", "m1_rmse_hz": 0.11, "m13_cpu_time_us": 101.0, "m2_mae_hz": 0.07},
            {"scenario": "S1", "estimator": "PLL", "m1_rmse_hz": 0.20, "m13_cpu_time_us": 40.0, "m2_mae_hz": 0.11},
            {"scenario": "S2", "estimator": "PLL", "m1_rmse_hz": 0.21, "m13_cpu_time_us": 42.0, "m2_mae_hz": 0.12},
        ],
        "aggregated_metrics": [
            {"scenario": "S1", "estimator": "EKF", "m1_rmse_hz_mean": 0.10},
            {"scenario": "S1", "estimator": "PLL", "m1_rmse_hz_mean": 0.20},
        ],
        "advanced_analysis": {"trends": {}, "robust_statistics": {"config": {"bootstrap_iters": 100}}},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fake_hypotheses(path: Path) -> None:
    path.write_text(
        """
hypotheses:
  - id: H1
    title: Model vs Loop RMSE
    metric: m1_rmse_hz
    group_a: family:Model-based
    group_b: family:Loop-based
    test: mw
    alpha: 0.05
    correction: holm
    mode: preregistered
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _fake_schema(path: Path) -> None:
    path.write_text(
        """
schema_version: "1.0"
required_fields: [id, metric, group_a, group_b, alpha, correction, mode, test]
allowed:
  correction: [holm, bh]
  mode: [preregistered, exploratory]
  test: [mw]
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_stats_and_report_generation(tmp_path: Path) -> None:
    report_json = tmp_path / "benchmark_full_report.json"
    hypotheses = tmp_path / "hypotheses.yaml"
    schema = tmp_path / "hypotheses_schema.yaml"
    out = tmp_path / "out"

    _fake_report(report_json)
    _fake_hypotheses(hypotheses)
    _fake_schema(schema)

    stats_result = run_hypotheses(
        hypotheses_path=hypotheses,
        schema_path=schema,
        input_json_path=report_json,
        output_dir=out,
        require_canonical_input=False,
    )
    assert Path(stats_result["json_path"]).exists()
    assert Path(stats_result["csv_path"]).exists()
    assert Path(stats_result["md_path"]).exists()

    rep_result = build_benchmark_report(input_json=report_json, output_dir=out, stats_json=Path(stats_result["json_path"]))
    assert Path(rep_result["markdown"]).exists()
    assert Path(rep_result["html"]).exists()
    assert Path(rep_result["appendix_csv"]).exists()
    assert Path(rep_result["run_manifest"]).exists()
