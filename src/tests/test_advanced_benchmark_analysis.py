from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis.advanced_benchmark_analysis import AdvancedBenchmarkAnalyzer, AdvancedStatsConfig  # noqa: E402


def _make_df_agg() -> pd.DataFrame:
    rows = []
    estimators = ["A", "B", "C"]
    for scenario_idx in range(1, 7):
        for est in estimators:
            rmse = {
                "A": 0.10 + 0.01 * scenario_idx,
                "B": 0.13 + 0.01 * scenario_idx,
                "C": 0.18 + 0.02 * scenario_idx,
            }[est]
            cpu = {"A": 11.0, "B": 6.0, "C": 9.0}[est]
            rows.append(
                {
                    "scenario": f"S{scenario_idx}",
                    "estimator": est,
                    "family": "X",
                    "m1_rmse_hz_mean": rmse,
                    "m13_cpu_time_us_mean": cpu,
                    "m2_mae_hz_mean": rmse * 0.8,
                    "m5_trip_risk_s_mean": rmse * 0.5,
                }
            )
    return pd.DataFrame(rows)


def _make_df_long() -> pd.DataFrame:
    rows = []
    estimators = ["A", "B"]
    rng = np.random.default_rng(123)
    for scenario_idx in range(4):
        for est in estimators:
            baseline = 0.1 if est == "A" else 0.2
            for run_idx in range(20):
                rows.append(
                    {
                        "scenario": f"S{scenario_idx}",
                        "estimator": est,
                        "family": "X",
                        "run_idx": run_idx,
                        "m1_rmse_hz": baseline + float(rng.normal(0.0, 0.01)),
                        "m13_cpu_time_us": (10.0 if est == "A" else 7.0) + float(rng.normal(0.0, 0.2)),
                    }
                )
    return pd.DataFrame(rows)


def test_bootstrap_ci_mean_and_cliffs_delta_contract() -> None:
    analyzer = AdvancedBenchmarkAnalyzer(AdvancedStatsConfig(bootstrap_iters=400, alpha=0.05, random_seed=7))
    ci = analyzer.bootstrap_ci_mean([1.0, 1.1, 0.9, 1.2, 1.0])
    assert ci["available"] is True
    assert float(ci["ci_low"]) <= float(ci["mean"]) <= float(ci["ci_high"])

    effect = analyzer.cliffs_delta([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    assert effect["available"] is True
    assert float(effect["delta"]) < 0.0


def test_ranked_statistics_outputs() -> None:
    analyzer = AdvancedBenchmarkAnalyzer(AdvancedStatsConfig(bootstrap_iters=300, alpha=0.05, random_seed=11))
    df_agg = _make_df_agg()
    df_long = _make_df_long()

    win = analyzer.pairwise_win_rate(df_agg, "m1_rmse_hz_mean", lower_better=True)
    assert win["available"] is True
    assert len(win["matrix"]) == 3

    dom = analyzer.dominance_score(
        df_agg,
        metric_cols=["m1_rmse_hz_mean", "m2_mae_hz_mean", "m5_trip_risk_s_mean", "m13_cpu_time_us_mean"],
        lower_better=True,
    )
    assert dom["available"] is True
    assert len(dom["scores"]) == 3

    boot = analyzer.estimator_bootstrap_summary(df_long, "m1_rmse_hz")
    assert boot["available"] is True
    assert len(boot["estimators"]) == 2

    friedman = analyzer.friedman_test(df_agg, "m1_rmse_hz_mean")
    assert "available" in friedman
