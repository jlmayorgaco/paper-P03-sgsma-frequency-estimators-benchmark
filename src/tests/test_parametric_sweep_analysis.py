from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis.parametric_sweep_analysis import build_parametric_regression_reports


def test_parametric_regression_report_builds(tmp_path: Path) -> None:
    manifest = {
        "families": [
            {
                "key": "ibr_harmonics",
                "display_name": "IBR Harmonics Severity",
                "sweep_label": "Harmonic level",
                "unit": "pu",
                "xscale": "linear",
                "scenarios": [
                    {"scenario_name": "SweepA", "sweep_value": 0.01, "is_anchor": False},
                    {"scenario_name": "SweepB", "sweep_value": 0.02, "is_anchor": True},
                    {"scenario_name": "SweepC", "sweep_value": 0.03, "is_anchor": False},
                ],
            }
        ]
    }
    (tmp_path / "scenario_sweep_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    df = pd.DataFrame(
        [
            {
                "scenario": "SweepA",
                "estimator": "PLL",
                "family": "Loop-based",
                "m1_rmse_hz_mean": 1.0,
                "m3_max_peak_hz_mean": 1.0,
                "m5_trip_risk_s_mean": 0.2,
                "m8_settling_time_s_mean": 0.4,
            },
            {
                "scenario": "SweepB",
                "estimator": "PLL",
                "family": "Loop-based",
                "m1_rmse_hz_mean": 1.1,
                "m3_max_peak_hz_mean": 1.1,
                "m5_trip_risk_s_mean": 0.22,
                "m8_settling_time_s_mean": 0.44,
            },
            {
                "scenario": "SweepC",
                "estimator": "PLL",
                "family": "Loop-based",
                "m1_rmse_hz_mean": 1.21,
                "m3_max_peak_hz_mean": 1.21,
                "m5_trip_risk_s_mean": 0.242,
                "m8_settling_time_s_mean": 0.484,
            },
        ]
    )
    df.to_csv(tmp_path / "global_metrics_report.csv", index=False)

    outputs = build_parametric_regression_reports(tmp_path)
    summary = pd.read_csv(outputs["csv"])
    rmse_row = summary[(summary["estimator"] == "PLL") & (summary["metric_col"] == "m1_rmse_hz_mean")].iloc[0]

    assert abs(float(rmse_row["pct_change_per_report_step"]) - 10.0) < 0.3
    assert "PLL: RMSE [Hz] increases" in str(rmse_row["statement"])
