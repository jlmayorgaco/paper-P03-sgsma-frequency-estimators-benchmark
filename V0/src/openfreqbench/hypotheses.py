from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _h(
    hypothesis_id: str,
    title: str,
    metric: str,
    group_a: str,
    group_b: str,
    correction: str = "holm",
) -> dict[str, Any]:
    return {
        "id": hypothesis_id,
        "title": title,
        "metric": metric,
        "group_a": group_a,
        "group_b": group_b,
        "test": "mw",
        "alpha": 0.05,
        "correction": correction,
        "mode": "preregistered",
    }


def build_hypothesis_bank(scope: str = "starter") -> dict[str, list[dict[str, Any]]]:
    global_rows = [
        _h(
            "H_AUTO_GLOBAL_001",
            "Global RMSE difference: Model-based vs Loop-based",
            "m1_rmse_hz",
            "family:Model-based",
            "family:Loop-based",
        ),
        _h(
            "H_AUTO_GLOBAL_002",
            "Global RMSE difference: Window-based vs Loop-based",
            "m1_rmse_hz",
            "family:Window-based",
            "family:Loop-based",
        ),
        _h(
            "H_AUTO_GLOBAL_003",
            "Global CPU difference: Model-based vs Window-based",
            "m13_cpu_time_us",
            "family:Model-based",
            "family:Window-based",
        ),
        _h(
            "H_AUTO_GLOBAL_004",
            "Global MAE difference: Data-driven vs Model-based",
            "m2_mae_hz",
            "family:Data-driven",
            "family:Model-based",
        ),
    ]
    estimator_rows = [
        _h("H_AUTO_EST_001", "ZCD vs IPDFT RMSE", "m1_rmse_hz", "estimator:ZCD", "estimator:IPDFT", "bh"),
        _h("H_AUTO_EST_002", "PLL vs SOGI-PLL trip risk", "m5_trip_risk_s", "estimator:PLL", "estimator:SOGI-PLL", "bh"),
        _h("H_AUTO_EST_003", "EKF vs UKF RMSE", "m1_rmse_hz", "estimator:EKF", "estimator:UKF", "bh"),
        _h("H_AUTO_EST_004", "LKF vs LKF2 RMSE", "m1_rmse_hz", "estimator:LKF", "estimator:LKF2", "bh"),
        _h("H_AUTO_EST_005", "Prony vs ESPRIT RMSE", "m1_rmse_hz", "estimator:Prony", "estimator:ESPRIT", "bh"),
    ]
    scenario_rows = [
        _h(
            "H_AUTO_SCENARIO_001",
            "Ramp 20 Hz/s vs 0.25 Hz/s RMSE",
            "m1_rmse_hz",
            "scenario:IEEE_Freq_Ramp_20Hzs",
            "scenario:IEEE_Freq_Ramp_0.25Hzs",
        ),
        _h(
            "H_AUTO_SCENARIO_002",
            "Ringdown severe vs low noise RMSE",
            "m1_rmse_hz",
            "scenario:IBR_Power_Imbalance_Ringdown_Severe_Noise",
            "scenario:IBR_Power_Imbalance_Ringdown_Low_Noise",
        ),
        _h(
            "H_AUTO_SCENARIO_003",
            "Phase jump 60 vs 20 RMSE",
            "m1_rmse_hz",
            "scenario:IEEE_Phase_Jump_60",
            "scenario:IEEE_Phase_Jump_20",
        ),
        _h(
            "H_AUTO_SCENARIO_004",
            "Harmonics large vs small RMSE",
            "m1_rmse_hz",
            "scenario:IBR_Harmonics_Large",
            "scenario:IBR_Harmonics_Small",
        ),
    ]

    if scope == "starter":
        rows = global_rows[:2] + estimator_rows[:2]
    elif scope == "canonical":
        rows = global_rows + estimator_rows + scenario_rows
    else:
        raise ValueError("scope must be `starter` or `canonical`.")
    return {"hypotheses": rows}


def write_hypothesis_bank(output_path: Path, scope: str = "starter") -> Path:
    payload = build_hypothesis_bank(scope=scope)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output_path

