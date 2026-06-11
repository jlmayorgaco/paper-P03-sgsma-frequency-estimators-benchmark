from __future__ import annotations

import json

import pandas as pd

from openfreqbench.schemas import validate_payload
from pipelines.full_mc_tuning_matrix import build_parser, run_matrix


def test_full_mc_tuning_matrix_dry_run_writes_plan(tmp_path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--run-id",
            "dry",
            "--output-dir",
            str(tmp_path),
            "--scenario",
            "IEEE_Single_SinWave",
            "--estimator",
            "ZCD",
            "--objective",
            "m1_rmse_hz",
            "--dry-run",
        ]
    )

    path = run_matrix(args)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["pipeline"] == "full_mc_tuning_matrix"
    assert payload["scenarios"] == ["IEEE_Single_SinWave"]
    assert payload["estimators"] == ["ZCD"]
    assert payload["objectives"] == ["m1_rmse_hz"]


def test_full_mc_tuning_matrix_smoke_writes_replay_artifact(tmp_path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--run-id",
            "smoke",
            "--output-dir",
            str(tmp_path),
            "--scenario",
            "IEEE_Single_SinWave",
            "--estimator",
            "ZCD",
            "--objective",
            "m1_rmse_hz",
            "--n-trials",
            "1",
            "--tune-runs",
            "1",
            "--eval-runs",
            "1",
            "--n-cost-reps",
            "1",
            "--no-capture-signals",
        ]
    )

    report_path = run_matrix(args)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    matrix = pd.read_csv(tmp_path / "smoke" / "tuning_matrix.csv")
    replay_spec = (
        tmp_path
        / "smoke"
        / "selected_replay"
        / "m1_rmse_hz"
        / "IEEE_Single_SinWave"
        / "ZCD"
        / "run_spec.json"
    )

    assert report["run_configuration"]["pipeline"] == "full_mc_tuning_matrix"
    assert validate_payload("benchmark-report", report) == []
    assert len(matrix) == 1
    assert matrix.loc[0, "objective_metric"] == "m1_rmse_hz"
    assert replay_spec.exists()
