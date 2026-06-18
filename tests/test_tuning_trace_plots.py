from __future__ import annotations

import json

import pandas as pd

from openfreqbench.tuning_trace_plots import build_tuning_trace_plots


def test_build_tuning_trace_plots_writes_ieee_artifacts(tmp_path) -> None:
    run_root = tmp_path / "run"
    spec_dir = run_root / "selected_replay" / "m1_rmse_hz" / "IEEE_Single_SinWave" / "ZCD"
    spec_dir.mkdir(parents=True)
    (spec_dir / "run_spec.json").write_text(
        json.dumps(
            {
                "objective_metric": "m1_rmse_hz",
                "scenario": "IEEE_Single_SinWave",
                "estimator": "ZCD",
                "best_params": {"nominal_f": 60.0},
            }
        ),
        encoding="utf-8",
    )

    manifest = build_tuning_trace_plots(
        run_root,
        formats=("png",),
        dpi=120,
        window_s=0.04,
        min_error_mhz=1.0,
        max_error_mhz=10.0,
    )

    out_dir = run_root / "plots" / "ieee"
    summary = pd.read_csv(out_dir / "trace_summary_ieee_mhz.csv")

    assert manifest["status"] == "pass"
    assert manifest["n_trace_specs"] == 1
    assert (out_dir / "ieee_plot_manifest.json").exists()
    assert (out_dir / "ground_truth_ieee_mhz.png").exists()
    assert (out_dir / "traces_m1_rmse_hz_ieee_mhz.png").exists()
    assert (out_dir / "trace_rmse_summary_ieee_mhz.png").exists()
    assert summary.loc[0, "estimator"] == "ZCD"
    assert summary.loc[0, "trace_rmse_mhz"] >= 0.0
