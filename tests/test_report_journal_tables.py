from __future__ import annotations

import json

import pandas as pd

from openfreqbench.reports import build_report_outputs


def test_report_build_writes_journal_tables(tmp_path) -> None:
    report_path = tmp_path / "benchmark_report.json"
    rows = []
    for run_idx in range(4):
        rows.append(
            {
                "run_idx": run_idx,
                "scenario": "IEEE_Single_SinWave",
                "estimator": "ZCD",
                "family": "Window-based",
                "m1_rmse_hz": 0.01 + run_idx * 0.001,
                "m2_mae_hz": 0.008,
                "m3_max_peak_hz": 0.02,
                "m5_trip_risk_s": 0.0,
                "m8_settling_time_s": 0.01,
                "m13_cpu_time_us": 2.0,
                "m14_struct_latency_ms": 16.7,
                "m15_pcb_compliant": 1,
                "m22_invalid_output_rate": 0.0,
            }
        )
        rows.append(
            {
                "run_idx": run_idx,
                "scenario": "IBR_Multi_Event",
                "estimator": "PI-GRU",
                "family": "Data-driven",
                "m1_rmse_hz": 0.02 + run_idx * 0.001,
                "m2_mae_hz": 0.015,
                "m3_max_peak_hz": 0.04,
                "m5_trip_risk_s": 0.0,
                "m8_settling_time_s": 0.02,
                "m13_cpu_time_us": 20.0,
                "m14_struct_latency_ms": 20.0,
                "m15_pcb_compliant": 1,
                "m22_invalid_output_rate": 0.0,
            }
        )
    raw = pd.DataFrame(rows)
    agg = raw.groupby(["scenario", "estimator", "family"], as_index=False).agg(
        {
            "m1_rmse_hz": "mean",
            "m13_cpu_time_us": "mean",
            "m14_struct_latency_ms": "mean",
        }
    )
    agg = agg.rename(
        columns={
            "m1_rmse_hz": "m1_rmse_hz_mean",
            "m13_cpu_time_us": "m13_cpu_time_us_mean",
            "m14_struct_latency_ms": "m14_struct_latency_ms_mean",
        }
    )
    report_path.write_text(
        json.dumps(
            {
                "metadata": {},
                "run_configuration": {
                    "run_id": "test",
                    "metric_profile": "canonical-single-phase-v1",
                    "parameter_policy": "artifact_tuned",
                    "tuned_artifacts_dir": "artifacts/full_mc_benchmark",
                },
                "reproducibility": {"git": {"commit": "abc"}, "config": {"path": "config.yaml"}},
                "raw_run_records": raw.to_dict(orient="records"),
                "aggregated_metrics": agg.to_dict(orient="records"),
                "artifacts": {"aggregated_metrics_csv": str(tmp_path / "aggregated_metrics.csv")},
            }
        ),
        encoding="utf-8",
    )
    agg.to_csv(tmp_path / "aggregated_metrics.csv", index=False)

    result = build_report_outputs(report_path)

    assert "metric_confidence_intervals" in result["scientific_tables"]
    assert (report_path.parent / "report" / "failure_analysis.csv").exists()
    assert (report_path.parent / "report" / "paper_traceability.csv").exists()
    assert (report_path.parent / "report" / "artifact_index.csv").exists()


def test_report_build_applies_main_vs_diagnostic_scope(tmp_path) -> None:
    report_path = tmp_path / "benchmark_report.json"
    rows = []
    for run_idx in range(3):
        rows.extend(
            [
                {
                    "run_idx": run_idx,
                    "scenario": "IBR_Multi_Event",
                    "estimator": "Prony",
                    "family": "Window-based",
                    "m1_rmse_hz": 0.001,
                    "m2_mae_hz": 0.001,
                    "m3_max_peak_hz": 0.002,
                    "m5_trip_risk_s": 0.0,
                    "m8_settling_time_s": 0.01,
                    "m13_cpu_time_us": 40.0,
                    "m14_struct_latency_ms": 34.0,
                    "m22_invalid_output_rate": 0.08,
                    "m36_post_startup_invalid_rate": 0.05,
                },
                {
                    "run_idx": run_idx,
                    "scenario": "IBR_Multi_Event",
                    "estimator": "ESPRIT",
                    "family": "Window-based",
                    "m1_rmse_hz": 1.0,
                    "m2_mae_hz": 0.8,
                    "m3_max_peak_hz": 5.0,
                    "m5_trip_risk_s": 0.1,
                    "m8_settling_time_s": 0.02,
                    "m13_cpu_time_us": 50.0,
                    "m14_struct_latency_ms": 17.0,
                    "m22_invalid_output_rate": 0.011267,
                    "m36_post_startup_invalid_rate": 0.0,
                },
                {
                    "run_idx": run_idx,
                    "scenario": "IBR_Multi_Event",
                    "estimator": "ZCD",
                    "family": "Classical",
                    "m1_rmse_hz": 2.0,
                    "m2_mae_hz": 1.8,
                    "m3_max_peak_hz": 9.0,
                    "m5_trip_risk_s": 0.2,
                    "m8_settling_time_s": 0.03,
                    "m13_cpu_time_us": 2.0,
                    "m14_struct_latency_ms": 0.0,
                    "m22_invalid_output_rate": 0.0,
                    "m36_post_startup_invalid_rate": 0.0,
                },
            ]
        )
    raw = pd.DataFrame(rows)
    agg = raw.groupby(["scenario", "estimator", "family"], as_index=False).agg(
        {
            "m1_rmse_hz": "mean",
            "m13_cpu_time_us": "mean",
            "m14_struct_latency_ms": "mean",
            "m22_invalid_output_rate": "mean",
            "m36_post_startup_invalid_rate": "mean",
        }
    )
    agg = agg.rename(columns={col: f"{col}_mean" for col in agg.columns if col.startswith("m")})
    report_path.write_text(
        json.dumps(
            {
                "metadata": {},
                "run_configuration": {
                    "run_id": "scope-test",
                    "metric_profile": "canonical-single-phase-v1",
                    "parameter_policy": "default",
                },
                "reproducibility": {"git": {"commit": "abc"}, "config": {"path": "config.yaml"}},
                "raw_run_records": raw.to_dict(orient="records"),
                "aggregated_metrics": agg.to_dict(orient="records"),
                "artifacts": {"aggregated_metrics_csv": str(tmp_path / "aggregated_metrics.csv")},
            }
        ),
        encoding="utf-8",
    )
    agg.to_csv(tmp_path / "aggregated_metrics.csv", index=False)

    result = build_report_outputs(report_path)

    scope = pd.read_csv(result["scientific_tables"]["paper_scope_classification"])
    diagnostic = pd.read_csv(result["scientific_tables"]["diagnostic_appendix"])
    ranking = pd.read_csv(result["scientific_tables"]["ranking_sensitivity"])
    failure = pd.read_csv(result["scientific_tables"]["failure_analysis"])
    summary = json.loads((report_path.parent / "report" / "analysis_summary.json").read_text())

    prony_scope = scope[scope["estimator"] == "Prony"].iloc[0]
    assert bool(prony_scope["diagnostic_appendix"]) is True
    assert prony_scope["classification_reason"] == "post_startup_invalid"
    assert set(diagnostic["estimator"]) == {"Prony"}

    rmse_ranking = ranking[ranking["metric"] == "m1_rmse_hz"]
    assert "Prony" not in set(rmse_ranking["estimator"])
    assert rmse_ranking.iloc[0]["estimator"] == "ESPRIT"
    assert summary["scenario_winners_by_rmse"][0]["estimator"] == "ESPRIT"
    assert summary["n_diagnostic_appendix_pairs"] == 1

    esprit_failure = failure[failure["estimator"] == "ESPRIT"].iloc[0]
    assert esprit_failure["invalid_output_metric"] == "m36_post_startup_invalid_rate"
    assert esprit_failure["invalid_output_rate"] == 0.0
