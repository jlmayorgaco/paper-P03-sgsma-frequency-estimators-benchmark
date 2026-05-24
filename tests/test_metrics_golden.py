from __future__ import annotations

import numpy as np

from analysis.metrics import calculate_all_metrics


def test_zero_error_metrics_are_zero_or_compliant() -> None:
    f_true = np.full(2000, 60.0)
    f_hat = f_true.copy()
    metrics = calculate_all_metrics(
        f_hat=f_hat,
        f_true=f_true,
        fs_dsp=10000.0,
        exec_time_s=0.01,
        structural_samples=0,
        noise_sigma=0.0,
    )
    assert metrics["m1_rmse_hz"] == 0.0
    assert metrics["m2_mae_hz"] == 0.0
    assert metrics["m3_max_peak_hz"] == 0.0
    assert metrics["m5_trip_risk_s"] == 0.0
    assert metrics["m7_pcb_hz"] == 0.0
    assert metrics["m15_pcb_compliant"] is True
    assert metrics["m16_heatmap_pass"] is True


def test_constant_bias_rmse_mae_peak_match_bias() -> None:
    f_true = np.full(2000, 60.0)
    f_hat = np.full(2000, 60.1)
    metrics = calculate_all_metrics(
        f_hat=f_hat,
        f_true=f_true,
        fs_dsp=10000.0,
        exec_time_s=0.01,
        structural_samples=0,
        noise_sigma=0.0,
    )
    assert abs(metrics["m1_rmse_hz"] - 0.1) < 1e-12
    assert abs(metrics["m2_mae_hz"] - 0.1) < 1e-12
    assert abs(metrics["m3_max_peak_hz"] - 0.1) < 1e-12

