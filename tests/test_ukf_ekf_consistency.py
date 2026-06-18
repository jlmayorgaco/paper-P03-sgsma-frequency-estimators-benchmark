from __future__ import annotations

import numpy as np

from estimators.ekf import EKF_Estimator
from estimators.ukf import UKF_Estimator


def test_ukf_and_ekf_agree_under_low_motion_harmonic_tuning() -> None:
    fs = 10_000.0
    t = np.arange(0.0, 0.30, 1.0 / fs, dtype=float)
    f_true = np.full_like(t, 60.0)
    phi = 2.0 * np.pi * 60.0 * t
    v = np.sin(phi) + 0.05 * np.sin(5.0 * phi + 0.37) + 0.03 * np.sin(7.0 * phi + 1.11)

    common = {
        "q_dc": 5e-10,
        "q_alpha": 3e-8,
        "q_beta": 1e-8,
        "q_omega": 1e-8,
        "r_meas": 1e-2,
        "output_smoothing": 0.03,
        "p_dc": 200.0,
        "p_alpha": 100.0,
        "p_beta": 100.0,
        "p_omega_hz": 0.01,
    }
    ekf = EKF_Estimator(**common)
    ukf = UKF_Estimator(**common, alpha_ut=0.7, beta_ut=3.5, kappa_ut=1.0)

    start = int(0.10 * fs)
    ekf_err = ekf.estimate(t, v)[start:] - f_true[start:]
    ukf_err = ukf.estimate(t, v)[start:] - f_true[start:]
    ekf_rmse = float(np.sqrt(np.mean(ekf_err**2)))
    ukf_rmse = float(np.sqrt(np.mean(ukf_err**2)))

    assert ukf_rmse <= ekf_rmse * 1.10 + 1e-6
    assert "f_nom=60.0Hz" in UKF_Estimator.describe_params({})
