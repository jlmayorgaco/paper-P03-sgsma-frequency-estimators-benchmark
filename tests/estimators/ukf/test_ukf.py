import sys
from pathlib import Path
import numpy as np

# Rutas
ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from estimators.ukf import UKF_Estimator


def test_ukf_nominal_pure_sine():
    """
    Valida seguimiento nominal razonable a 50 Hz.
    """
    fs = 10000.0
    t = np.arange(0.0, 1.0, 1.0 / fs)
    v = np.sin(2.0 * np.pi * 50.0 * t)

    estimator = UKF_Estimator(
        nominal_f=50.0,
        q_dc=1e-6,
        q_alpha=1e-4,
        q_beta=1e-4,
        q_omega=5e-2,
        r_meas=1e-3,
        output_smoothing=0.02,
        alpha_ut=0.1,
        beta_ut=2.0,
        kappa_ut=0.0,
    )
    f_est = estimator.estimate(t, v)

    f_clean = f_est[t > 0.2]
    f_mean = np.mean(f_clean)
    abs_err = np.abs(f_clean - 50.0)

    assert np.isclose(f_mean, 50.0, atol=0.12)
    assert np.percentile(abs_err, 95) < 0.30


def test_ukf_step_tracking():
    """
    Valida seguimiento de un escalón moderado 50 -> 52 Hz.
    """
    fs = 10000.0
    t = np.arange(0.0, 1.5, 1.0 / fs)

    f_true = np.where(t < 0.5, 50.0, 52.0)
    phase = np.cumsum(2.0 * np.pi * f_true / fs)
    v = np.sin(phase)

    estimator = UKF_Estimator(
        nominal_f=50.0,
        q_dc=1e-6,
        q_alpha=1e-4,
        q_beta=1e-4,
        q_omega=1e-1,
        r_meas=1e-3,
        output_smoothing=0.02,
        alpha_ut=0.1,
        beta_ut=2.0,
        kappa_ut=0.0,
    )
    f_est = estimator.estimate(t, v)

    f_post = f_est[t > 0.75]
    abs_err = np.abs(f_post - 52.0)

    assert np.isclose(np.mean(f_post), 52.0, atol=0.30)
    assert np.percentile(abs_err, 95) < 0.70


def test_ukf_robustness_to_noise():
    """
    Valida robustez razonable ante ruido aditivo.
    """
    fs = 10000.0
    t = np.arange(0.0, 1.5, 1.0 / fs)
    v = np.sin(2.0 * np.pi * 50.0 * t)

    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 0.05, size=len(t))
    v_noisy = v + noise

    estimator = UKF_Estimator(
        nominal_f=50.0,
        q_dc=1e-6,
        q_alpha=1e-4,
        q_beta=1e-4,
        q_omega=5e-2,
        r_meas=5e-3,
        output_smoothing=0.05,
        alpha_ut=0.1,
        beta_ut=2.0,
        kappa_ut=0.0,
    )
    f_est = estimator.estimate(t, v_noisy)

    f_clean = f_est[t > 0.25]
    abs_err = np.abs(f_clean - 50.0)

    f_mean = np.mean(f_clean)
    p95_err = np.percentile(abs_err, 95)
    rms_err = np.sqrt(np.mean((f_clean - 50.0) ** 2))

    assert np.isclose(f_mean, 50.0, atol=0.25)
    assert p95_err < 0.90
    assert rms_err < 0.45