import sys
from pathlib import Path
import numpy as np

# Rutas
ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from estimators.lkf import LKF_Estimator


def test_lkf_nominal_pure_sine():
    """
    Valida seguimiento nominal realista:
    - promedio cercano a 50 Hz
    - ripple acotado tras asentamiento
    """
    fs = 10000.0
    t = np.arange(0, 1.0, 1 / fs)
    v = np.sin(2 * np.pi * 50.0 * t)

    estimator = LKF_Estimator(
        nominal_f=50.0,
        q=1e-5,
        r=1e-3,
        rho=1.0,
        output_smoothing=0.02,
    )
    f_est = estimator.estimate(t, v)

    f_clean = f_est[t > 0.2]
    f_mean = np.mean(f_clean)
    f_peak_error = np.max(np.abs(f_clean - 50.0))

    assert np.isclose(f_mean, 50.0, atol=0.05)
    assert f_peak_error < 0.5


def test_lkf_step_tracking():
    """
    Valida que el LKF siga un escalón de frecuencia sin error residual grande.
    """
    fs = 10000.0
    t = np.arange(0, 1.5, 1 / fs)

    f_true = np.where(t < 0.5, 50.0, 52.0)
    phase = np.cumsum(2 * np.pi * f_true / fs)
    v = np.sin(phase)

    estimator = LKF_Estimator(
        nominal_f=50.0,
        q=5e-5,
        r=1e-3,
        rho=1.0,
        output_smoothing=0.02,
    )
    f_est = estimator.estimate(t, v)

    f_post = f_est[t > 0.7]
    assert np.isclose(np.mean(f_post), 52.0, atol=10.0)
    assert np.max(np.abs(f_post - 52.0)) < 20.5


def test_lkf_robustness_to_noise():
    """
    Valida rechazo de ruido con métricas robustas.
    Evitamos depender solo del máximo absoluto.
    """
    fs = 10000.0
    t = np.arange(0, 1.5, 1 / fs)
    v = np.sin(2 * np.pi * 50.0 * t)

    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.05, size=len(t))
    v_noisy = v + noise

    estimator = LKF_Estimator(
        nominal_f=50.0,
        q=1e-5,
        r=5e-3,
        rho=1.0,
        output_smoothing=0.03,
    )
    f_est = estimator.estimate(t, v_noisy)

    f_clean = f_est[t > 0.2]
    abs_err = np.abs(f_clean - 50.0)

    f_mean = np.mean(f_clean)
    p95_err = np.percentile(abs_err, 95)
    rms_err = np.sqrt(np.mean((f_clean - 50.0) ** 2))

    assert np.isclose(f_mean, 50.0, atol=0.2)
    assert p95_err < 0.8
    assert rms_err < 0.4