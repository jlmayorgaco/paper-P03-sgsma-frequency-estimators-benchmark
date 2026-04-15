import sys
from pathlib import Path
import numpy as np

# Rutas
ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from estimators.sogi_pll import SOGIPLLEstimator


def test_sogi_pll_nominal_pure_sine():
    """
    Valida seguimiento nominal realista en régimen permanente:
    - promedio muy cercano a 50 Hz
    - ripple acotado tras asentamiento
    """
    fs = 10000.0
    t = np.arange(0, 1.0, 1 / fs)
    v = np.sin(2 * np.pi * 50.0 * t)

    estimator = SOGIPLLEstimator(nominal_f=50.0)
    f_est = estimator.estimate(t, v)

    # Dejamos tiempo suficiente para que el lazo y el SOGI entren en régimen
    f_clean = f_est[t > 0.20]

    f_mean = np.mean(f_clean)
    f_peak_error = np.max(np.abs(f_clean - 50.0))

    assert np.isclose(f_mean, 50.0, atol=0.05)
    assert f_peak_error < 0.5


def test_sogi_pll_step_tracking():
    """Valida el asentamiento suave y sin error residual importante."""
    fs = 10000.0
    t = np.arange(0, 1.5, 1 / fs)

    f_true = np.where(t < 0.5, 50.0, 52.0)
    phase = np.cumsum(f_true) * (1 / fs) * 2 * np.pi
    v = np.sin(phase)

    estimator = SOGIPLLEstimator(nominal_f=50.0, settle_time=0.06)
    f_est = estimator.estimate(t, v)

    # A los 150 ms del escalón ya debe estar esencialmente asentado
    assert np.allclose(f_est[t > 0.65], 52.0, atol=0.05)


def test_sogi_pll_robustness_to_noise():
    """
    Valida rechazo de ruido con métricas robustas.
    Evitamos usar solo el máximo absoluto porque es sensible a outliers.
    """
    fs = 10000.0
    t = np.arange(0, 1.5, 1 / fs)
    v = np.sin(2 * np.pi * 50.0 * t)

    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.05, size=len(t))
    v_noisy = v + noise

    estimator = SOGIPLLEstimator(nominal_f=50.0)
    f_est = estimator.estimate(t, v_noisy)

    f_clean = f_est[t > 0.2]
    abs_err = np.abs(f_clean - 50.0)

    f_mean = np.mean(f_clean)
    p95_err = np.percentile(abs_err, 95)
    rms_err = np.sqrt(np.mean((f_clean - 50.0) ** 2))
    f_peak_error = np.max(abs_err)

    assert np.isclose(f_mean, 50.0, atol=0.2)
    assert p95_err < 0.8
    assert rms_err < 0.4
    assert f_peak_error < 1.5