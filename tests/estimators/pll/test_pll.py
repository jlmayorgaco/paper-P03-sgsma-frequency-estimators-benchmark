import sys
from pathlib import Path
import numpy as np

# Rutas
ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from estimators.pll import PLL_Estimator


def test_pll_nominal_pure_sine():
    """
    Valida seguimiento nominal razonable a 50 Hz.
    """
    fs = 10000.0
    t = np.arange(0.0, 1.0, 1.0 / fs)
    v = np.sin(2.0 * np.pi * 50.0 * t)

    estimator = PLL_Estimator(
        nominal_f=50.0,
        settle_time=0.08,
        amp_alpha=0.01,
        output_smoothing=0.02,
    )
    f_est = estimator.estimate(t, v)

    f_clean = f_est[t > 0.25]
    f_mean = np.mean(f_clean)
    abs_err = np.abs(f_clean - 50.0)

    assert np.isclose(f_mean, 50.0, atol=0.15)
    assert np.percentile(abs_err, 95) < 0.40


def test_pll_step_tracking():
    """
    Valida seguimiento de un escalón moderado 50 -> 52 Hz.
    """
    fs = 10000.0
    t = np.arange(0.0, 1.5, 1.0 / fs)

    f_true = np.where(t < 0.5, 50.0, 52.0)
    phase = np.cumsum(2.0 * np.pi * f_true / fs)
    v = np.sin(phase)

    estimator = PLL_Estimator(
        nominal_f=50.0,
        settle_time=0.08,
        amp_alpha=0.01,
        output_smoothing=0.02,
    )
    f_est = estimator.estimate(t, v)

    f_post = f_est[t > 0.8]
    abs_err = np.abs(f_post - 52.0)

    assert np.isclose(np.mean(f_post), 52.0, atol=0.30)
    assert np.percentile(abs_err, 95) < 0.80


def test_pll_robustness_to_noise():
    """
    Valida robustez razonable ante ruido aditivo.
    """
    fs = 10000.0
    t = np.arange(0.0, 1.5, 1.0 / fs)
    v = np.sin(2.0 * np.pi * 50.0 * t)

    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 0.05, size=len(t))
    v_noisy = v + noise

    estimator = PLL_Estimator(
        nominal_f=50.0,
        settle_time=0.08,
        amp_alpha=0.02,
        output_smoothing=0.05,
    )
    f_est = estimator.estimate(t, v_noisy)

    f_clean = f_est[t > 0.25]
    abs_err = np.abs(f_clean - 50.0)

    f_mean = np.mean(f_clean)
    p95_err = np.percentile(abs_err, 95)
    rms_err = np.sqrt(np.mean((f_clean - 50.0) ** 2))

    assert np.isclose(f_mean, 50.0, atol=0.30)
    assert p95_err < 1.0
    assert rms_err < 0.60