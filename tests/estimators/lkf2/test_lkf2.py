import sys
from pathlib import Path
import numpy as np

# Rutas
ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from estimators.lkf2 import LKF2_Estimator


def test_lkf2_nominal_pure_sine():
    """
    Valida seguimiento nominal razonable a 50 Hz:
    - media cercana a 50 Hz
    - error acotado tras asentamiento
    """
    fs = 10000.0
    t = np.arange(0.0, 1.0, 1.0 / fs)
    v = np.sin(2.0 * np.pi * 50.0 * t)

    estimator = LKF2_Estimator(
        nominal_f=50.0,
        q_dc=0.005,
        q_vc=0.05,
        q_vs=0.05,
        r=1.0,
        beta=50.0,
        lpf_mu=1.0,
    )
    f_est = estimator.estimate(t, v)

    f_clean = f_est[t > 0.25]
    f_mean = np.mean(f_clean)
    abs_err = np.abs(f_clean - 50.0)

    assert np.isclose(f_mean, 50.0, atol=0.2)
    assert np.percentile(abs_err, 95) < 0.8


def test_lkf2_step_tracking():
    """
    Valida seguimiento de un escalón moderado de frecuencia.
    Usamos 50 -> 52 Hz porque es un caso razonable y consistente
    con un baseline LKF basado en fase.
    """
    fs = 10000.0
    t = np.arange(0.0, 1.5, 1.0 / fs)

    f_true = np.where(t < 0.5, 50.0, 52.0)
    phase = np.cumsum(2.0 * np.pi * f_true / fs)
    v = np.sin(phase)

    estimator = LKF2_Estimator(
        nominal_f=50.0,
        q_dc=0.005,
        q_vc=0.05,
        q_vs=0.05,
        r=1.0,
        beta=50.0,
        lpf_mu=1.0,
    )
    f_est = estimator.estimate(t, v)

    f_post = f_est[t > 0.75]
    abs_err = np.abs(f_post - 52.0)

    assert np.isclose(np.mean(f_post), 52.0, atol=0.4)
    assert np.percentile(abs_err, 95) < 1.0


def test_lkf2_robustness_to_noise():
    """
    Valida robustez razonable ante ruido aditivo gaussiano.
    """
    fs = 10000.0
    t = np.arange(0.0, 1.5, 1.0 / fs)
    v = np.sin(2.0 * np.pi * 50.0 * t)

    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 0.05, size=len(t))
    v_noisy = v + noise

    estimator = LKF2_Estimator(
        nominal_f=50.0,
        q_dc=0.005,
        q_vc=0.05,
        q_vs=0.05,
        r=1.0,
        beta=50.0,
        lpf_mu=0.25,
    )
    f_est = estimator.estimate(t, v_noisy)

    f_clean = f_est[t > 0.25]
    abs_err = np.abs(f_clean - 50.0)

    f_mean = np.mean(f_clean)
    p95_err = np.percentile(abs_err, 95)
    rms_err = np.sqrt(np.mean((f_clean - 50.0) ** 2))

    assert np.isclose(f_mean, 50.0, atol=0.5)
    assert p95_err < 1.5
    assert rms_err < 0.8