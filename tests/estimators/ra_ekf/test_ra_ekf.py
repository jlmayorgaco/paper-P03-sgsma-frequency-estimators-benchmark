import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from estimators.ra_ekf import RAEKF_Estimator


def test_ra_ekf_nominal_pure_sine():
    """Validate reasonable steady-state tracking at 60 Hz nominal."""
    fs = 10000.0
    t = np.arange(0.0, 1.0, 1.0 / fs)
    v = np.sin(2.0 * np.pi * 60.0 * t)

    estimator = RAEKF_Estimator(nominal_f=60.0)
    f_est = estimator.estimate(t, v)

    f_clean = f_est[t > 0.2]
    f_mean = np.mean(f_clean)
    abs_err = np.abs(f_clean - 60.0)

    assert np.isclose(f_mean, 60.0, atol=0.1)
    assert np.percentile(abs_err, 95) < 0.3


def test_ra_ekf_step_tracking():
    """Validate tracking of a 60 -> 62 Hz step."""
    fs = 10000.0
    t = np.arange(0.0, 1.5, 1.0 / fs)

    f_true = np.where(t < 0.5, 60.0, 62.0)
    phase = np.cumsum(2.0 * np.pi * f_true / fs)
    v = np.sin(phase)

    estimator = RAEKF_Estimator(nominal_f=60.0)
    f_est = estimator.estimate(t, v)

    f_post = f_est[t > 0.75]
    abs_err = np.abs(f_post - 62.0)

    assert np.isclose(np.mean(f_post), 62.0, atol=0.5)
    assert np.percentile(abs_err, 95) < 1.0


def test_ra_ekf_robustness_to_noise():
    """Validate reasonable robustness under additive noise."""
    fs = 10000.0
    t = np.arange(0.0, 1.0, 1.0 / fs)
    v = np.sin(2.0 * np.pi * 60.0 * t)

    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 0.02, size=len(t))
    v_noisy = v + noise

    estimator = RAEKF_Estimator(nominal_f=60.0)
    f_est = estimator.estimate(t, v_noisy)

    f_clean = f_est[t > 0.2]
    abs_err = np.abs(f_clean - 60.0)
    f_mean = np.mean(f_clean)
    rms_err = np.sqrt(np.mean((f_clean - 60.0) ** 2))

    assert np.isclose(f_mean, 60.0, atol=0.2)
    assert np.percentile(abs_err, 95) < 0.5
    assert rms_err < 0.3


def test_ra_ekf_output_validity():
    """Validate RA-EKF returns finite outputs for a pure sine."""
    fs = 10000.0
    t = np.arange(0.0, 0.5, 1.0 / fs)
    v = np.sin(2.0 * np.pi * 60.0 * t)

    estimator = RAEKF_Estimator(nominal_f=60.0)
    f_est = estimator.estimate(t, v)

    finite_rate = np.mean(np.isfinite(f_est))
    assert finite_rate > 0.95
