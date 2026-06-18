import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from estimators.tkeo import TKEO_Estimator


def test_tkeo_nominal_pure_sine():
    """Validate reasonable steady-state tracking at 60 Hz nominal."""
    fs = 10000.0
    t = np.arange(0.0, 1.0, 1.0 / fs)
    v = np.sin(2.0 * np.pi * 60.0 * t)

    estimator = TKEO_Estimator(
        nominal_f=60.0,
        output_smoothing=0.01,
        input_smoothing=1.0,
        noise_power=0.0,
    )
    f_est = estimator.estimate(t, v)

    f_clean = f_est[t > 0.15]
    f_mean = np.mean(f_clean)
    abs_err = np.abs(f_clean - 60.0)

    assert np.isclose(f_mean, 60.0, atol=0.5)
    assert np.percentile(abs_err, 95) < 2.0


def test_tkeo_step_tracking():
    """Validate tracking of a moderate 60 -> 62 Hz step."""
    fs = 10000.0
    t = np.arange(0.0, 1.5, 1.0 / fs)

    f_true = np.where(t < 0.5, 60.0, 62.0)
    phase = np.cumsum(2.0 * np.pi * f_true / fs)
    v = np.sin(phase)

    estimator = TKEO_Estimator(
        nominal_f=60.0,
        output_smoothing=0.02,
        input_smoothing=1.0,
        noise_power=0.0,
    )
    f_est = estimator.estimate(t, v)

    f_post = f_est[t > 0.75]
    abs_err = np.abs(f_post - 62.0)

    assert np.isclose(np.mean(f_post), 62.0, atol=1.0)
    assert np.percentile(abs_err, 95) < 3.0


def test_tkeo_output_validity():
    """Validate TKEO returns finite outputs for a pure sine."""
    fs = 10000.0
    t = np.arange(0.0, 0.5, 1.0 / fs)
    v = np.sin(2.0 * np.pi * 60.0 * t)

    estimator = TKEO_Estimator(nominal_f=60.0)
    f_est = estimator.estimate(t, v)

    finite_rate = np.mean(np.isfinite(f_est))
    assert finite_rate > 0.95
