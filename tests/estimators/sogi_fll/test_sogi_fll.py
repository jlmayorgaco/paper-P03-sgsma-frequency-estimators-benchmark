import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from estimators.sogi_fll import SOGI_FLL_Estimator

def test_sogi_fll_nominal():
    fs = 10000.0
    t = np.arange(0.0, 1.0, 1.0 / fs)
    v = np.sin(2.0 * np.pi * 50.0 * t)

    estimator = SOGI_FLL_Estimator(nominal_f=50.0)
    f_est = estimator.estimate(t, v)

    f_clean = f_est[t > 0.15]
    assert np.isclose(np.mean(f_clean), 50.0, atol=0.01)
    assert np.max(np.abs(f_clean - 50.0)) < 0.05

def test_sogi_fll_step():
    fs = 10000.0
    t = np.arange(0.0, 1.5, 1.0 / fs)
    
    f_true = np.where(t < 0.5, 50.0, 52.0)
    phase = np.cumsum(2.0 * np.pi * f_true / fs)
    v = np.sin(phase)

    estimator = SOGI_FLL_Estimator(nominal_f=50.0, gamma=60.0)
    f_est = estimator.estimate(t, v)

    f_post = f_est[t > 0.9]
    assert np.isclose(np.mean(f_post), 52.0, atol=0.05)