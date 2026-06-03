import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from estimators.type3_sogi_pll import Type3_SOGI_PLL_Estimator

def test_type3_nominal():
    fs = 10000.0
    t = np.arange(0.0, 1.0, 1.0 / fs)
    v = np.sin(2.0 * np.pi * 60.0 * t)

    estimator = Type3_SOGI_PLL_Estimator(nominal_f=60.0)
    f_est = estimator.estimate(t, v)

    f_clean = f_est[t > 0.2]
    assert np.isclose(np.mean(f_clean), 60.0, atol=0.01)

def test_type3_step():
    fs = 10000.0
    t = np.arange(0.0, 1.5, 1.0 / fs)
    
    f_true = np.where(t < 0.5, 60.0, 62.0)
    phase = np.cumsum(2.0 * np.pi * f_true / fs)
    v = np.sin(phase)

    estimator = Type3_SOGI_PLL_Estimator(nominal_f=60.0)
    f_est = estimator.estimate(t, v)

    f_post = f_est[t > 1.0]
    assert np.isclose(np.mean(f_post), 62.0, atol=0.05)