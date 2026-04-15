import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from estimators.tft import TFT_Estimator

def test_tft_nominal():
    fs = 10000.0
    t = np.arange(0.0, 1.0, 1.0 / fs)
    v = np.sin(2.0 * np.pi * 60.0 * t)

    estimator = TFT_Estimator(nominal_f=60.0, n_cycles=2.0)
    f_est = estimator.estimate(t, v)

    # Evaluar después de llenar la ventana (2 ciclos a 60Hz = ~33.3 ms)
    f_clean = f_est[t > 0.05]
    assert np.isclose(np.mean(f_clean), 60.0, atol=0.005)
    assert np.max(np.abs(f_clean - 60.0)) < 0.01

def test_tft_rocof_ramp():
    fs = 10000.0
    t = np.arange(0.0, 1.0, 1.0 / fs)
    
    # Rampa sostenida de +5 Hz/s
    f_true = 60.0 + 5.0 * t
    phase = np.cumsum(2.0 * np.pi * f_true / fs)
    v = np.sin(phase)

    estimator = TFT_Estimator(nominal_f=60.0, n_cycles=2.0)
    f_est = estimator.estimate(t, v)

    # Debido a la latencia estructural (M muestras), la estimación 
    # de TFT rastreará la rampa pero desplazada en el tiempo.
    # Evaluamos que la pendiente general se mantenga consistente.
    f_steady_ramp = f_est[t > 0.1]
    rocof_est = np.gradient(f_steady_ramp, 1.0/fs)
    
    assert np.isclose(np.mean(rocof_est), 5.0, atol=0.2)