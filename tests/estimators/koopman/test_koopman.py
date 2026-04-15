import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from estimators.koopman import Koopman_Estimator

def test_koopman_dynamic_tracking():
    """Valida que RK-DPMU rastree un transitorio de red."""
    fs = 10000.0
    dt = 1.0 / fs
    t = np.arange(0, 0.2, dt)
    
    # Generamos una rampa suave de frecuencia (Dinámica transitoria)
    f_target = 60.0 + 5.0 * t  # RoCoF de 5 Hz/s
    phase = np.cumsum(2.0 * np.pi * f_target * dt)
    v = np.sin(phase)
    
    rk = Koopman_Estimator(nominal_f=60.0, n_cycles=1.5)
    f_est = rk.estimate(t, v)
    
    # Validamos el final de la rampa. 
    # Al ser un método de ventana, tendrá un ligero retraso teórico (latencia estructural)
    # Frecuencia real en t=0.2 es 61.0 Hz. La estimada debe estar muy cerca.
    final_f = f_est[-1]
    assert np.isclose(final_f, 61.0, atol=0.15)