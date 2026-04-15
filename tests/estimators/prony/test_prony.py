import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from estimators.prony import Prony_Estimator

def test_prony_steady_state():
    """Valida que Prony pueda detectar una frecuencia nominal pura."""
    fs = 10000.0
    dt = 1.0 / fs
    t = np.arange(0, 0.1, dt)
    f_target = 61.2
    v = np.sin(2.0 * np.pi * f_target * t)
    
    # Ventana de medio ciclo para mostrar su velocidad
    prony = Prony_Estimator(nominal_f=60.0, n_cycles=0.5, order=4)
    f_est = prony.estimate(t, v)
    
    assert np.isclose(f_est[-1], f_target, atol=0.01)

def test_prony_rejects_dc_offset():
    """Valida que un orden p=4 logre aislar la sinusoide de una componente DC."""
    fs = 10000.0
    dt = 1.0 / fs
    t = np.arange(0, 0.1, dt)
    f_target = 60.0
    
    # Señal + Offset DC que suele arruinar los métodos de cruce por cero
    v = np.sin(2.0 * np.pi * f_target * t) + 0.5 
    
    prony = Prony_Estimator(nominal_f=60.0, n_cycles=1.0, order=4)
    f_est = prony.estimate(t, v)
    
    # La estimación debe centrarse en 60Hz ignorando el polo DC
    assert np.isclose(f_est[-1], f_target, atol=0.05)