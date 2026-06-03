import sys
from pathlib import Path
import numpy as np

# Rutas
ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from estimators.epll import EPLL_Estimator


def test_epll_nominal_pure_sine():
    """Valida seguimiento perfecto en estado estacionario (sin rizado)."""
    fs = 10000.0
    t = np.arange(0.0, 1.0, 1.0 / fs)
    # EPLL usa seno internamente por defecto en la formulación clásica
    v = np.sin(2.0 * np.pi * 50.0 * t)

    estimator = EPLL_Estimator(nominal_f=50.0)
    f_est = estimator.estimate(t, v)

    f_clean = f_est[t > 0.15]
    f_mean = np.mean(f_clean)
    abs_err = np.abs(f_clean - 50.0)

    assert np.isclose(f_mean, 50.0, atol=0.01)
    # Tolerancia muchísimo más estricta que el Basic PLL porque no hay rizado 2w
    assert np.max(abs_err) < 0.05 


def test_epll_step_tracking():
    """Valida seguimiento rápido y sin sobreimpulso masivo de un escalón 50 -> 52 Hz."""
    fs = 10000.0
    t = np.arange(0.0, 1.0, 1.0 / fs)

    f_true = np.where(t < 0.3, 50.0, 52.0)
    phase = np.cumsum(2.0 * np.pi * f_true / fs)
    v = np.sin(phase)

    estimator = EPLL_Estimator(nominal_f=50.0)
    f_est = estimator.estimate(t, v)

    f_post = f_est[t > 0.6]
    abs_err = np.abs(f_post - 52.0)

    assert np.isclose(np.mean(f_post), 52.0, atol=0.05)
    assert np.max(abs_err) < 0.10


def test_epll_amplitude_immunity():
    """
    [SUPERPODER DEL EPLL] Valida que un hueco de tensión (voltage sag) severo 
    no destruya la estimación de frecuencia.
    """
    fs = 10000.0
    t = np.arange(0.0, 1.0, 1.0 / fs)
    
    # Frecuencia constante a 50Hz, pero la amplitud cae al 40% (hueco del 60%)
    amp = np.where((t > 0.3) & (t < 0.7), 0.4, 1.0)
    v = amp * np.sin(2.0 * np.pi * 50.0 * t)

    estimator = EPLL_Estimator(nominal_f=50.0)
    f_est = estimator.estimate(t, v)

    # Revisamos la frecuencia DURANTE el hueco de tensión
    f_during_sag = f_est[(t > 0.4) & (t < 0.6)]
    
    # La estimación debe mantenerse robusta cerca de 50Hz
    assert np.mean(f_during_sag) > 49.5
    assert np.mean(f_during_sag) < 50.5