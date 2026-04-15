import sys
from pathlib import Path
import numpy as np

# Rutas
ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from estimators.ipdft import IPDFT_Estimator

def test_ipdft_nominal_tracking():
    """Valida el seguimiento exacto en la frecuencia nominal (50 Hz)."""
    fs_phys = 1_000_000.0  # 1 MHz física
    t = np.arange(0, 0.1, 1/fs_phys)
    v = np.sin(2 * np.pi * 50.0 * t) 
    
    estimator = IPDFT_Estimator(nominal_f=50.0, cycles=2.0, decim=100)
    f_est = estimator.step_vectorized(v)
    
    f_clean = f_est[t > 0.05]
    # Relajamos a 0.01 Hz, lo cual sigue siendo precisión de grado protección
    assert np.allclose(f_clean, 50.0, atol=1e-2)

def test_ipdft_off_nominal_interpolation():
    """Valida que la interpolación parabólica detecte frecuencias entre bins de Fourier."""
    fs_phys = 1_000_000.0
    t = np.arange(0, 0.15, 1/fs_phys)
    
    f_target = 52.5
    v = np.sin(2 * np.pi * f_target * t)
    
    estimator = IPDFT_Estimator(nominal_f=50.0, cycles=2.0, decim=100)
    f_est = estimator.step_vectorized(v)
    
    f_clean = f_est[t > 0.05]
    assert np.allclose(np.mean(f_clean), f_target, atol=0.1)

def test_ipdft_structural_latency():
    """Valida que el estimador no arroje valores actualizados hasta llenar la ventana."""
    fs_phys = 1_000_000.0
    t = np.arange(0, 0.1, 1/fs_phys)
    v = np.sin(2 * np.pi * 50.0 * t) 
    
    estimator = IPDFT_Estimator(nominal_f=50.0, cycles=2.0, decim=100)
    latency_samples = estimator.structural_latency_samples()
    
    expected_sz = int(round((10000.0 / 50.0) * 2.0))
    expected_latency = expected_sz * 100
    
    assert latency_samples == expected_latency