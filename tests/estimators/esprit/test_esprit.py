import sys
import math
import numpy as np
import pytest
from pathlib import Path

# Configuración de rutas
ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from estimators.esprit import ESPRIT_Estimator, _esprit_core

# --- FIXTURE DE DATOS ---
@pytest.fixture
def signal_setup():
    """Genera una señal limpia y los parámetros de Hankel para los tests."""
    fs = 10000.0
    dt = 1.0 / fs
    f_target = 62.0
    t = np.arange(0, 0.02, dt) # 2 ciclos
    v = np.sin(2.0 * np.pi * f_target * t)
    
    N = len(v)
    L = N // 2
    M = N - L + 1
    
    # Construcción de Hankel manual
    H = np.zeros((L, M), dtype=np.float64)
    for i in range(L):
        for j in range(M):
            H[i, j] = v[i + j]
            
    return {"v": v, "dt": dt, "f": f_target, "H": H, "L": L, "M": M}

# --- TESTS GRANULARES ---

def test_hankel_dimensions(signal_setup):
    """¿La matriz de Hankel tiene el tamaño correcto para el algoritmo?"""
    H = signal_setup["H"]
    L, M = signal_setup["L"], signal_setup["M"]
    assert H.shape == (L, M)

def test_hankel_rank(signal_setup):
    """¿La señal es lo suficientemente rica para ESPRIT (Rango >= 2)?"""
    rank = np.linalg.matrix_rank(signal_setup["H"])
    assert rank >= 2

def test_svd_singular_values_count(signal_setup):
    """¿La SVD devuelve suficientes valores para identificar la sinusoide?"""
    U, S, Vh = np.linalg.svd(signal_setup["H"])
    assert len(S) >= 2

def test_svd_signal_to_noise_gap(signal_setup):
    """¿Los componentes de señal dominan a los de ruido? (SNR Gap)"""
    U, S, Vh = np.linalg.svd(signal_setup["H"])
    # En una sinusoide pura, el tercer valor singular debe ser despreciable
    gap = S[1] / (S[2] + 1e-15)
    assert gap > 100.0

def test_subspace_orthonormality(signal_setup):
    """¿El subespacio de señal extraído (Us) es ortonormal?"""
    U, S, Vh = np.linalg.svd(signal_setup["H"])
    Us = U[:, :2]
    identity_approx = Us.T @ Us
    assert np.allclose(identity_approx, np.eye(2), atol=1e-10)

def test_rotation_partition_shapes(signal_setup):
    """¿Las particiones para la invarianza rotacional tienen sentido?"""
    U, _, _ = np.linalg.svd(signal_setup["H"])
    Us = U[:, :2]
    Us1 = Us[:-1, :]
    Us2 = Us[1:, :]
    assert Us1.shape == Us2.shape
    assert Us1.shape[1] == 2

def test_phi_matrix_stability(signal_setup):
    """¿La matriz de rotación Phi se calcula sin errores numéricos?"""
    U, _, _ = np.linalg.svd(signal_setup["H"])
    Us = U[:, :2]
    Us1, Us2 = Us[:-1, :], Us[1:, :]
    phi = np.linalg.pinv(Us1) @ Us2
    assert np.isfinite(phi).all()

def test_phi_unitarity(signal_setup):
    """¿Phi es una matriz de rotación? (Determinante cerca de 1)"""
    U, _, _ = np.linalg.svd(signal_setup["H"])
    Us = U[:, :2]
    phi = np.linalg.pinv(Us[:-1, :]) @ Us[1:, :]
    det = np.linalg.det(phi)
    assert math.isclose(abs(det), 1.0, rel_tol=0.1)

def test_eigenvalues_unit_circle(signal_setup):
    """¿Los autovalores de Phi están en el círculo unitario? (Sin amortiguamiento)"""
    U, _, _ = np.linalg.svd(signal_setup["H"])
    Us = U[:, :2]
    phi = np.linalg.pinv(Us[:-1, :]) @ Us[1:, :]
    eigvals = np.linalg.eigvals(phi)
    for ev in eigvals:
        assert math.isclose(abs(ev), 1.0, abs_tol=0.1)

def test_frequency_estimation_accuracy(signal_setup):
    """¿La función core devuelve la frecuencia exacta esperada?"""
    f_calc = _esprit_core(signal_setup["v"], signal_setup["dt"])
    assert math.isclose(f_calc, signal_setup["f"], abs_tol=0.01)

def test_estimator_buffer_rolling():
    """¿El buffer de la clase se desplaza correctamente muestra a muestra?"""
    est = ESPRIT_Estimator(nominal_f=60.0, n_cycles=1.0)
    est.step(1.0)
    assert est.buffer[-1] == 1.0
    est.step(2.0)
    assert est.buffer[-2] == 1.0
    assert est.buffer[-1] == 2.0

def test_sanity_filter_logic():
    """¿El estimador ignora valores fuera de rango (40-80 Hz)?"""
    est = ESPRIT_Estimator(nominal_f=60.0)
    # Forzamos una salida previa
    est.f_out = 60.0
    # Simulamos un buffer que daría una frecuencia absurda (ej. 0 o 5000)
    # Pero el step debería mantener el 60.0 por el filtro de cordura
    absurd_val = est.step(0.0) 
    assert 40.0 <= absurd_val <= 80.0