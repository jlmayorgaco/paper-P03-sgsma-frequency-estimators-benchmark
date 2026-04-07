from __future__ import annotations

import math
import numpy as np
from numba import njit

from .base import BaseFrequencyEstimator
from .common import DT_DSP

@njit(cache=True)
def _esprit_core(buffer: np.ndarray, dt: float) -> float:
    """
    Algoritmo ESPRIT de Super-Resolución (Subspace-based).
    Optimizado para Numba con manejo explícito de dominios complejos.
    """
    N = len(buffer)
    # Dither minúsculo para evitar singularidad en señales puras
    buf_work = buffer + 1e-9 * np.random.standard_normal(N)
    
    # 1. Construcción de Matriz de Hankel
    L = N // 2
    M = N - L + 1
    H = np.zeros((L, M), dtype=np.float64)
    for i in range(L):
        for j in range(M):
            H[i, j] = buf_work[i + j]

    # 2. Descomposición en Valores Singulares (SVD)
    # Extraemos las 2 componentes principales (frecuencia positiva y negativa)
    U, S, Vh = np.linalg.svd(H)
    Us = U[:, :2]

    # 3. Invarianza Rotacional (LS-ESPRIT)
    Us1 = Us[:-1, :]
    Us2 = Us[1:, :]
    
    # Resolver Phi: Us1 * Phi = Us2 mediante pseudoinversa
    phi_mat = np.linalg.pinv(Us1) @ Us2
    
    # --- CORRECCIÓN DE DOMINIO PARA NUMBA ---
    # Forzamos conversión a complejo antes de eigvals para evitar ValueError
    phi_complex = phi_mat.astype(np.complex128)
    eigvals = np.linalg.eigvals(phi_complex)
    # ----------------------------------------
    
    # 4. Extracción de la fase del autovalor (Frecuencia)
    max_w = 0.0
    for ev in eigvals:
        # La frecuencia angular w corresponde al ángulo del autovalor
        w = abs(math.atan2(ev.imag, ev.real))
        if w > max_w:
            max_w = w
            
    return max_w / (2.0 * math.pi * dt)

class ESPRIT_Estimator(BaseFrequencyEstimator):
    """
    Estimador ESPRIT. 
    Proporciona super-resolución espectral superando el límite de la DFT.
    """
    name = "ESPRIT"

    def __init__(self, nominal_f: float = 60.0, n_cycles: float = 1.0, dt: float = DT_DSP) -> None:
        self.nominal_f = float(nominal_f)
        self.dt = float(dt)
        # Una ventana de 1 ciclo es suficiente para la precisión de ESPRIT
        self.N = int(round((1.0 / self.nominal_f) / self.dt * n_cycles))
        self.reset()

    def reset(self) -> None:
        self.buffer = np.zeros(self.N, dtype=np.float64)
        self.f_out = self.nominal_f

    def structural_latency_samples(self) -> int:
        return self.N // 2

    def step(self, z: float) -> float:
        """Procesa una muestra individual actualizando el buffer."""
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = z
        
        # Evitar cálculos en silencio absoluto para prevenir singularidades
        if np.abs(z) > 1e-4:
            try:
                val = _esprit_core(self.buffer, self.dt)
                # Filtro de cordura para evitar picos por inestabilidad numérica
                if 40.0 < val < 80.0:
                    self.f_out = val
            except:
                pass
        return self.f_out

    def step_vectorized(self, v_array: np.ndarray) -> np.ndarray:
        """
        Versión optimizada: actualiza el buffer siempre, 
        pero ejecuta el SVD pesado cada 10 muestras.
        """
        n = len(v_array)
        f_est = np.empty(n, dtype=np.float64)
        
        for i in range(n):
            z = v_array[i]
            # 1. Deslizar buffer SIEMPRE para mantener la coherencia de la señal
            self.buffer[:-1] = self.buffer[1:]
            self.buffer[-1] = z
            
            # 2. Decimación del cálculo (ejecutar SVD cada 10 pasos / 1ms a 10kHz)
            if i % 10 == 0 and np.abs(z) > 1e-4:
                try:
                    val = _esprit_core(self.buffer, self.dt)
                    if 40.0 < val < 80.0:
                        self.f_out = val
                except:
                    # Si falla (NaN o error de dominio), mantenemos la f_out previa
                    pass
            
            f_est[i] = self.f_out
            
        return f_est

    def estimate(self, t: np.ndarray, v: np.ndarray) -> np.ndarray:
        self.reset()
        return self.step_vectorized(v)