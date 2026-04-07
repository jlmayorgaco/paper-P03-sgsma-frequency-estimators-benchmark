from __future__ import annotations

import math
import numpy as np
from numba import njit

from .base import BaseFrequencyEstimator
from .common import DT_DSP

@njit(cache=True)
def _music_eval_spectrum(Un: np.ndarray, f_center: float, span: float, step: float, dt: float) -> tuple[float, float]:
    """
    Evalúa el pseudo-espectro de MUSIC en una región localizada.
    Busca minimizar el denominador: a(f)^H * Un * Un^H * a(f)
    """
    best_f = f_center
    min_val = 1e12
    two_pi_dt = 2.0 * math.pi * dt
    L = Un.shape[0]
    K = Un.shape[1] # Dimensión del subespacio de ruido

    f_start = f_center - span
    f_end = f_center + span + (step * 0.5)

    for f in np.arange(f_start, f_end, step):
        omega = f * two_pi_dt
        val = 0.0
        
        # Proyección del steering vector a(f) sobre cada vector del subespacio de ruido
        for col in range(K):
            dot_real = 0.0
            dot_imag = 0.0
            for row in range(L):
                # a(f) = [1, e^(jw), e^(j2w)...]^T
                # a(f)^H = [1, e^(-jw), e^(-j2w)...]
                c = math.cos(omega * row)
                s = -math.sin(omega * row)
                
                u_r = Un[row, col].real
                u_i = Un[row, col].imag
                
                # Multiplicación compleja
                dot_real += c * u_r - s * u_i
                dot_imag += c * u_i + s * u_r
                
            val += dot_real**2 + dot_imag**2

        if val < min_val:
            min_val = val
            best_f = f
            
    return best_f, min_val

@njit(cache=True)
def _music_core(buffer: np.ndarray, dt: float, f_nom: float) -> float:
    """Núcleo del estimador Spectral MUSIC."""
    N = len(buffer)
    # Dither para prevenir matrices singulares
    buf_work = buffer + 1e-9 * np.random.standard_normal(N)
    
    # 1. Matriz de Hankel
    L = N // 2
    M = N - L + 1
    H = np.zeros((L, M), dtype=np.float64)
    for i in range(L):
        for j in range(M):
            H[i, j] = buf_work[i + j]

    # 2. SVD para separar subespacios
    U, S, Vh = np.linalg.svd(H)
    
    # Subespacio de Ruido (Un): Descartamos las 2 componentes principales (señal)
    Un = U[:, 2:].astype(np.complex128)

    # 3. Búsqueda Espectral en 3 Etapas (Coarse -> Fine -> Ultra-fine)
    # Etapa 1: Búsqueda gruesa (40 a 80 Hz, pasos de 1 Hz)
    f1, _ = _music_eval_spectrum(Un, 60.0, 20.0, 1.0, dt)
    
    # Etapa 2: Búsqueda fina (+/- 1 Hz, pasos de 0.05 Hz)
    f2, _ = _music_eval_spectrum(Un, f1, 1.0, 0.05, dt)
    
    # Etapa 3: Búsqueda ultra-fina (+/- 0.05 Hz, pasos de 0.001 Hz)
    f_final, _ = _music_eval_spectrum(Un, f2, 0.05, 0.001, dt)

    return f_final

class MUSIC_Estimator(BaseFrequencyEstimator):
    """
    Estimador M.U.S.I.C. (Multiple Signal Classification).
    Super-resolución basada en la ortogonalidad del Subespacio de Ruido.
    """
    name = "MUSIC"

    def __init__(self, nominal_f: float = 60.0, n_cycles: float = 1.0, dt: float = DT_DSP) -> None:
        self.nominal_f = float(nominal_f)
        self.dt = float(dt)
        self.N = int(round((1.0 / self.nominal_f) / self.dt * n_cycles))
        self.reset()

    def reset(self) -> None:
        self.buffer = np.zeros(self.N, dtype=np.float64)
        self.f_out = self.nominal_f

    def structural_latency_samples(self) -> int:
        return self.N // 2

    def step(self, z: float) -> float:
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = z
        
        if np.abs(z) > 1e-4:
            try:
                val = _music_core(self.buffer, self.dt, self.nominal_f)
                if 40.0 < val < 80.0:
                    self.f_out = val
            except:
                pass
        return self.f_out

    def step_vectorized(self, v_array: np.ndarray) -> np.ndarray:
        n = len(v_array)
        f_est = np.empty(n, dtype=np.float64)
        
        for i in range(n):
            z = v_array[i]
            self.buffer[:-1] = self.buffer[1:]
            self.buffer[-1] = z
            
            # Ejecutar evaluación pesada cada 10 muestras (Decimación = 1ms)
            if i % 10 == 0 and np.abs(z) > 1e-4:
                try:
                    val = _music_core(self.buffer, self.dt, self.nominal_f)
                    if 40.0 < val < 80.0:
                        self.f_out = val
                except:
                    pass
            f_est[i] = self.f_out
            
        return f_est

    def estimate(self, t: np.ndarray, v: np.ndarray) -> np.ndarray:
        self.reset()
        return self.step_vectorized(v)