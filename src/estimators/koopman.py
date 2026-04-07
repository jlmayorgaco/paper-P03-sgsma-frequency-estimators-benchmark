from __future__ import annotations

import math
import numpy as np
from numba import njit

from .base import BaseFrequencyEstimator
from .common import DT_DSP

@njit(cache=True)
def _koopman_edmd_core(buffer: np.ndarray, dt: float) -> float:
    """
    Aproximación de Koopman mediante Hankel-DMD (Robust Koopman).
    Utiliza SVD para aislar el subespacio de señal antes de derivar la dinámica.
    """
    N = len(buffer)
    buf_work = buffer + 1e-9 * np.random.standard_normal(N)
    
    # 1. Matriz de Observables Profunda (Hankel)
    L = N // 2
    M = N - L + 1
    H = np.zeros((L, M), dtype=np.float64)
    for i in range(L):
        for j in range(M):
            H[i, j] = buf_work[i + j]

    # 2. Matrices desplazadas en el tiempo (Estado Actual X, Estado Futuro Y)
    X = H[:, :-1]
    Y = H[:, 1:]

    # 3. Truncamiento Robusto vía SVD (Aislamiento de la señal pura)
    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    
    # Añadimos .copy() para forzar matrices contiguas y eliminar el warning de Numba
    Ur_T = np.ascontiguousarray(U[:, :2].T)
    Sr_inv = np.diag(1.0 / S[:2])
    Vr = np.ascontiguousarray(Vh[:2, :].T)
    Y_contig = np.ascontiguousarray(Y)
    
    # 4. Operador de Koopman Reducido (K_tilde)
    K_tilde = Ur_T @ Y_contig @ Vr @ Sr_inv

    # 5. Autovalores del Operador Dinámico
    K_complex = K_tilde.astype(np.complex128)
    eigvals = np.linalg.eigvals(K_complex)
    
    # 6. Extracción de frecuencia de los modos de Koopman
    best_f = 0.0
    max_energy = -1.0
    two_pi_dt = 2.0 * math.pi * dt
    
    for ev in eigvals:
        w = abs(math.atan2(ev.imag, ev.real))
        f_hz = w / two_pi_dt
        
        if 40.0 < f_hz < 80.0:
            # Seleccionamos el polo más cercano al círculo unitario
            mag = abs(ev)
            energy_score = 1.0 / (abs(mag - 1.0) + 1e-4)
            
            if energy_score > max_energy:
                max_energy = energy_score
                best_f = f_hz
                
    return best_f
class Koopman_Estimator(BaseFrequencyEstimator):
    """
    Estimador RK-DPMU (Robust Koopman).
    Aproxima la dinámica no lineal mediante Extended Dynamic Mode Decomposition.
    Especializado en transitorios complejos de baja inercia.
    """
    name = "Koopman (RK-DPMU)"

    def __init__(self, nominal_f: float = 60.0, n_cycles: float = 1.5, dt: float = DT_DSP) -> None:
        self.nominal_f = float(nominal_f)
        self.dt = float(dt)
        # Koopman requiere un poco más de datos que ESPRIT para construir 
        # el mapeo Y = K*X robustamente. 1.5 ciclos es el punto dulce.
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
        
        if np.abs(z) > 1e-3:
            try:
                val = _koopman_edmd_core(self.buffer, self.dt)
                if val > 0.0:
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
            
            # Decimación del cálculo pesado (cada 10 muestras = 1ms)
            if i % 10 == 0 and np.abs(z) > 1e-3:
                try:
                    val = _koopman_edmd_core(self.buffer, self.dt)
                    if val > 0.0:
                        self.f_out = val
                except:
                    pass
            f_est[i] = self.f_out
            
        return f_est

    def estimate(self, t: np.ndarray, v: np.ndarray) -> np.ndarray:
        self.reset()
        return self.step_vectorized(v)