from __future__ import annotations

import math
import numpy as np
from numba import njit

from .base import BaseFrequencyEstimator
from .common import DT_DSP

@njit(cache=True)
def _prony_core(buffer: np.ndarray, dt: float, p: int) -> float:
    """
    Núcleo del Método de Prony.
    Utiliza predicción lineal (AR) y autovalores de la matriz compañera.
    """
    N = len(buffer)
    # T-102: deterministic dither replaces np.random.standard_normal (which used
    # Numba's unseeded per-thread RNG, making MC runs non-reproducible).
    # Linear ramp gives numerical well-posedness without statistical properties.
    buf_work = buffer + 1e-9 * (np.arange(N, dtype=np.float64) / N - 0.5)
    
    # 1. Ecuaciones de Predicción Lineal (LP): X * a = -x
    L = N - p
    X = np.zeros((L, p), dtype=np.float64)
    x_vec = np.zeros(L, dtype=np.float64)
    
    for i in range(L):
        for j in range(p):
            X[i, j] = buf_work[i + j]
        x_vec[i] = -buf_work[i + p]
        
    # 2. Resolución de coeficientes AR mediante Pseudoinversa
    a = np.linalg.pinv(X) @ x_vec
    
    # 3. Construcción de la Matriz Compañera
    # Las raíces del polinomio z^p + a_{p-1}z^{p-1} + ... + a_0 = 0
    # son los autovalores de C.
    C = np.zeros((p, p), dtype=np.float64)
    for i in range(p - 1):
        C[i, i+1] = 1.0
    for i in range(p):
        C[p-1, i] = -a[i]
        
    # 4. Extracción de polos (Raíces complejas)
    C_complex = C.astype(np.complex128)
    roots = np.linalg.eigvals(C_complex)
    
    # 5. Selección del polo de interés
    # FIX: Inicializamos en NaN para exponer fallos silenciosos
    best_f = np.nan 
    min_dist = 1e6
    two_pi_dt = 2.0 * math.pi * dt
    
    for r in roots:
        mag = abs(r)
        w = abs(math.atan2(r.imag, r.real))
        f_hz = w / two_pi_dt
        
        # Distancia al círculo unitario (1.0 = oscilación pura sin decaimiento)
        dist = abs(mag - 1.0)
        
        # Filtro de banda: entre 40 y 80 Hz
        if 40.0 < f_hz < 80.0:
            if dist < min_dist:
                min_dist = dist
                best_f = f_hz
                
    return best_f

class Prony_Estimator(BaseFrequencyEstimator):
    """
    Estimador de Frecuencia de Prony.
    Alta velocidad y precisión, pero extremadamente susceptible al ruido blanco.
    """
    name = "Prony"

    def __init__(self, nominal_f: float = 60.0, n_cycles: float = 1.0, order: int = 4, dt: float = DT_DSP) -> None:
        self.nominal_f = float(nominal_f)
        self.dt = float(dt)
        self.p = int(order) # Orden del modelo AR (4 asimila la red y componentes DC)
        
        # Ventana de datos (1 ciclo por defecto)
        self.N = int(round((1.0 / self.nominal_f) / self.dt * n_cycles))
        if self.N <= self.p * 2:
            self.N = self.p * 2 + 1 # Seguridad matemática mínima
            
        self.reset()

    def reset(self) -> None:
        self.buffer = np.zeros(self.N, dtype=np.float64)
        # FIX: Evitar "aprobar" el test regalando la frecuencia nominal
        self.f_out = np.nan

        # Métricas de diagnóstico
        self._valid_updates = 0
        self._total_calls = 0
        # T-101: shared decimation counter for step() / step_vectorized() parity
        self._step_counter = 0

    @classmethod
    def default_params(cls) -> dict[str, float]:
        return {"nominal_f": 60.0}

    @staticmethod
    def describe_params(params: dict[str, float]) -> str:
        return f"Prony f_nom={params.get('nominal_f', 60.0)}Hz"

    def structural_latency_samples(self) -> int:
        return self.N // 2

    def step(self, z: float) -> float:
        # T-101: mirrors step_vectorized() decimation via shared _step_counter
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = z

        if self._step_counter % 10 == 0 and np.abs(z) > 1e-4:
            self._total_calls += 1
            try:
                val = _prony_core(self.buffer, self.dt, self.p)
                if not np.isnan(val) and val > 0.0:
                    self.f_out = val
                    self._valid_updates += 1
            except:
                pass
        self._step_counter += 1
        return self.f_out

    def step_vectorized(self, v_array: np.ndarray) -> np.ndarray:
        # T-101: uses self._step_counter for decimation (same as step())
        n = len(v_array)
        f_est = np.empty(n, dtype=np.float64)

        for i in range(n):
            z = v_array[i]
            self.buffer[:-1] = self.buffer[1:]
            self.buffer[-1] = z

            if self._step_counter % 10 == 0 and np.abs(z) > 1e-4:
                self._total_calls += 1
                try:
                    val = _prony_core(self.buffer, self.dt, self.p)
                    if not np.isnan(val) and val > 0.0:
                        self.f_out = val
                        self._valid_updates += 1
                except:
                    pass
            self._step_counter += 1
            f_est[i] = self.f_out

        return f_est

    def estimate(self, t: np.ndarray, v: np.ndarray) -> np.ndarray:
        self.dt = float(t[1] - t[0])
        self.reset()
        return self.step_vectorized(v)