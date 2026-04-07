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
    # Dither para evitar singularidad exacta en señales puras
    buf_work = buffer + 1e-9 * np.random.standard_normal(N)
    
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
    # Buscamos la raíz más cercana al círculo unitario (menor amortiguamiento)
    # que tenga una frecuencia física lógica.
    best_f = 0.0
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
        self.f_out = self.nominal_f

    def structural_latency_samples(self) -> int:
        return self.N // 2

    def step(self, z: float) -> float:
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = z
        
        if np.abs(z) > 1e-4:
            try:
                val = _prony_core(self.buffer, self.dt, self.p)
                if val > 0.0: # Si encontró una raíz válida
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
            
            # Decimación: procesar cada 10 muestras para ahorrar CPU (1ms latencia)
            if i % 10 == 0 and np.abs(z) > 1e-4:
                try:
                    val = _prony_core(self.buffer, self.dt, self.p)
                    if val > 0.0:
                        self.f_out = val
                except:
                    pass
            f_est[i] = self.f_out
            
        return f_est

    def estimate(self, t: np.ndarray, v: np.ndarray) -> np.ndarray:
        self.reset()
        return self.step_vectorized(v)