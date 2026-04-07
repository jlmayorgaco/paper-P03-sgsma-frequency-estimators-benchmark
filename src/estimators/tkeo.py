from __future__ import annotations

import math
import numpy as np
from numba import njit

from .base import BaseFrequencyEstimator
from .common import DT_DSP

@njit(cache=True)
def _psi(x0: float, xm1: float, xp1: float) -> float:
    """Operador de Energía de Teager-Kaiser: Psi(x) = x(n)^2 - x(n-1)x(n+1)"""
    return x0**2 - xm1 * xp1

@njit(cache=True)
def _tkeo_vectorized_core(
    v_array: np.ndarray,
    dt: float,
    f_out: float,
    smooth_alpha: float,
    buffer: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Núcleo del estimador TKEO.
    """
    n = len(v_array)
    f_est = np.empty(n, dtype=np.float64)
    two_pi = 2.0 * math.pi

    for i in range(n):
        # Desplazar buffer circular
        buffer[0] = buffer[1]
        buffer[1] = buffer[2]
        buffer[2] = v_array[i]

        x_n = buffer[2]
        x_nm1 = buffer[1]
        x_nm2 = buffer[0]

        # Necesitamos 3 muestras para Psi(x). 
        # Nota: Para una implementación DES completa se requieren 4, 
        # pero esta aproximación es la estándar de latencia mínima.
        psi_x = _psi(x_nm1, x_nm2, x_n)
        
        if abs(psi_x) > 1e-9:
            # Algoritmo DES-1 simplificado:
            # f = (1/2pi*dt) * arccos(1 - (Psi(x[n]-x[n-1])) / (2*Psi(x[n])))
            # Aproximación de diferencia finita para el numerador:
            diff_psi = (x_n - x_nm1)**2
            arg = 1.0 - (diff_psi / (2.0 * psi_x + 1e-12))
            
            if arg > 1.0: arg = 1.0
            elif arg < -1.0: arg = -1.0
            
            f_raw = math.acos(arg) / (two_pi * dt)
        else:
            f_raw = f_out

        # Filtrado de salida (crucial por la amplificación de ruido del TKEO)
        f_out = (1.0 - smooth_alpha) * f_out + smooth_alpha * f_raw
        f_est[i] = f_out

    return f_est, f_out, buffer

class TKEO_Estimator(BaseFrequencyEstimator):
    """
    Teager-Kaiser Energy Operator (TKEO).
    Estimador de latencia sub-ciclo (P3) y costo O(1).
    """
    name = "TKEO"

    def __init__(
        self,
        nominal_f: float = 60.0,
        output_smoothing: float = 0.01,
        dt: float = DT_DSP,
    ) -> None:
        self.nominal_f = float(nominal_f)
        self.output_smoothing = float(output_smoothing)
        self.dt = float(dt)
        self.reset()

    def reset(self) -> None:
        self.buffer = np.zeros(3, dtype=np.float64)
        self.f_out = self.nominal_f

    @classmethod
    def default_params(cls) -> dict[str, float]:
        return {"nominal_f": 60.0, "output_smoothing": 0.01}

    def structural_latency_samples(self) -> int:
        return 1

    # --- MÉTODO FALTANTE AÑADIDO AQUÍ ---
    def step(self, z: float) -> float:
        """Procesa una muestra individual."""
        v_array = np.array([z], dtype=np.float64)
        return float(self.step_vectorized(v_array)[0])

    def step_vectorized(self, v_array: np.ndarray) -> np.ndarray:
        v_array = np.asarray(v_array, dtype=np.float64)
        if len(v_array) == 0:
            return np.empty(0, dtype=np.float64)

        f_est, self.f_out, self.buffer = _tkeo_vectorized_core(
            v_array=v_array,
            dt=self.dt,
            f_out=self.f_out,
            smooth_alpha=self.output_smoothing,
            buffer=self.buffer
        )
        return f_est

    def estimate(self, t: np.ndarray, v: np.ndarray) -> np.ndarray:
        self.dt = float(t[1] - t[0])
        self.reset()
        return self.step_vectorized(v)