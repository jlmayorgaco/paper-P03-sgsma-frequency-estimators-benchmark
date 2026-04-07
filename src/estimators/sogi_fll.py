from __future__ import annotations

import math
import numpy as np
from numba import njit

from .base import BaseFrequencyEstimator
from .common import DT_DSP


@njit(cache=True)
def _sogi_fll_vectorized_core(
    v_array: np.ndarray,
    dt: float,
    k_sogi: float,
    gamma: float,
    f_min: float,
    f_max: float,
    v_alpha: float,
    v_beta: float,
    w_hat: float,
) -> tuple[np.ndarray, float, float, float]:
    """
    Núcleo del SOGI-FLL (Frequency-Locked Loop).
    Usa Euler semi-implícito para garantizar estabilidad en el oscilador ortogonal.
    """
    n = len(v_array)
    f_est = np.empty(n, dtype=np.float64)
    two_pi = 2.0 * math.pi

    for i in range(n):
        v = v_array[i]
        
        # 1. Error del SOGI
        err = v - v_alpha
        
        # 2. Generador de Señal Ortogonal (OSG) - Euler Semi-implícito
        dv_alpha = w_hat * v_beta + k_sogi * w_hat * err
        v_alpha += dv_alpha * dt
        
        # Usamos el v_alpha actualizado para calcular v_beta (semi-implícito)
        dv_beta = -w_hat * v_alpha
        v_beta += dv_beta * dt
        
        # 3. Lazo FLL (Frequency-Locked Loop)
        # En lugar de un detector de fase y un PI, el FLL minimiza el error de frecuencia directo.
        e_f = err * v_beta
        
        # Actualización de frecuencia (el signo POSITIVO estabiliza el descenso de gradiente 
        # porque v_beta adelanta a v_alpha por 90 grados en nuestra formulación)
        dw = gamma * w_hat * e_f
        w_hat += dw * dt
        
        # Anti-windup
        w_min = f_min * two_pi
        w_max = f_max * two_pi
        if w_hat < w_min:
            w_hat = w_min
        elif w_hat > w_max:
            w_hat = w_max
            
        f_est[i] = w_hat / two_pi

    return f_est, v_alpha, v_beta, w_hat


class SOGI_FLL_Estimator(BaseFrequencyEstimator):
    """
    Second Order Generalized Integrator con Frequency-Locked Loop (SOGI-FLL).
    
    A diferencia de un PLL que rastrea la fase, el FLL rastrea directamente 
    la frecuencia. Su mayor ventaja competitiva en redes de baja inercia es
    su rechazo casi total a los saltos de fase (Phase Jumps).
    """

    name = "SOGI-FLL"

    def __init__(
        self,
        nominal_f: float = 50.0,
        k_sogi: float = 1.414,  # Ganancia típica de amortiguamiento SOGI (sqrt(2))
        gamma: float = 50.0,    # Ganancia adaptativa del FLL
        dt: float = DT_DSP,
    ) -> None:
        self.nominal_f = float(nominal_f)
        self.k_sogi = float(k_sogi)
        self.gamma = float(gamma)
        self.dt = float(dt)

        self.w_nom = 2.0 * math.pi * self.nominal_f
        self.f_min = self.nominal_f - 10.0
        self.f_max = self.nominal_f + 10.0

        self.reset()

    def reset(self) -> None:
        self.v_alpha = 0.0
        self.v_beta = 0.0
        self.w_hat = self.w_nom

    @classmethod
    def default_params(cls) -> dict[str, float]:
        return {
            "nominal_f": 50.0,
            "k_sogi": 1.414,
            "gamma": 50.0,
        }

    @staticmethod
    def describe_params(params: dict[str, float]) -> str:
        return (
            f"f_nom={params.get('nominal_f', 50.0)}Hz, "
            f"k={params.get('k_sogi', 1.414)}, "
            f"gamma={params.get('gamma', 50.0)}"
        )

    def structural_latency_samples(self) -> int:
        return 0

    def step(self, z: float) -> float:
        v_array = np.array([z], dtype=np.float64)
        return float(self.step_vectorized(v_array)[0])

    def step_vectorized(self, v_array: np.ndarray) -> np.ndarray:
        v_array = np.asarray(v_array, dtype=np.float64)
        if len(v_array) == 0:
            return np.empty(0, dtype=np.float64)

        (
            f_est,
            self.v_alpha,
            self.v_beta,
            self.w_hat,
        ) = _sogi_fll_vectorized_core(
            v_array=v_array,
            dt=self.dt,
            k_sogi=self.k_sogi,
            gamma=self.gamma,
            f_min=self.f_min,
            f_max=self.f_max,
            v_alpha=self.v_alpha,
            v_beta=self.v_beta,
            w_hat=self.w_hat,
        )
        return f_est

    def estimate(self, t: np.ndarray, v: np.ndarray) -> np.ndarray:
        dt = float(t[1] - t[0])
        self.dt = dt
        self.reset()
        return self.step_vectorized(v)