from __future__ import annotations

import math
import numpy as np
from numba import njit

from .base import BaseFrequencyEstimator
from .common import DT_DSP
@njit(cache=True)
def _type3_sogi_pll_vectorized_core(
    v_array: np.ndarray,
    dt: float,
    k_sogi: float,
    kp: float,
    ki: float,
    ki2: float,
    w_nom: float,
    f_min: float,
    f_max: float,
    v_alpha: float,
    v_beta: float,
    theta_hat: float,
    w_hat: float,
    err_int1: float,
    err_int2: float,
) -> tuple[np.ndarray, float, float, float, float, float, float]:
    """
    Núcleo del Type-3 SOGI-PLL.
    Utiliza un filtro de lazo PI-I (Proporcional + Doble Integral) para 
    lograr seguimiento de rampas de frecuencia (RoCoF) con error cero.
    """
    n = len(v_array)
    f_est = np.empty(n, dtype=np.float64)
    two_pi = 2.0 * math.pi
    w_min = f_min * two_pi
    w_max = f_max * two_pi

    for i in range(n):
        v = v_array[i]

        # 1. OSG (Orthogonal Signal Generator) - SOGI frontend
        err_sogi = v - v_alpha
        v_alpha += (w_hat * v_beta + k_sogi * w_hat * err_sogi) * dt
        v_beta += (-w_hat * v_alpha) * dt

        # 2. Park Transform (Detector de Fase)
        # CORRECCIÓN: Para obtener sin(theta - theta_hat) de forma estable
        # necesitamos esta combinación cruzada específica.
        v_q = v_alpha * math.cos(theta_hat) - v_beta * math.sin(theta_hat)

        # Normalización por amplitud para mantener ganancias constantes
        amp_sq = v_alpha * v_alpha + v_beta * v_beta
        amp = math.sqrt(amp_sq) if amp_sq > 1e-4 else 1e-2
        e_pd = v_q / amp

        # 3. Type-3 Loop Filter (PI-I)
        err_int1 += e_pd * dt
        err_int2 += err_int1 * dt

        dw = kp * e_pd + ki * err_int1 + ki2 * err_int2
        w_hat = w_nom + dw

        # Anti-windup básico
        if w_hat < w_min:
            w_hat = w_min
        elif w_hat > w_max:
            w_hat = w_max

        # 4. Integración de Fase (VCO)
        theta_hat += w_hat * dt
        
        while theta_hat > math.pi:
            theta_hat -= two_pi
        while theta_hat < -math.pi:
            theta_hat += two_pi

        f_est[i] = w_hat / two_pi

    return f_est, v_alpha, v_beta, theta_hat, w_hat, err_int1, err_int2

class Type3_SOGI_PLL_Estimator(BaseFrequencyEstimator):
    """
    Type-3 SOGI-PLL para redes de baja inercia.
    
    Resuelve la debilidad fundamental de los PLLs Tipo 2 (SOGI-PLL estándar), 
    los cuales presentan un error de seguimiento constante durante eventos 
    de alto RoCoF. El controlador PI-I garantiza error cero ante rampas.
    """

    name = "Type-3 SOGI-PLL"

    def __init__(
        self,
        nominal_f: float = 60.0,
        k_sogi: float = 1.414,
        # Sintonización para Type-3 (polos asignados a ~30 rad/s)
        kp: float = 90.0,
        ki: float = 2700.0,
        ki2: float = 27000.0,
        dt: float = DT_DSP,
    ) -> None:
        self.nominal_f = float(nominal_f)
        self.k_sogi = float(k_sogi)
        self.kp = float(kp)
        self.ki = float(ki)
        self.ki2 = float(ki2)
        self.dt = float(dt)

        self.w_nom = 2.0 * math.pi * self.nominal_f
        self.f_min = self.nominal_f - 20.0
        self.f_max = self.nominal_f + 20.0

        self.reset()

    def reset(self) -> None:
        self.v_alpha = 0.0
        self.v_beta = 0.0
        self.theta_hat = 0.0
        self.w_hat = self.w_nom
        self.err_int1 = 0.0
        self.err_int2 = 0.0

    @classmethod
    def default_params(cls) -> dict[str, float]:
        return {
            "nominal_f": 60.0,
            "k_sogi": 1.414,
            "kp": 90.0,
            "ki": 2700.0,
            "ki2": 27000.0,
        }

    @staticmethod
    def describe_params(params: dict[str, float]) -> str:
        return (
            f"f_nom={params.get('nominal_f', 60.0)}Hz, "
            f"kp={params.get('kp', 90.0)}, "
            f"ki={params.get('ki', 2700.0)}, "
            f"ki2={params.get('ki2', 27000.0)}"
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
            self.theta_hat,
            self.w_hat,
            self.err_int1,
            self.err_int2,
        ) = _type3_sogi_pll_vectorized_core(
            v_array=v_array,
            dt=self.dt,
            k_sogi=self.k_sogi,
            kp=self.kp,
            ki=self.ki,
            ki2=self.ki2,
            w_nom=self.w_nom,
            f_min=self.f_min,
            f_max=self.f_max,
            v_alpha=self.v_alpha,
            v_beta=self.v_beta,
            theta_hat=self.theta_hat,
            w_hat=self.w_hat,
            err_int1=self.err_int1,
            err_int2=self.err_int2,
        )
        return f_est

    def estimate(self, t: np.ndarray, v: np.ndarray) -> np.ndarray:
        dt = float(t[1] - t[0])
        self.dt = dt
        self.reset()
        return self.step_vectorized(v)