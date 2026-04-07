from __future__ import annotations

import math
import numpy as np
from numba import njit

from .base import BaseFrequencyEstimator
from .common import DT_DSP


@njit(cache=True)
def _wrap_pi(x: float) -> float:
    while x > math.pi:
        x -= 2.0 * math.pi
    while x < -math.pi:
        x += 2.0 * math.pi
    return x


@njit(cache=True)
def _epll_vectorized_core(
    v_array: np.ndarray,
    dt: float,
    w_nom: float,
    kp: float,
    ki: float,
    ka: float,
    f_min: float,
    f_max: float,
    amp_hat: float,
    w_hat: float,
    theta_hat: float,
) -> tuple[np.ndarray, float, float, float]:
    """
    Núcleo vectorizado del Enhanced PLL (EPLL) clásico de Karimi-Ghartemani.
    """
    n = len(v_array)
    f_est = np.empty(n, dtype=np.float64)
    two_pi = 2.0 * math.pi

    for i in range(n):
        u = v_array[i]

        # 1. Generación de la señal interna estimada
        y = amp_hat * math.sin(theta_hat)
        
        # 2. Cálculo del error
        e = u - y

        # 3. Lazo de Amplitud (estimación de amplitud pico)
        amp_hat += dt * ka * e * math.sin(theta_hat)
        # Prevenir amplitudes negativas por transitorios extremos
        if amp_hat < 1e-3:
            amp_hat = 1e-3

        # 4. Lazos de Fase y Frecuencia
        # El término pd_err actúa como el "Phase Detector" equivalente
        pd_err = e * math.cos(theta_hat)

        # Integrador de frecuencia
        w_pre = w_hat + dt * ki * pd_err
        f_pre = w_pre / two_pi

        # Anti-windup básico
        if f_min <= f_pre <= f_max:
            w_hat = w_pre
        elif f_pre < f_min:
            w_hat = f_min * two_pi
        elif f_pre > f_max:
            w_hat = f_max * two_pi

        # Actualización de fase
        theta_hat = _wrap_pi(theta_hat + dt * w_hat + dt * kp * pd_err)

        f_est[i] = w_hat / two_pi

    return f_est, amp_hat, w_hat, theta_hat


class EPLL_Estimator(BaseFrequencyEstimator):
    """
    Enhanced Phase-Locked Loop (EPLL).
    
    Estima simultáneamente amplitud, fase y frecuencia.
    Destaca por no tener rizado de 2*w en estado estacionario y por su
    robustez ante huecos de tensión (voltage sags).
    """

    name = "EPLL"

    def __init__(
        self,
        nominal_f: float = 50.0,
        kp: float = 120.0,      # Ganancia proporcional (Fase)
        ki: float = 4000.0,     # Ganancia integral (Frecuencia)
        ka: float = 100.0,      # Ganancia de Amplitud
        dt: float = DT_DSP,
    ) -> None:
        self.nominal_f = float(nominal_f)
        self.kp = float(kp)
        self.ki = float(ki)
        self.ka = float(ka)
        self.dt = float(dt)

        self.w_nom = 2.0 * math.pi * self.nominal_f
        self.f_min = self.nominal_f - 10.0
        self.f_max = self.nominal_f + 10.0

        self.reset()

    def reset(self) -> None:
        self.amp_hat = 1.0  # Asumimos 1.0 p.u. inicial
        self.w_hat = self.w_nom
        self.theta_hat = 0.0

    @classmethod
    def default_params(cls) -> dict[str, float]:
        return {
            "nominal_f": 50.0,
            "kp": 120.0,
            "ki": 4000.0,
            "ka": 100.0,
        }

    @staticmethod
    def describe_params(params: dict[str, float]) -> str:
        return (
            f"f_nom={params.get('nominal_f', 50.0)}Hz, "
            f"kp={params.get('kp', 120.0)}, "
            f"ki={params.get('ki', 4000.0)}, "
            f"ka={params.get('ka', 100.0)}"
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
            self.amp_hat,
            self.w_hat,
            self.theta_hat,
        ) = _epll_vectorized_core(
            v_array=v_array,
            dt=self.dt,
            w_nom=self.w_nom,
            kp=self.kp,
            ki=self.ki,
            ka=self.ka,
            f_min=self.f_min,
            f_max=self.f_max,
            amp_hat=self.amp_hat,
            w_hat=self.w_hat,
            theta_hat=self.theta_hat,
        )

        return f_est

    def estimate(self, t: np.ndarray, v: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)

        if len(t) == 0:
            return np.empty(0, dtype=np.float64)
        if len(t) == 1:
            return np.full(1, self.nominal_f, dtype=np.float64)

        dt = float(t[1] - t[0])
        self.dt = dt
        self.w_nom = 2.0 * math.pi * self.nominal_f

        self.reset()
        return self.step_vectorized(v)