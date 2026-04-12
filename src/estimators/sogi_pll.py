from __future__ import annotations

import math
import numpy as np
from numba import njit

from .base import BaseFrequencyEstimator
from .common import DT_DSP


# =====================================================================
# Numba JIT-compiled core logic
# =====================================================================
@njit(cache=True)
def _sogi_pll_vectorized_core(
    v_array: np.ndarray,
    dt: float,
    w_nom: float,
    k_sogi: float,
    kp: float,
    ki: float,
    w_min: float,
    w_max: float,
    smooth_alpha: float,
    theta: float,
    pll_int: float,
    v_alpha: float,
    x_beta: float,
    w_est: float,
    f_out: float,
) -> tuple[np.ndarray, float, float, float, float, float, float]:
    """
    SOGI-PLL core:
    - SOGI-QSG en espacio de estados
    - Park SRF estándar
    - PI sobre v_q normalizado
    - anti-windup por clamping
    - salida de frecuencia suavizada
    """
    n = len(v_array)
    f_est = np.empty(n, dtype=np.float64)

    two_pi = 2.0 * math.pi

    for i in range(n):
        z = v_array[i]

        # -----------------------------------------------------------------
        # 1) SOGI-QSG (state-space)
        # -----------------------------------------------------------------
        w = w_est
        if w < 1.0:
            w = 1.0

        dv_alpha = -k_sogi * w * v_alpha - (w * w) * x_beta + k_sogi * w * z
        dx_beta = v_alpha

        v_alpha = v_alpha + dt * dv_alpha
        x_beta = x_beta + dt * dx_beta

        v_beta = w * x_beta

        # -----------------------------------------------------------------
        # 2) SRF-PLL
        # -----------------------------------------------------------------
        s = math.sin(theta)
        c = math.cos(theta)

        v_d = v_alpha * c + v_beta * s
        v_q = -v_alpha * s + v_beta * c

        amp = math.hypot(v_d, v_q)
        if amp < 1e-9:
            amp = 1e-9

        err = v_q / amp

        up = kp * err
        pll_int_candidate = pll_int + ki * err * dt
        w_new = w_nom + up + pll_int_candidate

        # -----------------------------------------------------------------
        # 3) Anti-windup + clamping
        # -----------------------------------------------------------------
        if w_new > w_max:
            w_new = w_max
        elif w_new < w_min:
            w_new = w_min
        else:
            pll_int = pll_int_candidate

        w_est = w_new

        # -----------------------------------------------------------------
        # 4) Integración de fase
        # -----------------------------------------------------------------
        theta = theta + w_est * dt

        if theta > math.pi or theta < -math.pi:
            theta = (theta + math.pi) % (two_pi) - math.pi

        # -----------------------------------------------------------------
        # 5) Salida de frecuencia suavizada
        # -----------------------------------------------------------------
        f_raw = w_est / two_pi

        if smooth_alpha > 0.0:
            f_out = (1.0 - smooth_alpha) * f_out + smooth_alpha * f_raw
        else:
            f_out = f_raw

        f_est[i] = f_out

    return f_est, theta, pll_int, v_alpha, x_beta, w_est, f_out


# =====================================================================
# Clase Python
# =====================================================================
class SOGIPLLEstimator(BaseFrequencyEstimator):
    """
    SOGI-PLL frequency estimator.

    Características:
    - SOGI-QSG en espacio de estados
    - SRF-PLL estándar con error normalizado
    - salida suavizada para reducir ripple
    - núcleo vectorizado acelerado con Numba
    """
    name = "SOGI-PLL"

    def __init__(
        self,
        nominal_f: float = 60.0, # FIX: Unificado a 60 Hz
        k_sogi: float = 1.414,
        settle_time: float = 0.06,
        output_smoothing: float = 0.015,
        dt: float = DT_DSP,
    ) -> None:
        self.nominal_f = float(nominal_f)
        self.k = float(k_sogi)
        self.settle_time = float(settle_time)
        self.output_smoothing = float(output_smoothing)
        self.dt = float(dt)

        self.w_nom = 2.0 * math.pi * self.nominal_f

        # Sintonía tipo 2º orden equivalente
        zeta = 1.0 / math.sqrt(2.0)
        wn = 4.0 / (zeta * self.settle_time)

        self.kp = 2.0 * zeta * wn
        self.ki = wn * wn

        self.w_min = 2.0 * math.pi * (self.nominal_f - 10.0)
        self.w_max = 2.0 * math.pi * (self.nominal_f + 10.0)

        self.reset()

    def reset(self) -> None:
        # Para v(t)=sin(wt), este arranque reduce el transitorio
        self.theta = -0.5 * math.pi

        self.pll_int = 0.0

        # Estados del SOGI
        self.v_alpha = 0.0
        self.x_beta = -1.0 / self.w_nom

        self.w_est = self.w_nom
        self.f_out = self.nominal_f

    @classmethod
    def default_params(cls) -> dict[str, float]:
        return {
            "nominal_f": 60.0, # FIX: Unificado a 60 Hz
            "k_sogi": 1.414,
            "settle_time": 0.06,
            "output_smoothing": 0.015,
        }

    @staticmethod
    def describe_params(params: dict[str, float]) -> str:
        return (
            f"f_nom={params.get('nominal_f', 60.0)}Hz, " # FIX: Unificado a 60 Hz
            f"k_sogi={params.get('k_sogi', 1.414)}, "
            f"Ts={params.get('settle_time', 0.06)}s"
        )

    def structural_latency_samples(self) -> int:
        """
        T-104: SOGI-PLL settling time defines the transient window.
        Return 2x settle_time in samples so metric windows exclude the transient.
        Factor 2 is standard for 'settled to within ~2% of final value'.
        """
        return int(round(2.0 * self.settle_time / self.dt))

    def step(self, z: float) -> float:
        v_array = np.array([z], dtype=np.float64)
        return float(self.step_vectorized(v_array)[0])

    def step_vectorized(self, v_array: np.ndarray) -> np.ndarray:
        v_array = np.asarray(v_array, dtype=np.float64)

        if v_array.ndim != 1:
            raise ValueError("v_array must be a 1D array.")

        if len(v_array) == 0:
            return np.empty(0, dtype=np.float64)

        (
            f_est,
            self.theta,
            self.pll_int,
            self.v_alpha,
            self.x_beta,
            self.w_est,
            self.f_out,
        ) = _sogi_pll_vectorized_core(
            v_array=v_array,
            dt=self.dt,
            w_nom=self.w_nom,
            k_sogi=self.k,
            kp=self.kp,
            ki=self.ki,
            w_min=self.w_min,
            w_max=self.w_max,
            smooth_alpha=self.output_smoothing,
            theta=self.theta,
            pll_int=self.pll_int,
            v_alpha=self.v_alpha,
            x_beta=self.x_beta,
            w_est=self.w_est,
            f_out=self.f_out,
        )

        return f_est

    def estimate(self, t: np.ndarray, v: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)

        if t.ndim != 1 or v.ndim != 1:
            raise ValueError("t and v must be 1D arrays.")
        if len(t) != len(v):
            raise ValueError("t and v must have the same length.")
        if len(t) == 0:
            return np.empty(0, dtype=np.float64)
        if len(t) == 1:
            return np.full(1, self.nominal_f, dtype=np.float64)

        dt = float(t[1] - t[0])
        if dt <= 0.0:
            raise ValueError("Time vector must be strictly increasing.")

        if len(t) > 2:
            dt_all = np.diff(t)
            tol = max(1e-15, 1e-9 * abs(dt))
            if not np.all(np.abs(dt_all - dt) <= tol):
                raise ValueError("SOGIPLLEstimator requires uniformly sampled time vectors.")

        self.dt = dt
        self.reset()
        return self.step_vectorized(v)