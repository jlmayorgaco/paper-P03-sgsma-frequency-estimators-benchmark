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
def _pll_vectorized_core(
    v_array: np.ndarray,
    dt: float,
    w_nom: float,
    kp: float,
    ki: float,
    amp_alpha: float,
    pd_lpf_alpha: float,
    smooth_alpha: float,
    f_min: float,
    f_max: float,
    theta: float,
    pll_int: float,
    amp_sq_lp: float,
    pd_err_lp: float,
    f_out: float,
) -> tuple[np.ndarray, float, float, float, float, float]:
    """
    Simple single-phase multiplier PLL with:
    - amplitude normalization
    - phase-detector low-pass filtering
    - PI loop filter
    - basic anti-windup
    - optional output smoothing

    Detector:
        e_raw = v * cos(theta_hat) / amp_hat

    Important:
    The raw multiplier detector contains a strong 2*w ripple. A small LPF on the
    detector output is essential; otherwise the PI loop turns that ripple into a
    large oscillatory frequency estimate.
    """
    n = len(v_array)
    f_est = np.empty(n, dtype=np.float64)
    two_pi = 2.0 * math.pi

    for i in range(n):
        z = v_array[i]

        # -------------------------------------------------------------
        # 1) Amplitude normalization
        # -------------------------------------------------------------
        amp_sq_lp = (1.0 - amp_alpha) * amp_sq_lp + amp_alpha * (z * z)
        amp_hat = math.sqrt(max(amp_sq_lp, 1e-8))

        # -------------------------------------------------------------
        # 2) Raw multiplier phase detector
        # -------------------------------------------------------------
        e_raw = (z * math.cos(theta)) / amp_hat

        # Low-pass filter the detector output to remove 2*w ripple
        pd_err_lp = (1.0 - pd_lpf_alpha) * pd_err_lp + pd_lpf_alpha * e_raw

        # -------------------------------------------------------------
        # 3) PI loop filter with simple anti-windup
        # -------------------------------------------------------------
        up = kp * pd_err_lp
        w_pre = w_nom + up + pll_int
        f_pre = w_pre / two_pi

        # Integrate only if not saturated, or if error drives back inward
        if (
            (f_min <= f_pre <= f_max)
            or (f_pre < f_min and pd_err_lp > 0.0)
            or (f_pre > f_max and pd_err_lp < 0.0)
        ):
            pll_int += ki * pd_err_lp * dt

        w_hat = w_nom + up + pll_int
        f_hat = w_hat / two_pi

        if f_hat < f_min:
            f_hat = f_min
            w_hat = two_pi * f_hat
        elif f_hat > f_max:
            f_hat = f_max
            w_hat = two_pi * f_hat

        # -------------------------------------------------------------
        # 4) Phase integration
        # -------------------------------------------------------------
        theta = _wrap_pi(theta + w_hat * dt)

        # -------------------------------------------------------------
        # 5) Optional output smoothing
        # -------------------------------------------------------------
        if smooth_alpha > 0.0:
            f_out = (1.0 - smooth_alpha) * f_out + smooth_alpha * f_hat
        else:
            f_out = f_hat

        f_est[i] = f_out

    return f_est, theta, pll_int, amp_sq_lp, pd_err_lp, f_out


class PLL_Estimator(BaseFrequencyEstimator):
    """
    Simple single-phase multiplier PLL.

    This is intentionally simpler than a SOGI-PLL or SRF-PLL, but still needs:
    - amplitude normalization
    - filtered phase-detector output
    - conservative loop gains

    Without detector filtering, the estimate shows a strong 2*w oscillation.
    """

    name = "PLL"

    def __init__(
        self,
        nominal_f: float = 60.0, # FIX: Unificado a 60 Hz para evitar saturación en benchmarks
        settle_time: float = 0.08,
        amp_alpha: float = 0.01,
        output_smoothing: float = 0.02,
        pd_lpf_alpha: float = 0.02,
        kp_scale: float = 0.15,
        ki_scale: float = 0.05,
        dt: float = DT_DSP,
    ) -> None:
        self.nominal_f = float(nominal_f)
        self.settle_time = float(settle_time)
        self.amp_alpha = float(amp_alpha)
        self.output_smoothing = float(output_smoothing)
        self.pd_lpf_alpha = float(pd_lpf_alpha)
        self.kp_scale = float(kp_scale)
        self.ki_scale = float(ki_scale)
        self.dt = float(dt)

        self.w_nom = 2.0 * math.pi * self.nominal_f
        self.f_min = self.nominal_f - 10.0
        self.f_max = self.nominal_f + 10.0

        # Conservative PLL tuning for this simple sampled multiplier detector
        zeta = 1.0 / math.sqrt(2.0)
        wn = 4.0 / (zeta * self.settle_time)

        base_kp = 2.0 * zeta * wn
        base_ki = wn * wn

        self.kp = self.kp_scale * base_kp
        self.ki = self.ki_scale * base_ki

        self.reset()

    def reset(self) -> None:
        self.theta = 0.0
        self.pll_int = 0.0
        self.amp_sq_lp = 1.0
        self.pd_err_lp = 0.0
        self.f_out = self.nominal_f

    @classmethod
    def default_params(cls) -> dict[str, float]:
        return {
            "nominal_f": 60.0, # FIX: Unificado a 60 Hz
            "settle_time": 0.08,
            "amp_alpha": 0.01,
            "output_smoothing": 0.02,
            "pd_lpf_alpha": 0.02,
            "kp_scale": 0.15,
            "ki_scale": 0.05,
        }

    @staticmethod
    def describe_params(params: dict[str, float]) -> str:
        return (
            f"f_nom={params.get('nominal_f', 60.0)}Hz, " # FIX: Unificado a 60 Hz
            f"Ts={params.get('settle_time', 0.08)}s, "
            f"amp_alpha={params.get('amp_alpha', 0.01)}, "
            f"pd_lpf_alpha={params.get('pd_lpf_alpha', 0.02)}, "
            f"kp_scale={params.get('kp_scale', 0.15)}, "
            f"ki_scale={params.get('ki_scale', 0.05)}"
        )

    def structural_latency_samples(self) -> int:
        """
        T-104: PLL settling time defines the transient window.
        Return 2x settle_time in samples so metric windows exclude the transient.
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
            self.amp_sq_lp,
            self.pd_err_lp,
            self.f_out,
        ) = _pll_vectorized_core(
            v_array=v_array,
            dt=self.dt,
            w_nom=self.w_nom,
            kp=self.kp,
            ki=self.ki,
            amp_alpha=self.amp_alpha,
            pd_lpf_alpha=self.pd_lpf_alpha,
            smooth_alpha=self.output_smoothing,
            f_min=self.f_min,
            f_max=self.f_max,
            theta=self.theta,
            pll_int=self.pll_int,
            amp_sq_lp=self.amp_sq_lp,
            pd_err_lp=self.pd_err_lp,
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
                raise ValueError("PLL_Estimator requires uniformly sampled time vectors.")

        self.dt = dt
        self.w_nom = 2.0 * math.pi * self.nominal_f

        # Recompute gains in case settle_time changed externally
        zeta = 1.0 / math.sqrt(2.0)
        wn = 4.0 / (zeta * self.settle_time)
        base_kp = 2.0 * zeta * wn
        base_ki = wn * wn
        self.kp = self.kp_scale * base_kp
        self.ki = self.ki_scale * base_ki

        self.reset()
        return self.step_vectorized(v)