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
def _lkf_vectorized_core(
    v_array: np.ndarray,
    dt: float,
    nominal_f: float,
    q: float,
    r: float,
    rho: float,
    lag_samples: int,
    smooth_alpha: float,
    x1: float,
    x2: float,
    p11: float,
    p12: float,
    p21: float,
    p22: float,
    hist_x1: np.ndarray,
    hist_x2: np.ndarray,
    hist_idx: int,
    hist_count: int,
    f_out: float,
):
    """
    Linear Kalman Filter core for narrowband sinusoidal tracking.

    State convention:
        x = [sin(theta), cos(theta)]

    Frequency recovery:
        estimated from the phase difference between the current normalized
        state vector and the vector from 'lag_samples' samples ago.

    This is intentionally simple:
    - no clamping
    - no augmented state
    - no adaptive tuning
    """

    n = len(v_array)
    f_est = np.empty(n, dtype=np.float64)

    two_pi = 2.0 * math.pi
    w0 = two_pi * nominal_f

    c = math.cos(w0 * dt)
    s = math.sin(w0 * dt)

    # State transition for x = [sin(theta), cos(theta)]
    a11 = rho * c
    a12 = rho * s
    a21 = -rho * s
    a22 = rho * c

    for i in range(n):
        z = v_array[i]

        # -------------------------------------------------------------
        # 1) Predict
        # -------------------------------------------------------------
        xp1 = a11 * x1 + a12 * x2
        xp2 = a21 * x1 + a22 * x2

        # Pp = A P A' + Q, with Q = q * I
        ap11 = a11 * p11 + a12 * p21
        ap12 = a11 * p12 + a12 * p22
        ap21 = a21 * p11 + a22 * p21
        ap22 = a21 * p12 + a22 * p22

        pp11 = ap11 * a11 + ap12 * a12 + q
        pp12 = ap11 * a21 + ap12 * a22
        pp21 = ap21 * a11 + ap22 * a12
        pp22 = ap21 * a21 + ap22 * a22 + q

        # Soft symmetrization
        pp12pp21 = 0.5 * (pp12 + pp21)
        pp12 = pp12pp21
        pp21 = pp12pp21

        # -------------------------------------------------------------
        # 2) Update, H = [1 0]
        # -------------------------------------------------------------
        innov = z - xp1
        S = pp11 + r
        if S < 1e-15:
            S = 1e-15

        k1 = pp11 / S
        k2 = pp21 / S

        x1 = xp1 + k1 * innov
        x2 = xp2 + k2 * innov

        # Joseph-form covariance update
        a = 1.0 - k1
        b = -k2

        p11 = a * a * pp11 + r * k1 * k1
        p12 = -a * k2 * pp11 + a * pp12 + r * k1 * k2
        p21 = p12
        p22 = pp22 - 2.0 * k2 * pp12 + k2 * k2 * pp11 + r * k2 * k2

        if p11 < 1e-15:
            p11 = 1e-15
        if p22 < 1e-15:
            p22 = 1e-15

        # -------------------------------------------------------------
        # 3) Normalize ONLY for phase extraction
        # -------------------------------------------------------------
        norm_x = math.hypot(x1, x2)
        if norm_x > 1e-12:
            cur_x1 = x1 / norm_x
            cur_x2 = x2 / norm_x
        else:
            cur_x1 = 0.0
            cur_x2 = 1.0

        # -------------------------------------------------------------
        # 4) Frequency from lagged phase difference
        # -------------------------------------------------------------
        if hist_count >= lag_samples:
            old_x1 = hist_x1[hist_idx]
            old_x2 = hist_x2[hist_idx]

            # Correct sign for x = [sin(theta), cos(theta)]
            cross = old_x2 * cur_x1 - old_x1 * cur_x2
            dot = old_x1 * cur_x1 + old_x2 * cur_x2

            dtheta = math.atan2(cross, dot)
            f_raw = dtheta / (two_pi * lag_samples * dt)
        else:
            f_raw = nominal_f

        # Update lag buffer
        hist_x1[hist_idx] = cur_x1
        hist_x2[hist_idx] = cur_x2
        hist_idx += 1
        if hist_idx >= lag_samples:
            hist_idx = 0

        if hist_count < lag_samples:
            hist_count += 1

        # -------------------------------------------------------------
        # 5) Optional output smoothing
        # -------------------------------------------------------------
        if smooth_alpha > 0.0:
            f_out = (1.0 - smooth_alpha) * f_out + smooth_alpha * f_raw
        else:
            f_out = f_raw

        f_est[i] = f_out

    return (
        f_est,
        x1,
        x2,
        p11,
        p12,
        p21,
        p22,
        hist_x1,
        hist_x2,
        hist_idx,
        hist_count,
        f_out,
    )


# =====================================================================
# Class wrapper
# =====================================================================
class LKF_Estimator(BaseFrequencyEstimator):
    """
    Standard Linear Kalman Filter for narrowband sinusoidal tracking.

    State:
        x = [alpha, beta]^T ~ [sin(theta), cos(theta)]^T

    Dynamics:
        x[k+1] = A x[k] + w[k]

    Measurement:
        z[k] = [1, 0] x[k] + v[k]

    Frequency recovery:
        estimated from lagged phase difference of the filtered state.

    Notes
    -----
    - Simple narrowband LKF baseline.
    - No augmented frequency state.
    - Best near nominal and for moderate dynamics.
    """

    name = "LKF"

    def __init__(
        self,
        nominal_f: float = 60.0,
        q: float = 1e-5,
        r: float = 1e-3,
        rho: float = 1.0,
        output_smoothing: float = 0.02,
        phase_lag_samples: int = 0,
        dt: float = DT_DSP,
    ) -> None:
        self.nominal_f = float(nominal_f)
        self.q = float(q)
        self.r = float(r)
        self.rho = float(rho)
        self.output_smoothing = float(output_smoothing)
        self.phase_lag_samples = int(phase_lag_samples)
        self.dt = float(dt)

        self._lag_samples = 1
        self._hist_x1 = np.zeros(1, dtype=np.float64)
        self._hist_x2 = np.zeros(1, dtype=np.float64)

        self.reset()

    def _resolve_lag_samples(self, dt: float) -> int:
        if self.phase_lag_samples > 0:
            return self.phase_lag_samples

        fs = 1.0 / dt
        # Quarter-cycle lag at nominal frequency
        lag = int(round(fs / (4.0 * self.nominal_f)))
        return max(1, lag)

    def _configure_buffers(self) -> None:
        self._lag_samples = self._resolve_lag_samples(self.dt)
        self._hist_x1 = np.zeros(self._lag_samples, dtype=np.float64)
        self._hist_x2 = np.zeros(self._lag_samples, dtype=np.float64)
        self._hist_idx = 0
        self._hist_count = 0

    def reset(self) -> None:
        # Initial state consistent with v(0)=sin(0)=0
        self.x1 = 0.0   # sin(theta)
        self.x2 = 1.0   # cos(theta)

        # Large initial uncertainty
        self.p11 = 10.0
        self.p12 = 0.0
        self.p21 = 0.0
        self.p22 = 10.0

        self.f_out = self.nominal_f
        self._configure_buffers()

    @classmethod
    def default_params(cls) -> dict[str, float | int]:
        return {
            "nominal_f": 60.0,
            "q": 1e-5,
            "r": 1e-3,
            "rho": 1.0,
            "output_smoothing": 0.02,
            "phase_lag_samples": 0,
        }

    @staticmethod
    def describe_params(params: dict[str, float | int]) -> str:
        return (
            f"f_nom={params.get('nominal_f', 50.0)}Hz, "
            f"q={params.get('q', 1e-5):.1e}, "
            f"r={params.get('r', 1e-3):.1e}, "
            f"rho={params.get('rho', 1.0):.4f}"
        )

    def structural_latency_samples(self) -> int:
        return self._lag_samples

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
            self.x1,
            self.x2,
            self.p11,
            self.p12,
            self.p21,
            self.p22,
            self._hist_x1,
            self._hist_x2,
            self._hist_idx,
            self._hist_count,
            self.f_out,
        ) = _lkf_vectorized_core(
            v_array=v_array,
            dt=self.dt,
            nominal_f=self.nominal_f,
            q=self.q,
            r=self.r,
            rho=self.rho,
            lag_samples=self._lag_samples,
            smooth_alpha=self.output_smoothing,
            x1=self.x1,
            x2=self.x2,
            p11=self.p11,
            p12=self.p12,
            p21=self.p21,
            p22=self.p22,
            hist_x1=self._hist_x1,
            hist_x2=self._hist_x2,
            hist_idx=self._hist_idx,
            hist_count=self._hist_count,
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
                raise ValueError("LKF_Estimator requires uniformly sampled time vectors.")

        self.dt = dt
        self.reset()
        return self.step_vectorized(v)