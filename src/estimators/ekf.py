from __future__ import annotations

import math
import numpy as np
from numba import njit

from .base import BaseFrequencyEstimator
from .common import DT_DSP


@njit(cache=True)
def _ekf_vectorized_core(
    v_array: np.ndarray,
    dt: float,
    q_dc: float,
    q_alpha: float,
    q_beta: float,
    q_omega: float,
    r_meas: float,
    smooth_alpha: float,
    x: np.ndarray,
    P: np.ndarray,
    f_out: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    EKF core for single-phase sinusoid tracking.

    State:
        x = [v_dc, alpha, beta, omega]^T

    Signal model:
        y = v_dc + beta

    Dynamics:
        v_dc[k+1] = v_dc[k]
        [alpha, beta] rotate with omega[k]
        omega[k+1] = omega[k]
    """
    n = len(v_array)
    f_est = np.empty(n, dtype=np.float64)

    two_pi = 2.0 * math.pi

    for i in range(n):
        z = v_array[i]

        # -------------------------------------------------------------
        # 1) Current state
        # -------------------------------------------------------------
        vdc = x[0]
        alpha = x[1]
        beta = x[2]
        omega = x[3]

        wd = omega * dt
        c = math.cos(wd)
        s = math.sin(wd)

        # -------------------------------------------------------------
        # 2) Nonlinear state prediction
        # -------------------------------------------------------------
        x_pred = np.empty(4, dtype=np.float64)
        x_pred[0] = vdc
        x_pred[1] = alpha * c - beta * s
        x_pred[2] = alpha * s + beta * c
        x_pred[3] = omega

        # -------------------------------------------------------------
        # 3) Jacobian F = df/dx
        # -------------------------------------------------------------
        F = np.zeros((4, 4), dtype=np.float64)
        F[0, 0] = 1.0

        F[1, 1] = c
        F[1, 2] = -s
        F[1, 3] = dt * (-alpha * s - beta * c)

        F[2, 1] = s
        F[2, 2] = c
        F[2, 3] = dt * (alpha * c - beta * s)

        F[3, 3] = 1.0

        # -------------------------------------------------------------
        # 4) Covariance prediction: Pp = F P F^T + Q
        # -------------------------------------------------------------
        FP = np.zeros((4, 4), dtype=np.float64)
        for rr in range(4):
            for cc in range(4):
                acc = 0.0
                for kk in range(4):
                    acc += F[rr, kk] * P[kk, cc]
                FP[rr, cc] = acc

        Pp = np.zeros((4, 4), dtype=np.float64)
        for rr in range(4):
            for cc in range(4):
                acc = 0.0
                for kk in range(4):
                    acc += FP[rr, kk] * F[cc, kk]
                Pp[rr, cc] = acc

        Pp[0, 0] += q_dc
        Pp[1, 1] += q_alpha
        Pp[2, 2] += q_beta
        Pp[3, 3] += q_omega

        # Soft symmetrization
        for rr in range(4):
            for cc in range(rr + 1, 4):
                sym = 0.5 * (Pp[rr, cc] + Pp[cc, rr])
                Pp[rr, cc] = sym
                Pp[cc, rr] = sym

        # -------------------------------------------------------------
        # 5) Measurement update
        #    h(x) = v_dc + beta
        #    H = [1, 0, 1, 0]
        # -------------------------------------------------------------
        y_pred = x_pred[0] + x_pred[2]
        innov = z - y_pred

        # PH^T
        PHt = np.empty(4, dtype=np.float64)
        for rr in range(4):
            PHt[rr] = Pp[rr, 0] + Pp[rr, 2]

        # S = H P H^T + R
        S = Pp[0, 0] + Pp[0, 2] + Pp[2, 0] + Pp[2, 2] + r_meas
        if S < 1e-15:
            S = 1e-15

        K = np.empty(4, dtype=np.float64)
        for rr in range(4):
            K[rr] = PHt[rr] / S

        # State correction
        for rr in range(4):
            x[rr] = x_pred[rr] + K[rr] * innov

        # Covariance correction: P = Pp - K (H Pp)
        HP = np.empty(4, dtype=np.float64)
        for cc in range(4):
            HP[cc] = Pp[0, cc] + Pp[2, cc]

        for rr in range(4):
            for cc in range(4):
                P[rr, cc] = Pp[rr, cc] - K[rr] * HP[cc]

        # Symmetrize and keep diagonals positive
        for rr in range(4):
            if P[rr, rr] < 1e-15:
                P[rr, rr] = 1e-15
            for cc in range(rr + 1, 4):
                sym = 0.5 * (P[rr, cc] + P[cc, rr])
                P[rr, cc] = sym
                P[cc, rr] = sym

        # -------------------------------------------------------------
        # 6) Frequency output from omega state
        # -------------------------------------------------------------
        f_raw = x[3] / two_pi

        if smooth_alpha > 0.0:
            f_out = (1.0 - smooth_alpha) * f_out + smooth_alpha * f_raw
        else:
            f_out = f_raw

        f_est[i] = f_out

    return f_est, x, P, f_out


class EKF_Estimator(BaseFrequencyEstimator):
    """
    Extended Kalman Filter for single-phase voltage frequency estimation.

    State:
        x = [v_dc, alpha, beta, omega]^T

    where:
        alpha = V*cos(phi)
        beta  = V*sin(phi)

    Measurement:
        y = v_dc + beta

    Notes
    -----
    - Nonlinear state transition due to omega-dependent rotation.
    - Frequency is estimated directly from the omega state.
    - This is a clean baseline EKF with explicit frequency state.
    """

    name = "EKF"

    def __init__(
        self,
        nominal_f: float = 60.0, # FIX: Unificado a 60 Hz para evitar fallos de inicialización
        q_dc: float = 1e-6,
        q_alpha: float = 1e-4,
        q_beta: float = 1e-4,
        q_omega: float = 1e-4,   # FIX: Default conservador (era 5e-2) para señal limpia
        r_meas: float = 1e-3,
        output_smoothing: float = 0.02,
        dt: float = DT_DSP,
    ) -> None:
        self.nominal_f = float(nominal_f)
        self.q_dc = float(q_dc)
        self.q_alpha = float(q_alpha)
        self.q_beta = float(q_beta)
        self.q_omega = float(q_omega)
        self.r_meas = float(r_meas)
        self.output_smoothing = float(output_smoothing)
        self.dt = float(dt)

        self.w_nom = 2.0 * math.pi * self.nominal_f
        self.reset()

    def reset(self) -> None:
        self.x = np.array(
            [0.0, 1.0, 0.0, self.w_nom],
            dtype=np.float64,
        )

        self.P = np.diag(
            np.array(
                [100.0, 10.0, 10.0, (2.0 * math.pi * 5.0) ** 2],
                dtype=np.float64,
            )
        )

        self.f_out = self.nominal_f

    @classmethod
    def default_params(cls) -> dict[str, float]:
        return {
            "nominal_f": 60.0, # FIX: Unificado a 60 Hz
            "q_dc": 1e-6,
            "q_alpha": 1e-4,
            "q_beta": 1e-4,
            "q_omega": 1e-4,   # FIX: Conservador
            "r_meas": 1e-3,
            "output_smoothing": 0.02,
        }

    @staticmethod
    def describe_params(params: dict[str, float]) -> str:
        return (
            f"f_nom={params.get('nominal_f', 60.0)}Hz, "
            f"qω={params.get('q_omega', 1e-4):.1e}, "
            f"R={params.get('r_meas', 1e-3):.1e}"
        )

    def structural_latency_samples(self) -> int:
        return 0

    def step(self, z: float) -> float:
        v_array = np.array([z], dtype=np.float64)
        return float(self.step_vectorized(v_array)[0])

    def step_vectorized(self, v_array: np.ndarray) -> np.ndarray:
        v_array = np.asarray(v_array, dtype=np.float64)

        if v_array.ndim != 1:
            raise ValueError("v_array must be a 1D array.")
        if len(v_array) == 0:
            return np.empty(0, dtype=np.float64)

        f_est, self.x, self.P, self.f_out = _ekf_vectorized_core(
            v_array=v_array,
            dt=self.dt,
            q_dc=self.q_dc,
            q_alpha=self.q_alpha,
            q_beta=self.q_beta,
            q_omega=self.q_omega,
            r_meas=self.r_meas,
            smooth_alpha=self.output_smoothing,
            x=self.x,
            P=self.P,
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
                raise ValueError("EKF_Estimator requires uniformly sampled time vectors.")

        self.dt = dt
        self.w_nom = 2.0 * math.pi * self.nominal_f
        self.reset()
        return self.step_vectorized(v)