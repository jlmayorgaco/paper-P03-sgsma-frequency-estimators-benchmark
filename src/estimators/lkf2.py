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
def _lkf2_vectorized_core(
    v_array: np.ndarray,
    dt: float,
    w_nom: float,
    q0: float,
    q1: float,
    q2: float,
    r: float,
    beta: float,
    mu: float,
    x0: float,
    x1: float,
    x2: float,
    p00: float,
    p01: float,
    p02: float,
    p10: float,
    p11: float,
    p12: float,
    p20: float,
    p21: float,
    p22: float,
    phi_ref: float,
    prev_theta: float,
    have_prev_theta: bool,
    dtheta_lp: float,
    delta_w_hat: float,
) -> tuple[
    np.ndarray,
    float, float, float,
    float, float, float,
    float, float, float,
    float, float, float,
    float, bool, float, float
]:
    """
    Ahmed-style alternative LKF:
      x = [V0, V*cos(theta), V*sin(theta)]
      A = I
      C_n = [1, sin(phi_ref), cos(phi_ref)]
      theta_hat = atan2(x3, x2)
      omega_hat updated from differentiated theta_hat
    """

    n = len(v_array)
    f_est = np.empty(n, dtype=np.float64)
    two_pi = 2.0 * math.pi

    for i in range(n):
        z = v_array[i]

        # ---------------------------------------------------------
        # 1) Build time-varying measurement matrix C_n
        # ---------------------------------------------------------
        sphi = math.sin(phi_ref)
        cphi = math.cos(phi_ref)

        h0 = 1.0
        h1 = sphi
        h2 = cphi

        # ---------------------------------------------------------
        # 2) Predict: A = I
        # ---------------------------------------------------------
        xp0 = x0
        xp1 = x1
        xp2 = x2

        pp00 = p00 + q0
        pp01 = p01
        pp02 = p02

        pp10 = p10
        pp11 = p11 + q1
        pp12 = p12

        pp20 = p20
        pp21 = p21
        pp22 = p22 + q2

        # Soft symmetrization
        s01 = 0.5 * (pp01 + pp10)
        s02 = 0.5 * (pp02 + pp20)
        s12 = 0.5 * (pp12 + pp21)

        pp01 = s01
        pp10 = s01
        pp02 = s02
        pp20 = s02
        pp12 = s12
        pp21 = s12

        # ---------------------------------------------------------
        # 3) Update
        # ---------------------------------------------------------
        # P H^T
        ph0 = pp00 * h0 + pp01 * h1 + pp02 * h2
        ph1 = pp10 * h0 + pp11 * h1 + pp12 * h2
        ph2 = pp20 * h0 + pp21 * h1 + pp22 * h2

        S = h0 * ph0 + h1 * ph1 + h2 * ph2 + r
        if S < 1e-15:
            S = 1e-15

        k0 = ph0 / S
        k1 = ph1 / S
        k2 = ph2 / S

        yhat = h0 * xp0 + h1 * xp1 + h2 * xp2
        e = z - yhat

        x0 = xp0 + k0 * e
        x1 = xp1 + k1 * e
        x2 = xp2 + k2 * e

        # Standard covariance update: P = Pp - K(H Pp)
        # Since Pp is symmetric, H Pp = (Pp H^T)^T
        p00 = pp00 - k0 * ph0
        p01 = pp01 - k0 * ph1
        p02 = pp02 - k0 * ph2

        p10 = pp10 - k1 * ph0
        p11 = pp11 - k1 * ph1
        p12 = pp12 - k1 * ph2

        p20 = pp20 - k2 * ph0
        p21 = pp21 - k2 * ph1
        p22 = pp22 - k2 * ph2

        # Symmetrize again
        s01 = 0.5 * (p01 + p10)
        s02 = 0.5 * (p02 + p20)
        s12 = 0.5 * (p12 + p21)

        p01 = s01
        p10 = s01
        p02 = s02
        p20 = s02
        p12 = s12
        p21 = s12

        # Keep diagonal positive
        if p00 < 1e-15:
            p00 = 1e-15
        if p11 < 1e-15:
            p11 = 1e-15
        if p22 < 1e-15:
            p22 = 1e-15

        # ---------------------------------------------------------
        # 4) Phase estimation from states
        # ---------------------------------------------------------
        theta_hat = math.atan2(x2, x1)

        if have_prev_theta:
            dtheta = _wrap_pi(theta_hat - prev_theta)

            # Optional 1st-order LPF on phase difference
            # mu=1.0 => no filtering
            dtheta_lp = mu * dtheta + (1.0 - mu) * dtheta_lp

            # Discrete interpretation of the paper's phase-based loop:
            # delta_w_hat[k] = delta_w_hat[k-1] + beta * dtheta_lp[k]
            delta_w_hat = delta_w_hat + beta * dtheta_lp
        else:
            have_prev_theta = True
            dtheta_lp = 0.0

        prev_theta = theta_hat

        w_hat = w_nom + delta_w_hat
        f_est[i] = w_hat / two_pi

        # ---------------------------------------------------------
        # 5) Advance reference phase for next sample
        # ---------------------------------------------------------
        phi_ref = _wrap_pi(phi_ref + w_hat * dt)

    return (
        f_est,
        x0, x1, x2,
        p00, p01, p02,
        p10, p11, p12,
        p20, p21, p22,
        phi_ref, prev_theta, have_prev_theta, dtheta_lp, delta_w_hat
    )


class LKF2_Estimator(BaseFrequencyEstimator):
    """
    Alternative Linear Kalman Filter inspired by:

    H. Ahmed, S. Biricik, M. Benbouzid,
    "Linear Kalman Filter-Based Grid Synchronization Technique:
    An Alternative Implementation"

    States:
        x = [V0, V*cos(theta), V*sin(theta)]^T

    Measurement:
        y[n] = [1, sin(phi_ref), cos(phi_ref)] x[n] + w[n]

    Notes
    -----
    - Includes explicit DC-offset state.
    - Direct phase-angle estimation.
    - Frequency obtained from phase-difference loop.
    - Much closer to the Ahmed 2021 paper than the narrowband 2-state LKF.
    """

    name = "LKF2"

    def __init__(
        self,
        nominal_f: float = 50.0,
        q_dc: float = 0.005,
        q_vc: float = 0.05,
        q_vs: float = 0.05,
        r: float = 1.0,
        beta: float = 50.0,
        lpf_mu: float = 1.0,
        dt: float = DT_DSP,
    ) -> None:
        self.nominal_f = float(nominal_f)
        self.q_dc = float(q_dc)
        self.q_vc = float(q_vc)
        self.q_vs = float(q_vs)
        self.r = float(r)
        self.beta = float(beta)
        self.lpf_mu = float(lpf_mu)
        self.dt = float(dt)

        self.w_nom = 2.0 * math.pi * self.nominal_f
        self.reset()

    def reset(self) -> None:
        # Paper-inspired initialization
        self.x0 = 0.0   # V0
        self.x1 = 0.5   # V*cos(theta)
        self.x2 = 0.0   # V*sin(theta)

        p0 = 1000.0
        self.p00 = p0
        self.p01 = 0.0
        self.p02 = 0.0

        self.p10 = 0.0
        self.p11 = p0
        self.p12 = 0.0

        self.p20 = 0.0
        self.p21 = 0.0
        self.p22 = p0

        self.phi_ref = 0.0
        self.prev_theta = 0.0
        self.have_prev_theta = False
        self.dtheta_lp = 0.0
        self.delta_w_hat = 0.0

    @classmethod
    def default_params(cls) -> dict[str, float]:
        return {
            "nominal_f": 50.0,
            "q_dc": 0.005,
            "q_vc": 0.05,
            "q_vs": 0.05,
            "r": 1.0,
            "beta": 50.0,
            "lpf_mu": 1.0,
        }

    @staticmethod
    def describe_params(params: dict[str, float]) -> str:
        return (
            f"f_nom={params.get('nominal_f', 50.0)}Hz, "
            f"Q=diag([{params.get('q_dc', 0.005)}, "
            f"{params.get('q_vc', 0.05)}, "
            f"{params.get('q_vs', 0.05)}]), "
            f"R={params.get('r', 1.0)}, "
            f"beta={params.get('beta', 50.0)}"
        )

    def structural_latency_samples(self) -> int:
        return 0

    def step(self, z: float) -> float:
        return float(self.step_vectorized(np.array([z], dtype=np.float64))[0])

    def step_vectorized(self, v_array: np.ndarray) -> np.ndarray:
        v_array = np.asarray(v_array, dtype=np.float64)

        if v_array.ndim != 1:
            raise ValueError("v_array must be a 1D array.")
        if len(v_array) == 0:
            return np.empty(0, dtype=np.float64)

        (
            f_est,
            self.x0, self.x1, self.x2,
            self.p00, self.p01, self.p02,
            self.p10, self.p11, self.p12,
            self.p20, self.p21, self.p22,
            self.phi_ref, self.prev_theta,
            self.have_prev_theta, self.dtheta_lp, self.delta_w_hat
        ) = _lkf2_vectorized_core(
            v_array=v_array,
            dt=self.dt,
            w_nom=self.w_nom,
            q0=self.q_dc,
            q1=self.q_vc,
            q2=self.q_vs,
            r=self.r,
            beta=self.beta,
            mu=self.lpf_mu,
            x0=self.x0,
            x1=self.x1,
            x2=self.x2,
            p00=self.p00,
            p01=self.p01,
            p02=self.p02,
            p10=self.p10,
            p11=self.p11,
            p12=self.p12,
            p20=self.p20,
            p21=self.p21,
            p22=self.p22,
            phi_ref=self.phi_ref,
            prev_theta=self.prev_theta,
            have_prev_theta=self.have_prev_theta,
            dtheta_lp=self.dtheta_lp,
            delta_w_hat=self.delta_w_hat,
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
                raise ValueError("LKF2_Estimator requires uniformly sampled time vectors.")

        self.dt = dt
        self.w_nom = 2.0 * math.pi * self.nominal_f
        self.reset()
        return self.step_vectorized(v)