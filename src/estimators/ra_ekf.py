from __future__ import annotations

import math
import numpy as np
from numba import njit

from .base import BaseFrequencyEstimator
from .common import DT_DSP


@njit(cache=True)
def _ra_ekf_plus_core(
    v_array: np.ndarray,
    dt: float,
    w_nom: float,
    q_theta: float,
    q_omega: float,
    q_A: float,
    q_rocof: float,
    r_meas: float,
    sigma_v: float,
    gamma: float,
    deriv_lpf_alpha: float,
    rho_rocof: float,
    x: np.ndarray,
    P: np.ndarray,
    dv_filt: float,
    prev_z: float,
    has_prev: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, bool]:
    """
    Improved RA-EKF:
    - 4-state model: x = [theta, omega, A, rocof]
    - 2-channel causal measurement:
        z1 = v[k]
        z2 = LPF(dv/dt) / w_nom
    - soft robust weighting instead of hard gating
    - selective adaptive Q on omega/rocof only
    - RoCoF leakage (rho_rocof) to avoid random-walk drift
    - Joseph covariance update
    """
    n = len(v_array)
    out = np.empty(n, dtype=np.float64)

    two_pi = 2.0 * math.pi
    dt2_half = 0.5 * dt * dt

    w_min = two_pi * 40.0
    w_max = two_pi * 80.0
    a_min = 0.05
    a_max = 2.0
    rocof_max = two_pi * 20.0  # 20 Hz/s equivalent

    # derivative channel is noisier than raw voltage
    r_dv = max(4.0 * r_meas, 0.25 * sigma_v * sigma_v, 1e-8)
    sigma_d = max(sigma_v, 1e-4)

    I4 = np.eye(4, dtype=np.float64)

    for i in range(n):
        z = v_array[i]

        if not has_prev:
            prev_z = z
            has_prev = True
            out[i] = x[1] / two_pi
            continue

        # -------------------------------------------------------------
        # 0. Causal derivative pseudo-measurement
        # -------------------------------------------------------------
        raw_dv = (z - prev_z) / dt
        prev_z = z
        dv_filt = (1.0 - deriv_lpf_alpha) * dv_filt + deriv_lpf_alpha * raw_dv

        # -------------------------------------------------------------
        # 1. Predict
        # -------------------------------------------------------------
        theta = x[0]
        omega = x[1]
        amp = x[2]
        rocof = x[3]

        x_pred = np.empty(4, dtype=np.float64)
        x_pred[0] = theta + omega * dt + rocof * dt2_half
        x_pred[1] = omega + rocof * dt
        x_pred[2] = amp
        x_pred[3] = rho_rocof * rocof

        F = np.zeros((4, 4), dtype=np.float64)
        F[0, 0] = 1.0
        F[0, 1] = dt
        F[0, 3] = dt2_half
        F[1, 1] = 1.0
        F[1, 3] = dt
        F[2, 2] = 1.0
        F[3, 3] = rho_rocof

        FP = np.zeros((4, 4), dtype=np.float64)
        for rr in range(4):
            for cc in range(4):
                s = 0.0
                for kk in range(4):
                    s += F[rr, kk] * P[kk, cc]
                FP[rr, cc] = s

        Pp = np.zeros((4, 4), dtype=np.float64)
        for rr in range(4):
            for cc in range(4):
                s = 0.0
                for kk in range(4):
                    s += FP[rr, kk] * F[cc, kk]
                Pp[rr, cc] = s

        # selective adaptive Q: only omega / rocof get boosted
        theta_p = x_pred[0]
        omega_p = x_pred[1]
        amp_p = x_pred[2]

        z2_pred = dv_filt / w_nom
        h2_pred = -(amp_p * omega_p / w_nom) * math.sin(theta_p)
        dyn_mis = abs(z2_pred - h2_pred) / sigma_d
        q_scale = min(max(1.0 + 0.25 * dyn_mis, 1.0), 8.0)

        Pp[0, 0] += q_theta
        Pp[1, 1] += q_omega * q_scale
        Pp[2, 2] += q_A
        Pp[3, 3] += q_rocof * q_scale * q_scale

        for rr in range(4):
            for cc in range(rr + 1, 4):
                sym = 0.5 * (Pp[rr, cc] + Pp[cc, rr])
                Pp[rr, cc] = sym
                Pp[cc, rr] = sym

        # -------------------------------------------------------------
        # 2. Measurement model
        # -------------------------------------------------------------
        sin_t = math.sin(theta_p)
        cos_t = math.cos(theta_p)

        z1 = z
        z2 = dv_filt / w_nom

        h1 = amp_p * cos_t
        h2 = -(amp_p * omega_p / w_nom) * sin_t

        y0 = z1 - h1
        y1 = z2 - h2

        H = np.zeros((2, 4), dtype=np.float64)
        H[0, 0] = -amp_p * sin_t
        H[0, 2] = cos_t

        H[1, 0] = -(amp_p * omega_p / w_nom) * cos_t
        H[1, 1] = -(amp_p / w_nom) * sin_t
        H[1, 2] = -(omega_p / w_nom) * sin_t

        # PHT = Pp * H^T  => 4x2
        PHT = np.zeros((4, 2), dtype=np.float64)
        for rr in range(4):
            for cc in range(2):
                s = 0.0
                for kk in range(4):
                    s += Pp[rr, kk] * H[cc, kk]
                PHT[rr, cc] = s

        # S = H * Pp * H^T + R
        S00 = 0.0
        S01 = 0.0
        S10 = 0.0
        S11 = 0.0
        for kk in range(4):
            S00 += H[0, kk] * PHT[kk, 0]
            S01 += H[0, kk] * PHT[kk, 1]
            S10 += H[1, kk] * PHT[kk, 0]
            S11 += H[1, kk] * PHT[kk, 1]

        R00 = r_meas
        R11 = r_dv
        S00 += R00
        S11 += R11

        detS = S00 * S11 - S01 * S10
        if abs(detS) < 1e-15:
            x[:] = x_pred
            P[:, :] = Pp
            out[i] = x[1] / two_pi
            continue

        invS00 = S11 / detS
        invS01 = -S01 / detS
        invS10 = -S10 / detS
        invS11 = S00 / detS

        nis = (
            y0 * (invS00 * y0 + invS01 * y1)
            + y1 * (invS10 * y0 + invS11 * y1)
        )

        # -------------------------------------------------------------
        # 3. Soft robust weighting (no hard freeze)
        # -------------------------------------------------------------
        gate_ref = max(gamma * gamma, 1.0)
        scale = 1.0
        if nis > gate_ref:
            scale = min(50.0, nis / gate_ref)

        v_score = abs(y0) / max(sigma_v, 1e-6)
        d_score = abs(y1) / max(sigma_d, 1e-6)

        if scale > 1.0:
            if v_score >= d_score:
                R00 *= scale
                R11 *= math.sqrt(scale)
            else:
                R00 *= math.sqrt(scale)
                R11 *= scale

            S00 = 0.0
            S01 = 0.0
            S10 = 0.0
            S11 = 0.0
            for kk in range(4):
                S00 += H[0, kk] * PHT[kk, 0]
                S01 += H[0, kk] * PHT[kk, 1]
                S10 += H[1, kk] * PHT[kk, 0]
                S11 += H[1, kk] * PHT[kk, 1]

            S00 += R00
            S11 += R11

            detS = S00 * S11 - S01 * S10
            if abs(detS) < 1e-15:
                x[:] = x_pred
                P[:, :] = Pp
                out[i] = x[1] / two_pi
                continue

            invS00 = S11 / detS
            invS01 = -S01 / detS
            invS10 = -S10 / detS
            invS11 = S00 / detS

        # -------------------------------------------------------------
        # 4. Kalman gain
        # -------------------------------------------------------------
        K = np.zeros((4, 2), dtype=np.float64)
        for rr in range(4):
            K[rr, 0] = PHT[rr, 0] * invS00 + PHT[rr, 1] * invS10
            K[rr, 1] = PHT[rr, 0] * invS01 + PHT[rr, 1] * invS11

        x_new = np.empty(4, dtype=np.float64)
        for rr in range(4):
            x_new[rr] = x_pred[rr] + K[rr, 0] * y0 + K[rr, 1] * y1

        # -------------------------------------------------------------
        # 5. Joseph covariance update
        # -------------------------------------------------------------
        KH = np.zeros((4, 4), dtype=np.float64)
        for rr in range(4):
            for cc in range(4):
                KH[rr, cc] = K[rr, 0] * H[0, cc] + K[rr, 1] * H[1, cc]

        I_KH = np.zeros((4, 4), dtype=np.float64)
        for rr in range(4):
            for cc in range(4):
                I_KH[rr, cc] = I4[rr, cc] - KH[rr, cc]

        TMP = np.zeros((4, 4), dtype=np.float64)
        for rr in range(4):
            for cc in range(4):
                s = 0.0
                for kk in range(4):
                    s += I_KH[rr, kk] * Pp[kk, cc]
                TMP[rr, cc] = s

        P_new = np.zeros((4, 4), dtype=np.float64)
        for rr in range(4):
            for cc in range(4):
                s = 0.0
                for kk in range(4):
                    s += TMP[rr, kk] * I_KH[cc, kk]
                P_new[rr, cc] = s

        KRKT = np.zeros((4, 4), dtype=np.float64)
        for rr in range(4):
            for cc in range(4):
                KRKT[rr, cc] = K[rr, 0] * R00 * K[cc, 0] + K[rr, 1] * R11 * K[cc, 1]

        for rr in range(4):
            for cc in range(4):
                P[rr, cc] = P_new[rr, cc] + KRKT[rr, cc]

        for rr in range(4):
            for cc in range(rr + 1, 4):
                sym = 0.5 * (P[rr, cc] + P[cc, rr])
                P[rr, cc] = sym
                P[cc, rr] = sym

        # -------------------------------------------------------------
        # 6. Clamps / guards
        # -------------------------------------------------------------
        while x_new[0] > math.pi:
            x_new[0] -= two_pi
        while x_new[0] < -math.pi:
            x_new[0] += two_pi

        if x_new[1] < w_min:
            x_new[1] = w_min
        if x_new[1] > w_max:
            x_new[1] = w_max

        if x_new[2] < a_min:
            x_new[2] = a_min
        if x_new[2] > a_max:
            x_new[2] = a_max

        if x_new[3] < -rocof_max:
            x_new[3] = -rocof_max
        if x_new[3] > rocof_max:
            x_new[3] = rocof_max

        state_ok = True
        for rr in range(4):
            if not math.isfinite(x_new[rr]):
                state_ok = False
                break

        cov_ok = (
            math.isfinite(P[0, 0])
            and math.isfinite(P[1, 1])
            and math.isfinite(P[2, 2])
            and math.isfinite(P[3, 3])
        )

        if (not state_ok) or (not cov_ok):
            x[0] = 0.0
            x[1] = w_nom
            x[2] = 1.0
            x[3] = 0.0
            for rr in range(4):
                for cc in range(4):
                    P[rr, cc] = 0.0
            P[0, 0] = 0.5
            P[1, 1] = (2.0 * math.pi * 1.0) ** 2
            P[2, 2] = 0.1
            P[3, 3] = (2.0 * math.pi * 5.0) ** 2
        else:
            x[:] = x_new

        out[i] = x[1] / two_pi

    return out, x, P, dv_filt, prev_z, has_prev


class RAEKF_Estimator(BaseFrequencyEstimator):
    """
    Improved RoCoF-Augmented EKF.

    Changes vs old RA-EKF:
    - no hard gating that freezes omega/rocof
    - second causal measurement from filtered derivative
    - selective adaptive Q only on dynamic states
    - RoCoF leakage to avoid drift
    - Joseph covariance update
    """

    name = "RA-EKF"

    def __init__(
        self,
        nominal_f: float = 60.0,
        q_theta: float = 1e-6,
        q_omega: float = 5e-4,
        q_A: float = 1e-6,
        q_rocof: float = 5e-3,
        r_meas: float = 1e-4,
        sigma_v: float = 0.05,
        gamma: float = 8.0,
        dt: float = DT_DSP,
        deriv_lpf_alpha: float = 0.08,
        tau_rocof: float = 0.15,
    ) -> None:
        self.nominal_f = float(nominal_f)
        self.w_nom = 2.0 * math.pi * self.nominal_f
        self.dt = float(dt)

        self.q_theta = float(q_theta)
        self.q_omega = float(q_omega)
        self.q_A = float(q_A)
        self.q_rocof = float(q_rocof)

        self.r_meas = float(r_meas)
        self.sigma_v = float(sigma_v)
        self.gamma = float(gamma)

        self.deriv_lpf_alpha = float(deriv_lpf_alpha)
        self.tau_rocof = float(tau_rocof)

        self.reset()

    def reset(self) -> None:
        self.x = np.array(
            [0.0, self.w_nom, 1.0, 0.0],
            dtype=np.float64,
        )

        self.P = np.diag([
            0.5,
            (2.0 * math.pi * 1.0) ** 2,
            0.1,
            (2.0 * math.pi * 5.0) ** 2,
        ]).astype(np.float64)

        self.dv_filt = 0.0
        self.prev_z = 0.0
        self.has_prev = False

    @classmethod
    def default_params(cls) -> dict[str, float]:
        return {
            "nominal_f": 60.0,
            "q_theta": 1e-6,
            "q_omega": 5e-4,
            "q_A": 1e-6,
            "q_rocof": 5e-3,
            "r_meas": 1e-4,
            "sigma_v": 0.05,
            "gamma": 8.0,
            "deriv_lpf_alpha": 0.08,
            "tau_rocof": 0.15,
        }

    @staticmethod
    def describe_params(params: dict[str, float]) -> str:
        return (
            f"RA-EKF f_nom={params.get('nominal_f', 60.0)}Hz, "
            f"q_rocof={params.get('q_rocof', 5e-3):.1e}, "
            f"r_meas={params.get('r_meas', 1e-4):.1e}, "
            f"sigma_v={params.get('sigma_v', 0.05):.3f}, "
            f"gamma={params.get('gamma', 8.0):.1f}"
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

        rho_rocof = math.exp(-self.dt / max(self.tau_rocof, 1e-9))

        f_est, self.x, self.P, self.dv_filt, self.prev_z, self.has_prev = _ra_ekf_plus_core(
            v_array=v_array,
            dt=self.dt,
            w_nom=self.w_nom,
            q_theta=self.q_theta,
            q_omega=self.q_omega,
            q_A=self.q_A,
            q_rocof=self.q_rocof,
            r_meas=self.r_meas,
            sigma_v=self.sigma_v,
            gamma=self.gamma,
            deriv_lpf_alpha=self.deriv_lpf_alpha,
            rho_rocof=rho_rocof,
            x=self.x,
            P=self.P,
            dv_filt=self.dv_filt,
            prev_z=self.prev_z,
            has_prev=self.has_prev,
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

        self.dt = dt
        self.w_nom = 2.0 * math.pi * self.nominal_f
        self.reset()
        return self.step_vectorized(v)