from __future__ import annotations

import math
import numpy as np
from numba import njit

from .base import BaseFrequencyEstimator
from .common import DT_DSP


@njit(cache=True)
def _symmetrize_4x4(P: np.ndarray) -> None:
    for i in range(4):
        for j in range(i + 1, 4):
            s = 0.5 * (P[i, j] + P[j, i])
            P[i, j] = s
            P[j, i] = s


@njit(cache=True)
def _process_model(x: np.ndarray, dt: float) -> np.ndarray:
    """
    Nonlinear process model:
        x = [v_dc, alpha, beta, omega]^T

    where:
        alpha = V*cos(phi)
        beta  = V*sin(phi)
    """
    out = np.empty(4, dtype=np.float64)

    vdc = x[0]
    alpha = x[1]
    beta = x[2]
    omega = x[3]

    wd = omega * dt
    c = math.cos(wd)
    s = math.sin(wd)

    out[0] = vdc
    out[1] = alpha * c - beta * s
    out[2] = alpha * s + beta * c
    out[3] = omega

    return out


@njit(cache=True)
def _measurement_model(x: np.ndarray) -> float:
    """
    Measurement model:
        y = v_dc + beta
    """
    return x[0] + x[2]


@njit(cache=True)
def _ukf_vectorized_core(
    v_array: np.ndarray,
    dt: float,
    q_dc: float,
    q_alpha: float,
    q_beta: float,
    q_omega: float,
    r_meas: float,
    smooth_alpha: float,
    gamma: float,
    wm: np.ndarray,
    wc: np.ndarray,
    x: np.ndarray,
    P: np.ndarray,
    f_out: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    UKF core for single-phase sinusoid tracking.

    State:
        x = [v_dc, alpha, beta, omega]^T

    Measurement:
        y = v_dc + beta
    """
    n_samples = len(v_array)
    f_est = np.empty(n_samples, dtype=np.float64)

    n = 4
    n_sigma = 2 * n + 1
    two_pi = 2.0 * math.pi

    Q = np.zeros((4, 4), dtype=np.float64)
    Q[0, 0] = q_dc
    Q[1, 1] = q_alpha
    Q[2, 2] = q_beta
    Q[3, 3] = q_omega

    sigma = np.empty((n_sigma, n), dtype=np.float64)
    sigma_pred = np.empty((n_sigma, n), dtype=np.float64)
    y_sigma = np.empty(n_sigma, dtype=np.float64)

    for i in range(n_samples):
        z = v_array[i]

        # -------------------------------------------------------------
        # 1) Sigma points
        # -------------------------------------------------------------
        _symmetrize_4x4(P)

        P_work = P.copy()
        eps = 1e-12
        for d in range(4):
            if P_work[d, d] < eps:
                P_work[d, d] = eps
            P_work[d, d] += eps

        L = np.linalg.cholesky(P_work)

        for j in range(n):
            sigma[0, j] = x[j]

        for c in range(n):
            for r in range(n):
                sigma[c + 1, r] = x[r] + gamma * L[r, c]
                sigma[c + 1 + n, r] = x[r] - gamma * L[r, c]

        # -------------------------------------------------------------
        # 2) Propagate sigma points
        # -------------------------------------------------------------
        for k in range(n_sigma):
            sigma_pred[k, :] = _process_model(sigma[k, :], dt)

        # Predicted mean
        x_pred = np.zeros(4, dtype=np.float64)
        for k in range(n_sigma):
            for j in range(n):
                x_pred[j] += wm[k] * sigma_pred[k, j]

        # Predicted covariance
        P_pred = Q.copy()
        for k in range(n_sigma):
            dx0 = sigma_pred[k, 0] - x_pred[0]
            dx1 = sigma_pred[k, 1] - x_pred[1]
            dx2 = sigma_pred[k, 2] - x_pred[2]
            dx3 = sigma_pred[k, 3] - x_pred[3]

            w = wc[k]

            P_pred[0, 0] += w * dx0 * dx0
            P_pred[0, 1] += w * dx0 * dx1
            P_pred[0, 2] += w * dx0 * dx2
            P_pred[0, 3] += w * dx0 * dx3

            P_pred[1, 0] += w * dx1 * dx0
            P_pred[1, 1] += w * dx1 * dx1
            P_pred[1, 2] += w * dx1 * dx2
            P_pred[1, 3] += w * dx1 * dx3

            P_pred[2, 0] += w * dx2 * dx0
            P_pred[2, 1] += w * dx2 * dx1
            P_pred[2, 2] += w * dx2 * dx2
            P_pred[2, 3] += w * dx2 * dx3

            P_pred[3, 0] += w * dx3 * dx0
            P_pred[3, 1] += w * dx3 * dx1
            P_pred[3, 2] += w * dx3 * dx2
            P_pred[3, 3] += w * dx3 * dx3

        _symmetrize_4x4(P_pred)

        # -------------------------------------------------------------
        # 3) Measurement prediction
        # -------------------------------------------------------------
        y_pred = 0.0
        for k in range(n_sigma):
            y_sigma[k] = _measurement_model(sigma_pred[k, :])
            y_pred += wm[k] * y_sigma[k]

        S = r_meas
        Pxy = np.zeros(4, dtype=np.float64)

        for k in range(n_sigma):
            dy = y_sigma[k] - y_pred

            dx0 = sigma_pred[k, 0] - x_pred[0]
            dx1 = sigma_pred[k, 1] - x_pred[1]
            dx2 = sigma_pred[k, 2] - x_pred[2]
            dx3 = sigma_pred[k, 3] - x_pred[3]

            w = wc[k]

            S += w * dy * dy
            Pxy[0] += w * dx0 * dy
            Pxy[1] += w * dx1 * dy
            Pxy[2] += w * dx2 * dy
            Pxy[3] += w * dx3 * dy

        if S < 1e-15:
            S = 1e-15

        # -------------------------------------------------------------
        # 4) Update
        # -------------------------------------------------------------
        K0 = Pxy[0] / S
        K1 = Pxy[1] / S
        K2 = Pxy[2] / S
        K3 = Pxy[3] / S

        innov = z - y_pred

        x[0] = x_pred[0] + K0 * innov
        x[1] = x_pred[1] + K1 * innov
        x[2] = x_pred[2] + K2 * innov
        x[3] = x_pred[3] + K3 * innov

        P[0, 0] = P_pred[0, 0] - K0 * S * K0
        P[0, 1] = P_pred[0, 1] - K0 * S * K1
        P[0, 2] = P_pred[0, 2] - K0 * S * K2
        P[0, 3] = P_pred[0, 3] - K0 * S * K3

        P[1, 0] = P_pred[1, 0] - K1 * S * K0
        P[1, 1] = P_pred[1, 1] - K1 * S * K1
        P[1, 2] = P_pred[1, 2] - K1 * S * K2
        P[1, 3] = P_pred[1, 3] - K1 * S * K3

        P[2, 0] = P_pred[2, 0] - K2 * S * K0
        P[2, 1] = P_pred[2, 1] - K2 * S * K1
        P[2, 2] = P_pred[2, 2] - K2 * S * K2
        P[2, 3] = P_pred[2, 3] - K2 * S * K3

        P[3, 0] = P_pred[3, 0] - K3 * S * K0
        P[3, 1] = P_pred[3, 1] - K3 * S * K1
        P[3, 2] = P_pred[3, 2] - K3 * S * K2
        P[3, 3] = P_pred[3, 3] - K3 * S * K3

        _symmetrize_4x4(P)

        for d in range(4):
            if P[d, d] < 1e-12:
                P[d, d] = 1e-12

        # -------------------------------------------------------------
        # 5) Frequency output from omega state
        # -------------------------------------------------------------
        f_raw = x[3] / two_pi

        if smooth_alpha > 0.0:
            f_out = (1.0 - smooth_alpha) * f_out + smooth_alpha * f_raw
        else:
            f_out = f_raw

        f_est[i] = f_out

    return f_est, x, P, f_out


class UKF_Estimator(BaseFrequencyEstimator):
    """
    Unscented Kalman Filter for single-phase voltage frequency estimation.

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
    - No Jacobian required.
    """

    name = "UKF"

    def __init__(
        self,
        nominal_f: float = 50.0,
        q_dc: float = 1e-6,
        q_alpha: float = 1e-4,
        q_beta: float = 1e-4,
        q_omega: float = 5e-2,
        r_meas: float = 1e-3,
        output_smoothing: float = 0.02,
        alpha_ut: float = 0.1,
        beta_ut: float = 2.0,
        kappa_ut: float = 0.0,
        dt: float = DT_DSP,
    ) -> None:
        self.nominal_f = float(nominal_f)
        self.q_dc = float(q_dc)
        self.q_alpha = float(q_alpha)
        self.q_beta = float(q_beta)
        self.q_omega = float(q_omega)
        self.r_meas = float(r_meas)
        self.output_smoothing = float(output_smoothing)

        self.alpha_ut = float(alpha_ut)
        self.beta_ut = float(beta_ut)
        self.kappa_ut = float(kappa_ut)

        self.dt = float(dt)
        self.w_nom = 2.0 * math.pi * self.nominal_f

        self._configure_unscented_weights()
        self.reset()

    def _configure_unscented_weights(self) -> None:
        n = 4
        lam = self.alpha_ut ** 2 * (n + self.kappa_ut) - n
        c = n + lam

        if c <= 0.0:
            raise ValueError("Invalid UKF scaling parameters: n + lambda must be > 0.")

        self.gamma = math.sqrt(c)

        self.wm = np.full(2 * n + 1, 1.0 / (2.0 * c), dtype=np.float64)
        self.wc = np.full(2 * n + 1, 1.0 / (2.0 * c), dtype=np.float64)

        self.wm[0] = lam / c
        self.wc[0] = lam / c + (1.0 - self.alpha_ut ** 2 + self.beta_ut)

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
            "nominal_f": 50.0,
            "q_dc": 1e-6,
            "q_alpha": 1e-4,
            "q_beta": 1e-4,
            "q_omega": 5e-2,
            "r_meas": 1e-3,
            "output_smoothing": 0.02,
            "alpha_ut": 0.1,
            "beta_ut": 2.0,
            "kappa_ut": 0.0,
        }

    @staticmethod
    def describe_params(params: dict[str, float]) -> str:
        return (
            f"f_nom={params.get('nominal_f', 50.0)}Hz, "
            f"qω={params.get('q_omega', 5e-2):.1e}, "
            f"R={params.get('r_meas', 1e-3):.1e}, "
            f"α={params.get('alpha_ut', 0.1)}, "
            f"β={params.get('beta_ut', 2.0)}"
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

        f_est, self.x, self.P, self.f_out = _ukf_vectorized_core(
            v_array=v_array,
            dt=self.dt,
            q_dc=self.q_dc,
            q_alpha=self.q_alpha,
            q_beta=self.q_beta,
            q_omega=self.q_omega,
            r_meas=self.r_meas,
            smooth_alpha=self.output_smoothing,
            gamma=self.gamma,
            wm=self.wm,
            wc=self.wc,
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
                raise ValueError("UKF_Estimator requires uniformly sampled time vectors.")

        self.dt = dt
        self.w_nom = 2.0 * math.pi * self.nominal_f
        self._configure_unscented_weights()
        self.reset()
        return self.step_vectorized(v)