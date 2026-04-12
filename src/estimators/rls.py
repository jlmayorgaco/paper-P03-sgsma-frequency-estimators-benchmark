from __future__ import annotations

import math
import numpy as np
from numba import njit

from .base import BaseFrequencyEstimator
from .common import DT_DSP


@njit(cache=True)
def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


@njit(cache=True)
def _project_ar2_to_oscillator(
    a1: float,
    a2: float,
    r_min: float,
    r_max: float,
) -> tuple[float, float]:
    """
    Project AR(2) coefficients to a valid oscillatory region.

    For complex-conjugate poles:
        a1 = 2 r cos(w dt)
        a2 = -r^2

    Constraints:
        r in [r_min, r_max]
        |a1| <= 2 r

    This keeps the AR(2) model in a physically meaningful sinusoidal region.
    """
    if a2 >= -1e-12:
        a2 = -1e-12

    r = math.sqrt(-a2)
    r = _clamp(r, r_min, r_max)
    a2 = -(r * r)

    a1_lim = 2.0 * r * 0.999999
    a1 = _clamp(a1, -a1_lim, a1_lim)

    return a1, a2


@njit(cache=True)
def _extract_frequency_from_ar2(
    a1: float,
    a2: float,
    dt: float,
) -> float:
    """
    Extract frequency from AR(2) coefficients using both a1 and a2:

        a1 = 2 r cos(w dt)
        a2 = -r^2

    => r = sqrt(-a2)
    => cos(w dt) = a1 / (2 r)
    """
    two_pi = 2.0 * math.pi

    if a2 >= -1e-12:
        a2 = -1e-12

    r = math.sqrt(-a2)
    if r < 1e-12:
        r = 1e-12

    cos_w = a1 / (2.0 * r)
    cos_w = _clamp(cos_w, -1.0, 1.0)

    w_hat = math.acos(cos_w) / dt
    return w_hat / two_pi


@njit(cache=True)
def _rls_ar2_vectorized_core(
    v_array: np.ndarray,
    dt: float,
    is_vff: bool,
    lambda_fixed: float,
    alpha_vff: float,
    lambda_min: float,
    lambda_max: float,
    vff_beta: float,
    pole_radius_min: float,
    pole_radius_max: float,
    theta: np.ndarray,
    P: np.ndarray,
    y_km1: float,
    y_km2: float,
    err_pow: float,
    f_out: float,
    smooth_alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
    """
    RLS / VFF-RLS core for AR(2) sinusoidal tracking.

    Signal model:
        y[k] = a1 y[k-1] + a2 y[k-2] + e[k]

    Notes
    -----
    - Frequency extraction is consistent with both a1 and a2.
    - VFF adaptation is normalized by regressor energy.
    - When regressor energy is near zero (startup), VFF falls back to lambda_max
      to avoid pathological overreaction.
    - AR(2) coefficients are projected to a valid oscillatory region.
    """
    n = len(v_array)
    f_est = np.empty(n, dtype=np.float64)

    for i in range(n):
        y_k = v_array[i]

        # 1) Regressor phi = [y(k-1), y(k-2)]^T
        phi0 = y_km1
        phi1 = y_km2

        # 2) A priori prediction error
        y_pred = theta[0] * phi0 + theta[1] * phi1
        e_k = y_k - y_pred

        # 3) Forgetting factor: fixed or variable
        if is_vff:
            phi_energy = phi0 * phi0 + phi1 * phi1

            if phi_energy < 1e-10:
                lam = lambda_max
            else:
                e_norm_sq = (e_k * e_k) / (1e-10 + phi_energy)
                err_pow = (1.0 - vff_beta) * err_pow + vff_beta * e_norm_sq
                lam = 1.0 - alpha_vff * err_pow
                lam = _clamp(lam, lambda_min, lambda_max)
        else:
            lam = lambda_fixed

        # 4) RLS Kalman gain
        P_phi0 = P[0, 0] * phi0 + P[0, 1] * phi1
        P_phi1 = P[1, 0] * phi0 + P[1, 1] * phi1

        den = lam + phi0 * P_phi0 + phi1 * P_phi1
        if den < 1e-12:
            den = 1e-12

        K0 = P_phi0 / den
        K1 = P_phi1 / den

        # 5) Parameter update
        theta[0] += K0 * e_k
        theta[1] += K1 * e_k

        theta[0], theta[1] = _project_ar2_to_oscillator(
            theta[0],
            theta[1],
            pole_radius_min,
            pole_radius_max,
        )

        # 6) Covariance update
        P00 = (P[0, 0] - K0 * P_phi0) / lam
        P01 = (P[0, 1] - K0 * P_phi1) / lam
        P10 = (P[1, 0] - K1 * P_phi0) / lam
        P11 = (P[1, 1] - K1 * P_phi1) / lam

        P[0, 0] = P00
        P[0, 1] = 0.5 * (P01 + P10)
        P[1, 0] = P[0, 1]
        P[1, 1] = P11

        if P[0, 0] < 1e-12:
            P[0, 0] = 1e-12
        if P[1, 1] < 1e-12:
            P[1, 1] = 1e-12

        # 7) Frequency extraction
        f_raw = _extract_frequency_from_ar2(theta[0], theta[1], dt)

        # 8) Output smoothing
        if smooth_alpha > 0.0:
            f_out = (1.0 - smooth_alpha) * f_out + smooth_alpha * f_raw
        else:
            f_out = f_raw

        f_est[i] = f_out

        # 9) Shift memory
        y_km2 = y_km1
        y_km1 = y_k

    return f_est, theta, P, y_km1, y_km2, err_pow, f_out


class RLS_Estimator(BaseFrequencyEstimator):
    """
    Recursive Least Squares estimator using an AR(2) sinusoidal model.

    Supports:
    - Fixed-lambda RLS
    - Variable Forgetting Factor RLS (VFF-RLS)

    Model:
        y[k] = a1 y[k-1] + a2 y[k-2]

    For a sinusoid:
        a1 = 2 r cos(w dt)
        a2 = -r^2
    """

    name = "RLS"

    def __init__(
        self,
        nominal_f: float = 60.0,
        is_vff: bool = False,
        lambda_fixed: float = 0.995,
        alpha_vff: float = 0.20,
        lambda_min: float = 0.90,
        lambda_max: float = 0.9995,
        vff_beta: float = 0.02,
        output_smoothing: float = 0.03,
        pole_radius_min: float = 0.95,
        pole_radius_max: float = 1.0,
        p0: float = 0.01,
        dt: float = DT_DSP,
    ) -> None:
        self.nominal_f = float(nominal_f)
        self.is_vff = bool(is_vff)
        self.lambda_fixed = float(lambda_fixed)
        self.alpha_vff = float(alpha_vff)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.vff_beta = float(vff_beta)
        self.output_smoothing = float(output_smoothing)
        self.pole_radius_min = float(pole_radius_min)
        self.pole_radius_max = float(pole_radius_max)
        self.p0 = float(p0)
        self.dt = float(dt)

        self._validate_params()

        self.w_nom = 2.0 * math.pi * self.nominal_f
        self.reset()

    def _validate_params(self) -> None:
        if not np.isfinite(self.nominal_f) or self.nominal_f <= 0.0:
            raise ValueError("nominal_f must be a finite positive value.")
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be a finite positive value.")

        if not np.isfinite(self.lambda_fixed) or not (0.0 < self.lambda_fixed <= 1.0):
            raise ValueError("lambda_fixed must be in (0, 1].")

        if (
            not np.isfinite(self.lambda_min)
            or not np.isfinite(self.lambda_max)
            or not (0.0 < self.lambda_min <= self.lambda_max <= 1.0)
        ):
            raise ValueError("lambda_min and lambda_max must satisfy 0 < min <= max <= 1.")

        if not np.isfinite(self.alpha_vff) or self.alpha_vff < 0.0:
            raise ValueError("alpha_vff must be finite and >= 0.")
        if not np.isfinite(self.vff_beta) or not (0.0 < self.vff_beta <= 1.0):
            raise ValueError("vff_beta must be in (0, 1].")

        if (
            not np.isfinite(self.output_smoothing)
            or not (0.0 <= self.output_smoothing <= 1.0)
        ):
            raise ValueError("output_smoothing must be in [0, 1].")

        if (
            not np.isfinite(self.pole_radius_min)
            or not np.isfinite(self.pole_radius_max)
            or not (0.0 < self.pole_radius_min <= self.pole_radius_max <= 1.0)
        ):
            raise ValueError(
                "pole_radius_min and pole_radius_max must satisfy "
                "0 < min <= max <= 1."
            )

        if not np.isfinite(self.p0) or self.p0 <= 0.0:
            raise ValueError("p0 must be a finite value > 0.")

    def reset(self) -> None:
        a1_init = 2.0 * math.cos(self.w_nom * self.dt)
        a2_init = -1.0

        self.theta = np.array([a1_init, a2_init], dtype=np.float64)
        self.P = np.eye(2, dtype=np.float64) * self.p0

        self.y_km1 = 0.0
        self.y_km2 = 0.0
        self.err_pow = 0.0
        self.f_out = self.nominal_f

    @classmethod
    def default_params(cls) -> dict[str, float]:
        return {
            "nominal_f": 60.0,
            "is_vff": False,
            "lambda_fixed": 0.995,
            "alpha_vff": 0.20,
            "lambda_min": 0.90,
            "lambda_max": 0.9995,
            "vff_beta": 0.02,
            "output_smoothing": 0.03,
            "pole_radius_min": 0.95,
            "pole_radius_max": 1.0,
            "p0": 0.01, # FIX: Default conservador sincronizado
        }

    @staticmethod
    def describe_params(params: dict[str, float]) -> str:
        if params.get("is_vff", False):
            return (
                f"RLS (VFF), "
                f"f_nom={params.get('nominal_f', 60.0)}Hz, "
                f"lambda_min={params.get('lambda_min', 0.90)}, "
                f"lambda_max={params.get('lambda_max', 0.9995)}"
            )
        return (
            f"RLS (fixed λ), "
            f"f_nom={params.get('nominal_f', 60.0)}Hz, "
            f"lambda={params.get('lambda_fixed', 0.995)}"
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
        if not np.all(np.isfinite(v_array)):
            raise ValueError("v_array must contain only finite values.")

        (
            f_est,
            self.theta,
            self.P,
            self.y_km1,
            self.y_km2,
            self.err_pow,
            self.f_out,
        ) = _rls_ar2_vectorized_core(
            v_array=v_array,
            dt=self.dt,
            is_vff=self.is_vff,
            lambda_fixed=self.lambda_fixed,
            alpha_vff=self.alpha_vff,
            lambda_min=self.lambda_min,
            lambda_max=self.lambda_max,
            vff_beta=self.vff_beta,
            pole_radius_min=self.pole_radius_min,
            pole_radius_max=self.pole_radius_max,
            theta=self.theta,
            P=self.P,
            y_km1=self.y_km1,
            y_km2=self.y_km2,
            err_pow=self.err_pow,
            f_out=self.f_out,
            smooth_alpha=self.output_smoothing,
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
        if not np.all(np.isfinite(t)) or not np.all(np.isfinite(v)):
            raise ValueError("t and v must contain only finite values.")

        dt = float(t[1] - t[0])
        if dt <= 0.0:
            raise ValueError("Time vector must be strictly increasing.")

        if len(t) > 2:
            dt_all = np.diff(t)
            tol = max(1e-15, 1e-9 * abs(dt))
            if not np.all(np.abs(dt_all - dt) <= tol):
                raise ValueError("RLS_Estimator requires uniformly sampled time vectors.")

        self.dt = dt
        self.w_nom = 2.0 * math.pi * self.nominal_f
        self.reset()
        return self.step_vectorized(v)