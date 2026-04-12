from __future__ import annotations

import math
import numpy as np
from numba import njit

from .base import BaseFrequencyEstimator
from .common import DT_DSP


@njit(cache=True)
def _ra_ekf_vectorized_core(
    v_array: np.ndarray,
    dt: float,
    w_nom: float,
    q_base: np.ndarray,
    r_meas: float,
    sigma_v: float,
    gamma: float,
    x: np.ndarray,
    P: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    RA-EKF (RoCoF-Augmented EKF) Core.

    Implementa el modelo cinemático de 2do orden, Innovation-Scaling y Event-Gating.

    Estado:
        x = [theta, omega, A, rocof]^T

    Modelo cinemático:
        theta[k+1] = theta[k] + omega[k]*dt + 0.5*rocof[k]*dt^2
        omega[k+1] = omega[k] + rocof[k]*dt
        A[k+1]     = A[k]
        rocof[k+1] = rocof[k]

    Medida:
        y[k] = A[k] * cos(theta[k]) + v[k]
    """
    n = len(v_array)
    f_est = np.empty(n, dtype=np.float64)
    two_pi = 2.0 * math.pi
    dt_sq_half = 0.5 * dt * dt

    w_min = two_pi * 40.0
    w_max = two_pi * 80.0

    for i in range(n):
        z = v_array[i]

        theta = x[0]
        omega = x[1]
        A = x[2]
        rocof = x[3]

        # -------------------------------------------------------------
        # 1. Predicción Cinemática de Estado
        # -------------------------------------------------------------
        x_pred = np.empty(4, dtype=np.float64)
        x_pred[0] = theta + omega * dt + rocof * dt_sq_half
        x_pred[1] = omega + rocof * dt
        x_pred[2] = A
        x_pred[3] = rocof

        # -------------------------------------------------------------
        # 2. Jacobiana F
        # -------------------------------------------------------------
        F = np.zeros((4, 4), dtype=np.float64)
        F[0, 0] = 1.0
        F[0, 1] = dt
        F[0, 3] = dt_sq_half
        F[1, 1] = 1.0
        F[1, 3] = dt
        F[2, 2] = 1.0
        F[3, 3] = 1.0

        # Predicción de la medida e Innovación
        theta_p = x_pred[0]
        A_p = x_pred[2]
        y_pred = A_p * math.cos(theta_p)
        innov = z - y_pred

        # -------------------------------------------------------------
        # 3. Innovation-Driven Covariance Scaling
        # -------------------------------------------------------------
        rk = abs(innov) / sigma_v
        q_scale = max(0.25, min(4.0, rk))

        # Propagación de Covarianza: Pp = F P F^T + Q_eff
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

        # Inyección de ruido de proceso dinámico
        for j in range(4):
            Pp[j, j] += q_base[j] * q_scale

        # Simetrización suave
        for rr in range(4):
            for cc in range(rr + 1, 4):
                sym = 0.5 * (Pp[rr, cc] + Pp[cc, rr])
                Pp[rr, cc] = sym
                Pp[cc, rr] = sym

        # -------------------------------------------------------------
        # 4. Actualización de Medida (Jacobiana H)
        # y = A * cos(theta)
        # => dy/dtheta = -A*sin(theta), dy/dA = cos(theta)
        # -------------------------------------------------------------
        H = np.zeros(4, dtype=np.float64)
        H[0] = -A_p * math.sin(theta_p)
        H[2] = math.cos(theta_p)

        # S = H Pp H^T + R
        S = 0.0
        PHt = np.empty(4, dtype=np.float64)
        for rr in range(4):
            PHt[rr] = Pp[rr, 0] * H[0] + Pp[rr, 2] * H[2]
            S += H[rr] * PHt[rr]
        S += r_meas
        if S < 1e-15:
            S = 1e-15

        # Ganancia de Kalman
        K = np.empty(4, dtype=np.float64)
        for rr in range(4):
            K[rr] = PHt[rr] / S

        # -------------------------------------------------------------
        # 5. EVENT GATING: Bloqueo de estados ante saltos de fase
        # -------------------------------------------------------------
        if abs(innov) > gamma * sigma_v:
            K[1] = 0.0  # Preserva Omega
            K[3] = 0.0  # Preserva RoCoF

        # Corrección de Estado
        for rr in range(4):
            x[rr] = x_pred[rr] + K[rr] * innov

        # Corrección de Covarianza: P = Pp - K (H Pp)
        HP = np.empty(4, dtype=np.float64)
        for cc in range(4):
            HP[cc] = H[0] * Pp[0, cc] + H[2] * Pp[2, cc]

        for rr in range(4):
            for cc in range(4):
                P[rr, cc] = Pp[rr, cc] - K[rr] * HP[cc]

        # -------------------------------------------------------------
        # 6. GLOBAL NAN/INF GUARD
        #
        # Se verifica ANTES del wrap de fase, porque el wrap con theta=NaN
        # nunca termina de converger y deja x[0] contaminado.
        # También chequeamos P: si la covarianza explota, reseteamos todo.
        # -------------------------------------------------------------
        state_ok = (
            math.isfinite(x[0])
            and math.isfinite(x[1])
            and math.isfinite(x[2])
            and math.isfinite(x[3])
        )

        cov_ok = (
            math.isfinite(P[0, 0])
            and math.isfinite(P[1, 1])
            and math.isfinite(P[2, 2])
            and math.isfinite(P[3, 3])
        )

        if (not state_ok) or (not cov_ok):
            # Rescate completo a condiciones iniciales conservadoras
            x[0] = 0.0
            x[1] = w_nom
            x[2] = 1.0
            x[3] = 0.0

            for rr in range(4):
                for cc in range(4):
                    P[rr, cc] = 0.0
            P[0, 0] = 1.0
            P[1, 1] = 10.0
            P[2, 2] = 0.1
            P[3, 3] = 1.0

        # -------------------------------------------------------------
        # 7. Salida (Envolver fase y extraer frecuencia)
        # -------------------------------------------------------------
        while x[0] > math.pi:
            x[0] -= two_pi
        while x[0] < -math.pi:
            x[0] += two_pi

        # Clamps finales de frecuencia (después del guardián,
        # entonces son seguros: x[1] ya es finito aquí)
        if x[1] < w_min:
            x[1] = w_min
        if x[1] > w_max:
            x[1] = w_max

        f_est[i] = x[1] / two_pi

    return f_est, x, P


class RAEKF_Estimator(BaseFrequencyEstimator):
    """
    RoCoF-Augmented EKF (RA-EKF).

    Arquitectura propietaria para redes de baja inercia. Logra un seguimiento
    perfecto de rampas RoCoF (vía modelo cinemático de estado) y cancela los
    falsos disparos por saltos de fase usando lógica de Event Gating no lineal.
    """

    name = "RA-EKF"

    def __init__(
        self,
        nominal_f: float = 60.0,
        q_theta: float = 1e-4,
        q_omega: float = 1e-1,
        q_A: float = 1e-4,
        q_rocof: float = 1e-2,   # Default conservador
        r_meas: float = 1e-3,
        sigma_v: float = 0.1,    # Umbral base de ruido para Gating y Scaling
        gamma: float = 2.0,      # Multiplicador del umbral (Event Gating)
        dt: float = DT_DSP,
    ) -> None:
        self.nominal_f = float(nominal_f)
        self.w_nom = 2.0 * math.pi * self.nominal_f
        self.dt = float(dt)

        self.q_base = np.array(
            [q_theta, q_omega, q_A, q_rocof],
            dtype=np.float64,
        )
        self.r_meas = float(r_meas)
        self.sigma_v = float(sigma_v)
        self.gamma = float(gamma)

        self.reset()

    def reset(self) -> None:
        # Estado inicial: fase 0, omega nominal, amplitud 1.0, rocof 0
        self.x = np.array(
            [0.0, self.w_nom, 1.0, 0.0],
            dtype=np.float64,
        )
        # Covarianza inicial conservadora (P[3,3]=1.0 en lugar de 100)
        self.P = np.diag([1.0, 10.0, 0.1, 1.0]).astype(np.float64)

    @classmethod
    def default_params(cls) -> dict[str, float]:
        return {
            "nominal_f": 60.0,
            "q_theta": 1e-4,
            "q_omega": 1e-1,
            "q_A": 1e-4,
            "q_rocof": 1e-2,
            "r_meas": 1e-3,
            "sigma_v": 0.1,
            "gamma": 2.0,
        }

    @staticmethod
    def describe_params(params: dict[str, float]) -> str:
        return (
            f"RA-EKF f_nom={params.get('nominal_f', 60.0)}Hz, "
            f"q_rocof={params.get('q_rocof', 1e-2):.1e}, "
            f"sigma_v={params.get('sigma_v', 0.1):.2f}, "
            f"gamma={params.get('gamma', 2.0):.1f}"
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

        f_est, self.x, self.P = _ra_ekf_vectorized_core(
            v_array=v_array,
            dt=self.dt,
            w_nom=self.w_nom,
            q_base=self.q_base,
            r_meas=self.r_meas,
            sigma_v=self.sigma_v,
            gamma=self.gamma,
            x=self.x,
            P=self.P,
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