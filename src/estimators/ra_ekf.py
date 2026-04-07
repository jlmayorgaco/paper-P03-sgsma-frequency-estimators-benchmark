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
    q_base: np.ndarray,
    r_meas: float,
    sigma_v: float,
    gamma: float,
    x: np.ndarray,
    P: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    RA-EKF (RoCoF-Augmented EKF) Core.
    Implementa el modelo cinemático de 2do orden, Innovation-Scaling y Event-Gating.
    """
    n = len(v_array)
    f_est = np.empty(n, dtype=np.float64)
    two_pi = 2.0 * math.pi
    dt_sq_half = 0.5 * dt * dt

    for i in range(n):
        z = v_array[i]

        theta = x[0]
        omega = x[1]
        A = x[2]
        rocof = x[3]

        # 1. Predicción Cinemática de Estado (Ec. del paper)
        x_pred = np.empty(4, dtype=np.float64)
        x_pred[0] = theta + omega * dt + rocof * dt_sq_half
        x_pred[1] = omega + rocof * dt
        x_pred[2] = A
        x_pred[3] = rocof

        # 2. Jacobiana F
        F = np.zeros((4, 4), dtype=np.float64)
        F[0, 0] = 1.0; F[0, 1] = dt; F[0, 3] = dt_sq_half
        F[1, 1] = 1.0; F[1, 3] = dt
        F[2, 2] = 1.0
        F[3, 3] = 1.0

        # Predicción de la medida e Innovación
        theta_p = x_pred[0]
        A_p = x_pred[2]
        y_pred = A_p * math.cos(theta_p)
        innov = z - y_pred

        # 3. Innovation-Driven Covariance Scaling
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

        # 4. Actualización de Medida (Jacobiana H)
        # y = A * cos(theta) => dy/dtheta = -A*sin(theta), dy/dA = cos(theta)
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
        if S < 1e-15: S = 1e-15

        # Ganancia de Kalman
        K = np.empty(4, dtype=np.float64)
        for rr in range(4):
            K[rr] = PHt[rr] / S

        # 5. EVENT GATING: Bloqueo de estados ante saltos de fase
        is_gating = False
        if abs(innov) > gamma * sigma_v:
            K[1] = 0.0  # Preserva Omega
            K[3] = 0.0  # Preserva RoCoF
            is_gating = True

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

        # 6. Salida (Envolver fase y extraer frecuencia)
        while x[0] > math.pi: x[0] -= two_pi
        while x[0] < -math.pi: x[0] += two_pi

        # Si el salto de fase fue brutal, limitar omega artificialmente por un paso
        if x[1] < two_pi * 40.0: x[1] = two_pi * 40.0
        if x[1] > two_pi * 80.0: x[1] = two_pi * 80.0

        f_est[i] = x[1] / two_pi

    return f_est, x

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
        q_rocof: float = 5e0,
        r_meas: float = 1e-3,
        sigma_v: float = 0.1,  # Umbral base de ruido para Gating y Scaling
        gamma: float = 2.0,    # Multiplicador del umbral (Event Gating)
        dt: float = DT_DSP,
    ) -> None:
        self.nominal_f = float(nominal_f)
        self.w_nom = 2.0 * math.pi * self.nominal_f
        self.dt = float(dt)
        
        self.q_base = np.array([q_theta, q_omega, q_A, q_rocof], dtype=np.float64)
        self.r_meas = float(r_meas)
        self.sigma_v = float(sigma_v)
        self.gamma = float(gamma)

        self.reset()

    def reset(self) -> None:
        self.x = np.array([0.0, self.w_nom, 1.0, 0.0], dtype=np.float64)
        self.P = np.diag([1.0, 10.0, 0.1, 100.0]).astype(np.float64)

    @classmethod
    def default_params(cls) -> dict[str, float]:
        return {"nominal_f": 60.0}

    @staticmethod
    def describe_params(params: dict[str, float]) -> str:
        return f"RA-EKF f_nom={params.get('nominal_f', 60.0)}Hz"

    def structural_latency_samples(self) -> int:
        return 0

    def step(self, z: float) -> float:
        return float(self.step_vectorized(np.array([z], dtype=np.float64))[0])

    def step_vectorized(self, v_array: np.ndarray) -> np.ndarray:
        f_est, self.x = _ra_ekf_vectorized_core(
            v_array=v_array.astype(np.float64),
            dt=self.dt,
            q_base=self.q_base,
            r_meas=self.r_meas,
            sigma_v=self.sigma_v,
            gamma=self.gamma,
            x=self.x,
            P=self.P,
        )
        return f_est

    def estimate(self, t: np.ndarray, v: np.ndarray) -> np.ndarray:
        self.dt = float(t[1] - t[0])
        self.reset()
        return self.step_vectorized(v)