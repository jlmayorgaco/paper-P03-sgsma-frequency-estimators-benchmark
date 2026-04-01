from __future__ import annotations

import math
import numpy as np
from numba import njit

from .base import BaseFrequencyEstimator
from .common import DT_DSP, F_NOM, clamp_frequency_hz


# =====================================================================
# Numba JIT-compiled core logic (Ruta Rápida)
# =====================================================================
@njit(cache=True)
def _lkf_process_vectorized(
    v_array: np.ndarray,
    x: np.ndarray,
    P: np.ndarray,
    A: np.ndarray,
    H: np.ndarray,
    Q: np.ndarray,
    R_val: float,
    f_nom: float,
    smooth_win: int,
    initialized: bool,
    theta_prev: float,
    f_prev: float,
    f_buf: np.ndarray,
    buf_idx: int,
    buf_count: int,
    dt_dsp: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, float, float, int, int]:
    
    n_samples = len(v_array)
    f_hat_array = np.empty(n_samples, dtype=np.float64)

    # Pre-computamos valores para evitar crearlos dentro del bucle
    I = np.eye(2)
    H_T = np.ascontiguousarray(H.T)

    for k in range(n_samples):
        z = v_array[k]

        # --- 1. Predict ---
        x_pred = A @ x
        P_pred = A @ P @ A.T + Q

        # --- 2. Update ---
        # H es (1,2), x_pred es (2,). El resultado es un array de 1 elemento, sacamos el float con [0]
        y_pred = (H @ x_pred)[0] 
        innovation = z - y_pred

        s_mat = H @ P_pred @ H_T
        s_val = s_mat[0, 0] + R_val
        if s_val <= 0.0:
            s_val = 1e-12

        K = (P_pred @ H_T) / s_val  # K tiene forma (2,1)
        
        # Aplanamos K a (2,) para operar elemento a elemento
        K_flat = np.empty(2, dtype=np.float64)
        K_flat[0] = K[0, 0]
        K_flat[1] = K[1, 0]
        
        x = x_pred + (K_flat * innovation)
        P = (I - K @ H) @ P_pred

        alpha = x[0]
        beta = x[1]

        # --- 3. Extracción de Frecuencia ---
        mag = math.hypot(alpha, beta)
        if mag < 1e-6:
            f_hat_array[k] = f_nom
            continue

        theta = math.atan2(beta, alpha)

        if not initialized:
            theta_prev = theta
            initialized = True
            f_hat_array[k] = f_nom
            continue

        dtheta = theta - theta_prev

        # Unwrap a [-pi, pi]
        if dtheta > math.pi:
            dtheta -= 2.0 * math.pi
        elif dtheta < -math.pi:
            dtheta += 2.0 * math.pi

        f_inst = dtheta / (2.0 * math.pi * dt_dsp)
        
        # Clamp frequency inline para evitar llamadas a funciones externas
        if f_inst < 40.0:
            f_inst = 40.0
        elif f_inst > 80.0:
            f_inst = 80.0

        theta_prev = theta

        # Estabilización anti-salto
        if abs(f_inst - f_prev) > 5.0:
            f_inst = f_prev

        # --- 4. Buffer Circular ---
        f_buf[buf_idx] = f_inst
        buf_idx = (buf_idx + 1) % smooth_win
        if buf_count < smooth_win:
            buf_count += 1

        f_prev = f_inst

        if buf_count < smooth_win:
            f_hat_array[k] = f_nom
        else:
            # Promedio de la ventana
            sum_f = 0.0
            for i in range(smooth_win):
                sum_f += f_buf[i]
            f_hat_array[k] = sum_f / smooth_win

    return f_hat_array, x, P, initialized, theta_prev, f_prev, buf_idx, buf_count


# =====================================================================
# Clase Python
# =====================================================================
class LKF_Estimator(BaseFrequencyEstimator):
    """
    Standard linear Kalman filter for narrowband sinusoidal tracking.
    Accelerated with Numba via step_vectorized().
    """

    name = "LKF"

    def __init__(
        self,
        q_val: float = 1e-4,
        r_val: float = 1e-2,
        smooth_win: int = 10,
        f_nom: float = F_NOM,
    ) -> None:
        self.q_val = float(q_val)
        self.r_val = float(r_val)
        self.smooth_win = int(smooth_win)
        self.f_nom = float(f_nom)

        w0 = 2.0 * math.pi * self.f_nom
        c = math.cos(w0 * DT_DSP)
        s = math.sin(w0 * DT_DSP)

        # Matrices base (float64 crítico para Numba)
        self.A = np.array([[c, -s], [s, c]], dtype=np.float64)
        self.H = np.array([[1.0, 0.0]], dtype=np.float64)
        self.Q = np.eye(2, dtype=np.float64) * self.q_val

        self.reset()

    def reset(self) -> None:
        self.x = np.array([0.0, 0.0], dtype=np.float64)
        self.P = np.eye(2, dtype=np.float64)
        
        self._initialized = False
        self._theta_prev = 0.0
        self._f_prev = self.f_nom
        
        # Buffer circular (reemplazo del deque)
        self._f_buf = np.zeros(self.smooth_win, dtype=np.float64)
        self._buf_idx = 0
        self._buf_count = 0

    def structural_latency_samples(self) -> int:
        return self.smooth_win

    @classmethod
    def default_params(cls) -> dict[str, float | int]:
        return {
            "q_val": 1e-4,
            "r_val": 1e-2,
            "smooth_win": 10,
            "f_nom": F_NOM,
        }

    @staticmethod
    def describe_params(params: dict[str, float | int]) -> str:
        return (
            f"Q{params.get('q_val', 1e-4)},"
            f"R{params.get('r_val', 1e-2)},"
            f"Smooth{params.get('smooth_win', 10)},"
            f"Fnom{params.get('f_nom', F_NOM)}"
        )

    def step(self, z: float) -> float:
        """
        Ruta lenta. Muestra por muestra.
        Se usa automáticamente si el motor no tiene soporte para bloques.
        """
        # Envolvemos el float en un array de tamaño 1 y llamamos a Numba
        v_array = np.array([z], dtype=np.float64)
        f_hat_array = self.step_vectorized(v_array)
        return float(f_hat_array[0])

    def step_vectorized(self, v_array: np.ndarray) -> np.ndarray:
        """
        Ruta rápida. Procesa todo un array de mediciones en milisegundos.
        """
        (
            f_hat_array,
            self.x,
            self.P,
            self._initialized,
            self._theta_prev,
            self._f_prev,
            self._buf_idx,
            self._buf_count
        ) = _lkf_process_vectorized(
            v_array=v_array.astype(np.float64),
            x=self.x,
            P=self.P,
            A=self.A,
            H=self.H,
            Q=self.Q,
            R_val=self.r_val,
            f_nom=self.f_nom,
            smooth_win=self.smooth_win,
            initialized=self._initialized,
            theta_prev=self._theta_prev,
            f_prev=self._f_prev,
            f_buf=self._f_buf,
            buf_idx=self._buf_idx,
            buf_count=self._buf_count,
            dt_dsp=DT_DSP
        )
        return f_hat_array