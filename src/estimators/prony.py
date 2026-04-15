from __future__ import annotations

import math
import numpy as np
from numba import njit

from .base import BaseFrequencyEstimator
from .common import DT_DSP

@njit(cache=True)
def _prony_svd_core(
    buffer: np.ndarray,
    dt: float,
    L: int,
    order: int,
    f_out_prev: float,
) -> tuple[float, float]:
    
    N = len(buffer)
    if N < 2 * order:
        return f_out_prev, 0.0

    # 1. Construir matriz de Hankel
    H = np.empty((L, N - L + 1), dtype=np.float64)
    for i in range(L):
        H[i, :] = buffer[i : i + N - L + 1]

    # 2. Descomposición en Valores Singulares (SVD)
    U, s, Vh = np.linalg.svd(H, full_matrices=False)

    # Truncamiento dinámico (Noise Floor)
    threshold = 1e-6 * s[0]
    p_eff = 0
    for i in range(len(s)):
        if s[i] > threshold:
            p_eff += 1
            
    p_use = min(order, p_eff)
    if p_use < 2:
        p_use = 2

    U = U[:, :p_use]

    # 3. Shift-invariance (Mínimos Cuadrados)
    # FIX: ascontiguousarray para evitar el NumbaPerformanceWarning
    U1 = np.ascontiguousarray(U[:-1, :])
    U2 = np.ascontiguousarray(U[1:, :])

    U1_pinv = np.linalg.pinv(U1, rcond=1e-6)
    Z_mat = U1_pinv @ U2

    # 4. Encontrar eigenvalores (raíces)
    # FIX CRÍTICO: Evitar "domain change" forzando la matriz a ser compleja
    Z_mat_complex = Z_mat.astype(np.complex128)
    roots = np.linalg.eigvals(Z_mat_complex)

    # 5. Extraer frecuencias y buscar la dominante
    best_amp = -1.0
    best_f = f_out_prev

    for i in range(len(roots)):
        r = roots[i]
        mag = abs(r)
        
        if mag < 0.5 or mag > 1.5:
            continue
            
        phase = math.atan2(r.imag, r.real)
        
        if phase > 0:
            f_k = phase / (2.0 * math.pi * dt)
            if 10.0 <= f_k <= 120.0:
                amp_k = mag
                if amp_k > best_amp:
                    best_amp = amp_k
                    best_f = f_k

    return best_f, best_amp
@njit(cache=True)
def _prony_sliding_vectorized(
    v_array: np.ndarray,
    dt: float,
    window_size: int,
    L: int,
    order: int,
    f_out: float,
    buffer: np.ndarray,
    samples_seen: int,
) -> tuple[np.ndarray, float, np.ndarray, int]:
    """
    Procesa un vector de entrada utilizando una ventana deslizante.
    """
    n = len(v_array)
    f_est = np.empty(n, dtype=np.float64)

    for i in range(n):
        # Desplazar el buffer (FIFO)
        for j in range(window_size - 1):
            buffer[j] = buffer[j + 1]
        buffer[window_size - 1] = v_array[i]
        
        samples_seen += 1

        # Solo calculamos si la ventana ya se llenó
        if samples_seen >= window_size:
            best_f, _ = _prony_svd_core(buffer, dt, L, order, f_out)
            f_out = best_f
            
        f_est[i] = f_out

    return f_est, f_out, buffer, samples_seen


class Prony_Estimator(BaseFrequencyEstimator):
    """
    Estimador de Frecuencia basado en el método HSVD Prony.
    
    Características:
    - Alta resolución para estado estacionario y oscilaciones de baja frecuencia.
    - Implementa Truncamiento de Rango Dinámico para evitar divergencia con señales puras.
    - Utiliza una ventana deslizante paramétrica en Numba.
    """
    name = "Prony"

    def __init__(
        self,
        nominal_f: float = 60.0,
        order: int = 4,
        n_cycles: float = 2.0,
        dt: float = DT_DSP,
    ) -> None:
        self.nominal_f = float(nominal_f)
        self.order = int(order)
        self.n_cycles = float(n_cycles)
        self.dt = float(dt)

        # Cálculo del tamaño de la ventana
        self.window_size = int(round((self.n_cycles / self.nominal_f) / self.dt))
        if self.window_size < 10:
            self.window_size = 10  # Seguridad mínima

        # L (Pencil parameter): Usualmente se recomienda entre N/3 y N/2
        self.L = int(self.window_size // 2)

        self.reset()

    def reset(self) -> None:
        self.f_out = self.nominal_f
        self.buffer = np.zeros(self.window_size, dtype=np.float64)
        self.samples_seen = 0

    @classmethod
    def default_params(cls) -> dict[str, float]:
        return {
            "nominal_f": 60.0,
            "order": 4.0, 
            "n_cycles": 2.0,
        }

    @staticmethod
    def describe_params(params: dict[str, float]) -> str:
        return (
            f"Prony, "
            f"order={int(params.get('order', 4))}, "
            f"cycles={params.get('n_cycles', 2.0)}"
        )

    def structural_latency_samples(self) -> int:
        """
        La latencia estructural es exactamente el tamaño de la ventana.
        Antes de que se llene la ventana, el algoritmo reporta f_nom.
        """
        return self.window_size

    def step(self, z: float) -> float:
        v_array = np.array([z], dtype=np.float64)
        return float(self.step_vectorized(v_array)[0])

    def step_vectorized(self, v_array: np.ndarray) -> np.ndarray:
        v_array = np.asarray(v_array, dtype=np.float64)
        if len(v_array) == 0:
            return np.empty(0, dtype=np.float64)

        (
            f_est,
            self.f_out,
            self.buffer,
            self.samples_seen,
        ) = _prony_sliding_vectorized(
            v_array=v_array,
            dt=self.dt,
            window_size=self.window_size,
            L=self.L,
            order=self.order,
            f_out=self.f_out,
            buffer=self.buffer,
            samples_seen=self.samples_seen,
        )
        return f_est

    def estimate(self, t: np.ndarray, v: np.ndarray) -> np.ndarray:
        dt = float(t[1] - t[0])
        if dt <= 0.0:
            raise ValueError("El vector de tiempo debe ser estrictamente creciente.")
            
        self.dt = dt
        self.window_size = int(round((self.n_cycles / self.nominal_f) / self.dt))
        self.L = int(self.window_size // 2)
        
        self.reset()
        return self.step_vectorized(v)