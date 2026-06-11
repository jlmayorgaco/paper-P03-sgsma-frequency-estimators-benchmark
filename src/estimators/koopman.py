from __future__ import annotations

import math
import numpy as np
from numba import njit

from .base import BaseFrequencyEstimator
from .common import DT_DSP

REFERENCE_KEYS = ("williams2015_edmd_koopman",)

@njit(cache=True)
def _koopman_edmd_core(buffer: np.ndarray, dt: float) -> float:
    """
    Aproximación de Koopman mediante Hankel-DMD (Robust Koopman).
    Utiliza SVD para aislar el subespacio de señal antes de derivar la dinámica.
    """
    N = len(buffer)
    # T-102: deterministic dither replaces np.random.standard_normal (Numba's
    # unseeded per-thread RNG, making MC runs non-reproducible).
    buf_work = buffer + 1e-9 * (np.arange(N, dtype=np.float64) / N - 0.5)
    
    # 1. Matriz de Observables Profunda (Hankel)
    L = N // 2
    M = N - L + 1
    H = np.zeros((L, M), dtype=np.float64)
    for i in range(L):
        for j in range(M):
            H[i, j] = buf_work[i + j]

    # 2. Matrices desplazadas en el tiempo (Estado Actual X, Estado Futuro Y)
    X = H[:, :-1]
    Y = H[:, 1:]

    # 3. Truncamiento Robusto vía SVD (Aislamiento de la señal pura)
    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    
    # Añadimos .copy() para forzar matrices contiguas y eliminar el warning de Numba
    Ur_T = np.ascontiguousarray(U[:, :2].T)
    Sr_inv = np.diag(1.0 / S[:2])
    Vr = np.ascontiguousarray(Vh[:2, :].T)
    Y_contig = np.ascontiguousarray(Y)
    
    # 4. Operador de Koopman Reducido (K_tilde)
    K_tilde = Ur_T @ Y_contig @ Vr @ Sr_inv

    # 5. Autovalores del Operador Dinámico
    K_complex = K_tilde.astype(np.complex128)
    eigvals = np.linalg.eigvals(K_complex)
    
    # 6. Extracción de frecuencia de los modos de Koopman
    # FIX: Exponer fallo si no encuentra un modo físico válido
    best_f = np.nan 
    max_energy = -1.0
    two_pi_dt = 2.0 * math.pi * dt
    
    for ev in eigvals:
        w = abs(math.atan2(ev.imag, ev.real))
        f_hz = w / two_pi_dt
        
        if 40.0 < f_hz < 80.0:
            # Seleccionamos el polo más cercano al círculo unitario
            mag = abs(ev)
            energy_score = 1.0 / (abs(mag - 1.0) + 1e-4)
            
            if energy_score > max_energy:
                max_energy = energy_score
                best_f = f_hz
                
    return best_f

class Koopman_Estimator(BaseFrequencyEstimator):
    """
    Estimador RK-DPMU (Robust Koopman).
    Aproxima la dinámica no lineal mediante Extended Dynamic Mode Decomposition.
    Especializado en transitorios complejos de baja inercia.
    """
    name = "Koopman (RK-DPMU)"

    def __init__(
        self,
        nominal_f: float = 60.0,
        n_cycles: float = 1.5,
        dt: float = DT_DSP,
        execution_stride: int = 10,
    ) -> None:
        self.nominal_f = float(nominal_f)
        self.n_cycles = float(n_cycles)
        self.dt = float(dt)
        self.execution_stride = max(1, int(execution_stride))
        # Koopman requiere un poco más de datos que ESPRIT para construir 
        # el mapeo Y = K*X robustamente. 1.5 ciclos es el punto dulce.
        self.N = int(round((1.0 / self.nominal_f) / self.dt * self.n_cycles))
        self.reset()

    def reset(self) -> None:
        self.buffer = np.zeros(self.N, dtype=np.float64)
        # FIX: Arrancar a ciegas para evitar dar la respuesta nominal como un falso positivo
        self.f_out = np.nan

        # Métricas de diagnóstico
        self._valid_updates = 0
        self._total_calls = 0
        self.samples_seen = 0
        # T-101: both paths use samples_seen and execution_stride, so scalar
        # and vectorized decimation decisions stay identical.
        self._step_counter = 0

    @classmethod
    def default_params(cls) -> dict[str, float]:
        return {
            "nominal_f": 60.0,
            "n_cycles": 1.5,
            "execution_stride": 10.0,
        }

    @staticmethod
    def describe_params(params: dict[str, float]) -> str:
        return f"Koopman f_nom={params.get('nominal_f', 60.0)}Hz"

    def structural_latency_samples(self) -> int:
        return int(math.ceil(self.N / self.execution_stride) * self.execution_stride)

    def step(self, z: float) -> float:
        # T-101: mirrors the stride policy of step_vectorized().
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = z
        self.samples_seen += 1

        if (
            self.samples_seen >= self.N
            and self.samples_seen % self.execution_stride == 0
            and np.abs(z) > 1e-3
        ):
            self._total_calls += 1
            try:
                val = _koopman_edmd_core(self.buffer, self.dt)
                if not np.isnan(val) and 40.0 < val < 80.0:
                    self.f_out = val
                    self._valid_updates += 1
            except Exception:
                pass
        self._step_counter += 1
        return self.f_out

    def step_vectorized(self, v_array: np.ndarray) -> np.ndarray:
        # T-101: uses the same sample counter as step() for the stride policy.
        n = len(v_array)
        f_est = np.empty(n, dtype=np.float64)

        for i in range(n):
            z = v_array[i]
            self.buffer[:-1] = self.buffer[1:]
            self.buffer[-1] = z
            self.samples_seen += 1

            # Decimación del cálculo pesado (cada 10 muestras = 1ms)
            if (
                self.samples_seen >= self.N
                and self.samples_seen % self.execution_stride == 0
                and np.abs(z) > 1e-3
            ):
                self._total_calls += 1
                try:
                    val = _koopman_edmd_core(self.buffer, self.dt)
                    if not np.isnan(val) and 40.0 < val < 80.0:
                        self.f_out = val
                        self._valid_updates += 1
                except Exception:
                    pass
            self._step_counter += 1
            f_est[i] = self.f_out

        return f_est

    def estimate(self, t: np.ndarray, v: np.ndarray) -> np.ndarray:
        self.dt = float(t[1] - t[0])
        self.reset()
        return self.step_vectorized(v)
