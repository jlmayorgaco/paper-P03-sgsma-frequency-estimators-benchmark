from __future__ import annotations

import math
import numpy as np
from numba import njit

from .base import BaseFrequencyEstimator
from .common import DT_DSP


@njit(cache=True)
def _tft_vectorized_core(
    v_array: np.ndarray,
    w_nom: float,
    h0_real: np.ndarray,
    h0_imag: np.ndarray,
    h1_real: np.ndarray,
    h1_imag: np.ndarray,
    buffer: np.ndarray,
    buf_idx: int,
) -> tuple[np.ndarray, int]:
    """
    Núcleo vectorizado del Taylor-Fourier Transform (TFT) Multisignal.
    """
    n = len(v_array)
    N = len(buffer)
    f_est = np.empty(n, dtype=np.float64)
    two_pi = 2.0 * math.pi

    for i in range(n):
        buffer[buf_idx] = v_array[i]

        p0_r, p0_i = 0.0, 0.0
        p1_r, p1_i = 0.0, 0.0

        for k in range(N):
            idx = (buf_idx + 1 + k) % N
            v = buffer[idx]

            p0_r += v * h0_real[k]
            p0_i += v * h0_imag[k]
            p1_r += v * h1_real[k]
            p1_i += v * h1_imag[k]

        mag_sq = p0_r * p0_r + p0_i * p0_i
        if mag_sq > 1e-12:
            dw = (p1_i * p0_r - p1_r * p0_i) / mag_sq
        else:
            dw = 0.0

        f_est[i] = (w_nom + dw) / two_pi
        buf_idx = (buf_idx + 1) % N

    return f_est, buf_idx


class TFT_Estimator(BaseFrequencyEstimator):
    """
    Taylor-Fourier Transform (TFT) Multisignal de 2do orden.
    
    Implementa las bases conjugadas para forzar un rechazo total de la
    frecuencia negativa, eliminando el spectral leakage incluso con
    ventanas de longitud fraccionaria. (Ref: Platas-García).
    """

    name = "TFT"

    def __init__(
        self,
        nominal_f: float = 60.0,
        n_cycles: float = 2.0,
        dt: float = DT_DSP,
    ) -> None:
        self.nominal_f = float(nominal_f)
        self.n_cycles = float(n_cycles)
        self.dt = float(dt)
        self.w_nom = 2.0 * math.pi * self.nominal_f

        samples_per_cycle = (1.0 / self.nominal_f) / self.dt
        self.N = int(self.n_cycles * samples_per_cycle)
        if self.N % 2 == 0:
            self.N += 1

        self._compute_tft_weights()
        self.reset()

    def _compute_tft_weights(self) -> None:
        M = self.N // 2
        t_m = np.arange(-M, M + 1) * self.dt
        W = np.diag(np.hanning(self.N))

        # Frecuencia fundamental positiva y negativa
        ph = self.w_nom * t_m
        E_pos = np.exp(1j * ph)
        E_neg = np.exp(-1j * ph)

        # Matriz de bases Multisignal (B)
        B = np.column_stack([
            E_pos,
            t_m * E_pos,
            (t_m**2 / 2.0) * E_pos,
            E_neg,
            t_m * E_neg,
            (t_m**2 / 2.0) * E_neg
        ])

        # H = (B^H W B)^-1 B^H W
        BtWB = B.conj().T @ W @ B
        H_wls = np.linalg.inv(BtWB) @ B.conj().T @ W

        # Los renglones 0 y 1 extraen p0 y p1
        h0_c = H_wls[0, :]
        h1_c = H_wls[1, :]

        self.h0_real = np.ascontiguousarray(h0_c.real)
        self.h0_imag = np.ascontiguousarray(h0_c.imag)
        self.h1_real = np.ascontiguousarray(h1_c.real)
        self.h1_imag = np.ascontiguousarray(h1_c.imag)

    def reset(self) -> None:
        self.buffer = np.zeros(self.N, dtype=np.float64)
        self.buf_idx = 0

    @classmethod
    def default_params(cls) -> dict[str, float]:
        return {
            "nominal_f": 60.0,
            "n_cycles": 2.0,
        }

    @staticmethod
    def describe_params(params: dict[str, float]) -> str:
        return (
            f"f_nom={params.get('nominal_f', 60.0)}Hz, "
            f"Nc={params.get('n_cycles', 2.0)} cyc"
        )

    def structural_latency_samples(self) -> int:
        return self.N // 2

    def step(self, z: float) -> float:
        v_array = np.array([z], dtype=np.float64)
        return float(self.step_vectorized(v_array)[0])

    def step_vectorized(self, v_array: np.ndarray) -> np.ndarray:
        v_array = np.asarray(v_array, dtype=np.float64)
        if len(v_array) == 0:
            return np.empty(0, dtype=np.float64)

        f_est, self.buf_idx = _tft_vectorized_core(
            v_array=v_array,
            w_nom=self.w_nom,
            h0_real=self.h0_real,
            h0_imag=self.h0_imag,
            h1_real=self.h1_real,
            h1_imag=self.h1_imag,
            buffer=self.buffer,
            buf_idx=self.buf_idx,
        )
        return f_est

    def estimate(self, t: np.ndarray, v: np.ndarray) -> np.ndarray:
        dt = float(t[1] - t[0])
        self.dt = dt
        self.reset()
        return self.step_vectorized(v)