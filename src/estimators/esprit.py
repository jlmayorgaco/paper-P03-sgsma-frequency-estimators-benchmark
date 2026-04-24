from __future__ import annotations

import math

import numpy as np
from numba import njit

from .base import BaseFrequencyEstimator
from .common import DT_DSP


@njit(cache=True)
def _esprit_core(buffer: np.ndarray, H: np.ndarray, dt: float) -> float:
    """
    Deterministic LS-ESPRIT core.

    The caller owns the preallocated Hankel matrix H. No random dither is added
    here because the benchmark requires stable scalar/vectorized execution paths
    given the same input sequence.
    """
    L = H.shape[0]
    M = H.shape[1]

    for i in range(L):
        for j in range(M):
            H[i, j] = buffer[i + j]

    U, _S, _Vh = np.linalg.svd(H, full_matrices=False)

    Us = np.ascontiguousarray(U[:, :2])
    Us1 = np.ascontiguousarray(Us[:-1, :])
    Us2 = np.ascontiguousarray(Us[1:, :])

    phi_mat, _, _, _ = np.linalg.lstsq(Us1, Us2, rcond=1e-6)
    eigvals = np.linalg.eigvals(phi_mat.astype(np.complex128))

    max_w = 0.0
    for ev in eigvals:
        w = abs(math.atan2(ev.imag, ev.real))
        if w > max_w:
            max_w = w

    if max_w == 0.0:
        return np.nan

    return max_w / (2.0 * math.pi * dt)


@njit(cache=True)
def _esprit_sliding_vectorized(
    v_array: np.ndarray,
    buffer: np.ndarray,
    H_buffer: np.ndarray,
    dt: float,
    f_out: float,
    samples_seen: int,
    stride: int,
) -> tuple[np.ndarray, float, int]:
    """
    Sliding-window ESPRIT with a global execution counter.

    `samples_seen` is shared between scalar and vectorized paths so that
    decimation decisions are identical regardless of how the estimator is
    driven.
    """
    n = len(v_array)
    f_est = np.empty(n, dtype=np.float64)
    window_size = len(buffer)

    for i in range(n):
        for j in range(window_size - 1):
            buffer[j] = buffer[j + 1]
        buffer[window_size - 1] = v_array[i]

        samples_seen += 1

        if (
            samples_seen >= window_size
            and samples_seen % stride == 0
            and abs(v_array[i]) > 1e-4
        ):
            val = _esprit_core(buffer, H_buffer, dt)
            if not np.isnan(val) and 40.0 < val < 80.0:
                f_out = val

        f_est[i] = f_out

    return f_est, f_out, samples_seen


class ESPRIT_Estimator(BaseFrequencyEstimator):
    """ESPRIT estimator with deterministic scalar/vectorized behavior."""

    name = "ESPRIT"

    def __init__(
        self,
        nominal_f: float = 60.0,
        n_cycles: float = 1.0,
        dt: float = DT_DSP,
        execution_stride: int = 10,
    ) -> None:
        self.nominal_f = float(nominal_f)
        self.n_cycles = float(n_cycles)
        self.dt = float(dt)
        self.execution_stride = max(1, int(execution_stride))
        self._update_window_geometry()
        self.reset()

    def _update_window_geometry(self) -> None:
        self.N = int(round((self.n_cycles / self.nominal_f) / self.dt))
        self.N = max(self.N, 4)
        self.L = self.N // 2
        self.M = self.N - self.L + 1

    def reset(self) -> None:
        self.buffer = np.zeros(self.N, dtype=np.float64)
        self.H_buffer = np.zeros((self.L, self.M), dtype=np.float64)
        self.f_out = np.nan
        self.samples_seen = 0

    @classmethod
    def default_params(cls) -> dict[str, float]:
        return {
            "nominal_f": 60.0,
            "n_cycles": 1.0,
            "execution_stride": 10.0,
        }

    @staticmethod
    def describe_params(params: dict[str, float]) -> str:
        return f"ESPRIT f_nom={params.get('nominal_f', 60.0)}Hz"

    def structural_latency_samples(self) -> int:
        return self.N // 2

    def step(self, z: float) -> float:
        return float(self.step_vectorized(np.array([z], dtype=np.float64))[0])

    def step_vectorized(self, v_array: np.ndarray) -> np.ndarray:
        v_array = np.asarray(v_array, dtype=np.float64)
        if len(v_array) == 0:
            return np.empty(0, dtype=np.float64)

        f_est, self.f_out, self.samples_seen = _esprit_sliding_vectorized(
            v_array=v_array,
            buffer=self.buffer,
            H_buffer=self.H_buffer,
            dt=self.dt,
            f_out=self.f_out,
            samples_seen=self.samples_seen,
            stride=self.execution_stride,
        )
        return f_est

    def estimate(self, t: np.ndarray, v: np.ndarray) -> np.ndarray:
        self.dt = float(t[1] - t[0])
        self._update_window_geometry()
        self.reset()
        return self.step_vectorized(v)
