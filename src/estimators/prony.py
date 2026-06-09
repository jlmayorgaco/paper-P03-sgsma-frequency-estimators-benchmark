from __future__ import annotations

import math

import numpy as np
from numba import njit

from .base import BaseFrequencyEstimator
from .common import DT_DSP


@njit(cache=True)
def _prony_ls_core(
    buffer: np.ndarray,
    dt: float,
    order: int,
) -> tuple[float, float]:
    """
    Deterministic least-squares Prony core.

    Numerical failures are surfaced as NaN instead of being replaced by the
    previous output. This prevents silent failure masking in benchmark outputs.
    """
    N = len(buffer)
    if N < 2 * order:
        return np.nan, 0.0

    n_rows = N - order
    if n_rows < order:
        return np.nan, 0.0

    A = np.empty((n_rows, order), dtype=np.float64)
    b = np.empty(n_rows, dtype=np.float64)
    for row in range(n_rows):
        k = row + order
        b[row] = -buffer[k]
        for col in range(order):
            A[row, col] = buffer[k - col - 1]

    if np.isnan(A).any() or np.isinf(A).any():
        return np.nan, 0.0

    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=1e-8)
    if len(coeffs) != order or np.isnan(coeffs).any() or np.isinf(coeffs).any():
        return np.nan, 0.0

    companion = np.zeros((order, order), dtype=np.float64)
    for col in range(order):
        companion[0, col] = -coeffs[col]
    for row in range(1, order):
        companion[row, row - 1] = 1.0

    roots = np.linalg.eigvals(companion.astype(np.complex128))

    best_amp = -1.0
    best_f = np.nan

    for i in range(len(roots)):
        r = roots[i]
        mag = abs(r)
        if mag < 0.5 or mag > 1.5:
            continue

        phase = math.atan2(r.imag, r.real)
        if phase <= 0.0:
            continue

        f_k = phase / (2.0 * math.pi * dt)
        if 10.0 <= f_k <= 120.0 and mag > best_amp:
            best_amp = mag
            best_f = f_k

    return best_f, best_amp


@njit(cache=True)
def _prony_sliding_vectorized(
    v_array: np.ndarray,
    dt: float,
    window_size: int,
    order: int,
    f_out: float,
    buffer: np.ndarray,
    samples_seen: int,
    stride: int,
) -> tuple[np.ndarray, float, np.ndarray, int]:
    n = len(v_array)
    f_est = np.empty(n, dtype=np.float64)

    for i in range(n):
        for j in range(window_size - 1):
            buffer[j] = buffer[j + 1]
        buffer[window_size - 1] = v_array[i]

        samples_seen += 1

        if samples_seen >= window_size and samples_seen % stride == 0:
            best_f, _best_amp = _prony_ls_core(buffer, dt, order)
            f_out = best_f if np.isfinite(best_f) else np.nan

        f_est[i] = f_out

    return f_est, f_out, buffer, samples_seen


class Prony_Estimator(BaseFrequencyEstimator):
    name = "Prony"

    def __init__(
        self,
        nominal_f: float = 60.0,
        order: int = 4,
        n_cycles: float = 2.0,
        dt: float = DT_DSP,
        execution_stride: int = 10,
    ) -> None:
        self.nominal_f = float(nominal_f)
        self.order = int(order)
        self.n_cycles = float(n_cycles)
        self.dt = float(dt)
        self.execution_stride = max(1, int(execution_stride))

        self.window_size = int(round((self.n_cycles / self.nominal_f) / self.dt))
        if self.window_size < 10:
            self.window_size = 10

        self.L = int(self.window_size // 2)
        self.reset()

    def reset(self) -> None:
        self.f_out = np.nan
        self.buffer = np.zeros(self.window_size, dtype=np.float64)
        self.samples_seen = 0

    @classmethod
    def default_params(cls) -> dict[str, float]:
        return {
            "nominal_f": 60.0,
            "order": 4.0,
            "n_cycles": 2.0,
            "execution_stride": 10.0,
        }

    @staticmethod
    def describe_params(params: dict[str, float]) -> str:
        return (
            f"Prony, "
            f"order={int(params.get('order', 4))}, "
            f"cycles={params.get('n_cycles', 2.0)}"
        )

    def structural_latency_samples(self) -> int:
        return int(math.ceil(self.window_size / self.execution_stride) * self.execution_stride)

    def step(self, z: float) -> float:
        return float(self.step_vectorized(np.array([z], dtype=np.float64))[0])

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
            order=self.order,
            f_out=self.f_out,
            buffer=self.buffer,
            samples_seen=self.samples_seen,
            stride=self.execution_stride,
        )
        return f_est

    def estimate(self, t: np.ndarray, v: np.ndarray) -> np.ndarray:
        dt = float(t[1] - t[0])
        if dt <= 0.0:
            raise ValueError("El vector de tiempo debe ser estrictamente creciente.")

        self.dt = dt
        self.window_size = int(round((self.n_cycles / self.nominal_f) / self.dt))
        if self.window_size < 10:
            self.window_size = 10
        self.L = int(self.window_size // 2)

        self.reset()
        return self.step_vectorized(v)
