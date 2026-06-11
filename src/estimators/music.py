from __future__ import annotations

import math

import numpy as np
from numba import njit

from .base import BaseFrequencyEstimator
from .common import DT_DSP

REFERENCE_KEYS = ("schmidt1986_music",)


@njit(cache=True)
def _music_eval_spectrum(
    Un: np.ndarray,
    f_center: float,
    span: float,
    step: float,
    dt: float,
) -> tuple[float, float]:
    """Evaluate the MUSIC pseudo-spectrum around a local frequency band."""
    if span <= 0.0 or step <= 0.0:
        return np.nan, np.inf

    best_f = np.nan
    min_val = 1e18
    found_valid = False

    two_pi_dt = 2.0 * math.pi * dt
    L = Un.shape[0]
    K = Un.shape[1]

    f_start = f_center - span
    f_end = f_center + span + (step * 0.5)

    for f in np.arange(f_start, f_end, step):
        omega = f * two_pi_dt
        val = 0.0

        for col in range(K):
            dot_real = 0.0
            dot_imag = 0.0
            for row in range(L):
                c = math.cos(omega * row)
                s = -math.sin(omega * row)

                u_r = Un[row, col].real
                u_i = Un[row, col].imag

                dot_real += c * u_r - s * u_i
                dot_imag += c * u_i + s * u_r

            val += dot_real * dot_real + dot_imag * dot_imag

        if math.isfinite(val) and val < min_val:
            min_val = val
            best_f = f
            found_valid = True

    if not found_valid:
        return np.nan, np.inf

    return best_f, min_val


@njit(cache=True)
def _music_core(
    buffer: np.ndarray,
    dt: float,
    f_nom: float,
    signal_order: int,
    search_span_hz: float,
    coarse_step_hz: float,
    fine_span_hz: float,
    fine_step_hz: float,
    ultra_span_hz: float,
    ultra_step_hz: float,
) -> float:
    """Core localized MUSIC estimator."""
    N = len(buffer)

    # Deterministic dither prevents singular perfect-sine cases without
    # introducing run-to-run randomness.
    buf_work = buffer.copy()
    for k in range(N):
        buf_work[k] += 1e-12 * (k + 1)

    L = N // 2
    M = N - L + 1
    H = np.zeros((L, M), dtype=np.float64)
    for i in range(L):
        for j in range(M):
            H[i, j] = buf_work[i + j]

    U, _, _ = np.linalg.svd(H)

    order = max(1, int(signal_order))
    if U.shape[1] <= order:
        return np.nan

    Un = U[:, order:].astype(np.complex128)

    f1, _ = _music_eval_spectrum(Un, f_nom, search_span_hz, coarse_step_hz, dt)
    if math.isnan(f1):
        return np.nan

    f2, _ = _music_eval_spectrum(Un, f1, fine_span_hz, fine_step_hz, dt)
    if math.isnan(f2):
        return np.nan

    f_final, _ = _music_eval_spectrum(Un, f2, ultra_span_hz, ultra_step_hz, dt)
    if math.isnan(f_final):
        return np.nan

    return f_final


class MUSIC_Estimator(BaseFrequencyEstimator):
    """MUSIC spectral estimator using a local three-stage frequency search."""

    name = "MUSIC"

    def __init__(
        self,
        nominal_f: float = 60.0,
        n_cycles: float = 1.0,
        signal_order: int = 2,
        update_decimation: int = 40,
        search_span_hz: float = 20.0,
        coarse_step_hz: float = 1.0,
        fine_span_hz: float = 1.0,
        fine_step_hz: float = 0.05,
        ultra_span_hz: float = 0.05,
        ultra_step_hz: float = 0.01,
        dt: float = DT_DSP,
    ) -> None:
        self.nominal_f = float(nominal_f)
        self.dt = float(dt)
        self.n_cycles = float(n_cycles)
        self.signal_order = int(signal_order)
        self.update_decimation = max(1, int(update_decimation))
        self.search_span_hz = float(search_span_hz)
        self.coarse_step_hz = float(coarse_step_hz)
        self.fine_span_hz = float(fine_span_hz)
        self.fine_step_hz = float(fine_step_hz)
        self.ultra_span_hz = float(ultra_span_hz)
        self.ultra_step_hz = float(ultra_step_hz)
        self.N = max(8, int(round((1.0 / self.nominal_f) / self.dt * self.n_cycles)))
        self.reset()

    def reset(self) -> None:
        self.buffer = np.zeros(self.N, dtype=np.float64)
        self.f_out = np.nan
        self._valid_updates = 0
        self._total_calls = 0
        self._sample_index = 0

    @classmethod
    def default_params(cls) -> dict[str, float | int]:
        return {
            "nominal_f": 60.0,
            "n_cycles": 1.0,
            "signal_order": 2,
            "update_decimation": 40,
            "search_span_hz": 20.0,
            "coarse_step_hz": 1.0,
            "fine_span_hz": 1.0,
            "fine_step_hz": 0.05,
            "ultra_span_hz": 0.05,
            "ultra_step_hz": 0.01,
        }

    @staticmethod
    def describe_params(params: dict[str, float | int]) -> str:
        return (
            f"MUSIC f_nom={params.get('nominal_f', 60.0)}Hz, "
            f"Nc={params.get('n_cycles', 1.0)}, "
            f"order={params.get('signal_order', 2)}, "
            f"decim={params.get('update_decimation', 40)}"
        )

    def structural_latency_samples(self) -> int:
        return self.N // 2 + max(0, self.update_decimation - 1)

    def _process_sample(self, z: float) -> float:
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = z

        should_update = (
            self._sample_index % self.update_decimation == 0
            and abs(z) > 1e-4
        )
        if should_update:
            self._total_calls += 1
            try:
                val = _music_core(
                    self.buffer,
                    self.dt,
                    self.nominal_f,
                    self.signal_order,
                    self.search_span_hz,
                    self.coarse_step_hz,
                    self.fine_span_hz,
                    self.fine_step_hz,
                    self.ultra_span_hz,
                    self.ultra_step_hz,
                )
                if not np.isnan(val) and 40.0 < val < 80.0:
                    self.f_out = val
                    self._valid_updates += 1
            except Exception:
                pass

        self._sample_index += 1
        return self.f_out

    def step(self, z: float, t: float | None = None, mem: object | None = None) -> float:
        return self._process_sample(float(z))

    def step_vectorized(self, v_array: np.ndarray) -> np.ndarray:
        f_est = np.empty(len(v_array), dtype=np.float64)
        for i in range(len(v_array)):
            f_est[i] = self._process_sample(float(v_array[i]))
        return f_est

    def estimate(self, t: np.ndarray, v: np.ndarray) -> np.ndarray:
        self.dt = float(t[1] - t[0])
        self.reset()
        return self.step_vectorized(v)
