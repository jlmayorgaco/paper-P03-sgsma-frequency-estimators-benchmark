from __future__ import annotations

import numpy as np
from numba import njit

from .base import BaseFrequencyEstimator
from .common import DT_DSP


# =====================================================================
# Numba JIT-compiled core logic
# =====================================================================
@njit(cache=True)
def _zcd_vectorized_core(
    v_array: np.ndarray,
    dt: float,
    nominal_f: float,
    prev_v: float,
    have_prev: bool,
    sample_idx: int,
    last_cross_t: float,
    last_f: float,
) -> tuple[np.ndarray, float, bool, int, float, float]:
    """
    ZCD sample-by-sample core with:
    - positive-going zero-cross detection
    - linear interpolation for sub-sample crossing time
    - frequency = 1 / period between consecutive positive crossings
    - zero-order hold output
    """
    n = len(v_array)
    f_est = np.empty(n, dtype=np.float64)

    for i in range(n):
        z = v_array[i]

        # Primer sample: solo inicializamos estado
        if not have_prev:
            f_est[i] = last_f
            prev_v = z
            have_prev = True
            sample_idx = 1
            continue

        # Cruce positivo: prev_v <= 0 y z > 0
        if prev_v <= 0.0 and z > 0.0:
            dv = z - prev_v

            # Evitar división por cero o pendiente degenerada
            if abs(dv) > 1e-15:
                tau = -prev_v / dv  # fracción entre muestra previa y actual

                # Clip defensivo
                if tau < 0.0:
                    tau = 0.0
                elif tau > 1.0:
                    tau = 1.0

                # Tiempo exacto del cruce por interpolación lineal
                t_zc = (sample_idx - 1 + tau) * dt

                # Si ya había un cruce anterior, calculamos período/frecuencia
                if last_cross_t >= 0.0:
                    period = t_zc - last_cross_t
                    if period > 1e-12:
                        last_f = 1.0 / period

                last_cross_t = t_zc

        f_est[i] = last_f
        prev_v = z
        sample_idx += 1

    return f_est, prev_v, have_prev, sample_idx, last_cross_t, last_f


# =====================================================================
# Clase Python
# =====================================================================
class ZCDEstimator(BaseFrequencyEstimator):
    """
    Zero-Crossing Detector (ZCD) Frequency Estimator.

    Legacy baseline method:
    - positive-going zero crossings
    - linear interpolation for sub-sample timing
    - frequency from consecutive crossing intervals
    - zero-order hold output between updates
    """
    name = "ZCD"

    def __init__(self, nominal_f: float = 50.0, dt: float = DT_DSP) -> None:
        self.nominal_f = float(nominal_f)
        self.dt = float(dt)
        self.reset()

    def reset(self) -> None:
        self._prev_v = 0.0
        self._have_prev = False
        self._sample_idx = 0
        self._last_cross_t = -1.0
        self._last_f = self.nominal_f

    @classmethod
    def default_params(cls) -> dict[str, float]:
        return {"nominal_f": 50.0}

    @staticmethod
    def describe_params(params: dict[str, float]) -> str:
        return f"f_nom={params.get('nominal_f', 50.0)}Hz"

    def structural_latency_samples(self) -> int:
        """
        Approximate structural latency in samples:
        one nominal period is needed between consecutive positive crossings
        to produce an updated frequency estimate.
        """
        fs = 1.0 / self.dt
        return int(round(fs / self.nominal_f))

    def step(self, z: float) -> float:
        z = float(z)

        if not self._have_prev:
            self._prev_v = z
            self._have_prev = True
            self._sample_idx = 1
            return self._last_f

        if self._prev_v <= 0.0 and z > 0.0:
            dv = z - self._prev_v
            if abs(dv) > 1e-15:
                tau = -self._prev_v / dv
                tau = min(max(tau, 0.0), 1.0)

                t_zc = (self._sample_idx - 1 + tau) * self.dt

                if self._last_cross_t >= 0.0:
                    period = t_zc - self._last_cross_t
                    if period > 1e-12:
                        self._last_f = 1.0 / period

                self._last_cross_t = t_zc

        self._prev_v = z
        self._sample_idx += 1
        return self._last_f

    def step_vectorized(self, v_array: np.ndarray) -> np.ndarray:
        v_array = np.asarray(v_array, dtype=np.float64)

        if v_array.ndim != 1:
            raise ValueError("v_array must be a 1D array.")

        if len(v_array) == 0:
            return np.empty(0, dtype=np.float64)

        f_est, self._prev_v, self._have_prev, self._sample_idx, self._last_cross_t, self._last_f = (
            _zcd_vectorized_core(
                v_array=v_array,
                dt=self.dt,
                nominal_f=self.nominal_f,
                prev_v=self._prev_v,
                have_prev=self._have_prev,
                sample_idx=self._sample_idx,
                last_cross_t=self._last_cross_t,
                last_f=self._last_f,
            )
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

        # Validación suave de muestreo uniforme
        if len(t) > 2:
            dt_all = np.diff(t)
            tol = max(1e-15, 1e-9 * abs(dt))
            if not np.all(np.abs(dt_all - dt) <= tol):
                raise ValueError("ZCDEstimator requires uniformly sampled time vectors.")

        self.reset()
        self.dt = dt
        return self.step_vectorized(v)