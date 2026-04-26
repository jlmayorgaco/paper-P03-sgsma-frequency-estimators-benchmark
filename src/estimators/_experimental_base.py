from __future__ import annotations

import numpy as np


class ExperimentalFrequencyEstimator:
    """Small compatibility base for experimental estimators.

    These estimators expose a simple `step` and `step_vectorized` API so they can
    be promoted to active registry after dedicated validation/tuning.
    """

    name = "EXPERIMENTAL"

    @classmethod
    def default_params(cls) -> dict[str, float]:
        return {}

    def __init__(self, fs: float = 10_000.0, f_nominal: float = 60.0, **params: float):
        self.fs = float(fs)
        self.f_nominal = float(f_nominal)
        self.params = {**self.default_params(), **params}
        self._last = self.f_nominal
        self._sample_idx = 0
        self._last_crossing: float | None = None
        self._prev_lp: float | None = None
        self._half_periods: list[float] = []

        gain = abs(float(self.params.get("gain", 0.01)))
        aux = self._aux_param_value()
        self._alpha = float(np.clip(0.04 + 6.0 * gain, 0.04, 0.35))
        self._lp_beta = float(np.clip(0.12 + 0.04 * abs(aux), 0.08, 0.45))
        self._hysteresis = float(np.clip(0.004 + 0.6 * gain, 0.002, 0.03))
        self._window = int(np.clip(self.params.get("crossing_window", 6), 3, 20))

        self._min_half_period = float(self.fs / (2.0 * 120.0))
        self._max_half_period = float(self.fs / (2.0 * 20.0))

    def _aux_param_value(self) -> float:
        for k, v in self.params.items():
            if k != "gain":
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
        return 0.0

    def _lowpass(self, x: float) -> float:
        if self._prev_lp is None:
            return float(x)
        beta = self._lp_beta
        return float((1.0 - beta) * self._prev_lp + beta * x)

    def _crossing_position(self, prev_x: float, cur_x: float) -> float | None:
        h = self._hysteresis
        crossed = (prev_x * cur_x) <= 0.0 and max(abs(prev_x), abs(cur_x)) >= h
        if not crossed:
            return None

        den = cur_x - prev_x
        if abs(den) < 1e-12:
            frac = 0.5
        else:
            frac = float(np.clip((-prev_x) / den, 0.0, 1.0))
        return float((self._sample_idx - 1) + frac)

    def _estimate_scalar(self, x: float) -> float:
        x_lp = self._lowpass(float(x))
        if self._prev_lp is None:
            self._prev_lp = x_lp
            self._sample_idx += 1
            return float(self._last)

        crossing = self._crossing_position(self._prev_lp, x_lp)
        if crossing is not None:
            if self._last_crossing is not None:
                half_period = float(crossing - self._last_crossing)
                if self._min_half_period <= half_period <= self._max_half_period:
                    self._half_periods.append(half_period)
                    if len(self._half_periods) > self._window:
                        self._half_periods.pop(0)
                    hp_ref = float(np.median(self._half_periods))
                    f_inst = float(self.fs / (2.0 * hp_ref))
                    self._last = float((1.0 - self._alpha) * self._last + self._alpha * f_inst)
            self._last_crossing = crossing

        self._prev_lp = x_lp
        self._sample_idx += 1
        return self._last

    def step(self, x: float) -> float:
        return self._estimate_scalar(float(x))

    def step_vectorized(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        out = np.empty_like(x_arr, dtype=float)
        for i, v in enumerate(x_arr):
            out[i] = self._estimate_scalar(float(v))
        return out
