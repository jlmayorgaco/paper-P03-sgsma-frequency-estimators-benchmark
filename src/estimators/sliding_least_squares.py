from __future__ import annotations

REFERENCE_KEYS = ("besson1999_nonlinear_least_squares",)

from ._experimental_base import ExperimentalFrequencyEstimator


class Sliding_Least_Squares_Estimator(ExperimentalFrequencyEstimator):
    name = "Sliding-Least-Squares"

    @classmethod
    def default_params(cls) -> dict[str, float]:
        return {"gain": 0.009, "window_cycles": 3.0}
