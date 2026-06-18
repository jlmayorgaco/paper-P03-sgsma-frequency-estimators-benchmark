from __future__ import annotations

from ._experimental_base import ExperimentalFrequencyEstimator

REFERENCE_KEYS = ("kalman1960_linear_filtering", "mehra1970_adaptive_kalman")


class Adaptive_EKF_Estimator(ExperimentalFrequencyEstimator):
    name = "Adaptive-EKF"

    @classmethod
    def default_params(cls) -> dict[str, float]:
        return {"gain": 0.022, "adapt_rate": 0.01}
