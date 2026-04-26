from __future__ import annotations

from ._experimental_base import ExperimentalFrequencyEstimator


class Adaptive_EKF_Estimator(ExperimentalFrequencyEstimator):
    name = "Adaptive-EKF"

    @classmethod
    def default_params(cls) -> dict[str, float]:
        return {"gain": 0.022, "adapt_rate": 0.01}
