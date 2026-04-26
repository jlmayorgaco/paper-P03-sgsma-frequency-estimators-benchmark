from __future__ import annotations

from ._experimental_base import ExperimentalFrequencyEstimator


class SR_UKF_Estimator(ExperimentalFrequencyEstimator):
    name = "SR-UKF"

    @classmethod
    def default_params(cls) -> dict[str, float]:
        return {"gain": 0.018, "alpha": 1e-3, "beta": 2.0}
