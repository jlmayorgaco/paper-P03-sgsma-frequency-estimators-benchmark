from __future__ import annotations

from ._experimental_base import ExperimentalFrequencyEstimator


class CKF_Estimator(ExperimentalFrequencyEstimator):
    name = "CKF"

    @classmethod
    def default_params(cls) -> dict[str, float]:
        return {"gain": 0.02, "q": 1e-4, "r": 1e-3}
