from __future__ import annotations

from ._experimental_base import ExperimentalFrequencyEstimator


class Hinf_KF_Estimator(ExperimentalFrequencyEstimator):
    name = "Hinf-KF"

    @classmethod
    def default_params(cls) -> dict[str, float]:
        return {"gain": 0.017, "gamma": 1.2}
