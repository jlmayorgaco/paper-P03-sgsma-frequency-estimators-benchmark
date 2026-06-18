from __future__ import annotations

from ._experimental_base import ExperimentalFrequencyEstimator

REFERENCE_KEYS = ("julier2004_unscented_filtering", "blom1988_interacting_multiple_model")


class IMM_EKF_UKF_Estimator(ExperimentalFrequencyEstimator):
    name = "IMM-EKF/UKF"

    @classmethod
    def default_params(cls) -> dict[str, float]:
        return {"gain": 0.021, "mix_prob": 0.5}
