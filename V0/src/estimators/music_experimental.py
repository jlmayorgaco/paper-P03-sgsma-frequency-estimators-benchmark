from __future__ import annotations

from ._experimental_base import ExperimentalFrequencyEstimator


class MUSIC_Experimental_Estimator(ExperimentalFrequencyEstimator):
    name = "MUSIC"

    @classmethod
    def default_params(cls) -> dict[str, float]:
        return {"gain": 0.013, "subspace_order": 6.0}
