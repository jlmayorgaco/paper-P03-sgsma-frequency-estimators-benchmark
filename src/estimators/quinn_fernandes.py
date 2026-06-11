from __future__ import annotations

from ._experimental_base import ExperimentalFrequencyEstimator

REFERENCE_KEYS = ("quinn1991_frequency_estimation",)


class Quinn_Fernandes_Estimator(ExperimentalFrequencyEstimator):
    name = "Quinn-Fernandes"

    @classmethod
    def default_params(cls) -> dict[str, float]:
        return {"gain": 0.011, "window_cycles": 2.0}
