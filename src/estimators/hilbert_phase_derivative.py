from __future__ import annotations

REFERENCE_KEYS = ("boashash1992_instantaneous_frequency",)

from ._experimental_base import ExperimentalFrequencyEstimator


class Hilbert_Phase_Derivative_Estimator(ExperimentalFrequencyEstimator):
    name = "Hilbert-Phase-Derivative"

    @classmethod
    def default_params(cls) -> dict[str, float]:
        return {"gain": 0.016, "phase_smoothing": 0.2}
