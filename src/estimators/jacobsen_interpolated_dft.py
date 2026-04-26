from __future__ import annotations

from ._experimental_base import ExperimentalFrequencyEstimator


class Jacobsen_Interpolated_DFT_Estimator(ExperimentalFrequencyEstimator):
    name = "Jacobsen-Interpolated-DFT"

    @classmethod
    def default_params(cls) -> dict[str, float]:
        return {"gain": 0.012, "window_cycles": 2.0}
