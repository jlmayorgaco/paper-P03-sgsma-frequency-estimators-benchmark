from __future__ import annotations

from ._experimental_base import ExperimentalFrequencyEstimator

REFERENCE_KEYS = ("grandke1983_ipdft", "belega2008_weighted_ipdft")


class WLS_IPDFT_Estimator(ExperimentalFrequencyEstimator):
    name = "WLS-IpDFT"

    @classmethod
    def default_params(cls) -> dict[str, float]:
        return {"gain": 0.010, "window_cycles": 2.0}
