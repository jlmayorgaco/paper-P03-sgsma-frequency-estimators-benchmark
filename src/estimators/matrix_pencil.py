from __future__ import annotations

from ._experimental_base import ExperimentalFrequencyEstimator

REFERENCE_KEYS = ("hua1990_matrix_pencil",)


class Matrix_Pencil_Estimator(ExperimentalFrequencyEstimator):
    name = "Matrix-Pencil"

    @classmethod
    def default_params(cls) -> dict[str, float]:
        return {"gain": 0.014, "order": 4.0}
