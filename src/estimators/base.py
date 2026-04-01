from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseFrequencyEstimator(ABC):
    """
    Minimal common contract for all benchmark estimators.

    Every estimator should:
    - expose a human-readable name
    - implement step(z) -> estimated frequency in Hz
    - report its structural latency in samples
    """

    name: str = "BaseEstimator"

    @abstractmethod
    def step(self, z: float) -> float:
        """
        Process one input sample and return the current frequency estimate in Hz.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Optional state reset. Override when needed.
        """
        return None

    def structural_latency_samples(self) -> int:
        """
        Return the estimator's inherent latency in samples.
        Default = 0 for causal state-space estimators with immediate output.
        """
        return 0

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        """
        Optional: default constructor parameters for config-driven instantiation.
        """
        return {}

    @classmethod
    def tuning_grid(cls) -> list[dict[str, Any]]:
        """
        Optional: estimator-specific tuning configurations.
        Default empty list; external tuning logic may override this.
        """
        return []

    @staticmethod
    def describe_params(params: dict[str, Any]) -> str:
        """
        Optional helper for serializing tuned parameters in logs/JSON.
        """
        return ",".join(f"{k}={v}" for k, v in params.items())