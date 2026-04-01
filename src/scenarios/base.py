from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from estimators.common import F_NOM


@dataclass
class ScenarioData:
    """
    Standard container for scenario outputs produced by any benchmark scenario.
    """

    name: str
    t: np.ndarray
    v: np.ndarray
    f_true: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """
        Basic structural validation to ensure the scenario output is usable
        by the benchmark runner.
        """
        if not isinstance(self.t, np.ndarray):
            raise TypeError("t must be a numpy.ndarray")
        if not isinstance(self.v, np.ndarray):
            raise TypeError("v must be a numpy.ndarray")
        if not isinstance(self.f_true, np.ndarray):
            raise TypeError("f_true must be a numpy.ndarray")

        if self.t.ndim != 1:
            raise ValueError("t must be 1D")
        if self.v.ndim != 1:
            raise ValueError("v must be 1D")
        if self.f_true.ndim != 1:
            raise ValueError("f_true must be 1D")

        n = len(self.t)
        if len(self.v) != n:
            raise ValueError("v must have same length as t")
        if len(self.f_true) != n:
            raise ValueError("f_true must have same length as t")

        if n < 2:
            raise ValueError("scenario must contain at least 2 samples")

        dt = np.diff(self.t)
        if np.any(dt <= 0):
            raise ValueError("t must be strictly increasing")


class Scenario(ABC):
    """
    Abstract base class for all benchmark scenarios.

    Each scenario must define:
    - SCENARIO_NAME
    - DEFAULT_PARAMS
    - MONTE_CARLO_SPACE
    - generate(...)
    """

    SCENARIO_NAME: ClassVar[str] = "base"
    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {
        "duration_s": 1.5,
        "amplitude": 1.0,
        "freq_hz": F_NOM,
        "phase_rad": 0.0,
        "noise_sigma": 0.0,
        "seed": None,
    }
    MONTE_CARLO_SPACE: ClassVar[dict[str, Any]] = {}

    @classmethod
    def get_name(cls) -> str:
        return cls.SCENARIO_NAME

    @classmethod
    def get_default_params(cls) -> dict[str, Any]:
        return dict(cls.DEFAULT_PARAMS)

    @classmethod
    def get_monte_carlo_space(cls) -> dict[str, Any]:
        return dict(cls.MONTE_CARLO_SPACE)

    @classmethod
    def build_params(cls, **overrides: Any) -> dict[str, Any]:
        """
        Merge default parameters with user overrides.
        """
        params = cls.get_default_params()
        params.update(overrides)
        cls.validate_params(params)
        return params

    @classmethod
    def validate_params(cls, params: dict[str, Any]) -> None:
        """
        Optional per-scenario parameter validation.
        Subclasses may override this.
        """
        if params["duration_s"] <= 0:
            raise ValueError("duration_s must be > 0")
        if params["amplitude"] <= 0:
            raise ValueError("amplitude must be > 0")
        if params["freq_hz"] <= 0:
            raise ValueError("freq_hz must be > 0")
        if params["noise_sigma"] < 0:
            raise ValueError("noise_sigma must be >= 0")

    @classmethod
    @abstractmethod
    def generate(cls, **params: Any) -> ScenarioData:
        """
        Generate the scenario output.
        Must return a validated ScenarioData instance.
        """
        raise NotImplementedError

    @classmethod
    def run(cls, **overrides: Any) -> ScenarioData:
        """
        High-level helper used by runners:
        - merge defaults
        - validate params
        - generate scenario
        - validate output
        """
        params = cls.build_params(**overrides)
        data = cls.generate(**params)

        if data.name != cls.SCENARIO_NAME:
            data.name = cls.SCENARIO_NAME

        data.validate()
        return data