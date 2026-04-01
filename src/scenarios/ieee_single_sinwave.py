from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np

from .base import Scenario, ScenarioData
from estimators.common import FS_PHYSICS, F_NOM


class IEEESingleSinWaveScenario(Scenario):
    """
    Baseline single-tone sinusoidal scenario.

    Intended for:
    - estimator smoke tests
    - sanity validation
    - clean tracking checks
    - baseline debugging
    """

    SCENARIO_NAME: ClassVar[str] = "IEEE_Single_SinWave"

    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {
        "duration_s": 1.5,
        "amplitude": 1.0,
        "freq_hz": F_NOM,
        "phase_rad": 0.0,
        "noise_sigma": 0.0,
        "seed": None,
    }

    MONTE_CARLO_SPACE: ClassVar[dict[str, Any]] = {
        "amplitude": {
            "kind": "uniform",
            "low": 0.9,
            "high": 1.1,
        },
        "freq_hz": {
            "kind": "uniform",
            "low": 59.5,
            "high": 60.5,
        },
        "phase_rad": {
            "kind": "uniform",
            "low": 0.0,
            "high": 2.0 * math.pi,
        },
        "noise_sigma": {
            "kind": "uniform",
            "low": 0.0,
            "high": 0.02,
        },
    }

    @classmethod
    def validate_params(cls, params: dict[str, Any]) -> None:
        super().validate_params(params)

        if params["duration_s"] <= 0:
            raise ValueError("duration_s must be > 0")
        if params["amplitude"] < 0:
            raise ValueError("amplitude must be >= 0")
        if params["freq_hz"] <= 0:
            raise ValueError("freq_hz must be > 0")
        if params["noise_sigma"] < 0:
            raise ValueError("noise_sigma must be >= 0")

    @classmethod
    def generate(
        cls,
        duration_s: float = 1.5,
        amplitude: float = 1.0,
        freq_hz: float = F_NOM,
        phase_rad: float = 0.0,
        noise_sigma: float = 0.0,
        seed: int | None = None,
    ) -> ScenarioData:
        """
        Generate a clean single-tone sinusoidal scenario.
        """
        rng = np.random.default_rng(seed)

        t = np.arange(0.0, duration_s, 1.0 / FS_PHYSICS, dtype=float)
        f_true = np.full_like(t, freq_hz, dtype=float)

        phi = 2.0 * math.pi * freq_hz * t + phase_rad
        v = amplitude * np.sin(phi)

        if noise_sigma > 0.0:
            v = v + rng.normal(0.0, noise_sigma, size=t.shape)

        meta = {
            "description": "IEEE_Single_SinWave — clean constant-frequency sinusoid",
            "standard": "Synthetic baseline / sanity-check scenario",
            "parameters": {
                "duration_s": duration_s,
                "amplitude": amplitude,
                "freq_hz": freq_hz,
                "phase_rad": phase_rad,
                "noise_sigma": noise_sigma,
                "seed": seed,
            },
            "noise_sigma": noise_sigma,
            "dynamics": "Constant amplitude, constant frequency, constant phase slope.",
            "purpose": "Smoke test, clean tracking validation, estimator sanity baseline.",
            "monte_carlo_space": cls.MONTE_CARLO_SPACE,
            "fs_physics_hz": FS_PHYSICS,
        }

        return ScenarioData(
            name=cls.SCENARIO_NAME,
            t=t,
            v=v,
            f_true=f_true,
            meta=meta,
        )