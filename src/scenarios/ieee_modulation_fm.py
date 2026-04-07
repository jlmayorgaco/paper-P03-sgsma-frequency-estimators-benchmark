from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np

from .base import Scenario, ScenarioData
from estimators.common import FS_PHYSICS, F_NOM


class IEEEModulationFMScenario(Scenario):
    """
    Pure Phase/Frequency Modulation (PM/FM) Test.
    
    Intended for:
    - Accurately measuring estimator tracking bandwidth.
    - Observing attenuation and structural phase-lag without AM interference.
    """

    SCENARIO_NAME: ClassVar[str] = "IEEE_Modulation_FM"

    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {
        "duration_s": 2.0,
        "freq_nom_hz": F_NOM,
        "phase_rad": 0.0,
        "amplitude": 1.0,
        "ka": 0.1,               # 0.1 rad PM depth
        "fm_hz": 2.0,            # 2.0 Hz oscillation
        "noise_sigma": 0.001,
        "seed": None,
    }

    MONTE_CARLO_SPACE: ClassVar[dict[str, Any]] = {
        "fm_hz": {
            "kind": "uniform",
            "low": 0.5,
            "high": 5.0,
        },
        "ka": {
            "kind": "uniform",
            "low": 0.05,
            "high": 0.15,
        },
        "noise_sigma": {
            "kind": "uniform",
            "low": 0.0005,
            "high": 0.002,
        }
    }

    @classmethod
    def validate_params(cls, params: dict[str, Any]) -> None:
        if params.get("duration_s", 0) <= 0:
            raise ValueError("duration_s must be > 0")
        if params.get("fm_hz", 0) <= 0:
            raise ValueError("Modulation frequency fm_hz must be > 0")
        if params.get("noise_sigma", -1) < 0:
            raise ValueError("noise_sigma must be >= 0")

    @classmethod
    def generate(
        cls,
        duration_s: float = 2.0,
        freq_nom_hz: float = F_NOM,
        phase_rad: float = 0.0,
        amplitude: float = 1.0,
        ka: float = 0.1,
        fm_hz: float = 2.0,
        noise_sigma: float = 0.001,
        seed: int | None = None,
    ) -> ScenarioData:
        rng = np.random.default_rng(seed)
        t = np.arange(0.0, duration_s, 1.0 / FS_PHYSICS, dtype=float)

        # 1. Constant Amplitude
        amp_t = amplitude

        # 2. PM Modulated Phase
        phi_t = 2.0 * math.pi * freq_nom_hz * t + ka * np.cos(2.0 * math.pi * fm_hz * t) + phase_rad

        # 3. True Frequency (Derivative of Phase)
        f_true = freq_nom_hz - ka * fm_hz * np.sin(2.0 * math.pi * fm_hz * t)

        # 4. Voltage Signal
        v = amp_t * np.sin(phi_t)

        if noise_sigma > 0.0:
            v = v + rng.normal(0.0, noise_sigma, size=t.shape)

        meta = {
            "description": f"Pure PM/FM (ka={ka}, fm={fm_hz}Hz)",
            "standard": "Derived from IEC 60255-118-1",
            "parameters": {
                "duration_s": duration_s,
                "freq_nom_hz": freq_nom_hz,
                "ka": ka,
                "fm_hz": fm_hz,
                "noise_sigma": noise_sigma,
            },
            "dynamics": "Constant amplitude with oscillating frequency.",
            "purpose": "Evaluate pure frequency tracking bandwidth.",
            "monte_carlo_space": cls.MONTE_CARLO_SPACE,
            "fs_physics_hz": FS_PHYSICS,
        }

        return ScenarioData(name=cls.SCENARIO_NAME, t=t, v=v, f_true=f_true, meta=meta)