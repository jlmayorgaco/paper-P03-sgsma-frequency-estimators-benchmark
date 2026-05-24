from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np

from .base import Scenario, ScenarioData
from estimators.common import FS_PHYSICS, F_NOM


class IEEEModulationAMScenario(Scenario):
    """
    Pure Amplitude Modulation (AM) Test.
    
    Intended for:
    - Isolating AM-to-FM cross-coupling errors.
    - If the true frequency is strictly constant, any frequency oscillation 
      reported by the estimator is a cross-coupling artifact.
    """

    SCENARIO_NAME: ClassVar[str] = "IEEE_Modulation_AM"

    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {
        "duration_s": 2.0,
        "freq_nom_hz": F_NOM,
        "phase_rad": 0.0,
        "amplitude": 1.0,
        "kx": 0.1,               # 10% AM depth
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
        "kx": {
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
        kx: float = 0.1,
        fm_hz: float = 2.0,
        noise_sigma: float = 0.001,
        seed: int | None = None,
    ) -> ScenarioData:
        rng = np.random.default_rng(seed)
        t = np.arange(0.0, duration_s, 1.0 / FS_PHYSICS, dtype=float)

        # 1. AM Envelope
        amp_t = amplitude * (1.0 + kx * np.cos(2.0 * math.pi * fm_hz * t))

        # 2. Constant Phase and Frequency
        phi_t = 2.0 * math.pi * freq_nom_hz * t + phase_rad
        f_true = np.full_like(t, freq_nom_hz, dtype=float)

        # 3. Voltage Signal
        v = amp_t * np.sin(phi_t)

        if noise_sigma > 0.0:
            v = v + rng.normal(0.0, noise_sigma, size=t.shape)

        meta = {
            "description": f"Pure AM (kx={kx}, fm={fm_hz}Hz)",
            "standard": "Derived from IEC 60255-118-1",
            "parameters": {
                "duration_s": duration_s,
                "freq_nom_hz": freq_nom_hz,
                "kx": kx,
                "fm_hz": fm_hz,
                "noise_sigma": noise_sigma,
            },
            "dynamics": "Constant 60 Hz frequency with modulated amplitude.",
            "purpose": "Isolate AM-FM cross-coupling vulnerabilities.",
            "monte_carlo_space": cls.MONTE_CARLO_SPACE,
            "fs_physics_hz": FS_PHYSICS,
        }

        return ScenarioData(name=cls.SCENARIO_NAME, t=t, v=v, f_true=f_true, meta=meta)