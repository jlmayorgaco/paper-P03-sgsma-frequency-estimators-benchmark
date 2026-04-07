from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np

from .base import Scenario, ScenarioData
from estimators.common import FS_PHYSICS, F_NOM


class IEEEPhaseJump20Scenario(Scenario):
    """
    IEEE 1547-2018 Anti-Islanding Phase Jump Limit (20 degrees).
    
    Intended for:
    - Baseline testing of phase-jump ride-through capabilities.
    - Clean signal environment to isolate the phase discontinuity error.
    """

    SCENARIO_NAME: ClassVar[str] = "IEEE_Phase_Jump_20"

    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {
        "duration_s": 1.5,
        "freq_hz": F_NOM,
        "phase_rad": 0.0,
        "amplitude": 1.0,
        "phase_jump_rad": math.pi / 9.0,  # +20 grados
        "t_jump_s": 0.7,                  # Ocurre a los 0.7s
        "noise_sigma": 0.001,             # Ruido base estándar
        "seed": None,
    }

    MONTE_CARLO_SPACE: ClassVar[dict[str, Any]] = {
        "phase_jump_rad": {
            "kind": "uniform",
            "low": math.pi / 18.0,  # +10 grados
            "high": math.pi / 6.0,  # +30 grados
        },
        "t_jump_s": {
            "kind": "uniform",
            "low": 0.5,
            "high": 0.9,
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
        if params.get("t_jump_s", -1) < 0 or params.get("t_jump_s", 0) > params.get("duration_s", 0):
            raise ValueError("t_jump_s must be within duration_s")

    @classmethod
    def generate(
        cls,
        duration_s: float = 1.5,
        freq_hz: float = F_NOM,
        phase_rad: float = 0.0,
        amplitude: float = 1.0,
        phase_jump_rad: float = math.pi / 9.0,
        t_jump_s: float = 0.7,
        noise_sigma: float = 0.001,
        seed: int | None = None,
    ) -> ScenarioData:
        rng = np.random.default_rng(seed)
        t = np.arange(0.0, duration_s, 1.0 / FS_PHYSICS, dtype=float)

        f_true = np.full_like(t, freq_hz, dtype=float)
        phi = 2.0 * math.pi * freq_hz * t + phase_rad
        
        # Inyectar el salto de fase C0-discontinuo
        phi[t >= t_jump_s] += phase_jump_rad

        v = amplitude * np.sin(phi)

        if noise_sigma > 0.0:
            v = v + rng.normal(0.0, noise_sigma, size=t.shape)

        jump_deg = round(math.degrees(phase_jump_rad), 1)
        
        meta = {
            "description": f"IEEE 1547 Phase Jump (+{jump_deg}°)",
            "standard": "IEEE 1547-2018 anti-islanding phase limit (20°)",
            "parameters": {
                "duration_s": duration_s,
                "freq_hz": freq_hz,
                "phase_jump_rad": phase_jump_rad,
                "t_jump_s": t_jump_s,
                "noise_sigma": noise_sigma,
            },
            "dynamics": f"Constant 60 Hz; Instantaneous +{jump_deg}° step at t={t_jump_s}s. Clean signal.",
            "purpose": "Evaluate basic phase-jump tolerance without harmonic interference.",
            "monte_carlo_space": cls.MONTE_CARLO_SPACE,
            "fs_physics_hz": FS_PHYSICS,
        }

        return ScenarioData(name=cls.SCENARIO_NAME, t=t, v=v, f_true=f_true, meta=meta)