from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np

from .base import Scenario, ScenarioData
from estimators.common import FS_PHYSICS, F_NOM


class NERCPhaseJump60Scenario(Scenario):
    """
    NERC GADS Extreme Islanding Phase Jump (60 degrees + Harmonics).
    
    Intended for:
    - Stress-testing estimator robustness against severe Loss of Mains events.
    - Evaluating transient rejection in heavily distorted IBR harmonic environments 
      (IEEE 519 limits) combined with elevated switching noise.
    """

    SCENARIO_NAME: ClassVar[str] = "NERC_Phase_Jump_60"

    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {
        "duration_s": 1.5,
        "freq_hz": F_NOM,
        "phase_rad": 0.0,
        "amplitude": 1.0,
        "phase_jump_rad": math.pi / 3.0,  # +60 grados (3x el límite IEEE)
        "t_jump_s": 0.7,                  # Ocurre a los 0.7s
        "noise_sigma": 0.005,             # Ruido elevado (5x normal)
        "seed": None,
    }

    MONTE_CARLO_SPACE: ClassVar[dict[str, Any]] = {
        "phase_jump_rad": {
            "kind": "uniform",
            "low": math.pi / 4.5,   # +40 grados
            "high": math.pi / 2.25, # +80 grados
        },
        "t_jump_s": {
            "kind": "uniform",
            "low": 0.5,
            "high": 0.9,
        },
        "noise_sigma": {
            "kind": "uniform",
            "low": 0.003,
            "high": 0.01,
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
        phase_jump_rad: float = math.pi / 3.0,
        t_jump_s: float = 0.7,
        noise_sigma: float = 0.005,
        seed: int | None = None,
    ) -> ScenarioData:
        rng = np.random.default_rng(seed)
        t = np.arange(0.0, duration_s, 1.0 / FS_PHYSICS, dtype=float)

        f_true = np.full_like(t, freq_hz, dtype=float)
        phi = 2.0 * math.pi * freq_hz * t + phase_rad
        
        # Inyectar el salto de fase C0-discontinuo
        phi[t >= t_jump_s] += phase_jump_rad

        # Fundamental
        v = amplitude * np.sin(phi)

        # Inyección de Armónicos (Fondo sucio IBR - IEEE 519-2022 Límite PCC)
        # 5to armónico al 4%
        v += 0.04 * amplitude * np.sin(5.0 * 2.0 * math.pi * freq_hz * t + rng.uniform(0, 2*math.pi))
        # 7mo armónico al 2%
        v += 0.02 * amplitude * np.sin(7.0 * 2.0 * math.pi * freq_hz * t + rng.uniform(0, 2*math.pi))
        # Interarmónico a 32.5 Hz al 0.5% (Asíncrono)
        v += 0.005 * amplitude * np.sin(2.0 * math.pi * 32.5 * t + rng.uniform(0, 2*math.pi))

        if noise_sigma > 0.0:
            v = v + rng.normal(0.0, noise_sigma, size=t.shape)

        jump_deg = round(math.degrees(phase_jump_rad), 1)
        
        meta = {
            "description": f"NERC Extreme Phase Jump (+{jump_deg}°)",
            "standard": "Exceeds IEEE 1547-2018 (3x limit). Harmonics at IEEE 519 limit.",
            "parameters": {
                "duration_s": duration_s,
                "freq_hz": freq_hz,
                "phase_jump_rad": phase_jump_rad,
                "t_jump_s": t_jump_s,
                "noise_sigma": noise_sigma,
            },
            "dynamics": f"Constant 60 Hz; Instantaneous +{jump_deg}° step at t={t_jump_s}s. Harmonics: 5th (4%), 7th (2%), Inter: 32.5Hz (0.5%).",
            "purpose": "Drive headline RMSE and T_trip claims. Stress filter state-divergence under severe islanding.",
            "monte_carlo_space": cls.MONTE_CARLO_SPACE,
            "fs_physics_hz": FS_PHYSICS,
        }

        return ScenarioData(name=cls.SCENARIO_NAME, t=t, v=v, f_true=f_true, meta=meta)