from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np

from .base import Scenario, ScenarioData
from estimators.common import FS_PHYSICS, F_NOM


class IEEEMagStepScenario(Scenario):
    """
    IEEE 60255-118-1 Amplitude Step Scenario (Scenario A in the paper).
    
    Intended for:
    - Verifying AM-FM cross-coupling rejection.
    - Baseline dynamic response to voltage sags/swells.
    - No harmonics or impulsive noise (pure baseline test).
    """

    SCENARIO_NAME: ClassVar[str] = "IEEE_Mag_Step"

    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {
        "duration_s": 1.5,
        "freq_hz": F_NOM,
        "phase_rad": 0.0,
        "amp_pre_pu": 1.0,
        "amp_post_pu": 1.1,   # +10% step by default
        "t_step_s": 0.5,      # Step occurs at t=0.5s
        "noise_sigma": 0.001, # AWGN base noise (SNR ~ 57 dB)
        "seed": None,
    }

    # Monte Carlo Variations: Tamaño del salto, tiempo, ruido y fase
    MONTE_CARLO_SPACE: ClassVar[dict[str, Any]] = {
        "amp_post_pu": {
            "kind": "uniform",
            "low": 1.05,  # Salto del +5%
            "high": 1.15, # Salto del +15%
        },
        "t_step_s": {
            "kind": "uniform",
            "low": 0.4,
            "high": 0.6,
        },
        "phase_rad": {
            "kind": "uniform",
            "low": 0.0,
            "high": 2.0 * math.pi,
        },
        "noise_sigma": {
            "kind": "uniform",
            "low": 0.0005,
            "high": 0.002, # Ruido blanco variable
        },
        # amp_pre_pu y freq_hz se mantienen fijos para no mezclar variables en esta prueba
    }

    @classmethod
    def validate_params(cls, params: dict[str, Any]) -> None:
  
        if params["duration_s"] <= 0:
            raise ValueError("duration_s must be > 0")
        if params["amp_pre_pu"] < 0 or params["amp_post_pu"] < 0:
            raise ValueError("amplitudes must be >= 0")
        if params["t_step_s"] < 0 or params["t_step_s"] > params["duration_s"]:
            raise ValueError("t_step_s must be within duration_s")
        if params["noise_sigma"] < 0:
            raise ValueError("noise_sigma must be >= 0")

    @classmethod
    def generate(
        cls,
        duration_s: float = 1.5,
        freq_hz: float = F_NOM,
        phase_rad: float = 0.0,
        amp_pre_pu: float = 1.0,
        amp_post_pu: float = 1.1,
        t_step_s: float = 0.5,
        noise_sigma: float = 0.001,
        seed: int | None = None,
    ) -> ScenarioData:
        """
        Generate a pure amplitude step scenario.
        """
        rng = np.random.default_rng(seed)

        # 1. Vectores de Tiempo y Frecuencia Ideal
        t = np.arange(0.0, duration_s, 1.0 / FS_PHYSICS, dtype=float)
        f_true = np.full_like(t, freq_hz, dtype=float)

        # 2. Vector de Fase (Continuo)
        phi = 2.0 * math.pi * freq_hz * t + phase_rad
        
        # 3. Vector de Amplitud (Escalón Discreto)
        # Usamos np.where para crear el escalón exactamente en t_step_s
        amp_array = np.where(t < t_step_s, amp_pre_pu, amp_post_pu)
        
        # 4. Señal de Voltaje
        v = amp_array * np.sin(phi)

        # 5. Inyección de Ruido Blanco Gaussiano (AWGN)
        if noise_sigma > 0.0:
            v = v + rng.normal(0.0, noise_sigma, size=t.shape)

        meta = {
            "description": "IEEE 60255-118-1 Amplitude Step (+10% default)",
            "standard": "IEC/IEEE 60255-118-1 amplitude step test",
            "parameters": {
                "duration_s": duration_s,
                "freq_hz": freq_hz,
                "phase_rad": phase_rad,
                "amp_pre_pu": amp_pre_pu,
                "amp_post_pu": amp_post_pu,
                "t_step_s": t_step_s,
                "noise_sigma": noise_sigma,
                "seed": seed,
            },
            "noise_sigma": noise_sigma,
            "dynamics": f"Amplitude step {amp_pre_pu} -> {amp_post_pu} pu at t={t_step_s}s. Phase C0-continuous.",
            "purpose": "AM-FM cross-coupling error evaluation without harmonic interference.",
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