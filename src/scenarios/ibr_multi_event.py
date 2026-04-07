from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np

from .base import Scenario, ScenarioData
from estimators.common import FS_PHYSICS, F_NOM


class IBRMultiEventScenario(Scenario):
    """
    IBR Multi-Event (Composite Fault) Scenario.
    
    Intended for:
    - The ultimate stress test. Simulates a severe grid fault causing simultaneous:
      1. Voltage Sag (Amplitude Step)
      2. Phase Angle Jump
      3. Frequency Drop (Frequency Step)
    - Exposes cross-coupling vulnerabilities in Fourier and PLL-based algorithms.
    """

    SCENARIO_NAME: ClassVar[str] = "IBR_Multi_Event"

    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {
        "duration_s": 1.5,
        "freq_pre_hz": F_NOM,
        "freq_post_hz": F_NOM - 1.0,      # Cae 1.0 Hz de golpe
        "amp_pre_pu": 1.0,
        "amp_post_pu": 0.5,               # Hueco de tensión al 50%
        "phase_jump_rad": math.pi / 4.0,  # Salto de +45 grados
        "t_event_s": 0.5,                 # Todo explota a los 0.5s
        "noise_sigma": 0.005,             # Ruido de falla
        "seed": None,
    }

    MONTE_CARLO_SPACE: ClassVar[dict[str, Any]] = {
        "amp_post_pu": {
            "kind": "uniform",
            "low": 0.2,    # Falla profunda (20% de voltaje restante)
            "high": 0.8,   # Falla leve (80% restante)
        },
        "phase_jump_rad": {
            "kind": "uniform",
            "low": math.pi / 6.0,   # +30 grados
            "high": math.pi / 2.25, # +80 grados
        },
        "freq_post_hz": {
            "kind": "uniform",
            "low": 58.0,   # Caída masiva a 58 Hz
            "high": 59.5,  # Caída a 59.5 Hz
        },
        "t_event_s": {
            "kind": "uniform",
            "low": 0.4,
            "high": 0.8,
        }
    }

    @classmethod
    def validate_params(cls, params: dict[str, Any]) -> None:
        if params.get("duration_s", 0) <= 0:
            raise ValueError("duration_s must be > 0")
        if params.get("amp_pre_pu", -1) <= 0 or params.get("amp_post_pu", -1) <= 0:
            raise ValueError("Amplitudes must be > 0")
        if params.get("t_event_s", -1) < 0 or params.get("t_event_s", 0) > params.get("duration_s", 0):
            raise ValueError("t_event_s must be within duration_s")

    @classmethod
    def generate(
        cls,
        duration_s: float = 1.5,
        freq_pre_hz: float = F_NOM,
        freq_post_hz: float = 59.0,
        amp_pre_pu: float = 1.0,
        amp_post_pu: float = 0.5,
        phase_jump_rad: float = math.pi / 4.0,
        t_event_s: float = 0.5,
        noise_sigma: float = 0.005,
        seed: int | None = None,
    ) -> ScenarioData:
        rng = np.random.default_rng(seed)
        t = np.arange(0.0, duration_s, 1.0 / FS_PHYSICS, dtype=float)

        event_mask = t >= t_event_s

        # 1. Perfil de Frecuencia
        f_true = np.full_like(t, freq_pre_hz, dtype=float)
        f_true[event_mask] = freq_post_hz

        # 2. Perfil de Amplitud
        amp = np.full_like(t, amp_pre_pu, dtype=float)
        amp[event_mask] = amp_post_pu

        # 3. Perfil de Fase (Integración continua + el salto C0)
        phi = np.zeros_like(t, dtype=float)
        
        # Pre-evento
        phi[~event_mask] = 2.0 * math.pi * freq_pre_hz * t[~event_mask]
        
        # Post-evento
        if np.any(event_mask):
            # Fase acumulada justo antes del evento
            phi_event = 2.0 * math.pi * freq_pre_hz * t_event_s
            t_post = t[event_mask] - t_event_s
            # Reiniciamos la integración con la nueva frecuencia y sumamos el salto de fase
            phi[event_mask] = phi_event + phase_jump_rad + 2.0 * math.pi * freq_post_hz * t_post

        # 4. Señal de Voltaje Fundamental
        v = amp * np.sin(phi)

        # 5. Inyección de Armónicos (Fondo IBR - IEEE 519)
        v += 0.04 * amp * np.sin(5.0 * 2.0 * math.pi * f_true * t + rng.uniform(0, 2*math.pi))
        v += 0.02 * amp * np.sin(7.0 * 2.0 * math.pi * f_true * t + rng.uniform(0, 2*math.pi))

        if noise_sigma > 0.0:
            v = v + rng.normal(0.0, noise_sigma, size=t.shape)

        jump_deg = round(math.degrees(phase_jump_rad), 1)
        
        meta = {
            "description": f"IBR Composite Fault (Sag {amp_post_pu}pu, \u0394\u03C6 +{jump_deg}\u00B0, \u0394f {freq_post_hz-freq_pre_hz}Hz)",
            "standard": "Composite dynamic test (exceeds IEC 60255-118-1)",
            "parameters": {
                "amp_pre_pu": amp_pre_pu,
                "amp_post_pu": amp_post_pu,
                "freq_pre_hz": freq_pre_hz,
                "freq_post_hz": freq_post_hz,
                "phase_jump_rad": phase_jump_rad,
                "t_event_s": t_event_s,
                "noise_sigma": noise_sigma,
            },
            "dynamics": "Simultaneous amplitude step, phase jump, and frequency step.",
            "purpose": "Final stress test for structural cross-coupling and algorithm collapse.",
            "monte_carlo_space": cls.MONTE_CARLO_SPACE,
            "fs_physics_hz": FS_PHYSICS,
        }

        return ScenarioData(name=cls.SCENARIO_NAME, t=t, v=v, f_true=f_true, meta=meta)