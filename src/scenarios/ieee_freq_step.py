from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np

from .base import Scenario, ScenarioData
from estimators.common import FS_PHYSICS, F_NOM


class IEEEFreqStepScenario(Scenario):
    """
    IEC/IEEE 60255-118-1 Frequency Step Test.
    
    Intended for:
    - Measuring the "Response Time" and "Settling Time" of the estimator.
    - Evaluating transient overshoot when tracking a sudden loss of generation 
      or a sudden load rejection, without the C0-discontinuity of a phase jump.
    """

    SCENARIO_NAME: ClassVar[str] = "IEEE_Freq_Step"

    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {
        "duration_s": 1.5,
        "freq_pre_hz": F_NOM,
        "freq_post_hz": 59.0,     # Escalón estándar de -1.0 Hz
        "phase_rad": 0.0,
        "amplitude": 1.0,
        "t_step_s": 0.5,          # El cambio ocurre a los 0.5 segundos
        "noise_sigma": 0.001,     # Ruido blanco base
        "seed": None,
    }

    # Monte Carlo Variations: Profundidad del escalón, momento exacto y ruido
    MONTE_CARLO_SPACE: ClassVar[dict[str, Any]] = {
        "freq_post_hz": {
            "kind": "uniform",
            "low": 58.0,   # Caída severa (Under-frequency)
            "high": 62.0,  # Aumento severo (Over-frequency)
        },
        "t_step_s": {
            "kind": "uniform",
            "low": 0.4,
            "high": 0.8,
        },
        "phase_rad": {
            "kind": "uniform",
            "low": 0.0,
            "high": 2.0 * math.pi,
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
        if params.get("amplitude", -1) <= 0:
            raise ValueError("amplitude must be > 0")
        if params.get("t_step_s", -1) < 0 or params.get("t_step_s", 0) > params.get("duration_s", 0):
            raise ValueError("t_step_s must be within duration_s")

    @classmethod
    def generate(
        cls,
        duration_s: float = 1.5,
        freq_pre_hz: float = F_NOM,
        freq_post_hz: float = 59.0,
        phase_rad: float = 0.0,
        amplitude: float = 1.0,
        t_step_s: float = 0.5,
        noise_sigma: float = 0.001,
        seed: int | None = None,
    ) -> ScenarioData:
        """
        Generates a continuous voltage signal with an instantaneous frequency step.
        Phase integration must be strictly continuous (C0) to avoid artificial phase jumps.
        """
        rng = np.random.default_rng(seed)
        t = np.arange(0.0, duration_s, 1.0 / FS_PHYSICS, dtype=float)

        step_mask = t >= t_step_s

        # 1. Perfil de Frecuencia Verdadera
        f_true = np.full_like(t, freq_pre_hz, dtype=float)
        f_true[step_mask] = freq_post_hz

        # 2. Integración de Fase (Analítica y C0-Continua)
        phi = np.zeros_like(t, dtype=float)
        
        # Segmento A: Pre-Escalón
        phi[~step_mask] = 2.0 * math.pi * freq_pre_hz * t[~step_mask] + phase_rad
        
        # Segmento B: Post-Escalón
        if np.any(step_mask):
            # Fase exacta acumulada en el instante t_step_s
            phi_at_step = 2.0 * math.pi * freq_pre_hz * t_step_s + phase_rad
            # Tiempo transcurrido desde el escalón
            t_post = t[step_mask] - t_step_s
            # Nueva trayectoria de fase sin saltos
            phi[step_mask] = phi_at_step + 2.0 * math.pi * freq_post_hz * t_post

        # 3. Señal de Voltaje
        v = amplitude * np.sin(phi)

        # 4. Inyección de Ruido
        if noise_sigma > 0.0:
            v = v + rng.normal(0.0, noise_sigma, size=t.shape)

        delta_f = round(freq_post_hz - freq_pre_hz, 2)

        meta = {
            "description": f"IEEE 1547 Frequency Step ({delta_f} Hz)",
            "standard": "IEC 60255-118-1 frequency step test",
            "parameters": {
                "duration_s": duration_s,
                "freq_pre_hz": freq_pre_hz,
                "freq_post_hz": freq_post_hz,
                "phase_rad": phase_rad,
                "amplitude": amplitude,
                "t_step_s": t_step_s,
                "noise_sigma": noise_sigma,
                "seed": seed,
            },
            "noise_sigma": noise_sigma,
            "dynamics": f"C0-continuous phase with frequency stepping from {freq_pre_hz} Hz to {freq_post_hz} Hz at {t_step_s}s.",
            "purpose": "Determine algorithm response time, settling time, and overshoot under pure frequency changes.",
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