from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np

from .base import Scenario, ScenarioData
from estimators.common import FS_PHYSICS, F_NOM


class IEEEFreqRampScenario(Scenario):
    """
    IEEE 1547-2018 Category III Frequency Ramp Scenario (Scenario B).
    
    Intended for:
    - Verifying RoCoF tracking capabilities and structural lag.
    - Stressing the kinematic limits of window-based and loop-based estimators.
    - Baseline dynamic response to sustained low-inertia generation loss.
    """

    SCENARIO_NAME: ClassVar[str] = "IEEE_Freq_Ramp"

    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {
        "duration_s": 1.5,
        "freq_nom_hz": F_NOM,
        "rocof_hz_s": 3.0,      # IEEE Cat-III boundary (+3.0 Hz/s)
        "t_start_s": 0.3,       # Rampa inicia a los 0.3s
        "freq_cap_hz": 61.5,    # Techo de frecuencia (trip threshold)
        "phase_rad": 0.0,
        "amplitude": 1.0,
        "noise_sigma": 0.001,   # AWGN base noise
        "seed": None,
    }

    # Monte Carlo Variations: Pendiente del RoCoF, tiempo de inicio, ruido y fase
    MONTE_CARLO_SPACE: ClassVar[dict[str, Any]] = {
        "rocof_hz_s": {
            "kind": "uniform",
            "low": 1.0,   # Rampa moderada
            "high": 5.0,  # Rampa extrema (estrés de baja inercia)
        },
        "t_start_s": {
            "kind": "uniform",
            "low": 0.2,
            "high": 0.4,
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
        if params.get("t_start_s", -1) < 0 or params.get("t_start_s", 0) > params.get("duration_s", 0):
            raise ValueError("t_start_s must be within duration_s")
        if params.get("noise_sigma", -1) < 0:
            raise ValueError("noise_sigma must be >= 0")

    @classmethod
    def generate(
        cls,
        duration_s: float = 1.5,
        freq_nom_hz: float = F_NOM,
        rocof_hz_s: float = 3.0,
        t_start_s: float = 0.3,
        freq_cap_hz: float = 61.5,
        phase_rad: float = 0.0,
        amplitude: float = 1.0,
        noise_sigma: float = 0.001,
        seed: int | None = None,
    ) -> ScenarioData:
        """
        Generates a physically continuous frequency ramp test.
        The phase is calculated via exact analytical integration of the frequency profile
        to avoid any numerical cumulative sum errors.
        """
        rng = np.random.default_rng(seed)
        t = np.arange(0.0, duration_s, 1.0 / FS_PHYSICS, dtype=float)
        
        # 1. Determinar el instante donde la frecuencia choca con el límite (cap)
        delta_f = freq_cap_hz - freq_nom_hz
        
        # FIX: Proteger contra división por cero cuando RoCoF es 0.0 (como en el test de THD)
        if abs(rocof_hz_s) < 1e-9:
            t_end_s = duration_s
        else:
            t_end_s = t_start_s + (delta_f / rocof_hz_s)
            
        # Si la rampa es tan lenta que no alcanza el cap en el tiempo de simulación
        t_end_s = min(t_end_s, duration_s)

        # 2. Construir perfil de Frecuencia Real (Piecewise)
        f_true = np.full_like(t, freq_nom_hz, dtype=float)
        mask_ramp = (t >= t_start_s) & (t < t_end_s)
        mask_cap = (t >= t_end_s)

        f_true[mask_ramp] = freq_nom_hz + rocof_hz_s * (t[mask_ramp] - t_start_s)
        if np.any(mask_cap):
            f_true[mask_cap] = freq_cap_hz

        # 3. Construir perfil de Fase (Integración Analítica C1-continua)
        phi = np.zeros_like(t, dtype=float)
        
        # Segmento A: Pre-Rampa
        mask_pre = (t < t_start_s)
        phi[mask_pre] = 2.0 * math.pi * freq_nom_hz * t[mask_pre] + phase_rad
        
        # Segmento B: Durante la Rampa
        if np.any(mask_ramp):
            t_r = t[mask_ramp] - t_start_s
            phi_start = 2.0 * math.pi * freq_nom_hz * t_start_s + phase_rad
            phi[mask_ramp] = phi_start + 2.0 * math.pi * (freq_nom_hz * t_r + 0.5 * rocof_hz_s * (t_r ** 2))
            
        # Segmento C: Post-Rampa (Saturación)
        if np.any(mask_cap):
            t_c = t[mask_cap] - t_end_s
            # Fase acumulada justo al terminar la rampa
            delta_t_ramp = t_end_s - t_start_s
            phi_end = (2.0 * math.pi * freq_nom_hz * t_start_s + phase_rad) + \
                      2.0 * math.pi * (freq_nom_hz * delta_t_ramp + 0.5 * rocof_hz_s * (delta_t_ramp ** 2))
            phi[mask_cap] = phi_end + 2.0 * math.pi * freq_cap_hz * t_c

        # 4. Señal de Voltaje
        v = amplitude * np.sin(phi)

        # 5. Inyección de Ruido
        if noise_sigma > 0.0:
            v = v + rng.normal(0.0, noise_sigma, size=t.shape)

        meta = {
            "description": f"IEEE 1547 Frequency Ramp (+{rocof_hz_s} Hz/s)",
            "standard": "IEC 60255-118-1 frequency ramp test / IEEE 1547 Cat III",
            "parameters": {
                "duration_s": duration_s,
                "freq_nom_hz": freq_nom_hz,
                "rocof_hz_s": rocof_hz_s,
                "t_start_s": t_start_s,
                "freq_cap_hz": freq_cap_hz,
                "phase_rad": phase_rad,
                "amplitude": amplitude,
                "noise_sigma": noise_sigma,
                "seed": seed,
            },
            "noise_sigma": noise_sigma,
            "dynamics": f"Piecewise: {freq_nom_hz} Hz -> +{rocof_hz_s} Hz/s ramp -> hold at {freq_cap_hz} Hz.",
            "purpose": "Evaluate stochastic estimation lag and structural tracking latency under low-inertia RoCoF events.",
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