from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np

from .base import Scenario, ScenarioData
from estimators.common import FS_PHYSICS, F_NOM


class IEEEModulationScenario(Scenario):
    """
    IEC/IEEE 60255-118-1 AM/PM Modulation Test (Measurement Bandwidth).
    
    Intended for:
    - Verifying estimator tracking bandwidth.
    - Testing response to inter-area oscillations and sub-synchronous IBR resonance.
    - Observing the attenuation and phase-lag of the frequency estimate.
    """

    SCENARIO_NAME: ClassVar[str] = "IEEE_Modulation"

    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {
        "duration_s": 2.0,       # 2 segundos permite ver varios ciclos de modulación
        "freq_nom_hz": F_NOM,
        "phase_rad": 0.0,
        "amplitude": 1.0,
        "kx": 0.1,               # Profundidad de modulación AM (10%)
        "ka": 0.1,               # Profundidad de modulación de Fase (0.1 rad)
        "fm_hz": 2.0,            # Frecuencia de modulación (2.0 Hz)
        "noise_sigma": 0.001,    # AWGN base noise
        "seed": None,
    }

    # Monte Carlo Variations: Frecuencia de oscilación e índices de modulación
    MONTE_CARLO_SPACE: ClassVar[dict[str, Any]] = {
        "fm_hz": {
            "kind": "uniform",
            "low": 0.5,    # Oscilaciones inter-área lentas
            "high": 5.0,   # Resonancia de control de IBRs (rápida)
        },
        "kx": {
            "kind": "uniform",
            "low": 0.05,
            "high": 0.15,
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
        if params.get("amplitude", -1) <= 0:
            raise ValueError("amplitude must be > 0")
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
        ka: float = 0.1,
        fm_hz: float = 2.0,
        noise_sigma: float = 0.001,
        seed: int | None = None,
    ) -> ScenarioData:
        """
        Generates an AM and PM modulated signal per IEEE compliance tests.
        """
        rng = np.random.default_rng(seed)
        t = np.arange(0.0, duration_s, 1.0 / FS_PHYSICS, dtype=float)

        # 1. Modulación de Amplitud (AM)
        # amp(t) = A * (1 + kx * cos(2*pi*fm*t))
        amp_t = amplitude * (1.0 + kx * np.cos(2.0 * math.pi * fm_hz * t))

        # 2. Modulación de Fase (PM)
        # phi(t) = 2*pi*f0*t + ka * cos(2*pi*fm*t) + phi_0
        phi_t = 2.0 * math.pi * freq_nom_hz * t + ka * np.cos(2.0 * math.pi * fm_hz * t) + phase_rad

        # 3. Frecuencia Instantánea Teórica (Derivada de la fase)
        # f(t) = d(phi)/dt / (2*pi) = f0 - ka * fm * sin(2*pi*fm*t)
        f_true = freq_nom_hz - ka * fm_hz * np.sin(2.0 * math.pi * fm_hz * t)

        # 4. Señal de Voltaje Modulada
        v = amp_t * np.sin(phi_t)

        # 5. Inyección de Ruido
        if noise_sigma > 0.0:
            v = v + rng.normal(0.0, noise_sigma, size=t.shape)

        meta = {
            "description": f"IEEE 1547 Modulation AM/PM (fm={fm_hz}Hz)",
            "standard": "IEC 60255-118-1 modulation test",
            "parameters": {
                "duration_s": duration_s,
                "freq_nom_hz": freq_nom_hz,
                "phase_rad": phase_rad,
                "amplitude": amplitude,
                "kx": kx,
                "ka": ka,
                "fm_hz": fm_hz,
                "noise_sigma": noise_sigma,
                "seed": seed,
            },
            "noise_sigma": noise_sigma,
            "dynamics": f"AM (kx={kx}) and PM (ka={ka}) modulation at {fm_hz} Hz.",
            "purpose": "Evaluate estimator bandwidth, attenuation, and latency under low-frequency oscillations.",
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