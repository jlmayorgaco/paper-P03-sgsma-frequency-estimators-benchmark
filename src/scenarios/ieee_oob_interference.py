from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np

from .base import Scenario, ScenarioData
from estimators.common import FS_PHYSICS, F_NOM


class IEEEOOBInterferenceScenario(Scenario):
    """
    IEC/IEEE 60255-118-1 Out-of-Band (OOB) Interference Test.
    
    Intended for:
    - Testing the anti-aliasing and stop-band attenuation of the estimator.
    - Simulating Sub-Synchronous Resonance (SSR/SSCI) often seen in IBRs.
    - If the estimator lacks proper filtering, the OOB signal will alias into the 
      passband and cause severe artificial oscillations in the frequency estimate.
    """

    SCENARIO_NAME: ClassVar[str] = "IEEE_OOB_Interference"

    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {
        "duration_s": 1.5,
        "freq_hz": F_NOM,
        "phase_rad": 0.0,
        "amplitude": 1.0,
        "interf_freq_hz": 30.0,   # Interferencia subsíncrona estándar (30 Hz)
        "interf_amp_pu": 0.1,     # Amplitud masiva: 10% del voltaje nominal (Requisito IEEE)
        "noise_sigma": 0.001,
        "seed": None,
    }

    # Monte Carlo Variations: Frecuencia de la interferencia y su amplitud
    MONTE_CARLO_SPACE: ClassVar[dict[str, Any]] = {
        "interf_freq_hz": {
            "kind": "uniform",
            "low": 15.0,    # Resonancias de baja frecuencia
            "high": 45.0,   # Interarmónicos cercanos a la fundamental
        },
        "interf_amp_pu": {
            "kind": "uniform",
            "low": 0.05,    # 5% de amplitud
            "high": 0.15,   # 15% de amplitud (Estrés extremo)
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
        if params.get("interf_amp_pu", -1) < 0:
            raise ValueError("interf_amp_pu must be >= 0")
        if params.get("interf_freq_hz", 0) <= 0:
            raise ValueError("interf_freq_hz must be > 0")

    @classmethod
    def generate(
        cls,
        duration_s: float = 1.5,
        freq_hz: float = F_NOM,
        phase_rad: float = 0.0,
        amplitude: float = 1.0,
        interf_freq_hz: float = 30.0,
        interf_amp_pu: float = 0.1,
        noise_sigma: float = 0.001,
        seed: int | None = None,
    ) -> ScenarioData:
        rng = np.random.default_rng(seed)
        t = np.arange(0.0, duration_s, 1.0 / FS_PHYSICS, dtype=float)

        # 1. Frecuencia Verdadera (Es estricta y perfectamente constante)
        f_true = np.full_like(t, freq_hz, dtype=float)

        # 2. Señal Fundamental (60 Hz)
        phi_fund = 2.0 * math.pi * freq_hz * t + phase_rad
        v_fund = amplitude * np.sin(phi_fund)

        # 3. Inyección de la Señal Interfiriendo (OOB)
        # La fase de la interferencia es aleatoria respecto a la fundamental
        phi_interf = 2.0 * math.pi * interf_freq_hz * t + rng.uniform(0, 2.0 * math.pi)
        v_interf = interf_amp_pu * np.sin(phi_interf)

        # 4. Señal Total
        v = v_fund + v_interf

        # 5. Inyección de Ruido
        if noise_sigma > 0.0:
            v = v + rng.normal(0.0, noise_sigma, size=t.shape)

        meta = {
            "description": f"IEEE 1547 OOB Interference ({interf_amp_pu}pu at {interf_freq_hz}Hz)",
            "standard": "IEC 60255-118-1 out-of-band test",
            "parameters": {
                "duration_s": duration_s,
                "freq_hz": freq_hz,
                "phase_rad": phase_rad,
                "amplitude": amplitude,
                "interf_freq_hz": interf_freq_hz,
                "interf_amp_pu": interf_amp_pu,
                "noise_sigma": noise_sigma,
                "seed": seed,
            },
            "noise_sigma": noise_sigma,
            "dynamics": f"Constant {freq_hz} Hz fundamental heavily polluted by a {interf_freq_hz} Hz out-of-band signal.",
            "purpose": "Evaluate anti-aliasing filter performance and rejection of sub-synchronous IBR resonance.",
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