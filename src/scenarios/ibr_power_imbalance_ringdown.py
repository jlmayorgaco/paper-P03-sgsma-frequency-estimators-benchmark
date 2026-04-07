from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np

from .base import Scenario, ScenarioData
from estimators.common import FS_PHYSICS, F_NOM


class IBRPowerImbalanceRingdownScenario(Scenario):
    """
    IBR Power Imbalance & Control Ring-down Scenario.
    
    Intended for:
    - Simulating realistic power imbalances (load rejection / loss of generation).
    - Features a negative ROCOF ramp, followed by a simultaneous Phase Jump and Voltage Sag.
    - Concludes with an underdamped frequency recovery (PID/VSG Ring-down).
    - Rigorously tests the estimator's ability to track complex, realistic transient dynamics.
    """

    SCENARIO_NAME: ClassVar[str] = "IBR_Power_Imbalance_Ringdown"

    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {
        "duration_s": 2.5,
        "freq_nom_hz": F_NOM,
        "amp_pre_pu": 1.0,
        "amp_post_pu": 0.8,               # Caída de tensión por la falla
        "t_ramp_s": 0.5,                  # Inicio de la rampa negativa
        "rocof_hz_s": -2.0,               # Tasa de caída de frecuencia
        "t_event_s": 1.0,                 # Instante del evento principal (Salto + Ringdown)
        "phase_jump_rad": 80.0 * math.pi / 180.0, # Salto de +80 grados
        "ring_jump_hz": 1.5,              # Pico inicial del ring-down
        "ring_freq_hz": 4.0,              # Frecuencia de la oscilación PID
        "ring_tau_s": 0.25,               # Constante de decaimiento (amortiguamiento)
        "noise_sigma": 0.002,             # Ruido base IBR
        "seed": None,
    }

    MONTE_CARLO_SPACE: ClassVar[dict[str, Any]] = {
        "phase_jump_rad": {
            "kind": "uniform",
            "low": 40.0 * math.pi / 180.0,
            "high": 90.0 * math.pi / 180.0,
        },
        "amp_post_pu": {
            "kind": "uniform",
            "low": 0.5,
            "high": 0.9,
        },
        "ring_tau_s": {
            "kind": "uniform",
            "low": 0.1,    # Fuertemente amortiguado
            "high": 0.5,   # Pobremente amortiguado (muchas oscilaciones)
        },
        "t_event_s": {
            "kind": "uniform",
            "low": 0.8,
            "high": 1.2,
        }
    }

    @classmethod
    def validate_params(cls, params: dict[str, Any]) -> None:
        if params.get("duration_s", 0) <= 0:
            raise ValueError("duration_s must be > 0")
        if params.get("t_ramp_s", -1) >= params.get("t_event_s", 0):
            raise ValueError("t_ramp_s must be before t_event_s")

    @classmethod
    def generate(
        cls,
        duration_s: float = 2.5,
        freq_nom_hz: float = F_NOM,
        amp_pre_pu: float = 1.0,
        amp_post_pu: float = 0.8,
        t_ramp_s: float = 0.5,
        rocof_hz_s: float = -2.0,
        t_event_s: float = 1.0,
        phase_jump_rad: float = 80.0 * math.pi / 180.0,
        ring_jump_hz: float = 1.5,
        ring_freq_hz: float = 4.0,
        ring_tau_s: float = 0.25,
        noise_sigma: float = 0.002,
        seed: int | None = None,
    ) -> ScenarioData:
        rng = np.random.default_rng(seed)
        t = np.arange(0.0, duration_s, 1.0 / FS_PHYSICS, dtype=float)

        mask_pre = t < t_ramp_s
        mask_ramp = (t >= t_ramp_s) & (t < t_event_s)
        mask_ring = t >= t_event_s

        # 1. PERFIL DE FRECUENCIA VERDADERA
        f_true = np.zeros_like(t, dtype=float)
        
        # A. Nominal
        f_true[mask_pre] = freq_nom_hz
        
        # B. Rampa Negativa (Pérdida de Inercia/Desbalance)
        f_true[mask_ramp] = freq_nom_hz + rocof_hz_s * (t[mask_ramp] - t_ramp_s)
        
        # C. Ring-down (Recuperación y oscilación)
        # f(t) = f_nom + Delta_f * exp(-t/tau) * cos(2*pi*f_ring*t)
        alpha = 1.0 / ring_tau_s
        omega = 2.0 * math.pi * ring_freq_hz
        t_post = t[mask_ring] - t_event_s
        f_true[mask_ring] = freq_nom_hz + ring_jump_hz * np.exp(-alpha * t_post) * np.cos(omega * t_post)

        # 2. PERFIL DE FASE (INTEGRACIÓN ANALÍTICA EXACTA)
        phi = np.zeros_like(t, dtype=float)
        
        # A. Nominal
        phi[mask_pre] = 2.0 * math.pi * freq_nom_hz * t[mask_pre]
        
        # B. Rampa
        if np.any(mask_ramp):
            phi_ramp_start = 2.0 * math.pi * freq_nom_hz * t_ramp_s
            dt_ramp = t[mask_ramp] - t_ramp_s
            phi[mask_ramp] = phi_ramp_start + 2.0 * math.pi * (freq_nom_hz * dt_ramp + 0.5 * rocof_hz_s * dt_ramp**2)
            phi_event_pre = phi[mask_ramp][-1] + 2.0 * math.pi * (f_true[mask_ramp][-1] + f_true[mask_ring][0])/2.0 * (1.0/FS_PHYSICS)
        else:
            phi_event_pre = 2.0 * math.pi * freq_nom_hz * t_event_s

        # C. Ring-down
        if np.any(mask_ring):
            # Agregamos el Salto de Fase C0
            phi_event_post = phi_event_pre + phase_jump_rad
            
            # Integral exacta de exp(-alpha*t) * cos(omega*t)
            den = alpha**2 + omega**2
            def ring_integral(x: np.ndarray) -> np.ndarray:
                return (alpha + np.exp(-alpha * x) * (omega * np.sin(omega * x) - alpha * np.cos(omega * x))) / den
            
            phi[mask_ring] = phi_event_post + 2.0 * math.pi * (freq_nom_hz * t_post + ring_jump_hz * ring_integral(t_post))

        # 3. PERFIL DE AMPLITUD Y VOLTAJE
        amp = np.full_like(t, amp_pre_pu, dtype=float)
        amp[mask_ring] = amp_post_pu
        
        v = amp * np.sin(phi)

        # 4. Inyección de Ruido
        if noise_sigma > 0.0:
            v = v + rng.normal(0.0, noise_sigma, size=t.shape)

        jump_deg = round(math.degrees(phase_jump_rad), 1)
        
        meta = {
            "description": f"Power Imbalance (Sag to {amp_post_pu}pu, \u0394\u03C6 +{jump_deg}\u00B0, Ring-down)",
            "standard": "Complex transient based on GFM/VSG control responses",
            "parameters": {
                "amp_pre_pu": amp_pre_pu,
                "amp_post_pu": amp_post_pu,
                "t_ramp_s": t_ramp_s,
                "rocof_hz_s": rocof_hz_s,
                "t_event_s": t_event_s,
                "phase_jump_rad": phase_jump_rad,
                "ring_jump_hz": ring_jump_hz,
                "ring_tau_s": ring_tau_s,
            },
            "purpose": "Evaluate estimator stability under combined negative ROCOF, phase jump, and underdamped active power recovery.",
            "monte_carlo_space": cls.MONTE_CARLO_SPACE,
            "fs_physics_hz": FS_PHYSICS,
        }

        return ScenarioData(name=cls.SCENARIO_NAME, t=t, v=v, f_true=f_true, meta=meta)