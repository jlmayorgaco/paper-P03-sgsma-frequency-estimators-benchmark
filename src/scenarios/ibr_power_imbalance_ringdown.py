from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np

from .base import Scenario, ScenarioData
from estimators.common import FS_PHYSICS, F_NOM


class IBRPowerImbalanceRingdownScenario(Scenario):
    """
    IBR Power Imbalance & Control Ring-down Scenario — The Ultimate Stress Test.
    
    Modifications:
      - Phase jump 1 at fault onset (t_ramp_s) representing topological change.
      - Phase jump 2 at reconnection (t_event_s) representing sync error (e.g., 5-15 deg).
      - Amplitude coupled with frequency during the ringdown phase.
      - Parametrized inter-harmonics and multi-spectral noise for Pareto analysis.
    """

    SCENARIO_NAME: ClassVar[str] = "IBR_Power_Imbalance_Ringdown"

    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {
        "duration_s":       2.5,
        "freq_nom_hz":      F_NOM,
        "amp_pre_pu":       1.0,
        "amp_fault_pu":     0.6,   # Drop during RoCoF
        "amp_post_pu":      0.95,  # Base recovery level
        "amp_freq_coupling": 0.05, # K factor coupling V(t) to Delta f(t)
        
        "t_ramp_s":         0.5,
        "rocof_hz_s":       -2.0,
        "phase_jump_ramp_rad":  45.0 * math.pi / 180.0, # Jump at fault onset
        
        "t_event_s":        1.0,
        "phase_jump_event_rad": 10.0 * math.pi / 180.0, # Sync error jump at reconn
        
        "ring_jump_hz":     1.0,
        "ring_freq_hz":     4.0,
        "ring_tau_s":       0.25,
        
        "dc_offset_pu":     0.12,
        "dc_tau_s":         0.03,
        
        "interharmonic_hz": 32.5,
        "interharmonic_pu": 0.02,  # 2% interharmonic by default
        
        "white_noise_sigma": 0.005,
        "brown_noise_sigma": 0.000000001,
        "impulse_prob":      0.001,
        "impulse_mag":       0.2,
        "seed":              None,
    }

    MONTE_CARLO_SPACE: ClassVar[dict[str, Any]] = {
        "phase_jump_event_rad": {
            "kind": "uniform",
            "low":  5.0 * math.pi / 180.0,
            "high": 15.0 * math.pi / 180.0,
        },
        "amp_fault_pu": {
            "kind": "uniform",
            "low":  0.4,
            "high": 0.8,
        },
        "ring_tau_s": {
            "kind": "uniform",
            "low":  0.1,
            "high": 0.4,
        },
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
        duration_s:        float = 2.5,
        freq_nom_hz:       float = F_NOM,
        amp_pre_pu:        float = 1.0,
        amp_fault_pu:      float = 0.6,
        amp_post_pu:       float = 0.95,
        amp_freq_coupling: float = 0.05,
        t_ramp_s:          float = 0.5,
        rocof_hz_s:        float = -2.0,
        phase_jump_ramp_rad: float = 45.0 * math.pi / 180.0,
        t_event_s:         float = 1.0,
        phase_jump_event_rad: float = 10.0 * math.pi / 180.0,
        ring_jump_hz:      float = 1.0,
        ring_freq_hz:      float = 4.0,
        ring_tau_s:        float = 0.25,
        dc_offset_pu:      float = 0.12,
        dc_tau_s:          float = 0.03,
        interharmonic_hz:  float = 32.5,
        interharmonic_pu:  float = 0.02,
        white_noise_sigma: float = 0.005,
        brown_noise_sigma: float = 0.0003,
        impulse_prob:      float = 0.001,
        impulse_mag:       float = 0.2,
        seed:              int | None = None,
    ) -> ScenarioData:

        rng = np.random.default_rng(seed)
        dt  = 1.0 / FS_PHYSICS
        t   = np.arange(0.0, duration_s, dt, dtype=float)

        mask_pre  = t < t_ramp_s
        mask_ramp = (t >= t_ramp_s) & (t < t_event_s)
        mask_ring = t >= t_event_s

        # ── 1. Ground-truth frequency profile ──
        f_true = np.zeros_like(t, dtype=float)
        f_true[mask_pre]  = freq_nom_hz
        if np.any(mask_ramp):
            dt_r = t[mask_ramp] - t_ramp_s
            f_true[mask_ramp] = freq_nom_hz + rocof_hz_s * dt_r
        if np.any(mask_ring):
            alpha  = 1.0 / ring_tau_s
            omega  = 2.0 * math.pi * ring_freq_hz
            t_post = t[mask_ring] - t_event_s
            f_true[mask_ring] = (
                freq_nom_hz
                + ring_jump_hz * np.exp(-alpha * t_post) * np.cos(omega * t_post)
            )

        # ── 2. Phase Integration with TWO Jumps ──
        phi = np.zeros_like(t, dtype=float)

        phi[mask_pre] = 2.0 * math.pi * freq_nom_hz * t[mask_pre]
        phi_ramp_start = 2.0 * math.pi * freq_nom_hz * t_ramp_s

        # Jump 1 at t_ramp_s
        if np.any(mask_ramp):
            dt_r = t[mask_ramp] - t_ramp_s
            phi[mask_ramp] = (
                phi_ramp_start 
                + phase_jump_ramp_rad 
                + 2.0 * math.pi * (freq_nom_hz * dt_r + 0.5 * rocof_hz_s * dt_r ** 2)
            )
            phi_event_pre = phi[mask_ramp][-1]
        else:
            phi_event_pre = phi_ramp_start + phase_jump_ramp_rad

        # Jump 2 at t_event_s
        if np.any(mask_ring):
            phi_event_post = phi_event_pre + phase_jump_event_rad
            alpha  = 1.0 / ring_tau_s
            omega  = 2.0 * math.pi * ring_freq_hz
            t_post = t[mask_ring] - t_event_s
            
            den = alpha ** 2 + omega ** 2
            ring_int = (
                alpha
                - np.exp(-alpha * t_post) * (alpha * np.cos(omega * t_post) - omega * np.sin(omega * t_post))
            ) / den
            
            phi[mask_ring] = (
                phi_event_post
                + 2.0 * math.pi * (freq_nom_hz * t_post + ring_jump_hz * ring_int)
            )

        # ── 3. Amplitude Profile Coupled with Frequency ──
        amp = np.full_like(t, amp_pre_pu, dtype=float)
        amp[mask_ramp] = amp_fault_pu
        
        # El voltaje se recupera pero sigue la oscilación de la frecuencia
        if np.any(mask_ring):
            delta_f = f_true[mask_ring] - freq_nom_hz
            amp[mask_ring] = amp_post_pu + (amp_freq_coupling * delta_f)

        v = amp * np.sin(phi)

        # ── 4. Harmonics & Inter-harmonics ──
        h5_off  = rng.uniform(0.0, 2.0 * math.pi)
        h7_off  = rng.uniform(0.0, 2.0 * math.pi)
        v += 0.040 * amp * np.sin(5.0  * phi + h5_off)
        v += 0.025 * amp * np.sin(7.0  * phi + h7_off)

        ih_off = rng.uniform(0.0, 2.0 * math.pi)
        v += interharmonic_pu * amp * np.sin(2.0 * math.pi * interharmonic_hz * t + ih_off)

        # ── 5. Decaying DC ──
        if np.any(mask_ring):
            t_post_dc = t[mask_ring] - t_event_s
            v[mask_ring] += dc_offset_pu * np.exp(-t_post_dc / dc_tau_s)

        # ── 6. Noise ──
        noise = rng.normal(0.0, white_noise_sigma, size=t.shape)
        if brown_noise_sigma > 0.0:
            brown = np.cumsum(rng.normal(0.0, brown_noise_sigma, size=t.shape))
            noise += (brown - brown.mean())
        if impulse_prob > 0.0:
            signs  = rng.choice([-1, 1], size=t.shape)
            noise += (rng.random(size=t.shape) < impulse_prob).astype(float) * signs * impulse_mag

        v += noise

        # AQUÍ ESTÁ LA CORRECCIÓN: Agregar el diccionario "parameters" al meta
        meta = {
            "description": "IBR Ringdown: Dual phase jumps, coupled V-f dynamics.",
            "monte_carlo_space": cls.MONTE_CARLO_SPACE,
            "fs_physics_hz": FS_PHYSICS,
            "parameters": {
                "duration_s":        duration_s,
                "freq_nom_hz":       freq_nom_hz,
                "amp_pre_pu":        amp_pre_pu,
                "amp_fault_pu":      amp_fault_pu,
                "amp_post_pu":       amp_post_pu,
                "amp_freq_coupling": amp_freq_coupling,
                "t_ramp_s":          t_ramp_s,
                "rocof_hz_s":        rocof_hz_s,
                "phase_jump_ramp_rad": phase_jump_ramp_rad,
                "t_event_s":         t_event_s,
                "phase_jump_event_rad": phase_jump_event_rad,
                "ring_jump_hz":      ring_jump_hz,
                "ring_freq_hz":      ring_freq_hz,
                "ring_tau_s":        ring_tau_s,
                "dc_offset_pu":      dc_offset_pu,
                "dc_tau_s":          dc_tau_s,
                "interharmonic_hz":  interharmonic_hz,
                "interharmonic_pu":  interharmonic_pu,
                "white_noise_sigma": white_noise_sigma,
                "brown_noise_sigma": brown_noise_sigma,
                "impulse_prob":      impulse_prob,
                "impulse_mag":       impulse_mag,
                "seed":              seed,
            }
        }

        return ScenarioData(name=cls.SCENARIO_NAME, t=t, v=v, f_true=f_true, meta=meta)