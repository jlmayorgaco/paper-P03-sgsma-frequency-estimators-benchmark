from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np

from .base import Scenario, ScenarioData
from estimators.common import FS_PHYSICS, F_NOM


class IBRPowerImbalanceRingdownScenario(Scenario):
    """
    IBR Power Imbalance & Control Ring-down Scenario — IEEE/Journal Stress Test.

    Three-region waveform simulating a realistic VSG/GFM power imbalance event:

      Region 1 (nominal):   Clean pre-fault signal — establishes steady-state baseline.
      Region 2 (RoCoF ramp):  Negative frequency ramp from t_ramp_s → t_event_s,
                               mimicking inertia loss / generation dropout.
      Region 3 (ring-down):  Underdamped frequency recovery oscillation after
                               simultaneous phase jump + voltage sag, representing
                               PID/VSG controller transient response.

    Overlaid throughout:
      - IEEE 519 harmonic background (5th, 7th, 11th, 13th)
      - IBR interharmonic at 32.5 Hz (switching artefact)
      - Decaying DC offset at fault instant (inductive fault response)
      - Multi-spectral noise: white AWGN + brown drift + impulsive spikes

    Phase continuity is maintained analytically across all three region boundaries.
    """

    SCENARIO_NAME: ClassVar[str] = "IBR_Power_Imbalance_Ringdown"

    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {
        "duration_s":       2.5,
        "freq_nom_hz":      F_NOM,
        "amp_pre_pu":       1.0,
        "amp_post_pu":      0.8,
        "t_ramp_s":         0.5,
        "rocof_hz_s":       -2.0,
        "t_event_s":        1.0,
        # AJUSTADO: Salto de 30° es mucho más realista para un cambio de topología/desbalance
        "phase_jump_rad":   30.0 * math.pi / 180.0,  
        # AJUSTADO: Magnitud inicial del ring-down reducida a 1.0 Hz
        "ring_jump_hz":     1.0,
        "ring_freq_hz":     4.0,
        "ring_tau_s":       0.25,
        # Decaying DC
        "dc_offset_pu":     0.12,
        "dc_tau_s":         0.03,
        # Multi-spectral noise
        "white_noise_sigma": 0.005,
        "brown_noise_sigma": 0.000000001,
        "impulse_prob":      0.001,
        "impulse_mag":       0.2,
        "seed":              None,
    }

    MONTE_CARLO_SPACE: ClassVar[dict[str, Any]] = {
        "phase_jump_rad": {
            "kind": "uniform",
            "low":  10.0 * math.pi / 180.0, # Mínimo 10°
            "high": 45.0 * math.pi / 180.0, # Máximo 45°
        },
        "amp_post_pu": {
            "kind": "uniform",
            "low":  0.5,
            "high": 0.9,
        },
        "ring_tau_s": {
            "kind": "uniform",
            "low":  0.1,
            "high": 0.5,
        },
        "t_event_s": {
            "kind": "uniform",
            "low":  0.8,
            "high": 1.2,
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
        amp_post_pu:       float = 0.8,
        t_ramp_s:          float = 0.5,
        rocof_hz_s:        float = -2.0,
        t_event_s:         float = 1.0,
        phase_jump_rad:    float = 30.0 * math.pi / 180.0,  # Actualizado aquí también
        ring_jump_hz:      float = 1.0,                     # Actualizado aquí también
        ring_freq_hz:      float = 4.0,
        ring_tau_s:        float = 0.25,
        dc_offset_pu:      float = 0.12,
        dc_tau_s:          float = 0.03,
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

        # ── 1. Ground-truth frequency profile ───────────────────────────
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

        # ── 2. Phase — analytically integrated across all three regions ──
        phi = np.zeros_like(t, dtype=float)

        phi[mask_pre] = 2.0 * math.pi * freq_nom_hz * t[mask_pre]

        phi_ramp_start = 2.0 * math.pi * freq_nom_hz * t_ramp_s
        if np.any(mask_ramp):
            dt_r = t[mask_ramp] - t_ramp_s
            phi[mask_ramp] = (
                phi_ramp_start
                + 2.0 * math.pi * (freq_nom_hz * dt_r + 0.5 * rocof_hz_s * dt_r ** 2)
            )
            phi_event_pre = phi[mask_ramp][-1]
        else:
            phi_event_pre = phi_ramp_start

        if np.any(mask_ring):
            phi_event_post = phi_event_pre + phase_jump_rad
            alpha  = 1.0 / ring_tau_s
            omega  = 2.0 * math.pi * ring_freq_hz
            t_post = t[mask_ring] - t_event_s
            # Exact integral of ring_jump_hz * exp(-alpha*t)*cos(omega*t):
            #   ∫ = (alpha - exp(-a*t)*(a*cos(w*t) - w*sin(w*t))) / (a²+w²)
            den = alpha ** 2 + omega ** 2
            ring_int = (
                alpha
                - np.exp(-alpha * t_post) * (alpha * np.cos(omega * t_post) - omega * np.sin(omega * t_post))
            ) / den
            phi[mask_ring] = (
                phi_event_post
                + 2.0 * math.pi * (freq_nom_hz * t_post + ring_jump_hz * ring_int)
            )

        # ── 3. Amplitude profile ─────────────────────────────────────────
        amp = np.full_like(t, amp_pre_pu, dtype=float)
        amp[mask_ring] = amp_post_pu

        # ── 4. Fundamental voltage ───────────────────────────────────────
        v = amp * np.sin(phi)

        # ── 5. IEEE 519 harmonics (phase-coherent with fundamental) ──────
        h5_off  = rng.uniform(0.0, 2.0 * math.pi)
        h7_off  = rng.uniform(0.0, 2.0 * math.pi)
        h11_off = rng.uniform(0.0, 2.0 * math.pi)
        h13_off = rng.uniform(0.0, 2.0 * math.pi)
        v += 0.040 * amp * np.sin(5.0  * phi + h5_off)
        v += 0.025 * amp * np.sin(7.0  * phi + h7_off)
        v += 0.015 * amp * np.sin(11.0 * phi + h11_off)
        v += 0.010 * amp * np.sin(13.0 * phi + h13_off)

        # ── 6. IBR interharmonic at 32.5 Hz (switching artefact) ─────────
        ih_off = rng.uniform(0.0, 2.0 * math.pi)
        v += 0.020 * amp * np.sin(2.0 * math.pi * 32.5 * t + ih_off)

        # ── 7. Decaying DC offset at fault instant ────────────────────────
        if np.any(mask_ring):
            t_post_dc = t[mask_ring] - t_event_s
            v[mask_ring] += dc_offset_pu * np.exp(-t_post_dc / dc_tau_s)

        # ── 8. Multi-spectral noise ──────────────────────────────────────
        noise = rng.normal(0.0, white_noise_sigma, size=t.shape)

        if brown_noise_sigma > 0.0:
            brown = np.cumsum(rng.normal(0.0, brown_noise_sigma, size=t.shape))
            brown -= brown.mean()
            noise += brown

        if impulse_prob > 0.0:
            signs  = rng.choice([-1, 1], size=t.shape)
            spikes = (rng.random(size=t.shape) < impulse_prob).astype(float) * signs * impulse_mag
            noise += spikes

        v += noise

        jump_deg = round(math.degrees(phase_jump_rad), 1)
        meta = {
            "description": (
                f"IBR Power Imbalance Ring-down: {amp_post_pu} pu sag, "
                f"Δφ +{jump_deg}°, RoCoF {rocof_hz_s} Hz/s, "
                f"ring-down f={ring_freq_hz} Hz τ={ring_tau_s} s"
            ),
            "standard":   "Complex transient based on GFM/VSG control responses",
            "dynamics":   (
                "Nominal → RoCoF ramp → phase jump + sag + underdamped ring-down; "
                "IEEE 519 harmonics, 32.5 Hz interharmonic, decaying DC, "
                "white + brown + impulsive noise."
            ),
            "purpose":    "Evaluate estimator stability under combined RoCoF, phase jump, and ring-down.",
            "parameters": {
                "freq_nom_hz":    freq_nom_hz,
                "amp_pre_pu":     amp_pre_pu,
                "amp_post_pu":    amp_post_pu,
                "t_ramp_s":       t_ramp_s,
                "rocof_hz_s":     rocof_hz_s,
                "t_event_s":      t_event_s,
                "phase_jump_rad": phase_jump_rad,
                "ring_jump_hz":   ring_jump_hz,
                "ring_freq_hz":   ring_freq_hz,
                "ring_tau_s":     ring_tau_s,
                "dc_offset_pu":   dc_offset_pu,
                "dc_tau_s":       dc_tau_s,
            },
            "monte_carlo_space": cls.MONTE_CARLO_SPACE,
            "fs_physics_hz":     FS_PHYSICS,
        }

        return ScenarioData(name=cls.SCENARIO_NAME, t=t, v=v, f_true=f_true, meta=meta)