from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np

from .base import Scenario, ScenarioData
from estimators.common import FS_PHYSICS, F_NOM


class IBRMultiEventScenario(Scenario):
    """
    IBR Multi-Event Composite Fault Scenario — IEEE/Journal Stress Test.

    Simulates a severely degraded IBR-dominated grid combining seven simultaneous
    disturbances in a single 5-second waveform:

      1. Voltage Sag or Swell       — amplitude step at t_event_s
      2. Phase Angle Jump           — instantaneous Δφ at t_event_s
      3. Frequency RoCoF Ramp        — linear frequency ramp for rocof_duration_s,
                                       then clamps to f_steady (primary-response nadir)
      4. Low-frequency Oscillations  — AM (inter-area swing) + PM (angle oscillation)
      5. Decaying DC Offset          — inductive fault response, τ ≈ 30 ms
      6. Full IBR Harmonic Spectrum  — 3rd (triplen), 5th, 7th, 11th, 13th, 17th, 19th 
                                       and a 75 Hz inter-harmonic (MPPT hunting)
      7. Multi-spectral Noise        — white AWGN + brown drift + impulsive spikes

    Phase continuity is maintained analytically across all three time regions
    (pre-event, RoCoF ramp, steady nadir) to ensure correct instantaneous frequency.

    Reference: exceeds IEC 60255-118-1 dynamic test requirements; intended for
    worst-case IBR algorithm validation in low-inertia grids.
    """

    SCENARIO_NAME: ClassVar[str] = "IBR_Multi_Event"

    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {
        "duration_s":       5.0,
        "freq_pre_hz":      F_NOM,
        "rocof_hz_s":       -2.0,    # Hz/s — negative = frequency drops
        "rocof_duration_s": 1.0,     # s    — RoCoF active window; clamps after this
        "amp_pre_pu":       1.0,
        "amp_post_pu":      0.5,     # <1.0 = sag, >1.0 = swell
        "phase_jump_rad":   math.pi / 4.0,  # +45°
        "t_event_s":        0.5,
        # Decaying DC (inductive fault response)
        "dc_offset_pu":     0.15,    # peak DC at fault instant
        "dc_tau_s":         0.03,    # time constant — decays in ~2 fundamental cycles
        # Inter-area oscillations
        "am_mag_pu":        0.05,    # amplitude modulation depth
        "am_freq_hz":       1.5,     # typical inter-area frequency
        "pm_mag_rad":       0.05,    # phase modulation magnitude
        "pm_freq_hz":       1.0,
        # Multi-spectral noise
        "white_noise_sigma": 0.005,
        "brown_noise_sigma": 0.0001,
        "impulse_prob":      0.001,  # probability per sample
        "impulse_mag":       0.2,
        "seed":              None,
    }

    MONTE_CARLO_SPACE: ClassVar[dict[str, Any]] = {
        "amp_post_pu": {
            "kind": "uniform",
            "low":  0.2,   # deep sag
            "high": 1.3,   # swell
        },
        "phase_jump_rad": {
            "kind": "uniform",
            "low":  -math.pi / 3.0,  # -60°
            "high":  math.pi / 3.0,  # +60°
        },
        "rocof_hz_s": {
            "kind": "uniform",
            "low":  -5.0,  # extreme generation loss
            "high":  3.0,  # large load loss
        },
        "rocof_duration_s": {
            "kind": "uniform",
            "low":  0.3,   # fast primary response
            "high": 1.5,   # slow governor response
        },
        "dc_offset_pu": {
            "kind": "uniform",
            "low":  0.05,
            "high": 0.25,
        },
    }

    @classmethod
    def validate_params(cls, params: dict[str, Any]) -> None:
        if params.get("duration_s", 0) <= 0:
            raise ValueError("duration_s must be > 0")
        if params.get("amp_pre_pu", -1) <= 0 or params.get("amp_post_pu", -1) <= 0:
            raise ValueError("amp_pre_pu and amp_post_pu must be > 0")
        t_evt = params.get("t_event_s", -1)
        dur   = params.get("duration_s", 0)
        if t_evt < 0 or t_evt > dur:
            raise ValueError("t_event_s must be within [0, duration_s]")
        if params.get("rocof_duration_s", -1) <= 0:
            raise ValueError("rocof_duration_s must be > 0")

    @classmethod
    def generate(
        cls,
        duration_s:        float = 5.0,
        freq_pre_hz:       float = F_NOM,
        rocof_hz_s:        float = -2.0,
        rocof_duration_s:  float = 1.0,
        amp_pre_pu:        float = 1.0,
        amp_post_pu:       float = 0.5,
        phase_jump_rad:    float = math.pi / 4.0,
        t_event_s:         float = 0.5,
        dc_offset_pu:      float = 0.15,
        dc_tau_s:          float = 0.013,
        am_mag_pu:         float = 0.05,
        am_freq_hz:        float = 1.5,
        pm_mag_rad:        float = 0.05,
        pm_freq_hz:        float = 1.0,
        white_noise_sigma: float = 0.005,
        brown_noise_sigma: float = 0.000000005,
        impulse_prob:      float = 0.001,
        impulse_mag:       float = 0.2,
        seed:              int | None = None,
    ) -> ScenarioData:

        rng = np.random.default_rng(seed)
        dt  = 1.0 / FS_PHYSICS
        t   = np.arange(0.0, duration_s, dt, dtype=float)

        t_ramp_end = t_event_s + rocof_duration_s
        f_steady   = freq_pre_hz + rocof_hz_s * rocof_duration_s

        # ── Region masks ────────────────────────────────────────────────
        pre_mask  = t <  t_event_s
        ramp_mask = (t >= t_event_s) & (t < t_ramp_end)
        hold_mask = t >= t_ramp_end

        # ── 1. Ground-truth frequency profile ───────────────────────────
        f_true = np.full_like(t, freq_pre_hz, dtype=float)
        if np.any(ramp_mask):
            t_r = t[ramp_mask] - t_event_s
            f_true[ramp_mask] = freq_pre_hz + rocof_hz_s * t_r
        f_true[hold_mask] = f_steady

        # ── 2. Amplitude profile (step sag/swell at fault) ───────────────
        amp = np.full_like(t, amp_pre_pu, dtype=float)
        amp[ramp_mask] = amp_post_pu
        amp[hold_mask] = amp_post_pu

        # ── 3. Phase — analytically integrated across all three regions ──
        #
        #  Region 1 (pre):   φ = 2π f0 t
        #  Region 2 (ramp):  φ = φ_event + Δφ_jump + 2π(f0·t_r + ½R·t_r²)
        #  Region 3 (hold):  φ = φ_ramp_end + 2π f_steady · t_s
        #
        #  Continuity is exact at both boundaries.
        # ─────────────────────────────────────────────────────────────────
        phi = np.zeros_like(t, dtype=float)

        phi[pre_mask] = 2.0 * math.pi * freq_pre_hz * t[pre_mask]

        phi_at_event = 2.0 * math.pi * freq_pre_hz * t_event_s

        if np.any(ramp_mask):
            t_r = t[ramp_mask] - t_event_s
            phi[ramp_mask] = (
                phi_at_event
                + phase_jump_rad
                + 2.0 * math.pi * (freq_pre_hz * t_r + 0.5 * rocof_hz_s * t_r ** 2)
            )

        if np.any(hold_mask):
            phi_at_ramp_end = (
                phi_at_event
                + phase_jump_rad
                + 2.0 * math.pi * (
                    freq_pre_hz * rocof_duration_s
                    + 0.5 * rocof_hz_s * rocof_duration_s ** 2
                )
            )
            t_s = t[hold_mask] - t_ramp_end
            phi[hold_mask] = phi_at_ramp_end + 2.0 * math.pi * f_steady * t_s

        # ── 4. AM + PM modulation (inter-area oscillations) ─────────────
        am_mod = 1.0 + am_mag_pu * np.cos(2.0 * math.pi * am_freq_hz * t)
        pm_mod = pm_mag_rad * np.cos(2.0 * math.pi * pm_freq_hz * t)
        phi   += pm_mod
        # PM contribution to instantaneous frequency: d(pm_mod)/dt / 2π
        f_true += -pm_mag_rad * pm_freq_hz * np.sin(2.0 * math.pi * pm_freq_hz * t)

        # ── 5. Fundamental voltage ───────────────────────────────────────
        v = amp * am_mod * np.sin(phi)

        # ── 6. Extended IBR Harmonics & Inter-harmonics (The "Boss Fight") ──
        
        # A. Triplen Harmonic (3rd) - Represents unbalance during the fault
        h3_phase = rng.uniform(0.0, 2.0 * math.pi)
        v += 0.03 * amp * np.sin(3.0 * phi + h3_phase)

        # B. Standard IEEE 519 Characteristic Harmonics (5th, 7th, 11th, 13th, 17th, 19th)
        # Amplitudes strictly follow typical 1/h decay rules for 6-pulse/12-pulse inverters
        h5_phase  = rng.uniform(0.0, 2.0 * math.pi)
        h7_phase  = rng.uniform(0.0, 2.0 * math.pi)
        h11_phase = rng.uniform(0.0, 2.0 * math.pi)
        h13_phase = rng.uniform(0.0, 2.0 * math.pi)
        h17_phase = rng.uniform(0.0, 2.0 * math.pi)
        h19_phase = rng.uniform(0.0, 2.0 * math.pi)

        v += 0.040 * amp * np.sin(5.0 * phi + h5_phase)
        v += 0.020 * amp * np.sin(7.0 * phi + h7_phase)
        v += 0.015 * amp * np.sin(11.0 * phi + h11_phase)
        v += 0.010 * amp * np.sin(13.0 * phi + h13_phase)
        v += 0.005 * amp * np.sin(17.0 * phi + h17_phase)
        v += 0.005 * amp * np.sin(19.0 * phi + h19_phase)

        # C. Inter-harmonic (75 Hz) - Sub-synchronous control interactions (e.g. MPPT hunting)
        # We use strict time (t) here instead of phi because inter-harmonics 
        # often operate at fixed frequencies decoupled from the grid RoCoF.
        ih75_phase = rng.uniform(0.0, 2.0 * math.pi)
        v += 0.015 * amp * np.sin(2.0 * math.pi * 75.0 * t + ih75_phase)

        # ── 7. Decaying DC offset (inductive fault response) ─────────────
        if np.any(ramp_mask) or np.any(hold_mask):
            post_mask = ramp_mask | hold_mask
            t_post    = t[post_mask] - t_event_s
            v[post_mask] += dc_offset_pu * np.exp(-t_post / dc_tau_s)

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

        # ── Metadata ─────────────────────────────────────────────────────
        jump_deg = round(math.degrees(phase_jump_rad), 1)
        meta = {
            "description": (
                f"IBR Multi-Event: {amp_post_pu} pu sag/swell, Δφ {jump_deg}°, "
                f"RoCoF {rocof_hz_s} Hz/s for {rocof_duration_s} s "
                f"→ f_nadir = {f_steady:.3f} Hz"
            ),
            "standard":    "Extreme composite dynamic test (beyond IEC 60255-118-1)",
            "dynamics":    (
                "Amplitude step, phase jump, RoCoF ramp with nadir clamp, "
                "decaying DC, AM/PM oscillations, Full IBR Spectrum "
                "(3rd, 5th, 7th, 11th, 13th, 17th, 19th + 75Hz Inter-harmonic), "
                "white + brown + impulsive noise."
            ),
            "purpose":     "IBR algorithm failure-mode analysis; worst-case stress test.",
            "parameters": {
                "freq_pre_hz":      freq_pre_hz,
                "f_steady_hz":      f_steady,
                "rocof_hz_s":       rocof_hz_s,
                "rocof_duration_s": rocof_duration_s,
                "t_event_s":        t_event_s,
                "amp_pre_pu":       amp_pre_pu,
                "amp_post_pu":      amp_post_pu,
                "phase_jump_rad":   phase_jump_rad,
                "dc_offset_pu":     dc_offset_pu,
                "dc_tau_s":         dc_tau_s,
            },
            "monte_carlo_space": cls.MONTE_CARLO_SPACE,
            "fs_physics_hz":     FS_PHYSICS,
        }

        return ScenarioData(name=cls.SCENARIO_NAME, t=t, v=v, f_true=f_true, meta=meta)