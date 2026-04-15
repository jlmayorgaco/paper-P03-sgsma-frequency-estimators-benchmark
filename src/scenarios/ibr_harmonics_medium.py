from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np

from .base import Scenario, ScenarioData
from estimators.common import FS_PHYSICS, F_NOM


class IBRHarmonicsMediumScenario(Scenario):
    """
    IBR Harmonics — Medium (Level 2 / Realistic Grid Edge).

    Represents a realistic grid connection at the edge of IEEE 519 limits:
    moderate harmonic distortion (THD ≈ 8 %) with an interharmonic component
    and a small RoCoF event. Sensor drift (brown noise) is included.

    Harmonic content (relative to fundamental amplitude):
        3rd  : 2.0 %   (3rd-order triplen from unbalanced load)
        5th  : 5.0 %   (dominant IBR switching harmonic)
        7th  : 3.0 %
        11th : 1.5 %
        13th : 1.0 %
    Interharmonic:
        75 Hz: 2.0 %   (1.25 f₀ — non-integer switching frequency alias)
    Noise:
        White AWGN σ = 0.005
        Brown drift  σ_step = 0.0003

    Event:
        RoCoF = -1.0 Hz/s for 0.4 s starting at t_event_s, then holds at nadir.

    Purpose: test robustness under moderate harmonic pollution + drift.
    """

    SCENARIO_NAME: ClassVar[str] = "IBR_Harmonics_Medium"

    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {
        "duration_s":         2.0,
        "freq_nom_hz":        F_NOM,
        "rocof_hz_s":         -1.0,
        "rocof_duration_s":   0.4,
        "t_event_s":          0.8,
        "amp_pu":             1.0,
        # Harmonics
        "h3_pct":             0.020,
        "h5_pct":             0.050,
        "h7_pct":             0.030,
        "h11_pct":            0.015,
        "h13_pct":            0.010,
        # Interharmonic
        "ih75_pct":           0.020,
        # Noise
        "white_noise_sigma":  0.005,
        "brown_noise_sigma":  0.0001,
        "seed":               None,
    }

    MONTE_CARLO_SPACE: ClassVar[dict[str, Any]] = {
        "rocof_hz_s": {
            "kind": "uniform",
            "low":  -2.0,
            "high": -0.3,
        },
        "rocof_duration_s": {
            "kind": "uniform",
            "low":  0.2,
            "high": 0.6,
        },
        "h5_pct": {
            "kind": "uniform",
            "low":  0.030,
            "high": 0.070,
        },
        "ih75_pct": {
            "kind": "uniform",
            "low":  0.010,
            "high": 0.035,
        },
    }

    @classmethod
    def validate_params(cls, params: dict[str, Any]) -> None:
        if params.get("duration_s", 0) <= 0:
            raise ValueError("duration_s must be > 0")

    @classmethod
    def generate(
        cls,
        duration_s:        float = 2.0,
        freq_nom_hz:       float = F_NOM,
        rocof_hz_s:        float = -1.0,
        rocof_duration_s:  float = 0.4,
        t_event_s:         float = 0.8,
        amp_pu:            float = 1.0,
        h3_pct:            float = 0.020,
        h5_pct:            float = 0.050,
        h7_pct:            float = 0.030,
        h11_pct:           float = 0.015,
        h13_pct:           float = 0.010,
        ih75_pct:          float = 0.020,
        white_noise_sigma: float = 0.005,
        brown_noise_sigma: float = 0.0003,
        seed:              int | None = None,
    ) -> ScenarioData:

        rng = np.random.default_rng(seed)
        dt  = 1.0 / FS_PHYSICS
        t   = np.arange(0.0, duration_s, dt, dtype=float)

        t_ramp_end = t_event_s + rocof_duration_s
        f_steady   = freq_nom_hz + rocof_hz_s * rocof_duration_s

        pre_mask  = t <  t_event_s
        ramp_mask = (t >= t_event_s) & (t < t_ramp_end)
        hold_mask = t >= t_ramp_end

        # ── 1. Frequency profile ─────────────────────────────────────────
        f_true = np.full_like(t, freq_nom_hz, dtype=float)
        if np.any(ramp_mask):
            t_r = t[ramp_mask] - t_event_s
            f_true[ramp_mask] = freq_nom_hz + rocof_hz_s * t_r
        f_true[hold_mask] = f_steady

        # ── 2. Phase (continuous across all regions) ─────────────────────
        phi = np.zeros_like(t, dtype=float)
        phi[pre_mask] = 2.0 * math.pi * freq_nom_hz * t[pre_mask]
        phi_at_event  = 2.0 * math.pi * freq_nom_hz * t_event_s

        if np.any(ramp_mask):
            t_r = t[ramp_mask] - t_event_s
            phi[ramp_mask] = (
                phi_at_event
                + 2.0 * math.pi * (freq_nom_hz * t_r + 0.5 * rocof_hz_s * t_r ** 2)
            )

        if np.any(hold_mask):
            phi_at_ramp_end = (
                phi_at_event
                + 2.0 * math.pi * (
                    freq_nom_hz * rocof_duration_s
                    + 0.5 * rocof_hz_s * rocof_duration_s ** 2
                )
            )
            t_s = t[hold_mask] - t_ramp_end
            phi[hold_mask] = phi_at_ramp_end + 2.0 * math.pi * f_steady * t_s

        # ── 3. Fundamental voltage ───────────────────────────────────────
        v = amp_pu * np.sin(phi)

        # ── 4. Harmonics (phase-coherent) ────────────────────────────────
        v += h3_pct  * amp_pu * np.sin(3.0  * phi + rng.uniform(0, 2 * math.pi))
        v += h5_pct  * amp_pu * np.sin(5.0  * phi + rng.uniform(0, 2 * math.pi))
        v += h7_pct  * amp_pu * np.sin(7.0  * phi + rng.uniform(0, 2 * math.pi))
        v += h11_pct * amp_pu * np.sin(11.0 * phi + rng.uniform(0, 2 * math.pi))
        v += h13_pct * amp_pu * np.sin(13.0 * phi + rng.uniform(0, 2 * math.pi))

        # ── 5. Interharmonic at 75 Hz ────────────────────────────────────
        v += ih75_pct * amp_pu * np.sin(2.0 * math.pi * 75.0 * t + rng.uniform(0, 2 * math.pi))

        # ── 6. Multi-spectral noise ──────────────────────────────────────
        noise = rng.normal(0.0, white_noise_sigma, size=t.shape)
        if brown_noise_sigma > 0.0:
            brown = np.cumsum(rng.normal(0.0, brown_noise_sigma, size=t.shape))
            brown -= brown.mean()
            noise += brown
        v += noise

        thd = math.sqrt(h3_pct**2 + h5_pct**2 + h7_pct**2 + h11_pct**2 + h13_pct**2)
        meta = {
            "description": (
                f"IBR Harmonics Medium: RoCoF {rocof_hz_s} Hz/s for {rocof_duration_s} s "
                f"→ f_nadir={f_steady:.2f} Hz, THD ≈ {thd*100:.1f}%"
            ),
            "standard":   "Near IEEE 519 limit (THD ≈ 8 %)",
            "dynamics":   "RoCoF ramp + nadir hold, moderate harmonics, interharmonic at 75 Hz, brown noise.",
            "purpose":    "Harmonic robustness under realistic IBR grid edge conditions.",
            "parameters": {
                "freq_nom_hz":      freq_nom_hz,
                "f_steady_hz":      f_steady,
                "rocof_hz_s":       rocof_hz_s,
                "rocof_duration_s": rocof_duration_s,
                "t_event_s":        t_event_s,
                "thd_pct":          round(thd * 100, 2),
            },
            "monte_carlo_space": cls.MONTE_CARLO_SPACE,
            "fs_physics_hz":     FS_PHYSICS,
        }

        return ScenarioData(name=cls.SCENARIO_NAME, t=t, v=v, f_true=f_true, meta=meta)
