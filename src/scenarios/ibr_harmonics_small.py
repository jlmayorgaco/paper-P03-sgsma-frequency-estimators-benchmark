from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np

from .base import Scenario, ScenarioData
from estimators.common import FS_PHYSICS, F_NOM


class IBRHarmonicsSmallScenario(Scenario):
    """
    IBR Harmonics — Small (Level 1 / IEEE 519 Compliant).

    Represents a good-quality IBR grid connection: low harmonic distortion
    (THD < 5 %, within IEEE 519 limits) with baseline sensor noise only.
    A small frequency step at t_event_s tests harmonic rejection capability
    when the perturbation is barely above noise floor.

    Harmonic content:
        5th  : 2.0 %   (dominant odd harmonic from 6-pulse rectifier)
        7th  : 1.5 %
        11th : 0.5 %   (reduced — good input filter)
    Noise:
        White AWGN σ = 0.002 (SNR ≈ 54 dB)
        No brown drift, no impulses.

    Purpose: baseline harmonic robustness benchmark under clean conditions.
    """

    SCENARIO_NAME: ClassVar[str] = "IBR_Harmonics_Small"

    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {
        "duration_s":         2.0,
        "freq_nom_hz":        F_NOM,
        "freq_step_hz":       0.3,    # Small step — hard to detect above harmonic noise
        "t_event_s":          1.0,
        "amp_pu":             1.0,
        # Harmonics (% of fundamental)
        "h5_pct":             0.020,
        "h7_pct":             0.015,
        "h11_pct":            0.005,
        # Noise
        "white_noise_sigma":  0.002,
        "seed":               None,
    }

    MONTE_CARLO_SPACE: ClassVar[dict[str, Any]] = {
        "freq_step_hz": {
            "kind": "uniform",
            "low":  -0.5,
            "high":  0.5,
        },
        "h5_pct": {
            "kind": "uniform",
            "low":  0.010,
            "high": 0.030,
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
        freq_step_hz:      float = 0.3,
        t_event_s:         float = 1.0,
        amp_pu:            float = 1.0,
        h5_pct:            float = 0.020,
        h7_pct:            float = 0.015,
        h11_pct:           float = 0.005,
        white_noise_sigma: float = 0.002,
        seed:              int | None = None,
    ) -> ScenarioData:

        rng = np.random.default_rng(seed)
        dt  = 1.0 / FS_PHYSICS
        t   = np.arange(0.0, duration_s, dt, dtype=float)

        pre_mask  = t <  t_event_s
        post_mask = t >= t_event_s

        # ── 1. Frequency profile (step) ──────────────────────────────────
        f_pre  = freq_nom_hz
        f_post = freq_nom_hz + freq_step_hz
        f_true = np.where(pre_mask, f_pre, f_post)

        # ── 2. Phase (analytically continuous at step) ───────────────────
        phi = np.zeros_like(t, dtype=float)
        phi[pre_mask] = 2.0 * math.pi * f_pre * t[pre_mask]
        if np.any(post_mask):
            phi_at_event = 2.0 * math.pi * f_pre * t_event_s
            t_post = t[post_mask] - t_event_s
            phi[post_mask] = phi_at_event + 2.0 * math.pi * f_post * t_post

        # ── 3. Fundamental voltage ───────────────────────────────────────
        v = amp_pu * np.sin(phi)

        # ── 4. Harmonics (phase-coherent, random initial offsets) ────────
        v += h5_pct  * amp_pu * np.sin(5.0  * phi + rng.uniform(0, 2 * math.pi))
        v += h7_pct  * amp_pu * np.sin(7.0  * phi + rng.uniform(0, 2 * math.pi))
        v += h11_pct * amp_pu * np.sin(11.0 * phi + rng.uniform(0, 2 * math.pi))

        # ── 5. White noise ───────────────────────────────────────────────
        v += rng.normal(0.0, white_noise_sigma, size=t.shape)

        meta = {
            "description": (
                f"IBR Harmonics Small: {freq_step_hz:+.2f} Hz step at t={t_event_s}s, "
                f"THD ≈ {math.sqrt(h5_pct**2 + h7_pct**2 + h11_pct**2)*100:.1f}%"
            ),
            "standard":   "IEEE 519 compliant (THD < 5 %)",
            "dynamics":   "Small frequency step with low harmonic distortion and white noise only.",
            "purpose":    "Baseline harmonic rejection benchmark under clean IBR conditions.",
            "parameters": {
                "freq_nom_hz":  freq_nom_hz,
                "freq_step_hz": freq_step_hz,
                "t_event_s":    t_event_s,
                "amp_pu":       amp_pu,
                "h5_pct":       h5_pct,
                "h7_pct":       h7_pct,
                "h11_pct":      h11_pct,
            },
            "monte_carlo_space": cls.MONTE_CARLO_SPACE,
            "fs_physics_hz":     FS_PHYSICS,
        }

        return ScenarioData(name=cls.SCENARIO_NAME, t=t, v=v, f_true=f_true, meta=meta)
