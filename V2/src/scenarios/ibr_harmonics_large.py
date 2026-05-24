from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np

from .base import Scenario, ScenarioData
from estimators.common import FS_PHYSICS, F_NOM


class IBRHarmonicsLargeScenario(Scenario):
    """
    IBR Harmonics — Large (Level 3 / Severe IBR Environment).

    Represents a severely polluted grid: high harmonic distortion (THD ≈ 15 %,
    well beyond IEEE 519), multiple interharmonics, sub-harmonics, large noise,
    and impulsive spikes. Combined with a frequency step event.

    This scenario is specifically designed to break estimators that rely on
    spectral purity or narrow DFT windows.

    Harmonic content (relative to fundamental amplitude):
        2nd  : 2.0 %   (half-wave asymmetry from single-phase loading)
        3rd  : 4.0 %   (triplen — zero-sequence, transformer saturation)
        5th  : 8.0 %   (dominant 6-pulse IBR harmonic)
        7th  : 5.0 %
        11th : 3.0 %
        13th : 2.0 %
    Sub-harmonic:
        0.5·f₀ = 30 Hz : 1.5 %  (sub-synchronous oscillation / flicker)
    Interharmonics:
        32.5 Hz : 2.0 %  (IBR switching artefact — GaN converter)
        85.0 Hz : 1.5 %  (non-integer alias)
    Noise:
        White AWGN σ = 0.015
        Brown drift  σ_step = 0.001
        Impulsive:    prob = 0.001, mag = 0.3 pu

    Event:
        Frequency step +0.8 Hz at t_event_s (load rejection).

    Purpose: worst-case IBR harmonic / noise stress test.
    """

    SCENARIO_NAME: ClassVar[str] = "IBR_Harmonics_Large"

    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {
        "duration_s":         2.0,
        "freq_nom_hz":        F_NOM,
        "freq_step_hz":       0.8,    # Load rejection
        "t_event_s":          1.0,
        "amp_pu":             1.0,
        # Harmonics
        "h2_pct":             0.020,
        "h3_pct":             0.040,
        "h5_pct":             0.080,
        "h7_pct":             0.050,
        "h11_pct":            0.030,
        "h13_pct":            0.020,
        # Sub-harmonic
        "sub_pct":            0.015,  # at 0.5 * freq_nom_hz
        # Interharmonics
        "ih325_pct":          0.020,  # 32.5 Hz
        "ih85_pct":           0.015,  # 85.0 Hz
        # Noise
        "white_noise_sigma":  0.015,
        "brown_noise_sigma":  0.0003,
        "impulse_prob":       0.001,
        "impulse_mag":        0.30,
        "seed":               None,
    }

    MONTE_CARLO_SPACE: ClassVar[dict[str, Any]] = {
        "freq_step_hz": {
            "kind": "uniform",
            "low":  0.3,
            "high": 1.5,
        },
        "h5_pct": {
            "kind": "uniform",
            "low":  0.060,
            "high": 0.120,
        },
        "white_noise_sigma": {
            "kind": "uniform",
            "low":  0.008,
            "high": 0.025,
        },
        "impulse_prob": {
            "kind": "uniform",
            "low":  0.0005,
            "high": 0.002,
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
        freq_step_hz:      float = 0.8,
        t_event_s:         float = 1.0,
        amp_pu:            float = 1.0,
        h2_pct:            float = 0.020,
        h3_pct:            float = 0.040,
        h5_pct:            float = 0.080,
        h7_pct:            float = 0.050,
        h11_pct:           float = 0.030,
        h13_pct:           float = 0.020,
        sub_pct:           float = 0.015,
        ih325_pct:         float = 0.020,
        ih85_pct:          float = 0.015,
        white_noise_sigma: float = 0.015,
        brown_noise_sigma: float = 0.001,
        impulse_prob:      float = 0.001,
        impulse_mag:       float = 0.30,
        seed:              int | None = None,
    ) -> ScenarioData:

        rng = np.random.default_rng(seed)
        dt  = 1.0 / FS_PHYSICS
        t   = np.arange(0.0, duration_s, dt, dtype=float)

        pre_mask  = t <  t_event_s
        post_mask = t >= t_event_s

        f_pre  = freq_nom_hz
        f_post = freq_nom_hz + freq_step_hz

        # ── 1. Frequency profile (step) ──────────────────────────────────
        f_true = np.where(pre_mask, f_pre, f_post)

        # ── 2. Phase (continuous at step) ────────────────────────────────
        phi = np.zeros_like(t, dtype=float)
        phi[pre_mask] = 2.0 * math.pi * f_pre * t[pre_mask]
        if np.any(post_mask):
            phi_at_event = 2.0 * math.pi * f_pre * t_event_s
            t_post = t[post_mask] - t_event_s
            phi[post_mask] = phi_at_event + 2.0 * math.pi * f_post * t_post

        # ── 3. Fundamental voltage ───────────────────────────────────────
        v = amp_pu * np.sin(phi)

        # ── 4. Integer harmonics (phase-coherent) ────────────────────────
        v += h2_pct  * amp_pu * np.sin(2.0  * phi + rng.uniform(0, 2 * math.pi))
        v += h3_pct  * amp_pu * np.sin(3.0  * phi + rng.uniform(0, 2 * math.pi))
        v += h5_pct  * amp_pu * np.sin(5.0  * phi + rng.uniform(0, 2 * math.pi))
        v += h7_pct  * amp_pu * np.sin(7.0  * phi + rng.uniform(0, 2 * math.pi))
        v += h11_pct * amp_pu * np.sin(11.0 * phi + rng.uniform(0, 2 * math.pi))
        v += h13_pct * amp_pu * np.sin(13.0 * phi + rng.uniform(0, 2 * math.pi))

        # ── 5. Sub-harmonic at 0.5 * freq_nom_hz ─────────────────────────
        f_sub = 0.5 * freq_nom_hz
        v += sub_pct * amp_pu * np.sin(2.0 * math.pi * f_sub * t + rng.uniform(0, 2 * math.pi))

        # ── 6. Interharmonics (absolute time — not phase-locked) ─────────
        v += ih325_pct * amp_pu * np.sin(2.0 * math.pi * 32.5 * t + rng.uniform(0, 2 * math.pi))
        v += ih85_pct  * amp_pu * np.sin(2.0 * math.pi * 85.0 * t + rng.uniform(0, 2 * math.pi))

        # ── 7. Multi-spectral noise ──────────────────────────────────────
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

        thd = math.sqrt(h2_pct**2 + h3_pct**2 + h5_pct**2 + h7_pct**2 + h11_pct**2 + h13_pct**2)
        meta = {
            "description": (
                f"IBR Harmonics Large: {freq_step_hz:+.2f} Hz step at t={t_event_s}s, "
                f"THD ≈ {thd*100:.1f}%, sub-harmonic + 2 interharmonics + impulsive noise"
            ),
            "standard":   "Exceeds IEEE 519 (THD ≈ 15 %) — worst-case IBR stress test",
            "dynamics":   "Freq step, high harmonic distortion, sub-harmonic, interharmonics, impulsive noise.",
            "purpose":    "Worst-case harmonic/noise stress test to expose estimator collapse.",
            "parameters": {
                "freq_nom_hz":  freq_nom_hz,
                "freq_post_hz": f_post,
                "freq_step_hz": freq_step_hz,
                "t_event_s":    t_event_s,
                "thd_pct":      round(thd * 100, 2),
                "f_sub_hz":     f_sub,
            },
            "monte_carlo_space": cls.MONTE_CARLO_SPACE,
            "fs_physics_hz":     FS_PHYSICS,
        }

        return ScenarioData(name=cls.SCENARIO_NAME, t=t, v=v, f_true=f_true, meta=meta)
