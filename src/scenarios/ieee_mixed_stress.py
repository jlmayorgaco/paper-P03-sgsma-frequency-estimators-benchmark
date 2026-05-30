from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np

from .base import Scenario, ScenarioData
from estimators.common import FS_PHYSICS, F_NOM

# Harmonic weight profile (unnormalized) — same ratios as atlas_sweep harmonics sweep.
_HARM_WEIGHTS = {2: 0.25, 3: 0.50, 5: 1.00, 7: 0.75, 11: 0.375, 13: 0.25}
_HARM_WEIGHT_NORM = math.sqrt(sum(w * w for w in _HARM_WEIGHTS.values()))

# Bernoulli-Gaussian spike parameters (variance-matched to noise_sigma_pu²).
_BG_P_SPIKE = 0.05
_BG_SPIKE_RATIO = 3.0
_BG_VARIANCE_FACTOR = 1.0 + _BG_SPIKE_RATIO**2 * _BG_P_SPIKE  # = 1.45


def _harmonic_coefficients(thd_pu: float) -> dict[int, float]:
    """Return per-harmonic amplitudes (in pu) that achieve the requested THD."""
    return {k: float(max(0.0, thd_pu) * w / _HARM_WEIGHT_NORM) for k, w in _HARM_WEIGHTS.items()}


class IEEEMixedStressScenario(Scenario):
    """
    IBR-Centric Mixed Stress Scenario (ATLAS-14).

    Combines four concurrent stressors in a single signal:

    1. **RoCoF ramp** starting at ``t_ramp_s``, lasting ``ramp_duration_s``.
       Frequency rises from ``freq_nom_hz`` at ``rocof_hz_s`` Hz/s, then holds
       at the peak.  Phase is C¹-continuous (analytical integration).

    2. **Phase jump** of ``phase_jump_deg`` degrees at ``t_jump_s``.  The jump
       is instantaneous (C⁰-discontinuous) and may co-occur with or precede the
       RoCoF ramp — estimators must track both simultaneously.

    3. **Integer harmonics** at 2f, 3f, 5f, 7f, 11f, 13f scaled proportionally
       to reach ``thd_pct``% total harmonic distortion.  Harmonic phases are
       random (randomised each trial via the Monte Carlo seed).

    4. **Additive noise** parametrised by ``noise_sigma_pu``.  ``noise_type``
       selects ``"gaussian"`` (AWGN) or ``"bernoulli_gaussian"`` (sparse
       impulsive spikes on top of a Gaussian background, variance-matched to
       ``noise_sigma_pu²``).

    The ``t_jump_s`` key in ``DEFAULT_PARAMS`` enables the MC engine to compute
    post-event metrics (m24–m30) automatically.
    """

    SCENARIO_NAME: ClassVar[str] = "IEEE_Mixed_Stress"

    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {
        "duration_s":       2.0,
        "freq_nom_hz":      F_NOM,
        "amplitude":        1.0,
        "phase_rad":        0.0,
        # RoCoF ramp
        "rocof_hz_s":       5.0,
        "t_ramp_s":         0.30,
        "ramp_duration_s":  0.40,
        # Phase jump
        "phase_jump_deg":   20.0,
        "t_jump_s":         0.50,
        # Harmonics
        "thd_pct":          8.0,
        # Noise
        "noise_sigma_pu":   0.003,
        "noise_type":       "gaussian",
        "seed":             None,
    }

    MONTE_CARLO_SPACE: ClassVar[dict[str, Any]] = {
        "phase_rad": {"kind": "uniform", "low": 0.0, "high": 2.0 * math.pi},
    }

    @classmethod
    def validate_params(cls, params: dict[str, Any]) -> None:
        if params["duration_s"] <= 0:
            raise ValueError("duration_s must be > 0")
        if params["amplitude"] <= 0:
            raise ValueError("amplitude must be > 0")
        if params["rocof_hz_s"] < 0:
            raise ValueError("rocof_hz_s must be >= 0")
        if params["ramp_duration_s"] < 0:
            raise ValueError("ramp_duration_s must be >= 0")
        if params["t_ramp_s"] < 0:
            raise ValueError("t_ramp_s must be >= 0")
        if params["t_jump_s"] < 0:
            raise ValueError("t_jump_s must be >= 0")
        if params["thd_pct"] < 0:
            raise ValueError("thd_pct must be >= 0")
        if params["noise_sigma_pu"] < 0:
            raise ValueError("noise_sigma_pu must be >= 0")
        if params["noise_type"] not in {"gaussian", "bernoulli_gaussian"}:
            raise ValueError("noise_type must be 'gaussian' or 'bernoulli_gaussian'")

    @classmethod
    def generate(
        cls,
        duration_s: float = 2.0,
        freq_nom_hz: float = F_NOM,
        amplitude: float = 1.0,
        phase_rad: float = 0.0,
        rocof_hz_s: float = 5.0,
        t_ramp_s: float = 0.30,
        ramp_duration_s: float = 0.40,
        phase_jump_deg: float = 20.0,
        t_jump_s: float = 0.50,
        thd_pct: float = 8.0,
        noise_sigma_pu: float = 0.003,
        noise_type: str = "gaussian",
        seed: int | None = None,
        **_extra: Any,
    ) -> ScenarioData:
        rng = np.random.default_rng(seed)
        t = np.arange(0.0, duration_s, 1.0 / FS_PHYSICS, dtype=float)

        # ── 1. Frequency profile (piecewise: nominal → ramp → hold) ──────────
        t_ramp_end = t_ramp_s + max(0.0, ramp_duration_s)
        f_peak = freq_nom_hz + rocof_hz_s * ramp_duration_s

        f_true = np.full_like(t, freq_nom_hz, dtype=float)
        mask_ramp = (t >= t_ramp_s) & (t < t_ramp_end)
        mask_post = t >= t_ramp_end
        f_true[mask_ramp] = freq_nom_hz + rocof_hz_s * (t[mask_ramp] - t_ramp_s)
        f_true[mask_post] = f_peak

        # ── 2. Phase — C¹-continuous analytical integration ───────────────────
        phi = np.zeros_like(t, dtype=float)

        # Pre-ramp: uniform nominal frequency
        mask_pre = t < t_ramp_s
        phi[mask_pre] = 2.0 * math.pi * freq_nom_hz * t[mask_pre] + phase_rad

        # During ramp: quadratic phase
        if np.any(mask_ramp):
            phi_at_ramp_start = 2.0 * math.pi * freq_nom_hz * t_ramp_s + phase_rad
            tau = t[mask_ramp] - t_ramp_s
            phi[mask_ramp] = (
                phi_at_ramp_start
                + 2.0 * math.pi * (freq_nom_hz * tau + 0.5 * rocof_hz_s * tau**2)
            )

        # Post-ramp: constant peak frequency
        if np.any(mask_post):
            phi_at_ramp_end = (
                (2.0 * math.pi * freq_nom_hz * t_ramp_s + phase_rad)
                + 2.0 * math.pi * (freq_nom_hz * ramp_duration_s + 0.5 * rocof_hz_s * ramp_duration_s**2)
            )
            tau_post = t[mask_post] - t_ramp_end
            phi[mask_post] = phi_at_ramp_end + 2.0 * math.pi * f_peak * tau_post

        # ── 3. Instantaneous phase jump (C⁰ discontinuity) ───────────────────
        phase_jump_rad = math.radians(phase_jump_deg)
        phi[t >= t_jump_s] += phase_jump_rad

        # ── 4. Fundamental voltage ────────────────────────────────────────────
        v = amplitude * np.sin(phi)

        # ── 5. Integer harmonics (phase-coherent, random phase offsets) ───────
        thd_pu = thd_pct / 100.0
        if thd_pu > 0.0:
            coeffs = _harmonic_coefficients(thd_pu)
            for k, amp_k in coeffs.items():
                v += amplitude * amp_k * np.sin(float(k) * phi + rng.uniform(0.0, 2.0 * math.pi))

        # ── 6. Noise ──────────────────────────────────────────────────────────
        sigma = noise_sigma_pu
        if sigma > 0.0:
            n = t.shape[0]
            if noise_type == "bernoulli_gaussian":
                sigma_bg = sigma / math.sqrt(_BG_VARIANCE_FACTOR)
                sigma_spike = _BG_SPIKE_RATIO * sigma_bg
                background = rng.normal(0.0, sigma_bg, size=n)
                indicators = rng.binomial(1, _BG_P_SPIKE, size=n)
                spikes = rng.normal(0.0, sigma_spike, size=n) * indicators
                v = v + background + spikes
            else:
                v = v + rng.normal(0.0, sigma, size=n)

        # ── 7. Metadata ───────────────────────────────────────────────────────
        actual_thd = math.sqrt(sum(c**2 for c in _harmonic_coefficients(thd_pu).values()))
        meta = {
            "description": (
                f"IEEE Mixed Stress: RoCoF={rocof_hz_s:+g} Hz/s, "
                f"phase_jump={phase_jump_deg:+g}°, THD={thd_pct:.1f}%, "
                f"noise={noise_sigma_pu:.4f} pu ({noise_type})"
            ),
            "rocof_hz_s": rocof_hz_s,
            "phase_jump_deg": phase_jump_deg,
            "thd_pct": thd_pct,
            "noise_sigma_pu": noise_sigma_pu,
            "noise_type": noise_type,
            "actual_thd_pu": actual_thd,
            "parameters": {
                "duration_s": duration_s,
                "freq_nom_hz": freq_nom_hz,
                "amplitude": amplitude,
                "phase_rad": phase_rad,
                "rocof_hz_s": rocof_hz_s,
                "t_ramp_s": t_ramp_s,
                "ramp_duration_s": ramp_duration_s,
                "phase_jump_deg": phase_jump_deg,
                "t_jump_s": t_jump_s,
                "thd_pct": thd_pct,
                "noise_sigma_pu": noise_sigma_pu,
                "noise_type": noise_type,
                "seed": seed,
            },
            "dynamics": (
                f"RoCoF ramp {rocof_hz_s:+g} Hz/s at t={t_ramp_s}s, "
                f"phase jump {phase_jump_deg:+g}° at t={t_jump_s}s, "
                f"THD {thd_pct:.1f}%, noise sigma {noise_sigma_pu:.4f} pu."
            ),
            "fs_physics_hz": FS_PHYSICS,
        }

        return ScenarioData(
            name=cls.SCENARIO_NAME,
            t=t,
            v=v,
            f_true=f_true,
            meta=meta,
        )
