from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np

from .base import Scenario, ScenarioData
from estimators.common import FS_PHYSICS, F_NOM

NONGAUSSIAN_NOISE_TYPES: tuple[str, ...] = (
    "gaussian",
    "laplace",
    "student_t_df3",
    "student_t_df5",
    "student_t_df10",
    "bernoulli_gaussian",
)

# Bernoulli-Gaussian parameters (fixed, variance-matched by construction):
# spike rate p=0.05, spike amplitude = 3 × background amplitude.
# Background std = sigma / sqrt(1 + 9*p), spike std = 3 × background std.
# Variance: sigma_bg^2 + p * sigma_spike^2 = sigma_bg^2 * (1 + 9p) = sigma^2.
_BG_P_SPIKE = 0.05
_BG_SPIKE_RATIO = 3.0
_BG_VARIANCE_FACTOR = 1.0 + _BG_SPIKE_RATIO**2 * _BG_P_SPIKE  # = 1.45


class IEEENonGaussianNoiseScenario(Scenario):
    """
    Single-tone sinusoid with selectable non-Gaussian noise model.

    All noise models are parametrised by ``noise_sigma`` and scaled so that
    their variance equals ``noise_sigma**2``, allowing fair comparison at
    matched nominal power. ``noise_type`` selects the model:

    - ``"gaussian"``         — N(0, sigma^2), same as noise_snr baseline
    - ``"laplace"``          — Laplace(0, b) with b = sigma/sqrt(2)
    - ``"student_t_df3"``    — t(df=3) scaled to variance sigma^2
    - ``"bernoulli_gaussian"``— background Gaussian + sparse Bernoulli spikes
    """

    SCENARIO_NAME: ClassVar[str] = "IEEE_NonGaussian_Noise"

    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {
        "duration_s": 1.5,
        "amplitude": 1.0,
        "freq_hz": F_NOM,
        "phase_rad": 0.0,
        "noise_sigma": 0.001,
        "noise_type": "gaussian",
        "seed": None,
    }

    MONTE_CARLO_SPACE: ClassVar[dict[str, Any]] = {
        "phase_rad": {"kind": "uniform", "low": 0.0, "high": 2.0 * math.pi},
    }

    @classmethod
    def validate_params(cls, params: dict[str, Any]) -> None:
        if params["duration_s"] <= 0:
            raise ValueError("duration_s must be > 0")
        if params["amplitude"] < 0:
            raise ValueError("amplitude must be >= 0")
        if params["freq_hz"] <= 0:
            raise ValueError("freq_hz must be > 0")
        if params["noise_sigma"] < 0:
            raise ValueError("noise_sigma must be >= 0")
        noise_type = params.get("noise_type", "gaussian")
        if noise_type not in NONGAUSSIAN_NOISE_TYPES:
            raise ValueError(
                f"noise_type must be one of {NONGAUSSIAN_NOISE_TYPES}, got {noise_type!r}"
            )

    @classmethod
    def generate(
        cls,
        duration_s: float = 1.5,
        amplitude: float = 1.0,
        freq_hz: float = F_NOM,
        phase_rad: float = 0.0,
        noise_sigma: float = 0.001,
        noise_type: str = "gaussian",
        seed: int | None = None,
        **_extra: Any,
    ) -> ScenarioData:
        rng = np.random.default_rng(seed)

        t = np.arange(0.0, duration_s, 1.0 / FS_PHYSICS, dtype=float)
        f_true = np.full_like(t, freq_hz, dtype=float)

        phi = 2.0 * math.pi * freq_hz * t + phase_rad
        v = amplitude * np.sin(phi)

        if noise_sigma > 0.0:
            n = t.shape[0]
            if noise_type == "gaussian":
                noise = rng.normal(0.0, noise_sigma, size=n)
            elif noise_type == "laplace":
                # Laplace variance = 2b^2 = sigma^2  →  b = sigma / sqrt(2)
                b = noise_sigma / math.sqrt(2.0)
                noise = rng.laplace(0.0, b, size=n)
            elif noise_type == "student_t_df3":
                df = 3
                scale = noise_sigma / math.sqrt(df / (df - 2))
                noise = rng.standard_t(df, size=n).astype(float) * scale
            elif noise_type == "student_t_df5":
                df = 5
                scale = noise_sigma / math.sqrt(df / (df - 2))
                noise = rng.standard_t(df, size=n).astype(float) * scale
            elif noise_type == "student_t_df10":
                df = 10
                scale = noise_sigma / math.sqrt(df / (df - 2))
                noise = rng.standard_t(df, size=n).astype(float) * scale
            elif noise_type == "bernoulli_gaussian":
                sigma_bg = noise_sigma / math.sqrt(_BG_VARIANCE_FACTOR)
                sigma_spike = _BG_SPIKE_RATIO * sigma_bg
                background = rng.normal(0.0, sigma_bg, size=n)
                indicators = rng.binomial(1, _BG_P_SPIKE, size=n)
                spikes = rng.normal(0.0, sigma_spike, size=n) * indicators
                noise = background + spikes
            else:
                noise = rng.normal(0.0, noise_sigma, size=n)
            v = v + noise

        meta = {
            "description": (
                f"IEEE_NonGaussian_Noise — noise_type={noise_type}, sigma={noise_sigma}"
            ),
            "noise_type": noise_type,
            "noise_sigma": noise_sigma,
            "parameters": {
                "duration_s": duration_s,
                "amplitude": amplitude,
                "freq_hz": freq_hz,
                "phase_rad": phase_rad,
                "noise_sigma": noise_sigma,
                "noise_type": noise_type,
                "seed": seed,
            },
            "dynamics": "Constant frequency with non-Gaussian additive noise.",
            "fs_physics_hz": FS_PHYSICS,
        }

        return ScenarioData(
            name=cls.SCENARIO_NAME,
            t=t,
            v=v,
            f_true=f_true,
            meta=meta,
        )
