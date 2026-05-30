from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np

from .base import Scenario, ScenarioData
from estimators.common import FS_PHYSICS, F_NOM

# Default background AWGN present even when no spikes occur.
_DEFAULT_AWGN_SIGMA = 0.001


class IEEEImpulsiveNoiseScenario(Scenario):
    """
    Single-tone sinusoid with Bernoulli-Gaussian impulsive noise.

    Signal model:
        v[k] = A*sin(θ[k]) + w[k] + b[k]*s[k]

    where:
        w[k] ~ N(0, awgn_sigma²)           background AWGN, always present
        b[k] ~ Bernoulli(impulse_probability)
        s[k] ~ N(0, impulse_magnitude_pu²) spike amplitude when b[k]=1

    ``impulse_probability`` and ``impulse_magnitude_pu`` are independent sweep
    axes, unlike ATLAS-09 where all models are variance-matched at a common σ.
    """

    SCENARIO_NAME: ClassVar[str] = "IEEE_Impulsive_Noise"

    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {
        "duration_s": 1.5,
        "amplitude": 1.0,
        "freq_hz": F_NOM,
        "phase_rad": 0.0,
        "awgn_sigma": _DEFAULT_AWGN_SIGMA,
        "impulse_probability": 0.001,
        "impulse_magnitude_pu": 0.10,
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
        awgn = params.get("awgn_sigma", 0.0)
        if awgn < 0:
            raise ValueError("awgn_sigma must be >= 0")
        prob = params.get("impulse_probability", 0.0)
        if not (0.0 <= prob <= 1.0):
            raise ValueError("impulse_probability must be in [0, 1]")
        mag = params.get("impulse_magnitude_pu", 0.0)
        if mag < 0:
            raise ValueError("impulse_magnitude_pu must be >= 0")

    @classmethod
    def generate(
        cls,
        duration_s: float = 1.5,
        amplitude: float = 1.0,
        freq_hz: float = F_NOM,
        phase_rad: float = 0.0,
        awgn_sigma: float = _DEFAULT_AWGN_SIGMA,
        impulse_probability: float = 0.001,
        impulse_magnitude_pu: float = 0.10,
        seed: int | None = None,
        **_extra: Any,
    ) -> ScenarioData:
        rng = np.random.default_rng(seed)

        t = np.arange(0.0, duration_s, 1.0 / FS_PHYSICS, dtype=float)
        f_true = np.full_like(t, freq_hz, dtype=float)

        phi = 2.0 * math.pi * freq_hz * t + phase_rad
        v = amplitude * np.sin(phi)

        n = t.shape[0]
        if awgn_sigma > 0.0:
            v = v + rng.normal(0.0, awgn_sigma, size=n)

        if impulse_probability > 0.0 and impulse_magnitude_pu > 0.0:
            indicators = rng.binomial(1, impulse_probability, size=n)
            spikes = rng.normal(0.0, impulse_magnitude_pu, size=n) * indicators
            v = v + spikes

        meta = {
            "description": (
                f"IEEE_Impulsive_Noise — p={impulse_probability}, "
                f"mag={impulse_magnitude_pu} pu, awgn={awgn_sigma} pu"
            ),
            "impulse_probability": impulse_probability,
            "impulse_magnitude_pu": impulse_magnitude_pu,
            "awgn_sigma": awgn_sigma,
            "parameters": {
                "duration_s": duration_s,
                "amplitude": amplitude,
                "freq_hz": freq_hz,
                "phase_rad": phase_rad,
                "awgn_sigma": awgn_sigma,
                "impulse_probability": impulse_probability,
                "impulse_magnitude_pu": impulse_magnitude_pu,
                "seed": seed,
            },
            "dynamics": "Constant frequency with background AWGN plus sparse Bernoulli-Gaussian spikes.",
            "fs_physics_hz": FS_PHYSICS,
        }

        return ScenarioData(
            name=cls.SCENARIO_NAME,
            t=t,
            v=v,
            f_true=f_true,
            meta=meta,
        )
