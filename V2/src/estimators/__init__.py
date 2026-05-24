"""
Estimator package.

Keep this file lightweight.
It should only expose shared base utilities and, progressively,
the concrete estimator classes as they are migrated.
"""

from .base import BaseFrequencyEstimator
from .common import (
    FS_PHYSICS,
    FS_DSP,
    RATIO,
    DT_DSP,
    F_NOM,
    IIR_Bandpass,
    FastRMS_Normalizer,
    clamp_frequency_hz,
    ar2_to_freq,
)

__all__ = [
    "BaseFrequencyEstimator",
    "FS_PHYSICS",
    "FS_DSP",
    "RATIO",
    "DT_DSP",
    "F_NOM",
    "IIR_Bandpass",
    "FastRMS_Normalizer",
    "clamp_frequency_hz",
    "ar2_to_freq",
]