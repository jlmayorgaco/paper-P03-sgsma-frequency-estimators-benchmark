from __future__ import annotations

import math
from collections import deque

import numpy as np

# ============================================================================
# Shared sampling constants
# ============================================================================

FS_PHYSICS = 1_000_000.0
FS_DSP = 10_000.0
RATIO = int(FS_PHYSICS / FS_DSP)
DT_DSP = 1.0 / FS_DSP
F_NOM = 60.0


# ============================================================================
# Shared helper classes
# ============================================================================

class IIR_Bandpass:
    """
    2nd-order IIR bandpass centered at 60 Hz.
    """

    def __init__(self, center_hz: float = 60.0, bandwidth_hz: float = 40.0) -> None:
        w0 = 2.0 * np.pi * center_hz / FS_DSP
        bw = 2.0 * np.pi * bandwidth_hz / FS_DSP
        r = 1.0 - (bw / 2.0)

        self.a1 = 2.0 * r * np.cos(w0)
        self.a2 = -(r * r)
        self.b0 = (1.0 - self.a2) / 4.0
        self.b2 = -self.b0

        self.x = deque([0.0, 0.0], maxlen=2)
        self.y = deque([0.0, 0.0], maxlen=2)

    def reset(self) -> None:
        self.x = deque([0.0, 0.0], maxlen=2)
        self.y = deque([0.0, 0.0], maxlen=2)

    def step(self, v_in: float) -> float:
        v_out = (
            self.b0 * v_in
            + self.b2 * self.x[1]
            + self.a1 * self.y[0]
            + self.a2 * self.y[1]
        )
        self.x.appendleft(v_in)
        self.y.appendleft(v_out)
        return float(v_out)


class FastRMS_Normalizer:
    """
    Sliding-RMS normalizer / simple AGC.
    """

    def __init__(self, window_len: int | None = None, floor: float = 0.1) -> None:
        if window_len is None:
            window_len = int(FS_DSP / F_NOM / 2.0)
        self.win_len = max(1, int(window_len))
        self.floor = float(floor)
        self.buf: deque[float] = deque(maxlen=self.win_len)

    def reset(self) -> None:
        self.buf = deque(maxlen=self.win_len)

    def step(self, val: float) -> float:
        sq_val = val * val
        self.buf.append(float(sq_val))
        rms = np.sqrt(np.mean(self.buf)) if self.buf else 1.0
        rms = max(float(rms), self.floor)
        return float(val / (rms * 1.41421356))


# ============================================================================
# Shared numeric helpers
# ============================================================================

def clamp_frequency_hz(f_hz: float, f_min: float = 40.0, f_max: float = 80.0) -> float:
    if not np.isfinite(f_hz):
        return F_NOM
    return float(np.clip(f_hz, f_min, f_max))


def ar2_to_freq(theta0: float, dt: float = DT_DSP) -> float:
    a1 = float(np.clip(theta0, -1.9999, 1.9999))
    val = float(np.clip(a1 / 2.0, -0.9999, 0.9999))

    try:
        w = math.acos(val) / dt
    except ValueError:
        return F_NOM

    f_inst = w / (2.0 * math.pi)
    return clamp_frequency_hz(f_inst)