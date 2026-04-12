#!/usr/bin/env python3
"""
T-101 Unit Test — step() vs step_vectorized() agreement
========================================================
For every estimator in the canonical 11-method set, verifies that
running step(z) in a loop and step_vectorized(v) on the same signal
from the same initial state produce output within atol=1e-6 Hz for
every sample.

Usage:
    python src/tests/test_scalar_vs_vector.py
    # or via pytest
    pytest src/tests/test_scalar_vs_vector.py -v

Acceptance criteria (T-101):
- All estimators pass for the canonical 11-method set.
- Exit code 0 on success, 1 on any failure.
"""
from __future__ import annotations

import sys
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

from estimators.common import FS_PHYSICS, FS_DSP, RATIO, DT_DSP, F_NOM
from estimators.ekf      import EKF_Estimator
from estimators.ra_ekf   import RAEKF_Estimator
from estimators.ukf      import UKF_Estimator
from estimators.pll      import PLL_Estimator
from estimators.sogi_fll import SOGI_FLL_Estimator
from estimators.ipdft    import IPDFT_Estimator
from estimators.tft      import TFT_Estimator
from estimators.rls      import RLS_Estimator
from estimators.tkeo     import TKEO_Estimator
from estimators.koopman  import Koopman_Estimator

DT_ACTUAL = 1.0 / FS_DSP
ATOL      = 1e-6   # Hz — agreement tolerance

# Canonical 11-method set (PI-GRU excluded — step_vectorized is sample-by-sample)
ESTIMATOR_CONFIGS = [
    ("EKF",      EKF_Estimator,      {}),
    ("EKF2",     RAEKF_Estimator,    {}),
    ("UKF",      UKF_Estimator,      {}),
    ("PLL",      PLL_Estimator,      {}),
    ("SOGI",     SOGI_FLL_Estimator, {}),
    ("IpDFT",    IPDFT_Estimator,    {}),
    ("TFT",      TFT_Estimator,      {}),
    ("RLS",      RLS_Estimator,      {"is_vff": False}),
    ("RLS-VFF",  RLS_Estimator,      {"is_vff": True}),
    ("Teager",   TKEO_Estimator,     {}),
    ("Koopman",  Koopman_Estimator,  {}),
]


def make_estimator(cls, extra):
    params = cls.default_params() if hasattr(cls, "default_params") else {}
    params["dt"] = DT_ACTUAL
    params.update(extra)
    try:
        est = cls(**params)
    except TypeError:
        params.pop("dt", None)
        est = cls(**params)
    if hasattr(est, "reset"):
        est.reset()
    return est


def generate_test_signal(n_samples: int = 3000) -> np.ndarray:
    """Clean 60 Hz sine at 10 kHz."""
    t = np.arange(n_samples) * DT_ACTUAL
    return np.sin(2.0 * math.pi * F_NOM * t)


@pytest.mark.parametrize("name,cls,extra", ESTIMATOR_CONFIGS,
                         ids=[c[0] for c in ESTIMATOR_CONFIGS])
def test_step_equals_step_vectorized(name, cls, extra):
    """step(z) in a loop must agree with step_vectorized(v) within ATOL for all samples."""
    v = generate_test_signal()

    # Path A: step_vectorized on full signal
    est_vec = make_estimator(cls, extra)
    f_hat_vec = np.asarray(est_vec.step_vectorized(v), dtype=float)

    # Path B: step() in a loop (fresh identical state)
    est_step = make_estimator(cls, extra)
    f_hat_step = np.array([est_step.step(float(z)) for z in v], dtype=float)

    assert len(f_hat_vec) == len(v),  f"{name}: step_vectorized length mismatch"
    assert len(f_hat_step) == len(v), f"{name}: step loop length mismatch"

    # Compare only finite samples (NaN during structural warm-up is allowed)
    both_finite = np.isfinite(f_hat_vec) & np.isfinite(f_hat_step)
    # At least the second half of the signal should be finite
    half = len(v) // 2
    assert np.all(both_finite[half:]), (
        f"{name}: NaN/Inf in second half of output — "
        f"vec_nan={np.sum(~np.isfinite(f_hat_vec[half:]))}, "
        f"step_nan={np.sum(~np.isfinite(f_hat_step[half:]))}"
    )

    diff = np.abs(f_hat_vec[both_finite] - f_hat_step[both_finite])
    max_diff = float(np.max(diff)) if len(diff) > 0 else 0.0
    assert max_diff <= ATOL, (
        f"{name}: max |step_vectorized - step| = {max_diff:.6g} Hz > atol={ATOL} Hz"
    )


# ── Standalone runner ──────────────────────────────────────────────────────
if __name__ == "__main__":
    v = generate_test_signal()
    print(f"Signal: {len(v)} samples at 10 kHz, 60 Hz sine")
    print(f"Tolerance: {ATOL} Hz\n")
    print(f"{'Estimator':<12} {'status':<8} {'max_diff_Hz':<14} note")
    print("-" * 55)

    all_pass = True
    for name, cls, extra in ESTIMATOR_CONFIGS:
        try:
            est_vec  = make_estimator(cls, extra)
            f_hat_vec = np.asarray(est_vec.step_vectorized(v), dtype=float)

            est_step = make_estimator(cls, extra)
            f_hat_step = np.array([est_step.step(float(z)) for z in v], dtype=float)

            both_finite = np.isfinite(f_hat_vec) & np.isfinite(f_hat_step)
            if np.any(both_finite):
                diff = np.abs(f_hat_vec[both_finite] - f_hat_step[both_finite])
                max_diff = float(np.max(diff))
            else:
                max_diff = float("nan")

            half = len(v) // 2
            second_half_ok = np.all(both_finite[half:])

            if not second_half_ok:
                status = "FAIL"
                note   = "NaN in second half"
                all_pass = False
            elif not math.isfinite(max_diff) or max_diff > ATOL:
                status = "FAIL"
                note   = f"diff={max_diff:.4g} Hz > {ATOL} Hz"
                all_pass = False
            else:
                status = "PASS"
                note   = ""

            print(f"{name:<12} {status:<8} {max_diff:<14.6g} {note}")

        except Exception as e:
            print(f"{name:<12} {'ERROR':<8} {'n/a':<14} {e}")
            all_pass = False

    print()
    if all_pass:
        print("ALL PASS")
        sys.exit(0)
    else:
        print("FAILURES DETECTED")
        sys.exit(1)
