#!/usr/bin/env python3
"""
T-001 — Minimal Smoke Test
==========================
Verifies estimator wiring, step/step_vectorized agreement,
steady-state accuracy, and metrics sanity.

Usage (from repo root):
    python src/tests/smoke_test_minimal.py

Acceptance criteria:
- Completes in < 30 s
- Prints table: estimator | status | mean_err | max_err | step==step_vec | has_nan | time_ms
- Exits with non-zero code if any estimator fails
"""

from __future__ import annotations

import sys
import time
import math
import copy
from pathlib import Path

import numpy as np

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

# ── Imports ──────────────────────────────────────────────────────────────────
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
from estimators.pi_gru   import PI_GRU_Estimator
from analysis.metrics    import calculate_all_metrics

# ── Constants ────────────────────────────────────────────────────────────────
FS_DSP_ACTUAL = 10_000.0   # Target sample rate reaching estimators after decimation
DT_ACTUAL     = 1.0 / FS_DSP_ACTUAL
DURATION_S    = 1.5
FREQ_HZ       = 60.0
STEP_CMP_N    = 2000       # Samples used for step() vs step_vectorized() comparison
MEAN_ERR_TOL  = 0.1        # Hz — steady-state mean absolute error tolerance
STEP_AGREE_TOL = 0.05      # Hz — step vs step_vec agreement tolerance

# Canonical 11-method set + PI-GRU (marked as MC-excluded)
ESTIMATORS: dict[str, tuple[type, dict, bool]] = {
    # name:  (class, extra_kwargs, mc_included)
    "EKF":      (EKF_Estimator,      {},                  True),
    "EKF2":     (RAEKF_Estimator,    {},                  True),
    "UKF":      (UKF_Estimator,      {},                  True),
    "PLL":      (PLL_Estimator,      {},                  True),
    "SOGI":     (SOGI_FLL_Estimator, {},                  True),
    "IpDFT":    (IPDFT_Estimator,    {},                  True),
    "TFT":      (TFT_Estimator,      {},                  True),
    "RLS":      (RLS_Estimator,      {"is_vff": False},   True),
    "RLS-VFF":  (RLS_Estimator,      {"is_vff": True},    True),
    "Teager":   (TKEO_Estimator,     {},                  True),
    "Koopman":  (Koopman_Estimator,  {},                  True),
    "PI-GRU":   (PI_GRU_Estimator,   {},                  False),  # MC-excluded
}


def make_estimator(cls: type, extra: dict):
    """Instantiate estimator with default_params + dt injection + extra overrides."""
    params = cls.default_params() if hasattr(cls, "default_params") else {}
    params["dt"] = DT_ACTUAL
    params.update(extra)
    try:
        est = cls(**params)
    except TypeError:
        # Some estimators may not accept dt — try without it
        params.pop("dt", None)
        est = cls(**params)
    if hasattr(est, "reset"):
        est.reset()
    return est


def generate_signal(duration_s: float = DURATION_S) -> tuple[np.ndarray, np.ndarray]:
    """Generate a decimated 60 Hz sine at FS_DSP_ACTUAL."""
    # Generate at FS_PHYSICS then decimate (mirrors what T-100 will fix in the engine)
    t_hi = np.arange(0.0, duration_s, 1.0 / FS_PHYSICS)
    v_hi = np.sin(2.0 * math.pi * FREQ_HZ * t_hi)
    t    = t_hi[::RATIO]
    v    = v_hi[::RATIO]
    return t, v


# ── Section 1: Sample-rate consistency check ─────────────────────────────────
def check_constants() -> list[str]:
    issues = []
    if FS_PHYSICS != 1_000_000.0:
        issues.append(f"FS_PHYSICS={FS_PHYSICS} (expected 1e6)")
    if FS_DSP != 10_000.0:
        issues.append(f"FS_DSP={FS_DSP} (expected 1e4)")
    if RATIO != 100:
        issues.append(f"RATIO={RATIO} (expected 100)")
    if abs(DT_DSP - 1e-4) > 1e-12:
        issues.append(f"DT_DSP={DT_DSP} (expected 1e-4)")
    return issues


# ── Section 2: Estimator wiring tests ────────────────────────────────────────
def test_estimator(name: str, cls: type, extra: dict, v: np.ndarray
                   ) -> dict:
    result = {
        "name":         name,
        "status":       "PASS",
        "mean_err":     float("nan"),
        "max_err":      float("nan"),
        "step_agree":   False,
        "has_nan":      False,
        "time_ms":      float("nan"),
        "note":         "",
    }

    # ── 2a. step_vectorized on full signal ──────────────────────────────────
    try:
        est_vec = make_estimator(cls, extra)
        t0 = time.perf_counter()
        f_hat = est_vec.step_vectorized(v)
        result["time_ms"] = (time.perf_counter() - t0) * 1000.0
        f_hat = np.asarray(f_hat, dtype=float)
    except Exception as e:
        result["status"] = "FAIL"
        result["note"]   = f"step_vectorized raised: {e}"
        return result

    # Length check
    if len(f_hat) != len(v):
        result["status"] = "FAIL"
        result["note"]   = f"output length {len(f_hat)} != signal length {len(v)}"
        return result

    # NaN check: allow NaN only within structural latency window
    struct_lat = 0
    if hasattr(est_vec, "structural_latency_samples"):
        struct_lat = est_vec.structural_latency_samples()
    post_latency = f_hat[struct_lat:]
    nan_in_latency   = not np.all(np.isfinite(f_hat[:struct_lat]))
    nan_post_latency = not np.all(np.isfinite(post_latency))
    result["has_nan"] = nan_post_latency
    if nan_in_latency and not nan_post_latency:
        result["note"] += f"NaN during warm-up only (struct_lat={struct_lat}) "
    if nan_post_latency:
        result["status"] = "FAIL"
        result["note"]   = f"NaN/Inf after structural latency ({struct_lat} samples)"
        return result

    # ── 2b. step() vs step_vectorized() agreement ───────────────────────────
    v_short = v[:STEP_CMP_N]
    try:
        est_step = make_estimator(cls, extra)
        f_hat_step = np.array([est_step.step(float(z)) for z in v_short])

        est_vec2 = make_estimator(cls, extra)
        f_hat_vec2 = est_vec2.step_vectorized(v_short)

        # Compare only finite (post-latency) samples
        both_finite = np.isfinite(f_hat_step) & np.isfinite(f_hat_vec2)
        if np.any(both_finite):
            diff = np.abs(f_hat_step[both_finite] - f_hat_vec2[both_finite])
            max_diff = float(np.max(diff))
        else:
            max_diff = 0.0  # Nothing to compare
        result["step_agree"] = (max_diff <= STEP_AGREE_TOL)
        if not result["step_agree"] and math.isfinite(max_diff):
            result["note"] += f"step/vec diff={max_diff:.4f}Hz "
    except Exception as e:
        result["note"] += f"step-cmp failed: {e} "

    # ── 2c. Steady-state mean error ─────────────────────────────────────────
    half = len(f_hat) // 2
    f_true_const = np.full(len(v), FREQ_HZ)
    err = f_hat[half:] - f_true_const[half:]
    result["mean_err"] = float(np.mean(np.abs(err)))
    result["max_err"]  = float(np.max(np.abs(err[np.isfinite(err)])) if len(err) > 0 else float("nan"))

    if result["mean_err"] > MEAN_ERR_TOL:
        if result["status"] == "PASS":
            result["status"] = "WARN"
        result["note"] += f"mean_err={result['mean_err']:.4f}Hz>{MEAN_ERR_TOL}Hz "

    if not result["step_agree"] and result["status"] == "PASS":
        result["status"] = "WARN"

    return result


# ── Section 3: Metrics sanity check ──────────────────────────────────────────
def check_metrics_sanity() -> list[str]:
    """Run calculate_all_metrics with f_hat == f_true; all accuracy metrics must be 0."""
    n = int(FS_DSP_ACTUAL * 2.0)   # 2 s of signal
    f_perfect = np.full(n, FREQ_HZ)

    try:
        m = calculate_all_metrics(
            f_hat=f_perfect,
            f_true=f_perfect,
            fs_dsp=FS_DSP_ACTUAL,
            exec_time_s=0.001,
            structural_samples=0,
            noise_sigma=0.0,
            interharmonic_hz=32.5,
        )
    except Exception as e:
        return [f"calculate_all_metrics raised: {e}"]

    failures = []
    accuracy_keys = ["m1_rmse_hz", "m2_mae_hz", "m3_max_peak_hz", "m4_std_error_hz", "m5_trip_risk_s"]
    for k in accuracy_keys:
        v = m.get(k, None)
        if v is None:
            failures.append(f"{k}: missing from output")
        elif not math.isfinite(v):
            failures.append(f"{k}={v} (not finite)")
        elif abs(v) > 1e-9:
            failures.append(f"{k}={v:.6g} (expected 0 for perfect oracle)")
    return failures


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    t_start = time.perf_counter()
    print("=" * 70)
    print("T-001 Minimal Smoke Test")
    print("=" * 70)

    # 1. Constants
    print("\n[1] Sample-rate constants:")
    print(f"    FS_PHYSICS={FS_PHYSICS:.0f}  FS_DSP={FS_DSP:.0f}  RATIO={RATIO}  DT_DSP={DT_DSP:.1e}")
    const_issues = check_constants()
    if const_issues:
        for iss in const_issues:
            print(f"    FAIL: {iss}")
    else:
        print("    OK — constants consistent with dual-rate design")

    # 2. Generate signal (at 10 kHz after decimation)
    print(f"\n[2] Signal: {FREQ_HZ} Hz sine, {DURATION_S} s, FS={FS_DSP_ACTUAL:.0f} Hz (decimated from {FS_PHYSICS:.0f})")
    t, v = generate_signal()
    n_sig = len(v)
    print(f"    N={n_sig} samples, dt={t[1]-t[0]:.2e} s")

    # 3. Metrics sanity
    print("\n[3] Metrics sanity (f_hat == f_true -> all accuracy metrics = 0):")
    metric_failures = check_metrics_sanity()
    if metric_failures:
        for f in metric_failures:
            print(f"    FAIL: {f}")
    else:
        print("    OK — all accuracy metrics are 0 for perfect oracle")

    # 4. Estimator tests
    print(f"\n[4] Estimator tests (signal N={n_sig}, step_cmp first {STEP_CMP_N}):")
    print()
    COL = "{:<12} {:<6} {:>10} {:>10} {:>12} {:>8} {:>10}  {}"
    header = COL.format("Estimator", "MC", "mean_err", "max_err",
                        "step==vec", "has_nan", "time_ms", "note")
    print(header)
    print("-" * len(header))

    results = []
    any_fail = bool(const_issues) or bool(metric_failures)

    for name, (cls, extra, mc) in ESTIMATORS.items():
        mc_tag = "YES" if mc else "NO "
        # PI-GRU is slow — limit signal length for smoke test
        v_run = v[:STEP_CMP_N] if name == "PI-GRU" else v
        r = test_estimator(name, cls, extra, v_run)
        results.append(r)

        status_sym = {"PASS": "OK", "WARN": " !", "FAIL": "XX"}.get(r["status"], "??")
        print(COL.format(
            f"{status_sym} {name}",
            mc_tag,
            f"{r['mean_err']:.4f}" if math.isfinite(r["mean_err"]) else "n/a",
            f"{r['max_err']:.4f}"  if math.isfinite(r["max_err"])  else "n/a",
            str(r["step_agree"]),
            str(r["has_nan"]),
            f"{r['time_ms']:.1f}"  if math.isfinite(r["time_ms"])  else "n/a",
            r["note"].strip(),
        ))

        if r["status"] == "FAIL":
            any_fail = True

    # 5. Summary
    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_warn = sum(1 for r in results if r["status"] == "WARN")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    n_total = len(results)

    elapsed = time.perf_counter() - t_start
    print()
    print(f"Elapsed: {elapsed:.1f} s  (limit: 30 s{'  OVER LIMIT' if elapsed > 30 else ''})")
    print(f"Summary: {n_pass}/{n_total} PASS  {n_warn} WARN  {n_fail} FAIL")
    if any_fail:
        print("\nSMOKE TEST FAILED — see FAIL rows above")
        return 1
    elif n_warn:
        print("\nSMOKE TEST PASSED WITH WARNINGS — see WARN rows above")
        return 0
    else:
        print("\nSMOKE TEST PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
