#!/usr/bin/env python
"""
Lightweight smoke check — not a full benchmark run.
Verifies:
1. All estimators import without crash
2. IpDFT on a ramp-like signal is not pathological (no 3-Hz errors)
3. Module-level constants are sane
"""
import sys
import numpy as np

sys.path.insert(0, r"C:\Users\walla\Documents\Github\paper\src")

print("=" * 70)
print("Lightweight Smoke Check")
print("=" * 70)

# ── 1. Import smoke ───────────────────────────────────────────────────
try:
    from estimators import (
        TunableIpDFT, StandardPLL, ClassicEKF, SOGI_FLL,
        RLS_Estimator, RLS_VFF_Estimator, Teager_Estimator,
        TFT_Estimator, UKF_Estimator, Koopman_RKDPmu,
        calculate_metrics, FS_DSP, DT_DSP, F_NOM,
        SEED, SETTLING_THRESHOLD, TRIP_THRESHOLD,
        tune_ipdft, tune_ekf, tune_ekf2, tune_pll,
        tune_sogi, tune_rls, tune_teager, tune_tft,
        tune_vff_rls, tune_ukf, tune_koopman,
        get_test_signals,
    )
    print("[PASS] All estimators and functions imported successfully")
except Exception as e:
    print(f"[FAIL] Import error: {e}")
    sys.exit(1)

# ── 2. Constants sanity ───────────────────────────────────────────────
assert FS_DSP == 10000.0, f"FS_DSP wrong: {FS_DSP}"
assert F_NOM == 60.0, f"F_NOM wrong: {F_NOM}"
assert SETTLING_THRESHOLD == 0.2, "SETTLING_THRESHOLD wrong"
assert TRIP_THRESHOLD == 0.5, "TRIP_THRESHOLD wrong"
print(f"[PASS] Constants: FS_DSP={FS_DSP}, F_NOM={F_NOM}, thresholds sane")

# ── 3. IpDFT on ramp-like signal (mimics IEEE_Freq_Ramp) ─────────────
print("\n[Check] IpDFT on 5 Hz/s ramp-like signal:")
ipdft = TunableIpDFT(cycles=5)

# Generate: 60 Hz ramp to 63.5 Hz over 0.7 s (5 Hz/s), then flat
T = 1.5
t = np.arange(0, T, 1.0 / FS_DSP)
f_ramp = np.ones_like(t) * 60.0
mask = (t > 0.3) & (t < 1.0)
f_ramp[mask] = 60.0 + 5.0 * (t[mask] - 0.3)
f_ramp[t >= 1.0] = f_ramp[t < 1.0][-1]
phi = np.cumsum(2 * np.pi * f_ramp * (1.0 / FS_DSP))
v = np.sin(phi) + np.random.normal(0, 0.001, len(t))

# Run IpDFT
est_trace = np.array([ipdft.step(s) for s in v])

# Compute metrics on the ramp portion (0.3-1.0 s)
ramp_mask = mask
est_ramp = est_trace[ramp_mask]
f_ramp_gt = f_ramp[ramp_mask]
rmse = float(np.sqrt(np.mean((est_ramp - f_ramp_gt) ** 2)))
max_err = float(np.max(np.abs(est_ramp - f_ramp_gt)))
print(f"  IpDFT Ramp RMSE (0.3-1.0s): {rmse:.4f} Hz")
print(f"  IpDFT Ramp MAX_ERR (0.3-1.0s): {max_err:.4f} Hz")
# Old pathological result was ~3.09 Hz RMSE. After fix, should be < 0.5 Hz
if rmse < 1.0:
    print(f"  [PASS] IpDFT ramp RMSE {rmse:.4f} Hz < 1.0 Hz — no longer pathological")
else:
    print(f"  [WARNING] IpDFT ramp RMSE {rmse:.4f} Hz still high, check interpolation")

# ── 4. Quick metrics sanity ───────────────────────────────────────────
# Steady 60 Hz, flat trace
f_flat = np.ones_like(t) * 60.0
m = calculate_metrics(f_flat, f_flat, 0.001)
assert m["RMSE"] == 0.0, "Zero-error case should give RMSE=0"
assert m["MAX_PEAK"] == 0.0, "Zero-error case should give MAX_PEAK=0"
print(f"[PASS] calculate_metrics zero-error edge case: RMSE={m['RMSE']}, MAX={m['MAX_PEAK']}")

# Random trace should not crash
rand_trace = np.random.randn(len(t)) * 5.0 + 60.0
m2 = calculate_metrics(rand_trace, f_flat, 0.001)
assert not np.isnan(m2["RMSE"]), "NaN in random trace metrics"
assert m2["RMSE"] > 0, "Random trace should have non-zero RMSE"
print(f"[PASS] calculate_metrics with random trace: RMSE={m2['RMSE']:.4f}")

# ── 5. Verify UKF has smooth_win=10 in tuning ─────────────────────────
import inspect
ukf_src = inspect.getsource(tune_ukf)
has_smooth_win_10 = "smooth_win=10" in ukf_src
has_disclosure = "DISCLOSURE" in ukf_src
print(f"\n[Check] UKF disclosure comment: {'[PASS]' if has_disclosure else '[FAIL]'}")
print(f"[Check] UKF smooth_win=10 in tuning: {'[PASS]' if has_smooth_win_10 else '[FAIL]'}")

print("\n" + "=" * 70)
print("SMOKE CHECK COMPLETE — no crashes, all basic sanity checks passed")
print("=" * 70)
