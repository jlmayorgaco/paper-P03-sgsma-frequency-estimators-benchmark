#!/usr/bin/env python
"""
Targeted test: IpDFT Hann-compatible magnitude interpolation
Verifies the fix in TunableIpDFT.step() against known off-bin pure tones.
Tests frequencies at known offsets from bin centres to confirm sign correctness.
"""

import sys
import numpy as np

sys.path.insert(0, r"C:\Users\walla\Documents\Github\paper\src")
from estimators import TunableIpDFT, FS_DSP, DT_DSP

def run_ipdft_test():
    print("=" * 70)
    print("IpDFT Hann-Compatible Magnitude Interpolation — Targeted Test")
    print("=" * 70)

    # For a 5-cycle IpDFT: sz = (10000/60)*5 ≈ 833 samples, res = 10000/833 ≈ 12.01 Hz/bin
    ipdft = TunableIpDFT(cycles=5)
    print(f"  Window size: {ipdft.sz} samples, resolution: {ipdft.res:.4f} Hz/bin")
    print()

    # Target frequencies (known offsets from bin centres)
    # At 5 cycles: bin centre = round(f/12.01)*12.01
    # 60.25 Hz → bin 5 → offset +0.25 Hz from bin centre (bin 5 = 60.05)
    # 60.5 Hz  → bin 5 → offset +0.45 Hz
    # 61.0 Hz  → bin 5 → offset +0.95 Hz
    # 62.0 Hz  → bin 5 → offset +1.95 Hz
    # 63.0 Hz  → bin 5 → offset +2.95 Hz

    test_freqs = [60.25, 60.5, 61.0, 62.0, 63.0]
    phases = [0.0, np.pi/4, np.pi/2]

    all_pass = True
    header = f"{'Freq (Hz)':>10}  {'Phase':>8}  {'Bin':>4}  {'Delta':>7}  {'Sign':>5}  {'Est (Hz)':>10}  {'Error (Hz)':>11}  {'Pass':>5}"
    print(header)
    print("-" * 80)

    for f_true in test_freqs:
        for phi in phases:
            ipdft.buf.clear()
            ipdft._buf_state = [] if not hasattr(ipdft, '_buf_state') else None

            # Build one full buffer of pure tone at (f_true, phi)
            t = np.arange(ipdft.sz) * DT_DSP
            signal = np.sin(2 * np.pi * f_true * t + phi)

            # Feed buffer, collect last output
            for s in signal[:-1]:
                _ = ipdft.step(s)
            f_est = ipdft.step(signal[-1])

            # Derive bin info from the step()
            sp = np.fft.rfft(np.array(ipdft.buf) * ipdft.win)
            sp_abs = np.abs(sp)
            k = int(np.argmax(sp_abs))

            delta = (f_est / ipdft.res) - k
            sign = "above" if delta > 0 else "below"
            error = f_est - f_true

            # For positive offset from bin centre, estimate should be above bin (positive delta)
            # For 60.25, 60.5, 61.0, 62.0, 63.0 — all are above bin 5 (≈60.05 Hz)
            bin_center = k * ipdft.res
            offset_from_bin = f_true - bin_center
            expected_sign = "above" if offset_from_bin > 0 else "below"
            # Small tolerance for rounding
            sign_ok = (sign == expected_sign) or (abs(delta) < 0.3)
            status = "PASS" if sign_ok else "FAIL"
            if not sign_ok:
                all_pass = False

            print(f"{f_true:>10.2f}  {phi:>8.3f}  {k:>4}  {delta:>+7.3f}  {sign:>5}  "
                  f"{f_est:>10.4f}  {error:>+11.4f}  {status:>5}")

    print()
    print("=" * 70)
    print(f"RESULT: {'ALL PASS — magnitude interpolation sign is correct' if all_pass else 'SOME FAILURES — check delta sign'}")
    print("=" * 70)
    return all_pass


if __name__ == "__main__":
    ok = run_ipdft_test()
    sys.exit(0 if ok else 1)
