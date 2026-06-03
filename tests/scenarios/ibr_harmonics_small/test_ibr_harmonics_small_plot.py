import sys
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenarios.ibr_harmonics_small import IBRHarmonicsSmallScenario


def test_harmonics_small_plot_and_csv():
    out_dir = ROOT / "tests" / "scenarios" / "ibr_harmonics_small" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    t_evt = 1.0

    sc = IBRHarmonicsSmallScenario.run(
        white_noise_sigma = 0.002,
        seed              = 42,
    )

    df = pd.DataFrame({"t_s": sc.t, "v_pu": sc.v, "f_true_hz": sc.f_true})
    df.to_csv(out_dir / "ibr_harmonics_small.csv", index=False)

    p  = sc.meta["parameters"]
    dt = float(sc.t[1] - sc.t[0])
    f0 = p["freq_nom_hz"]
    f1 = f0 + p["freq_step_hz"]

    plt.rcParams.update({
        "font.family":    "serif",
        "font.size":      9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi":     300,
        "savefig.bbox":   "tight",
    })

    # ═══════════════════════════════════════════════════════════════════════
    # FIGURE 1 — Overview
    # ═══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.5), sharex=True)
    plt.subplots_adjust(hspace=0.12)

    ax_v = axes[0]
    ax_v.plot(sc.t, sc.v, linewidth=0.5, color="steelblue")
    ax_v.axvline(t_evt, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
    ax_v.annotate(
        f"Freq step {p['freq_step_hz']:+.2f} Hz",
        xy=(t_evt, 0.6), xytext=(t_evt + 0.1, 0.85),
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8), fontsize=7.5,
    )
    ax_v.axvspan(0,     t_evt,    alpha=0.06, color="green",  label=f"Pre: f={f0:.1f} Hz")
    ax_v.axvspan(t_evt, sc.t[-1], alpha=0.06, color="orange", label=f"Post: f={f1:.2f} Hz")
    ax_v.set_ylabel("Voltage [pu]")
    ax_v.set_ylim(-1.3, 1.3)
    ax_v.grid(True, alpha=0.25)
    ax_v.legend(loc="upper right", ncol=2, fontsize=7)
    ax_v.set_title(
        "IBR Harmonics Small (IEEE 519 Compliant)\n"
        f"5th={p['h5_pct']*100:.1f}% + 7th={p['h7_pct']*100:.1f}% + 11th={p['h11_pct']*100:.1f}%  |  White noise only",
        fontsize=9,
    )

    ax_f = axes[1]
    ax_f.plot(sc.t, sc.f_true, linewidth=1.2, color="teal", label="$f_{true}(t)$")
    ax_f.axvline(t_evt, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
    ax_f.axhline(f0, color="green",  linestyle="--", linewidth=0.8,
                 alpha=0.7, label=f"$f_0$ = {f0:.1f} Hz")
    ax_f.axhline(f1, color="orange", linestyle="--", linewidth=0.8,
                 alpha=0.7, label=f"$f_1$ = {f1:.2f} Hz")
    ax_f.axvspan(0,     t_evt,    alpha=0.06, color="green")
    ax_f.axvspan(t_evt, sc.t[-1], alpha=0.06, color="orange")
    ax_f.set_ylim(f0 - 0.5, f1 + 0.5)
    ax_f.set_xlim(sc.t[0], sc.t[-1])
    ax_f.set_xlabel("Time [s]")
    ax_f.set_ylabel("Frequency [Hz]")
    ax_f.set_title("Ground-Truth Frequency: Small Step + Low Harmonic Environment", fontsize=9)
    ax_f.legend(loc="center right")
    ax_f.grid(True, alpha=0.25)

    fig.savefig(out_dir / "ibr_harmonics_small.png")
    plt.close(fig)
    print(f"[DIAGNOSTIC] Overview saved to: {out_dir / 'ibr_harmonics_small.png'}")

    # ═══════════════════════════════════════════════════════════════════════
    # FIGURE 2 — Zoom: harmonic waveform detail around step
    # ═══════════════════════════════════════════════════════════════════════
    # Tight zoom — just 5 cycles each side so harmonic distortion is visible
    z_lo = t_evt - 0.08
    z_hi = t_evt + 0.15

    mask_z = (sc.t >= z_lo) & (sc.t <= z_hi)
    t_z    = sc.t[mask_z]
    v_z    = sc.v[mask_z]
    f_z    = sc.f_true[mask_z]

    phi0  = np.arcsin(np.clip(v_z[0], -1, 1))
    v_ref = np.sin(2.0 * math.pi * f0 * (t_z - t_z[0]) + phi0)

    fig2, axes2 = plt.subplots(2, 1, figsize=(8.5, 5.0), sharex=True)
    plt.subplots_adjust(hspace=0.12)

    ax_zv = axes2[0]
    ax_zv.plot(t_z, v_ref, linewidth=1.0, color="green",    linestyle="--",
               alpha=0.7, label="Pure sinusoid reference")
    ax_zv.plot(t_z, v_z,   linewidth=1.0, color="steelblue", label="v(t) — with harmonics")
    ax_zv.axvline(t_evt, color="black", linestyle="--", linewidth=1.0, alpha=0.9)
    ax_zv.annotate(
        "Harmonic distortion\nvisible on waveform peaks",
        xy=(t_z[len(t_z)//4], v_z[len(t_z)//4]),
        xytext=(t_z[len(t_z)//4] - 0.03, 0.85),
        arrowprops=dict(arrowstyle="->", color="steelblue", lw=0.8), fontsize=7.5,
    )
    ax_zv.axvspan(z_lo,  t_evt, alpha=0.07, color="green")
    ax_zv.axvspan(t_evt, z_hi,  alpha=0.07, color="orange")
    ax_zv.set_ylabel("Voltage [pu]")
    ax_zv.set_ylim(-1.2, 1.2)
    ax_zv.grid(True, alpha=0.25)
    ax_zv.legend(loc="upper left", fontsize=7.5)
    ax_zv.set_title(
        "Harmonic Waveform Detail — Zoom Around Frequency Step\n"
        "(Harmonic distortion vs. pure sinusoid reference)",
        fontsize=9,
    )

    ax_zf = axes2[1]
    ax_zf.plot(t_z, f_z, linewidth=1.5, color="teal", label="$f_{true}(t)$")
    ax_zf.axvline(t_evt, color="black", linestyle="--", linewidth=1.0, alpha=0.9)
    ax_zf.axhline(f0, color="green",  linestyle="--", linewidth=0.8,
                  alpha=0.7, label=f"f_0 = {f0:.1f} Hz")
    ax_zf.axhline(f1, color="orange", linestyle="--", linewidth=0.8,
                  alpha=0.7, label=f"f_1 = {f1:.2f} Hz")
    ax_zf.set_xlabel("Time [s]")
    ax_zf.set_ylabel("Frequency [Hz]")
    ax_zf.set_ylim(f0 - 0.3, f1 + 0.3)
    ax_zf.set_xlim(z_lo, z_hi)
    ax_zf.grid(True, alpha=0.25)
    ax_zf.legend(loc="center left", fontsize=7.5)
    ax_zf.set_title("Frequency Step Detail (Small, +0.3 Hz)", fontsize=9)

    ax_zv.axvspan(z_lo,  t_evt, alpha=0.07, color="green")
    ax_zv.axvspan(t_evt, z_hi,  alpha=0.07, color="orange")
    ax_zf.axvspan(z_lo,  t_evt, alpha=0.07, color="green")
    ax_zf.axvspan(t_evt, z_hi,  alpha=0.07, color="orange")

    fig2.savefig(out_dir / "ibr_harmonics_small_zoom.png")
    plt.close(fig2)
    print(f"[DIAGNOSTIC] Zoom saved to:     {out_dir / 'ibr_harmonics_small_zoom.png'}")
    print(f"  f0={f0:.3f} Hz  f1={f1:.3f} Hz  step={p['freq_step_hz']:+.3f} Hz  "
          f"THD approx {math.sqrt(p['h5_pct']**2+p['h7_pct']**2+p['h11_pct']**2)*100:.1f}%")


test_harmonics_small_plot_and_csv()
