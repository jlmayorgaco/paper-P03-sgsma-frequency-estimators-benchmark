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

from scenarios.ibr_harmonics_medium import IBRHarmonicsMediumScenario


def test_harmonics_medium_plot_and_csv():
    out_dir = ROOT / "tests" / "scenarios" / "ibr_harmonics_medium" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    sc = IBRHarmonicsMediumScenario.run(seed=42)

    df = pd.DataFrame({"t_s": sc.t, "v_pu": sc.v, "f_true_hz": sc.f_true})
    df.to_csv(out_dir / "ibr_harmonics_medium.csv", index=False)

    p       = sc.meta["parameters"]
    dt      = float(sc.t[1] - sc.t[0])
    f0      = p["freq_nom_hz"]
    f_nadir = p["f_steady_hz"]
    t_evt   = p["t_event_s"]
    t_nadir = t_evt + p["rocof_duration_s"]

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
    ax_v.plot(sc.t, sc.v, linewidth=0.4, color="darkorange")
    ax_v.axvline(t_evt,   color="black",   linestyle="--", linewidth=1.0, alpha=0.8)
    ax_v.axvline(t_nadir, color="dimgray", linestyle=":",  linewidth=1.0, alpha=0.8)
    ax_v.annotate(
        "RoCoF onset",
        xy=(t_evt, 0.6), xytext=(t_evt + 0.1, 0.9),
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8), fontsize=7.5,
    )
    ax_v.annotate(
        f"Nadir hold\n(f={f_nadir:.2f} Hz)",
        xy=(t_nadir, 0.4), xytext=(t_nadir + 0.08, 0.75),
        arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.8), fontsize=7.5,
    )
    ax_v.axvspan(0,       t_evt,    alpha=0.06, color="green",  label="Nominal")
    ax_v.axvspan(t_evt,   t_nadir,  alpha=0.06, color="orange", label="RoCoF ramp")
    ax_v.axvspan(t_nadir, sc.t[-1], alpha=0.06, color="red",    label="Nadir hold")
    ax_v.set_ylabel("Voltage [pu]")
    ax_v.set_ylim(-1.4, 1.4)
    ax_v.grid(True, alpha=0.25)
    ax_v.legend(loc="upper right", ncol=3, fontsize=7)
    ax_v.set_title(
        "IBR Harmonics Medium — Realistic Grid Edge (THD ≈ 8 %)\n"
        "3rd + 5th + 7th + 11th + 13th  |  75 Hz Interharmonic  |  White + Brown Noise",
        fontsize=9,
    )

    ax_f = axes[1]
    ax_f.plot(sc.t, sc.f_true, linewidth=1.2, color="saddlebrown", label="$f_{true}(t)$")
    ax_f.axvline(t_evt,   color="black",   linestyle="--", linewidth=1.0, alpha=0.8)
    ax_f.axvline(t_nadir, color="dimgray", linestyle=":",  linewidth=1.0, alpha=0.8)
    ax_f.axhline(f0,      color="green",  linestyle="--", linewidth=0.8,
                 alpha=0.7, label=f"$f_0$ = {f0:.1f} Hz")
    ax_f.axhline(f_nadir, color="red",    linestyle="--", linewidth=0.8,
                 alpha=0.7, label=f"$f_{{nadir}}$ = {f_nadir:.2f} Hz")
    ax_f.annotate(
        f"RoCoF = {p['rocof_hz_s']:+.1f} Hz/s",
        xy=((t_evt + t_nadir) / 2, (f0 + f_nadir) / 2),
        xytext=((t_evt + t_nadir) / 2 + 0.05, (f0 + f_nadir) / 2 + 0.1),
        arrowprops=dict(arrowstyle="->", color="saddlebrown", lw=0.8),
        color="saddlebrown", fontsize=7.5,
    )
    ax_f.axvspan(0,       t_evt,    alpha=0.06, color="green")
    ax_f.axvspan(t_evt,   t_nadir,  alpha=0.06, color="orange")
    ax_f.axvspan(t_nadir, sc.t[-1], alpha=0.06, color="red")
    ax_f.set_ylim(f_nadir - 0.3, f0 + 0.3)
    ax_f.set_xlim(sc.t[0], sc.t[-1])
    ax_f.set_xlabel("Time [s]")
    ax_f.set_ylabel("Frequency [Hz]")
    ax_f.set_title("Ground-Truth Frequency: RoCoF Ramp + Nadir Hold", fontsize=9)
    ax_f.legend(loc="lower left", ncol=2)
    ax_f.grid(True, alpha=0.25)

    fig.savefig(out_dir / "ibr_harmonics_medium.png")
    plt.close(fig)
    print(f"[DIAGNOSTIC] Overview saved to: {out_dir / 'ibr_harmonics_medium.png'}")

    # ═══════════════════════════════════════════════════════════════════════
    # FIGURE 2 — Zoom: RoCoF onset + harmonic waveform detail
    # ═══════════════════════════════════════════════════════════════════════
    z_lo = t_evt - 0.08
    z_hi = t_evt + 0.30

    mask_z = (sc.t >= z_lo) & (sc.t <= z_hi)
    t_z    = sc.t[mask_z]
    v_z    = sc.v[mask_z]
    f_z    = sc.f_true[mask_z]

    phi0  = np.arcsin(np.clip(v_z[0], -1, 1))
    v_ref = np.sin(2.0 * math.pi * f0 * (t_z - t_z[0]) + phi0)

    fig2, axes2 = plt.subplots(2, 1, figsize=(8.5, 5.0), sharex=True)
    plt.subplots_adjust(hspace=0.12)

    ax_zv = axes2[0]
    ax_zv.plot(t_z, v_ref, linewidth=0.9, color="green",       linestyle="--",
               alpha=0.6, label="Pure sinusoid reference")
    ax_zv.plot(t_z, v_z,   linewidth=0.9, color="darkorange",  label="v(t) — harmonics + noise")
    ax_zv.axvline(t_evt, color="black", linestyle="--", linewidth=1.0, alpha=0.9)
    ax_zv.annotate(
        f"5th ({p.get('h5_pct',0.05)*100:.0f}%) + 7th ({p.get('h7_pct',0.03)*100:.0f}%)\nwaveform distortion",
        xy=(t_z[len(t_z)//5], v_z[len(t_z)//5]),
        xytext=(t_z[len(t_z)//5] - 0.04, 0.9),
        arrowprops=dict(arrowstyle="->", color="darkorange", lw=0.8), fontsize=7.5,
    )
    ax_zv.axvspan(z_lo,  t_evt, alpha=0.07, color="green")
    ax_zv.axvspan(t_evt, z_hi,  alpha=0.07, color="orange")
    ax_zv.set_ylabel("Voltage [pu]")
    ax_zv.set_ylim(-1.3, 1.3)
    ax_zv.grid(True, alpha=0.25)
    ax_zv.legend(loc="upper left", fontsize=7.5)
    ax_zv.set_title(
        "Harmonic Waveform Detail — Zoom Around RoCoF Onset",
        fontsize=9,
    )

    ax_zf = axes2[1]
    ax_zf.plot(t_z, f_z, linewidth=1.2, color="saddlebrown", label="$f_{true}(t)$")
    ax_zf.axvline(t_evt, color="black", linestyle="--", linewidth=1.0, alpha=0.9)
    ax_zf.axhline(f0, color="green", linestyle="--", linewidth=0.8,
                  alpha=0.7, label=f"f_0 = {f0:.1f} Hz")
    ax_zf.annotate(
        f"RoCoF = {p['rocof_hz_s']:+.1f} Hz/s",
        xy=(t_evt + 0.12, f0 + p["rocof_hz_s"] * 0.12),
        xytext=(t_evt + 0.04, f0 - 0.05),
        arrowprops=dict(arrowstyle="->", color="saddlebrown", lw=0.8),
        color="saddlebrown", fontsize=7.5,
    )
    ax_zf.axvspan(z_lo,  t_evt, alpha=0.07, color="green")
    ax_zf.axvspan(t_evt, z_hi,  alpha=0.07, color="orange")
    ax_zf.set_xlabel("Time [s]")
    ax_zf.set_ylabel("Frequency [Hz]")
    ax_zf.set_ylim(f_z.min() - 0.15, f0 + 0.15)
    ax_zf.set_xlim(z_lo, z_hi)
    ax_zf.grid(True, alpha=0.25)
    ax_zf.legend(loc="lower left", fontsize=7.5)
    ax_zf.set_title("RoCoF Onset Detail", fontsize=9)

    fig2.savefig(out_dir / "ibr_harmonics_medium_zoom.png")
    plt.close(fig2)
    print(f"[DIAGNOSTIC] Zoom saved to:     {out_dir / 'ibr_harmonics_medium_zoom.png'}")
    print(f"  f0={f0:.3f} Hz  f_nadir={f_nadir:.3f} Hz  THD={p['thd_pct']:.1f}%")


test_harmonics_medium_plot_and_csv()
