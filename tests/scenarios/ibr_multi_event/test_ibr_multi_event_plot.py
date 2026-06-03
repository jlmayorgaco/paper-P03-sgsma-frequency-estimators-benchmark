import sys
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenarios.ibr_multi_event import IBRMultiEventScenario


def test_ibr_multi_event_plot_and_csv():
    out_dir = ROOT / "tests" / "scenarios" / "ibr_multi_event" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Scenario parameters ───────────────────────────────────────────────
    t_evt          = 0.5            # s   — fault onset
    rocof          = -2.0           # Hz/s
    rocof_dur      = 1.0            # s   — RoCoF window; frequency clamps after this
    amp_post       = 0.75            # pu  — 50 % voltage sag
    jump_rad       = math.pi / 4.0  # rad — +45° phase jump

    # Clean signal (zero noise) so all structural features are visible
    sc = IBRMultiEventScenario.run(
        duration_s        = 5,
        amp_post_pu       = amp_post,
        phase_jump_rad    = jump_rad,
        t_event_s         = t_evt,
        rocof_hz_s        = rocof,
        rocof_duration_s  = rocof_dur,
        white_noise_sigma = 0.025,
        brown_noise_sigma = 0.000000001,
        impulse_prob      = 0.001,
        seed              = 42,
    )

    df = pd.DataFrame({"t_s": sc.t, "v_pu": sc.v, "f_true_hz": sc.f_true})
    df.to_csv(out_dir / "ibr_multi_event.csv", index=False)

    # ── Derived display values ────────────────────────────────────────────
    t_nadir   = t_evt + rocof_dur           # s   — RoCoF clamp point
    f_nadir   = sc.meta["parameters"]["f_steady_hz"]
    jump_deg  = round(math.degrees(jump_rad), 1)
    dt        = float(sc.t[1] - sc.t[0])

    # ── Figure layout ────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family":    "serif",
        "font.size":      9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi":     300,
        "savefig.bbox":   "tight",
    })

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.5), sharex=True)
    plt.subplots_adjust(hspace=0.12)

    # ── Panel A: Voltage ─────────────────────────────────────────────────
    ax_v = axes[0]
    ax_v.plot(sc.t, sc.v, linewidth=0.5, color="steelblue", label="v(t)")

    ax_v.axvline(t_evt,   color="black",   linestyle="--", linewidth=1.0, alpha=0.8)
    ax_v.axvline(t_nadir, color="dimgray", linestyle=":",  linewidth=1.0, alpha=0.8)

    ax_v.annotate(
        f"Fault onset\nSag {amp_post} pu | Δφ +{jump_deg}°\n+DC offset",
        xy=(t_evt, 0.55), xytext=(t_evt + 0.18, 1.1),
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
        fontsize=7.5,
    )
    ax_v.annotate(
        f"RoCoF clamp\n({rocof_dur} s after fault)",
        xy=(t_nadir, 0.3), xytext=(t_nadir + 0.15, 0.85),
        arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.8),
        fontsize=7.5,
    )

    ax_v.set_ylabel("Voltage [pu]")
    ax_v.set_ylim(-1.4, 1.55)
    ax_v.grid(True, alpha=0.25)

    # Region shading
    ax_v.axvspan(0,       t_evt,   alpha=0.06, color="green",  label="Pre-fault (nominal)")
    ax_v.axvspan(t_evt,   t_nadir, alpha=0.06, color="orange", label="RoCoF ramp region")
    ax_v.axvspan(t_nadir, sc.t[-1],alpha=0.06, color="red",    label="Nadir hold region")
    ax_v.legend(loc="upper right", ncol=3, fontsize=7)

    ax_v.set_title(
        "IBR Multi-Event Composite Fault — Voltage Waveform\n"
        "(Sag + Phase Jump + Decaying DC + AM/PM Modulation + IEEE 519 Harmonics)",
        fontsize=9,
    )

    # ── Panel B: True Frequency ───────────────────────────────────────────
    ax_f = axes[1]
    ax_f.plot(sc.t, sc.f_true, linewidth=1.2, color="crimson", label="$f_{true}(t)$")

    ax_f.axvline(t_evt,   color="black",   linestyle="--", linewidth=1.0, alpha=0.8)
    ax_f.axvline(t_nadir, color="dimgray", linestyle=":",  linewidth=1.0, alpha=0.8)

    # Nominal and nadir reference lines
    ax_f.axhline(sc.f_true[0],  color="green",  linestyle="--", linewidth=0.8,
                 alpha=0.7, label=f"$f_0$ = {sc.f_true[0]:.1f} Hz (nominal)")
    ax_f.axhline(f_nadir, color="red",    linestyle="--", linewidth=0.8,
                 alpha=0.7, label=f"$f_{{nadir}}$ = {f_nadir:.2f} Hz")

    ax_f.annotate(
        f"RoCoF = {rocof} Hz/s",
        xy=((t_evt + t_nadir) / 2, f_nadir + (sc.f_true[0] - f_nadir) / 2),
        xytext=((t_evt + t_nadir) / 2 + 0.05, f_nadir + (sc.f_true[0] - f_nadir) * 0.7),
        arrowprops=dict(arrowstyle="->", color="crimson", lw=0.8),
        color="crimson", fontsize=7.5,
    )
    ax_f.annotate(
        f"PM ripple\n(±{sc.meta['parameters'].get('pm_mag_rad', 0.05)*1e3:.0f} mHz)",
        xy=(t_nadir + 0.8, f_nadir - 0.03),
        xytext=(t_nadir + 1.0, f_nadir - 0.25),
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
        fontsize=7.5,
    )

    # Region shading (same colors)
    ax_f.axvspan(0,       t_evt,    alpha=0.06, color="green")
    ax_f.axvspan(t_evt,   t_nadir,  alpha=0.06, color="orange")
    ax_f.axvspan(t_nadir, sc.t[-1], alpha=0.06, color="red")

    f_min = f_nadir - 0.5
    f_max = sc.f_true[0] + 0.3
    ax_f.set_ylim(f_min, f_max)
    ax_f.set_xlim(sc.t[0], sc.t[-1])
    ax_f.set_xlabel("Time [s]")
    ax_f.set_ylabel("Frequency [Hz]")
    ax_f.set_title(
        "Ground-Truth Frequency: RoCoF Ramp → Nadir Clamp + PM Oscillation",
        fontsize=9,
    )
    ax_f.legend(loc="lower left", ncol=2)
    ax_f.grid(True, alpha=0.25)

    out_path = out_dir / "ibr_multi_event2.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[DIAGNOSTIC] Plot saved to: {out_path}")
    print(f"  f0 = {sc.f_true[0]:.3f} Hz  |  f_nadir = {f_nadir:.3f} Hz  "
          f"|  df = {f_nadir - sc.f_true[0]:.3f} Hz  "
          f"|  RoCoF clamps at t = {t_nadir:.2f} s")

    # ── Zoom figure: fault transition detail ─────────────────────────────
    # Window: 5 pre-fault cycles + enough post to see DC decay (3*tau ~ 90 ms)
    z_lo = t_evt - 0.08    # ~5 cycles before at 60 Hz
    z_hi = t_evt + 0.35    # covers DC decay + ~20 post-fault cycles

    mask_z = (sc.t >= z_lo) & (sc.t <= z_hi)
    t_z    = sc.t[mask_z]
    v_z    = sc.v[mask_z]
    f_z    = sc.f_true[mask_z]

    # Reference: clean pre-fault sinusoid (no sag, no noise) for comparison
    amp_pre = sc.meta["parameters"]["amp_pre_pu"]
    v_ref   = amp_pre * __import__("numpy").sin(
        2.0 * math.pi * sc.f_true[0] * t_z
        + 2.0 * math.pi * sc.f_true[0] * t_z[0] * 0  # phase anchored at t_z[0]
    )
    # anchor phase to match the actual signal just before the event
    # use first sample of zoom window for offset
    import numpy as _np
    phi0_ref  = 2.0 * math.pi * sc.f_true[0] * t_z[0]
    phi0_sig  = _np.arcsin(_np.clip(v_z[0] / amp_pre, -1, 1))
    v_ref     = amp_pre * _np.sin(2.0 * math.pi * sc.f_true[0] * (t_z - t_z[0]) + phi0_sig)

    fig2, axes2 = plt.subplots(2, 1, figsize=(8.5, 5.0), sharex=True)
    plt.subplots_adjust(hspace=0.12)

    # ── Zoom panel A: voltage ────────────────────────────────────────────
    ax_zv = axes2[0]
    ax_zv.plot(t_z, v_ref, linewidth=0.8, color="green",  linestyle="--",
               alpha=0.6, label="Pre-fault reference (no disturbance)")
    ax_zv.plot(t_z, v_z,   linewidth=0.9, color="steelblue", label="v(t) — composite fault")

    ax_zv.axvline(t_evt, color="black", linestyle="--", linewidth=1.0, alpha=0.9)

    # Annotate DC offset peak (first post-fault sample)
    idx_evt   = int((t_evt - z_lo) / dt)
    dc_peak   = sc.meta["parameters"]["dc_offset_pu"]
    dc_tau    = sc.meta["parameters"]["dc_tau_s"]
    ax_zv.annotate(
        f"DC offset peak\n+{dc_peak:.0%} pu  (tau={dc_tau*1e3:.0f} ms)",
        xy=(t_evt + 0.002, v_z[idx_evt + 1] if idx_evt + 1 < len(v_z) else v_z[-1]),
        xytext=(t_evt + 0.04, 0.85),
        arrowprops=dict(arrowstyle="->", color="navy", lw=0.8),
        fontsize=7.5,
    )
    ax_zv.annotate(
        f"Sag to {sc.meta['parameters']['amp_post_pu']} pu\nDelta-phi = +{jump_deg} deg",
        xy=(t_evt + 0.01, -0.3),
        xytext=(t_evt + 0.08, -0.75),
        arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8),
        fontsize=7.5,
    )

    ax_zv.set_ylabel("Voltage [pu]")
    ax_zv.set_ylim(-1.2, 1.2)
    ax_zv.grid(True, alpha=0.25)
    ax_zv.legend(loc="upper left", fontsize=7.5)
    ax_zv.set_title(
        "Fault Transition — Zoom Detail\n"
        "(Phase Jump + Sag + Decaying DC  |  Pre-fault reference overlaid)",
        fontsize=9,
    )

    # Shade pre / post regions
    ax_zv.axvspan(z_lo,  t_evt, alpha=0.07, color="green",  label="_")
    ax_zv.axvspan(t_evt, z_hi,  alpha=0.07, color="orange", label="_")

    # ── Zoom panel B: frequency ──────────────────────────────────────────
    ax_zf = axes2[1]
    ax_zf.plot(t_z, f_z, linewidth=1.2, color="crimson", label="$f_{true}(t)$")
    ax_zf.axvline(t_evt, color="black", linestyle="--", linewidth=1.0, alpha=0.9)
    ax_zf.axhline(sc.f_true[0], color="green", linestyle="--", linewidth=0.8,
                  alpha=0.7, label=f"$f_0$ = {sc.f_true[0]:.1f} Hz")

    # Annotate RoCoF slope with a bracket-style arrow
    t_mid  = t_evt + 0.15
    f_mid  = sc.f_true[0] + rocof * 0.15
    ax_zf.annotate(
        f"RoCoF = {rocof:+.1f} Hz/s",
        xy=(t_mid, f_mid),
        xytext=(t_evt + 0.04, sc.f_true[0] - 0.08),
        arrowprops=dict(arrowstyle="->", color="crimson", lw=0.8),
        color="crimson", fontsize=7.5,
    )

    ax_zf.set_xlabel("Time [s]")
    ax_zf.set_ylabel("Frequency [Hz]")
    # Tight y-window: only show what changes within the zoom
    f_lo_z = f_z.min() - 0.15
    f_hi_z = sc.f_true[0] + 0.15
    ax_zf.set_ylim(f_lo_z, f_hi_z)
    ax_zf.set_xlim(z_lo, z_hi)
    ax_zf.grid(True, alpha=0.25)
    ax_zf.legend(loc="lower left", fontsize=7.5)
    ax_zf.set_title(
        "Instantaneous Frequency — Zoom  (PM ripple + RoCoF onset)",
        fontsize=9,
    )

    ax_zv.axvspan(z_lo,  t_evt, alpha=0.07, color="green")
    ax_zv.axvspan(t_evt, z_hi,  alpha=0.07, color="orange")
    ax_zf.axvspan(z_lo,  t_evt, alpha=0.07, color="green")
    ax_zf.axvspan(t_evt, z_hi,  alpha=0.07, color="orange")

    zoom_path = out_dir / "ibr_multi_event_zoom.png"
    fig2.savefig(zoom_path)
    plt.close(fig2)
    print(f"[DIAGNOSTIC] Zoom plot saved to: {zoom_path}")


test_ibr_multi_event_plot_and_csv()
