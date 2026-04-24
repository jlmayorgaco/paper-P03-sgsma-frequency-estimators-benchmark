import sys
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenarios.ibr_power_imbalance_ringdown import IBRPowerImbalanceRingdownScenario

def test_ringdown_plot_and_csv():
    out_dir = ROOT / "tests" / "scenarios" / "ibr_power_imbalance_ringdown" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Generar Escenario ──────────────────────────────────────────────
    # Usamos parámetros de ruido moderados para que la gráfica sea legible 
    # pero muestre la distorsión.
    sc = IBRPowerImbalanceRingdownScenario.run(
        white_noise_sigma = 0.01,
        brown_noise_sigma = 0.000001,
        interharmonic_pu  = 0.03,
        impulse_prob      = 0.002,
        seed              = 42,
    )

    df = pd.DataFrame({"t_s": sc.t, "v_pu": sc.v, "f_true_hz": sc.f_true})
    df.to_csv(out_dir / "ibr_ringdown.csv", index=False)

    # Extraer parámetros actualizados para las anotaciones
    p  = sc.meta["parameters"]
    dt = float(sc.t[1] - sc.t[0])
    f0 = p["freq_nom_hz"]
    f_ring_peak = f0 + p["ring_jump_hz"]
    
    t_ramp = p["t_ramp_s"]
    t_evt  = p["t_event_s"]
    jump_ramp_deg = math.degrees(p["phase_jump_ramp_rad"])
    jump_evt_deg  = math.degrees(p["phase_jump_event_rad"])

    # ═══════════════════════════════════════════════════════════════════════
    # FIGURE 1 — Overview (full signal)
    # ═══════════════════════════════════════════════════════════════════════
    plt.rcParams.update({
        "font.family":    "serif",
        "font.size":      9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi":     300,
        "savefig.bbox":   "tight",
    })

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.0), sharex=True)
    plt.subplots_adjust(hspace=0.15)

    # ── Voltage panel ────────────────────────────────────────────────────
    ax_v = axes[0]
    ax_v.plot(sc.t, sc.v, linewidth=0.4, color="mediumpurple")
    ax_v.axvline(t_ramp, color="steelblue", linestyle="--", linewidth=1.0, alpha=0.8)
    ax_v.axvline(t_evt,  color="black",     linestyle="--", linewidth=1.0, alpha=0.9)
    
    # Anotación Salto 1 (Falla)
    ax_v.annotate(
        f"Fault Onset:\nDrop to {p['amp_fault_pu']} pu\nΔφ +{jump_ramp_deg:.0f}°",
        xy=(t_ramp, 0.4), xytext=(t_ramp - 0.35, 0.9),
        arrowprops=dict(arrowstyle="->", color="steelblue", lw=0.8), fontsize=7.5,
    )
    # Anotación Salto 2 (Reconexión / Ringdown)
    ax_v.annotate(
        f"Reconnection:\nBase {p['amp_post_pu']} pu\nΔφ +{jump_evt_deg:.0f}°\n+DC & V-f coupling",
        xy=(t_evt, 0.4), xytext=(t_evt + 0.1, 0.9),
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8), fontsize=7.5,
    )
    
    ax_v.axvspan(0,      t_ramp, alpha=0.06, color="green",  label="Nominal")
    ax_v.axvspan(t_ramp, t_evt,  alpha=0.06, color="orange", label="Fault (RoCoF)")
    ax_v.axvspan(t_evt, sc.t[-1],alpha=0.06, color="red",    label="Ring-down")
    ax_v.set_ylabel("Voltage [pu]")
    ax_v.set_ylim(-1.5, 1.5)
    ax_v.grid(True, alpha=0.25)
    ax_v.legend(loc="upper right", ncol=3, fontsize=7)
    ax_v.set_title(
        "IBR Power Imbalance — Dual Phase Jump & V-f Coupling\n"
        "(IEEE 519 Harmonics + 32.5 Hz Interharmonic + Multi-spectral Noise)",
        fontsize=9, fontweight="bold"
    )

    # ── Frequency panel ──────────────────────────────────────────────────
    ax_f = axes[1]
    ax_f.plot(sc.t, sc.f_true, linewidth=1.2, color="darkred", label="$f_{true}(t)$")
    ax_f.axvline(t_ramp, color="steelblue", linestyle="--", linewidth=1.0, alpha=0.8)
    ax_f.axvline(t_evt,  color="black",     linestyle="--", linewidth=1.0, alpha=0.9)
    ax_f.axhline(f0, color="green", linestyle="--", linewidth=0.8,
                 alpha=0.7, label=f"$f_0$ = {f0:.1f} Hz")
                 
    ax_f.annotate(
        f"RoCoF = {p['rocof_hz_s']:+.1f} Hz/s",
        xy=((t_ramp + t_evt) / 2, f0 + p['rocof_hz_s'] * (t_evt - t_ramp) / 2),
        xytext=(t_ramp + 0.05, f0 + p['rocof_hz_s'] * (t_evt - t_ramp) * 0.7),
        arrowprops=dict(arrowstyle="->", color="steelblue", lw=0.8),
        color="steelblue", fontsize=7.5,
    )
    ax_f.annotate(
        f"Ring-down: peak {f_ring_peak:.2f} Hz\nf_ring={p['ring_freq_hz']} Hz, tau={p['ring_tau_s']} s",
        xy=(t_evt + 0.05, f_ring_peak),
        xytext=(t_evt + 0.3, f_ring_peak + 0.3),
        arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8),
        fontsize=7.5,
    )
    
    ax_f.axvspan(0,      t_ramp, alpha=0.06, color="green")
    ax_f.axvspan(t_ramp, t_evt,  alpha=0.06, color="orange")
    ax_f.axvspan(t_evt, sc.t[-1],alpha=0.06, color="red")
    f_vals = sc.f_true
    ax_f.set_ylim(f_vals.min() - 0.3, f_vals.max() + 0.5)
    ax_f.set_xlim(sc.t[0], sc.t[-1])
    ax_f.set_xlabel("Time [s]")
    ax_f.set_ylabel("Frequency [Hz]")
    ax_f.legend(loc="lower right", ncol=2)
    ax_f.grid(True, alpha=0.25)

    fig.savefig(out_dir / "ibr_ringdown.png")
    plt.close(fig)
    print(f"[DIAGNOSTIC] Overview plot saved to: {out_dir / 'ibr_ringdown.png'}")

    # ═══════════════════════════════════════════════════════════════════════
    # FIGURE 2 — Zoom: Reconnection transition detail (t = 1.0s)
    # ═══════════════════════════════════════════════════════════════════════
    # Enfocamos el zoom en el segundo evento (reconexión)
    z_lo = t_evt - 0.75
    z_hi = t_evt + 0.75

    mask_z = (sc.t >= z_lo) & (sc.t <= z_hi)
    t_z    = sc.t[mask_z]
    v_z    = sc.v[mask_z]
    f_z    = sc.f_true[mask_z]

    fig2, axes2 = plt.subplots(2, 1, figsize=(8.5, 5.0), sharex=True)
    plt.subplots_adjust(hspace=0.15)

    ax_zv = axes2[0]
    ax_zv.plot(t_z, v_z, linewidth=0.9, color="mediumpurple", label="v(t) — ring-down scenario")
    ax_zv.axvline(t_evt, color="black", linestyle="--", linewidth=1.0, alpha=0.9)
    
    ax_zv.annotate(
        f"DC peak {p['dc_offset_pu']*100:.0f}% (tau {p['dc_tau_s']*1e3:.0f} ms)",
        xy=(t_evt + 0.003, v_z[int(0.003/dt)] if int(0.003/dt) < len(v_z) else v_z[-1]),
        xytext=(t_evt + 0.05, 1.0),
        arrowprops=dict(arrowstyle="->", color="navy", lw=0.8), fontsize=7.5,
    )
    ax_zv.annotate(
        f"Base amp {p['amp_post_pu']} pu | +{jump_evt_deg:.0f}° phase jump",
        xy=(t_evt + 0.01, -0.4),
        xytext=(t_evt + 0.06, -0.95),
        arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8), fontsize=7.5,
    )
    
    ax_zv.axvspan(z_lo,  t_evt, alpha=0.07, color="orange", label="Fault (RoCoF)")
    ax_zv.axvspan(t_evt, z_hi,  alpha=0.07, color="red", label="Ring-down (Coupled)")
    ax_zv.set_ylabel("Voltage [pu]")
    ax_zv.set_ylim(-1.3, 1.3)
    ax_zv.grid(True, alpha=0.25)
    ax_zv.legend(loc="upper left", fontsize=7.5)
    ax_zv.set_title(
        "Reconnection Transition — Zoom Detail\n"
        "(Second Phase Jump + DC Transient + Amplitude-Frequency Coupling)",
        fontsize=9, fontweight="bold"
    )

    ax_zf = axes2[1]
    ax_zf.plot(t_z, f_z, linewidth=1.2, color="darkred", label="$f_{true}(t)$")
    ax_zf.axvline(t_evt, color="black", linestyle="--", linewidth=1.0, alpha=0.9)
    ax_zf.axhline(f0, color="green", linestyle="--", linewidth=0.8,
                  alpha=0.7, label=f"$f_0$ = {f0:.1f} Hz")
                  
    ax_zf.annotate(
        f"Ring-down onset\npeak = {f_ring_peak:.2f} Hz",
        xy=(t_evt + 0.01, f_ring_peak - 0.1),
        xytext=(t_evt + 0.08, f_ring_peak - 0.4),
        arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8), fontsize=7.5,
    )
    
    ax_zf.axvspan(z_lo,  t_evt, alpha=0.07, color="orange")
    ax_zf.axvspan(t_evt, z_hi,  alpha=0.07, color="red")
    ax_zf.set_xlabel("Time [s]")
    ax_zf.set_ylabel("Frequency [Hz]")
    ax_zf.set_ylim(f_z.min() - 0.3, f_z.max() + 0.5)
    ax_zf.set_xlim(z_lo, z_hi)
    ax_zf.grid(True, alpha=0.25)
    ax_zf.legend(loc="lower left", fontsize=7.5)

    fig2.savefig(out_dir / "ibr_ringdown_zoom.png")
    plt.close(fig2)
    print(f"[DIAGNOSTIC] Zoom plot saved to:     {out_dir / 'ibr_ringdown_zoom.png'}")

if __name__ == "__main__":
    test_ringdown_plot_and_csv()