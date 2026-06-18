"""Two scenario-suite dashboards from the full Monte-Carlo benchmark scenario
CSVs (artifacts/full_mc_benchmark/<scn>/<scn>_scenario.csv, cols t_s,v_pu,
f_true_hz). Each column is one scenario: TOP row = voltage waveform v(t),
BOTTOM row = true frequency f(t). A faint 60 Hz reference makes frequency
events visible in v(t); flat f(t) panels are annotated to show the stress is
not a frequency event. Saves slides/figures/scenario_suite_{1,2}.{pdf,png}.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "artifacts" / "full_mc_benchmark"
FIG = Path(__file__).with_name("figures")

INK = "#0E2841"; BLUE = "#156082"; GOLD = "#E97132"
GREEN = "#196B24"; VIOLET = "#A02B93"; RED = "#B23A48"; MUTED = "#5B677A"
REFC = "#AEBAC4"
W = 2 * np.pi * 60.0

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9.0,
    "axes.titlesize": 9.8, "axes.titleweight": "bold", "axes.titlecolor": "black",
    "axes.labelsize": 8.6, "axes.edgecolor": "black", "axes.linewidth": 0.9,
    "text.color": "black", "axes.labelcolor": "black",
    "xtick.color": "black", "ytick.color": "black",
    "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
    "xtick.direction": "in", "ytick.direction": "in",
    "legend.fontsize": 7.2, "figure.dpi": 300,
    "lines.antialiased": True, "patch.antialiased": True,
    "savefig.dpi": 300,
})


def load(scn):
    df = pd.read_csv(BASE / scn / f"{scn}_scenario.csv")
    return df["t_s"].to_numpy(), df["v_pu"].to_numpy(), df["f_true_hz"].to_numpy()


def envelope(v, w=167):
    return pd.Series(np.abs(v)).rolling(w, min_periods=1, center=True).max().to_numpy()


def ref60(t, v, t0, t1):
    m = (t >= t0) & (t < t1)
    A = np.c_[np.cos(W * t[m]), np.sin(W * t[m])]
    a, b = np.linalg.lstsq(A, v[m], rcond=None)[0]
    return a * np.cos(W * t) + b * np.sin(W * t)


def style(ax):
    # publication-style full box frame (matches the paper figures)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("black")
        s.set_linewidth(0.9)
    ax.grid(True, color="#c7c7c7", lw=0.55, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(direction="in", length=3.2, width=0.8)
    ax.margins(x=0.02)


def note(ax, txt, color=INK):
    ax.annotate(txt, xy=(0.5, 0.035), xycoords="axes fraction", ha="center",
                va="bottom", fontsize=6.2, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.78))


def vpanel(ax, scn, t0, t1, color, *, ref=None, env=False, raw_alpha=1.0,
           ann=None, vline=None):
    t, v, _ = load(scn)
    m = (t >= t0) & (t <= t1)
    x = (t[m] - t0) * 1e3
    if ref is not None:
        r = ref60(t, v, ref[0], ref[1])
        ax.plot(x, r[m], lw=1.2, ls=(0, (4, 2)), color="#7C8B97", zorder=1)
    ax.plot(x, v[m], lw=1.7, color=color, alpha=raw_alpha, zorder=2,
            solid_capstyle="round", solid_joinstyle="round")
    if env:
        e = envelope(v)
        ax.plot(x, e[m], lw=2.0, color="black", zorder=3, solid_capstyle="round")
        ax.plot(x, -e[m], lw=2.0, color="black", zorder=3, solid_capstyle="round")
    if vline is not None:
        ax.axvline((vline - t0) * 1e3, color="black", lw=1.1, ls=(0, (3, 2)), zorder=4)
    ax.set_xlabel(f"time from {t0:.2f} s [ms]")
    if ann:
        note(ax, ann)
    style(ax)


def fpanel(ax, scn, t0, t1, color, *, flat=False, ann=None, vline=None, ylim=None):
    t, _, f = load(scn)
    m = (t >= t0) & (t <= t1)
    ax.plot(t[m], f[m], lw=1.9, color=color, solid_capstyle="round", solid_joinstyle="round")
    ax.axhline(60, color="#7C8B97", lw=0.9, ls="--")
    if vline is not None:
        ax.axvline(vline, color="black", lw=1.1, ls=(0, (3, 2)), zorder=4)
    if ylim is not None:
        ax.set_ylim(*ylim)
    elif flat:
        ax.set_ylim(59.45, 60.55)
    ax.set_xlabel("time [s]")
    if ann:
        note(ax, ann, color=(RED if flat else INK))
    style(ax)


def phase_window(scn, half=0.013):
    t, v, _ = load(scn)
    d = np.abs(np.diff(v, prepend=v[0])); lo, hi = int(0.05*len(v)), int(0.95*len(v))
    tc = t[lo + int(np.argmax(d[lo:hi]))]
    return tc, half


def dashboard(specs, fname, suptitle):
    n = len(specs)
    fig, ax = plt.subplots(2, n, figsize=(2.65 * n, 5.1))
    fig.subplots_adjust(left=0.058, right=0.99, top=0.84, bottom=0.10,
                        wspace=0.40, hspace=0.46)
    for j, sp in enumerate(specs):
        av, af = ax[0, j], ax[1, j]
        sp["v"](av)
        av.set_title(sp["title"])
        sp["f"](af)
        if j == 0:
            av.set_ylabel("VOLTAGE\n$v(t)$  [pu]", fontweight="bold", fontsize=8.6)
            af.set_ylabel("FREQUENCY\n$f_\\mathrm{true}(t)$  [Hz]", fontweight="bold", fontsize=8.6)
    fig.suptitle(suptitle, fontsize=10.5, fontweight="bold", color=INK, y=0.955)
    FIG.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{fname}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / f"{fname}.pdf")


def main():
    # ---------------- Dashboard I: single-mechanism stresses ----------------
    tcpj, hpj = phase_window("IEEE_Phase_Jump_60")
    d1 = [
        dict(title="A  Magnitude step",
             v=lambda a: vpanel(a, "IEEE_Mag_Step_25pct", 0.42, 0.60, BLUE,
                                ann="amplitude $+25\\%$", vline=0.50),
             f=lambda a: fpanel(a, "IEEE_Mag_Step_25pct", 0.42, 0.60, BLUE,
                                flat=True, ann="f stays 60 Hz", vline=0.50)),
        dict(title="B  Frequency ramp",
             v=lambda a: vpanel(a, "IEEE_Freq_Ramp_10Hzs", 0.30, 0.50, GREEN,
                                ref=(0.05, 0.30), vline=0.305, ann="speeds up vs 60 Hz ref"),
             f=lambda a: fpanel(a, "IEEE_Freq_Ramp_10Hzs", 0.0, 0.8, GREEN,
                                ann="ramp $\\to$ 61.5 Hz")),
        dict(title="Frequency step",
             v=lambda a: vpanel(a, "IEEE_Freq_Step", 0.46, 0.63, GOLD,
                                ref=(0.05, 0.50), vline=0.50, ann="slows after step"),
             f=lambda a: fpanel(a, "IEEE_Freq_Step", 0.0, 1.0, GOLD,
                                ann="step $-1$ Hz", ylim=(58.7, 60.3))),
        dict(title="C  Modulation: AM",
             v=lambda a: vpanel(a, "IEEE_Modulation_AM", 0.50, 1.00, VIOLET,
                                env=True, raw_alpha=0.55, ann="amplitude modulated"),
             f=lambda a: fpanel(a, "IEEE_Modulation_AM", 0.0, 2.0, VIOLET,
                                flat=True, ann="f stays 60 Hz")),
        dict(title="C  Modulation: FM",
             v=lambda a: vpanel(a, "IEEE_Modulation_FM", 0.50, 0.75, RED,
                                ann="constant amplitude"),
             f=lambda a: fpanel(a, "IEEE_Modulation_FM", 0.0, 1.0, RED,
                                ann="$\\pm0.2$ Hz swing")),
    ]
    dashboard(d1[:3], "scenario_suite_1a",
              "Standard stresses (1 of 2): VOLTAGE v(t) on top, true FREQUENCY f(t) below")
    dashboard(d1[3:], "scenario_suite_1b",
              "Standard stresses (2 of 2): VOLTAGE v(t) on top, true FREQUENCY f(t) below")

    # ---------------- Dashboard II: composite IBR stresses ----------------
    d2 = [
        dict(title="Harmonic distortion",
             v=lambda a: vpanel(a, "IBR_Harmonics_Large", 1.00, 1.033, GREEN,
                                ann="THD-distorted shape"),
             f=lambda a: fpanel(a, "IBR_Harmonics_Large", 0.0, 2.0, GREEN,
                                ann="apparent f ripple")),
        dict(title="D  Phase jump 60$^\\circ$",
             v=lambda a: vpanel(a, "IEEE_Phase_Jump_60", tcpj - hpj, tcpj + hpj, RED,
                                ref=(tcpj - 0.030, tcpj - 0.002), ann="phase discontinuity"),
             f=lambda a: fpanel(a, "IEEE_Phase_Jump_60", 0.0, 2.0, RED,
                                flat=True, ann="phase only: f stays 60 Hz")),
        dict(title="Out-of-band interference",
             v=lambda a: vpanel(a, "IEEE_OOB_Interference", 0.50, 0.60, GOLD,
                                env=True, raw_alpha=0.9, ann="beating waveform"),
             f=lambda a: fpanel(a, "IEEE_OOB_Interference", 0.0, 2.0, GOLD,
                                flat=True, ann="f stays 60 Hz")),
        dict(title="Power-imbalance ringdown",
             v=lambda a: vpanel(a, "IBR_Power_Imbalance_Ringdown", 0.45, 1.50, BLUE,
                                env=True, raw_alpha=0.40, ann="amplitude dip"),
             f=lambda a: fpanel(a, "IBR_Power_Imbalance_Ringdown", 0.45, 1.80, BLUE,
                                ann="damped 59$\\leftrightarrow$61 Hz")),
        dict(title="E  Multi-event IBR",
             v=lambda a: vpanel(a, "IBR_Multi_Event", 0.45, 2.00, VIOLET,
                                env=True, raw_alpha=0.30, ann="stacked amplitude"),
             f=lambda a: fpanel(a, "IBR_Multi_Event", 0.45, 2.00, VIOLET,
                                ann="ramp + jumps + ringdown")),
    ]
    dashboard(d2[:3], "scenario_suite_2a",
              "Composite IBR stresses (1 of 2): VOLTAGE v(t) on top, true FREQUENCY f(t) below")
    dashboard(d2[3:], "scenario_suite_2b",
              "Composite IBR stresses (2 of 2): VOLTAGE v(t) on top, true FREQUENCY f(t) below")


if __name__ == "__main__":
    main()
