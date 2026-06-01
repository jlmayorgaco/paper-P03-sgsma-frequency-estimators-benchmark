"""Two high-value synthesis figures from the paper-grade ATLAS data (n=30, 18 est):

1) Safe-operating-envelope heatmap: for each estimator x stress family, the
   critical severity where frequency RMSE first crosses the 0.05 Hz guide.
   Greener = survives to higher severity. Source: atlas_critical_thresholds.csv.

2) Scaling-exponent chart: the log-log slope of RMSE vs severity per estimator,
   averaged over sweeps, with the slope=1 (Cramer-Rao-efficient) reference.
   slope~1 = error grows linearly with stress; slope~0 = saturated/broken;
   slope>1 = super-linear collapse. Source: hypothesis_results.csv.
"""
from __future__ import annotations
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

REPO = Path(__file__).resolve().parents[1]
ATLAS = REPO / "artifacts" / "atlas-papergrade-missing-v1"
FIG = Path(__file__).with_name("figures")
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9.0,
    "axes.titlesize": 10.0, "axes.titleweight": "bold", "axes.titlecolor": "black",
    "axes.labelsize": 8.8, "axes.edgecolor": "black", "axes.linewidth": 0.9,
    "text.color": "black", "axes.labelcolor": "black",
    "xtick.color": "black", "ytick.color": "black",
    "xtick.labelsize": 7.6, "ytick.labelsize": 7.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "legend.fontsize": 7.6, "figure.dpi": 300, "savefig.dpi": 300,
})

# deck estimator order (loop, window/spectral, model, adaptive, data-driven)
EST_ORDER = ["PLL", "SOGI-PLL", "SOGI-FLL", "Type-3 SOGI-PLL", "ZCD",
             "IPDFT", "TFT", "ESPRIT", "Prony", "MUSIC",
             "EKF", "UKF", "RA-EKF", "LKF", "LKF2",
             "RLS", "TKEO", "Koopman (RK-DPMU)"]
SHORT = {"Koopman (RK-DPMU)": "Koopman", "Type-3 SOGI-PLL": "T3-SOGI-PLL"}

# sweeps to display + a human label + the unit shown in each cell
SWEEPS = [
    ("frequency_step",     "Freq step",      "abs_step_hz",          "Hz"),
    ("rocof",              "RoCoF",          "abs_rocof_hz_s",       "Hz/s"),
    ("phase_jump_sweep",   "Phase jump",     "abs_phase_jump_deg",   "deg"),
    ("magnitude_step",     "Mag step",       "abs_step_percent",     "%"),
    ("harmonics",          "Harmonics",      "thd_percent",          "% THD"),
    ("interharmonics",     "Interharm.",     "interharmonic_percent", "%"),
    ("noise_snr",          "Noise",          "noise_sigma_pu",       "pu"),
    ("modulation_am_sweep", "AM",            "modulation_frequency_hz", "Hz"),
    ("modulation_fm_sweep", "FM",            "modulation_frequency_hz", "Hz"),
]


def _fmt(v):
    if v >= 100: return f"{v:.0f}"
    if v >= 1:   return f"{v:.0f}" if v == int(v) else f"{v:.1f}"
    if v >= 0.01: return f"{v:.2f}"
    return f"{v:.0e}"


def safe_envelope():
    rows = list(csv.DictReader(open(ATLAS / "atlas_critical_thresholds.csv")))
    crit = {}      # (est,sweep) -> critical_level
    maxlvl = {}    # sweep -> max tested
    for r in rows:
        try:
            cl = float(r["critical_level"])
        except (ValueError, KeyError):
            cl = np.nan
        crit[(r["estimator"], r["sweep_key"])] = cl
        try:
            maxlvl[r["sweep_key"]] = float(r["max_level_tested"])
        except (ValueError, KeyError):
            pass

    ests = [e for e in EST_ORDER if any((e, s[0]) in crit for s in SWEEPS)]
    nE, nS = len(ests), len(SWEEPS)
    # color = log-rank of how far it survives within each sweep (per-column normalized)
    Z = np.full((nE, nS), np.nan)
    labels = [["" for _ in range(nS)] for _ in range(nE)]
    for j, (sk, _, _, unit) in enumerate(SWEEPS):
        col = [crit.get((e, sk), np.nan) for e in ests]
        vals = np.array([c for c in col if not np.isnan(c)])
        lo, hi = (vals.min(), vals.max()) if len(vals) else (0, 1)
        mx = maxlvl.get(sk, hi)
        for i, c in enumerate(col):
            if np.isnan(c):
                continue
            # normalized survival: higher critical level => greener (survives more)
            Z[i, j] = (np.log10(c) - np.log10(lo)) / (np.log10(hi) - np.log10(lo) + 1e-9)
            cap = "" if c < mx else "+"   # '+' = never crossed guide in range
            labels[i][j] = f"{_fmt(c)}{cap}"

    cmap = LinearSegmentedColormap.from_list(
        "ofb", ["#B23A48", "#E97132", "#F2D479", "#7FB069", "#196B24"])
    fig, ax = plt.subplots(figsize=(11.6, 6.4))
    im = ax.imshow(Z, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(nS)); ax.set_xticklabels([s[1] for s in SWEEPS], rotation=18, ha="right")
    ax.set_yticks(range(nE)); ax.set_yticklabels([SHORT.get(e, e) for e in ests])
    for i in range(nE):
        for j in range(nS):
            if labels[i][j]:
                z = Z[i, j]
                tc = "white" if (z < 0.22 or z > 0.82) else "black"
                ax.text(j, i, labels[i][j], ha="center", va="center",
                        fontsize=6.6, color=tc, fontweight="bold")
    ax.set_title("Safe-operating envelope: severity where frequency RMSE first exceeds 0.05 Hz")
    ax.set_xlabel("stress family (cell = critical severity, in that family's units;  $+$ = never exceeded)")
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.9)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cb.set_label("relative survival (greener = tolerates more severe stress)", fontsize=7.6)
    cb.set_ticks([0, 1]); cb.set_ticklabels(["first to fail", "last to fail"])
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"atlas_safe_envelope.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / "atlas_safe_envelope.pdf")


def scaling_exponents():
    rows = list(csv.DictReader(open(ATLAS / "hypothesis_results.csv")))
    # sweeps where a log-log slope is physically meaningful (monotone severity)
    keep = {"frequency_step", "rocof", "phase_jump_sweep", "magnitude_step",
            "harmonics", "interharmonics", "noise_snr"}
    by_est = {}
    for r in rows:
        if r["sweep_key"] not in keep:
            continue
        try:
            s = float(r["trend_slope_loglog"])
        except ValueError:
            continue
        # drop degenerate near-zero-divide blowups (|slope|>2.5 are numeric artifacts)
        if abs(s) > 2.5:
            continue
        by_est.setdefault(r["estimator"], []).append(s)

    ests = [e for e in EST_ORDER if e in by_est]
    means = [float(np.mean(by_est[e])) for e in ests]
    stds = [float(np.std(by_est[e])) for e in ests]
    order = np.argsort(means)
    ests = [ests[i] for i in order]; means = [means[i] for i in order]; stds = [stds[i] for i in order]

    def col(m):
        if m < 0.3:  return "#B23A48"   # saturated / broken
        if m < 0.8:  return "#E97132"   # sub-linear (sluggish)
        if m <= 1.15: return "#196B24"  # ~linear: Cramer-Rao-like tracking
        return "#7A1FA0"                # super-linear collapse

    fig, ax = plt.subplots(figsize=(11.4, 5.4))
    y = np.arange(len(ests))
    ax.barh(y, means, xerr=stds, color=[col(m) for m in means],
            edgecolor="black", linewidth=0.6, height=0.66,
            error_kw=dict(ecolor="#555555", lw=0.8, capsize=2))
    ax.axvline(1.0, color="#0E2841", lw=1.4, ls="--")
    ax.text(1.02, len(ests) - 0.4, "slope $=1$\n(error $\\propto$ stress)",
            fontsize=7.6, color="#0E2841", va="top")
    ax.set_yticks(y); ax.set_yticklabels([SHORT.get(e, e) for e in ests])
    ax.set_xlabel("mean log--log scaling exponent of RMSE vs.\\ severity (across 7 stress families)")
    ax.set_title("How error grows with stress: the scaling exponent classifies estimators")
    ax.set_xlim(-0.05, 1.6)
    for s in ax.spines.values():
        s.set_linewidth(0.9)
    ax.grid(True, axis="x", color="#c7c7c7", lw=0.5, alpha=0.8); ax.set_axisbelow(True)
    # legend
    from matplotlib.patches import Patch
    leg = [Patch(fc="#B23A48", ec="black", label="$<0.3$ saturated / broken"),
           Patch(fc="#E97132", ec="black", label="$0.3$--$0.8$ sub-linear"),
           Patch(fc="#196B24", ec="black", label="$\\approx 1$ linear (efficient)"),
           Patch(fc="#7A1FA0", ec="black", label="$>1.15$ super-linear collapse")]
    ax.legend(handles=leg, loc="lower right", frameon=True, framealpha=0.92,
              edgecolor="#888888", fontsize=7.0)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"atlas_scaling_exponent.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / "atlas_scaling_exponent.pdf")


if __name__ == "__main__":
    safe_envelope()
    scaling_exponents()
