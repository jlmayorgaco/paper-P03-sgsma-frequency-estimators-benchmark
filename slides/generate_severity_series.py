"""Generate paired ATLAS severity-response curves (RMSE vs severity, log-log).

Reads the paper-grade consolidated per-estimator table and produces three
two-panel figures matching the look of ``dynamic_sensitivity_curves``:

  severity_curves_step.pdf      -> magnitude step + frequency step
  severity_curves_dynamic.pdf   -> RoCoF ramp + phase jump
  severity_curves_spectral.pdf  -> harmonics + broadband noise

Source: artifacts/atlas-papergrade-missing-v1/rmse_by_estimator.csv (n=30).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "slides" / "figures"
DATA = REPO / "artifacts" / "atlas-papergrade-missing-v1" / "rmse_by_estimator.csv"

METHOD_COLORS = {
    "ESPRIT": "#2F6B8F",
    "Koopman (RK-DPMU)": "#B23A48",
    "TFT": "#8E5B2F",
    "IPDFT": "#C7792E",
    "SOGI-FLL": "#3F7D20",
    "SOGI-PLL": "#5A8F29",
    "ZCD": "#256D85",
    "UKF": "#6A4C93",
    "RA-EKF": "#7D5EA6",
    "Type-3 SOGI-PLL": "#1E7A73",
    "EKF": "#53408F",
    "TKEO": "#7A7A2C",
}

PAPER_STYLE = {
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.edgecolor": "black",
    "axes.linewidth": 0.9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.titleweight": "bold",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "grid.color": "#c7c7c7",
    "grid.linewidth": 0.55,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
}
plt.rcParams.update(PAPER_STYLE)

SELECTED = list(METHOD_COLORS.keys())

# sweep_key -> (severity column, x-axis label, panel title)
SWEEPS = {
    "magnitude_step": ("abs_step_percent", "Voltage step [%]", "Magnitude step"),
    "frequency_step": ("abs_step_hz", "Step magnitude [Hz]", "Frequency step"),
    "rocof": ("abs_rocof_hz_s", "Absolute RoCoF [Hz/s]", "RoCoF ramp"),
    "phase_jump_sweep": ("abs_phase_jump_deg", "Phase jump [deg]", "Phase jump"),
    "harmonics": ("thd_percent", "Integer harmonic THD [%]", "Harmonics"),
    "noise_snr": ("noise_sigma_pu", "Noise sigma [pu]", "Broadband noise"),
}

PAIRS = [
    ("severity_curves_step", ["magnitude_step", "frequency_step"]),
    ("severity_curves_dynamic", ["rocof", "phase_jump_sweep"]),
    ("severity_curves_spectral", ["harmonics", "noise_snr"]),
]

df = pd.read_csv(DATA)


def draw_panel(ax, sweep):
    xcol, xlabel, title = SWEEPS[sweep]
    sub = df[(df["sweep_key"] == sweep) & (df["estimator"].isin(SELECTED))].copy()
    sub[xcol] = pd.to_numeric(sub[xcol], errors="coerce").abs()
    sub["rmse"] = pd.to_numeric(sub["m1_rmse_hz_mean"], errors="coerce")
    sub = sub.dropna(subset=[xcol, "rmse"])
    sub = sub[(sub[xcol] > 0) & (sub["rmse"] > 0)]
    agg = (
        sub.groupby(["estimator", xcol], as_index=False)["rmse"].mean()
        .sort_values(xcol)
    )
    n = 0
    for est, g in agg.groupby("estimator", sort=False):
        if len(g) < 2:
            continue
        ax.plot(
            g[xcol], g["rmse"], marker="o", linewidth=1.7, markersize=3.5,
            color=METHOD_COLORS.get(est, "#555555"), label=est,
        )
        n += 1
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.22)
    return n


for name, sweeps in PAIRS:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, sw in zip(axes, sweeps):
        cnt = draw_panel(ax, sw)
        print(f"{name}: {sw} -> {cnt} estimator curves")
    axes[0].set_ylabel("Mean RMSE [Hz] (log scale)")
    # unified, de-duplicated legend across both panels
    seen = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    fig.legend(
        list(seen.values()), list(seen.keys()),
        loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7, frameon=False,
    )
    fig.suptitle("Severity response curves: RMSE grows with disturbance severity", y=1.02)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)

print("OK")
