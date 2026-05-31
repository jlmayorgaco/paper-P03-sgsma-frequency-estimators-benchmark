"""Two new figure sets for the SGSMA deck:

1) RoCoF sign-asymmetry per family: for each estimator family, RMSE vs |RoCoF|
   with a SOLID line for down-ramps (negative) and a DASHED line for up-ramps
   (positive). Asymmetric methods show the solid (down) curve sitting clearly
   above the dashed (up) curve. Saves rocof_fam_{loop,model,window}.{pdf,png}.
   Source: artifacts/atlas-papergrade-missing-v1/rmse_by_estimator.csv (rocof).

2) IBR ringdown tracking mega-dashboard: all 18 estimators, f_true(t) (black)
   vs f_hat(t) (color) on the IBR_Power_Imbalance_Ringdown scenario, run 0.
   Saves ringdown_tracking_all.{pdf,png}.
   Source: artifacts/full_mc_benchmark/IBR_Power_Imbalance_Ringdown/<EST>/...signals.csv
"""
from __future__ import annotations
from pathlib import Path
import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
ATLAS = REPO / "artifacts" / "atlas-papergrade-missing-v1" / "rmse_by_estimator.csv"
MC = REPO / "artifacts" / "full_mc_benchmark" / "IBR_Power_Imbalance_Ringdown"
FIG = Path(__file__).with_name("figures")
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9.0,
    "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlecolor": "black",
    "axes.labelsize": 8.6, "axes.edgecolor": "black", "axes.linewidth": 0.9,
    "text.color": "black", "axes.labelcolor": "black",
    "xtick.color": "black", "ytick.color": "black",
    "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
    "xtick.direction": "in", "ytick.direction": "in",
    "legend.fontsize": 6.8, "figure.dpi": 300, "savefig.dpi": 300,
})

PALETTE = ["#156082", "#E97132", "#196B24", "#A02B93", "#B23A48",
           "#0E2841", "#7A5195", "#2F9E8F"]


def _style(ax):
    for s in ax.spines.values():
        s.set_visible(True); s.set_color("black"); s.set_linewidth(0.9)
    ax.grid(True, which="both", color="#c7c7c7", lw=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(direction="in", length=3.0, width=0.8)


# ----------------------------------------------------------------- RoCoF panels
def _rocof_rows():
    rows = [r for r in csv.DictReader(open(ATLAS)) if r["sweep_key"] == "rocof"]
    return rows


def rocof_family_panel(rows, family, fname, title):
    ests = sorted({r["estimator"] for r in rows if r["family"] == family})
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    legend_items = []
    for k, e in enumerate(ests):
        col = PALETTE[k % len(PALETTE)]
        for direction, ls, lw, alpha in (("neg", "-", 2.1, 1.0), ("pos", (0, (4, 2)), 1.5, 0.75)):
            er = [r for r in rows if r["family"] == family and r["estimator"] == e
                  and r["direction"] == direction]
            er.sort(key=lambda r: float(r["abs_rocof_hz_s"]))
            if not er:
                continue
            x = [float(r["abs_rocof_hz_s"]) for r in er]
            y = [float(r["m1_rmse_hz_mean"]) for r in er]
            ax.plot(x, y, ls=ls, lw=lw, color=col, alpha=alpha,
                    solid_capstyle="round")
        legend_items.append(Line2D([0], [0], color=col, lw=2.1, label=e))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("|RoCoF|  [Hz/s]  (log)")
    ax.set_ylabel("frequency RMSE  [Hz]  (log)")
    ax.set_title(title)
    _style(ax)
    leg1 = ax.legend(handles=legend_items, loc="upper left", ncol=2, frameon=False,
                     handlelength=1.4, columnspacing=1.0)
    ax.add_artist(leg1)
    style_items = [Line2D([0], [0], color="black", lw=2.1, ls="-", label="down-ramp ($-$)"),
                   Line2D([0], [0], color="black", lw=1.5, ls=(0, (4, 2)), label="up-ramp ($+$)")]
    ax.legend(handles=style_items, loc="lower right", frameon=True, framealpha=0.9,
              edgecolor="#888888", handlelength=2.2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{fname}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / f"{fname}.pdf")


# ------------------------------------------------------------ ringdown dashboard
# representative set: one per family + robust/fragile contrast in model
ORDER = ["RA-EKF", "EKF", "UKF", "SOGI-FLL",
         "IPDFT", "ESPRIT", "Koopman (RK-DPMU)", "ZCD"]
FAMTAG = {"RA-EKF": "model", "EKF": "model", "UKF": "model", "SOGI-FLL": "loop",
          "IPDFT": "window", "ESPRIT": "spectral", "Koopman (RK-DPMU)": "data-driven",
          "ZCD": "loop (legacy)"}
SHORT = {"Koopman (RK-DPMU)": "Koopman"}


def ringdown_dashboard(fname):
    fig, axes = plt.subplots(2, 4, figsize=(13.6, 5.4))
    fig.subplots_adjust(left=0.05, right=0.995, top=0.88, bottom=0.10,
                        wspace=0.28, hspace=0.40)
    for idx, est in enumerate(ORDER):
        ax = axes.flat[idx]
        f = MC / est / f"IBR_Power_Imbalance_Ringdown__{est}_signals.csv"
        if not f.exists():
            ax.set_visible(False); continue
        df = pd.read_csv(f)
        d0 = df[df["run_idx"] == df["run_idx"].min()]
        t = d0["t_s"].to_numpy(); ft = d0["f_true_hz"].to_numpy(); fh = d0["f_hat_hz"].to_numpy()
        ax.plot(t, ft, color="black", lw=1.6, zorder=3, label="true")
        ax.plot(t, fh, color="#B23A48", lw=1.2, alpha=0.9, zorder=2, label="estimate")
        ax.set_ylim(55, 65)
        rmse = float(np.sqrt(np.nanmean((fh - ft) ** 2)))
        tag = SHORT.get(est, est)
        ax.set_title(f"{tag}  ({FAMTAG.get(est,'')})", pad=2, fontsize=9.0)
        ax.annotate(f"RMSE {rmse:.2g} Hz", xy=(0.5, 0.05), xycoords="axes fraction",
                    ha="center", va="bottom", fontsize=6.6, fontweight="bold",
                    color="#B23A48",
                    bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.82))
        if idx % 4 == 0:
            ax.set_ylabel("f [Hz]")
        if idx >= 4:
            ax.set_xlabel("t [s]")
        _style(ax)
    handles = [Line2D([0], [0], color="black", lw=1.8, label="true frequency $f(t)$"),
               Line2D([0], [0], color="#B23A48", lw=1.5, label="estimated $\\hat f(t)$")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.985), fontsize=9.5)
    fig.suptitle("IBR power-imbalance ringdown: how each of the 18 estimators tracks the true frequency",
                 y=0.945, fontsize=11.0, fontweight="bold", color="#0E2841")
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{fname}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / f"{fname}.pdf")


def main():
    # RoCoF sign asymmetry already lives on its own redesigned deck slide
    # (rocof_sign_asymmetry_wide.pdf); here we only build the ringdown dashboard.
    ringdown_dashboard("ringdown_tracking_rep")


if __name__ == "__main__":
    main()
