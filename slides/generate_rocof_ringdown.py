"""Ringdown tracking figures for the SGSMA deck.

Produces:
  - ringdown_tracking_rep.{pdf,png}: 8 representative estimators.
  - ringdown_tracking_all18.{pdf,png}: all available ringdown estimators in a
    3x6 family-ordered audit grid.

Source:
  artifacts/full_mc_benchmark/IBR_Power_Imbalance_Ringdown/<EST>/...signals.csv
  artifacts/full_mc_benchmark/IBR_Power_Imbalance_Ringdown/<EST>/...summary.csv
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
# Representative set: one per family + robust/fragile contrast in model.
REP_ORDER = ["RA-EKF", "EKF", "UKF", "SOGI-FLL",
             "IPDFT", "ESPRIT", "Koopman (RK-DPMU)", "ZCD"]

ALL_ORDER = [
    "PLL", "SOGI-PLL", "SOGI-FLL", "Type-3 SOGI-PLL", "ZCD", "IPDFT",
    "TFT", "ESPRIT", "Prony", "MUSIC", "EKF", "UKF",
    "RA-EKF", "LKF", "LKF2", "RLS", "TKEO", "Koopman (RK-DPMU)",
]

FAMTAG = {
    "PLL": "loop", "SOGI-PLL": "loop", "SOGI-FLL": "loop",
    "Type-3 SOGI-PLL": "loop", "ZCD": "loop",
    "IPDFT": "window", "TFT": "window", "ESPRIT": "spectral",
    "Prony": "window", "MUSIC": "spectral",
    "EKF": "model", "UKF": "model", "RA-EKF": "model", "LKF": "model",
    "LKF2": "model", "RLS": "adaptive", "TKEO": "adaptive",
    "Koopman (RK-DPMU)": "data-driven",
}
FAMILY_COLORS = {
    "loop": "#156082",
    "window": "#E97132",
    "spectral": "#E97132",
    "model": "#6A4C93",
    "adaptive": "#196B24",
    "data-driven": "#B23A48",
}
SHORT = {"Koopman (RK-DPMU)": "Koopman", "Type-3 SOGI-PLL": "T3-SOGI"}
EVENTS = [(0.5, "fault"), (1.0, "ringdown")]
FAIL_COLOR = "#B23A48"


def _signals_path(est: str) -> Path:
    return MC / est / f"IBR_Power_Imbalance_Ringdown__{est}_signals.csv"


def _summary_path(est: str) -> Path:
    return MC / est / f"IBR_Power_Imbalance_Ringdown__{est}_summary.csv"


def _first_run(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["run_idx"] == df["run_idx"].min()]


def _run_summary(est: str, run_idx: int) -> dict[str, float]:
    path = _summary_path(est)
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "run_idx" in df:
        df = df[df["run_idx"] == run_idx]
    if df.empty:
        return {}
    row = df.iloc[0]
    out = {}
    for key in ("m1_rmse_hz", "m5_trip_risk_s", "m22_invalid_output_rate",
                "m31_freq_bound_hit_rate"):
        try:
            out[key] = float(row.get(key, np.nan))
        except (TypeError, ValueError):
            out[key] = float("nan")
    return out


def _series(est: str):
    path = _signals_path(est)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    d0 = _first_run(df)
    run_idx = int(d0["run_idx"].iloc[0]) if "run_idx" in d0 else 0
    t = d0["t_s"].to_numpy(float)
    ft = d0["f_true_hz"].to_numpy(float)
    fh = d0["f_hat_hz"].to_numpy(float)
    finite = np.isfinite(fh)
    rmse_trace = float(np.sqrt(np.nanmean((fh - ft) ** 2))) if finite.any() else float("nan")
    lo = float(np.nanmin(fh)) if finite.any() else float("nan")
    hi = float(np.nanmax(fh)) if finite.any() else float("nan")
    summary = _run_summary(est, run_idx)
    rmse = summary.get("m1_rmse_hz", rmse_trace)
    invalid = summary.get("m22_invalid_output_rate", 1.0 - finite.mean())
    bound = summary.get("m31_freq_bound_hit_rate", 0.0)
    bad = (
        (not np.isfinite(rmse))
        or rmse > 2.0
        or invalid > 0.02
        or bound > 0.02
        or lo < 55.0
        or hi > 65.0
    )
    return {
        "t": t, "f_true": ft, "f_hat": fh, "run_idx": run_idx,
        "rmse": rmse, "rmse_trace": rmse_trace, "invalid": invalid,
        "bound": bound, "lo": lo, "hi": hi, "bad": bad,
    }


def _event_lines(ax, annotate: bool = False):
    for x, label in EVENTS:
        ax.axvline(x, color="#5B677A", lw=0.8, ls=(0, (3, 2)), alpha=0.85, zorder=1)
        if annotate:
            ax.text(x + 0.015, 64.25, label, fontsize=6.2, color="#5B677A",
                    ha="left", va="top", rotation=90)


def ringdown_dashboard(fname):
    fig, axes = plt.subplots(2, 4, figsize=(13.6, 5.4))
    fig.subplots_adjust(left=0.05, right=0.995, top=0.82, bottom=0.10,
                        wspace=0.28, hspace=0.40)
    for idx, est in enumerate(REP_ORDER):
        ax = axes.flat[idx]
        item = _series(est)
        if item is None:
            ax.set_visible(False)
            continue
        t = item["t"]; ft = item["f_true"]; fh = item["f_hat"]
        family = FAMTAG.get(est, "")
        color = FAMILY_COLORS.get(family, "#B23A48")
        ax.plot(t, ft, color="black", lw=1.6, zorder=3, label="true")
        ax.plot(t, fh, color=color, lw=1.2, alpha=0.95, zorder=2, label="estimate")
        _event_lines(ax, annotate=(idx == 0))
        ax.set_ylim(55, 65)
        tag = SHORT.get(est, est)
        ax.set_title(f"{tag}  ({family})", pad=2, fontsize=9.0,
                     color=(FAIL_COLOR if item["bad"] else "black"))
        ax.annotate(f"RMSE {item['rmse']:.2g} Hz", xy=(0.5, 0.05), xycoords="axes fraction",
                    ha="center", va="bottom", fontsize=6.6, fontweight="bold",
                    color=(FAIL_COLOR if item["bad"] else color),
                    bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.82))
        if idx % 4 == 0:
            ax.set_ylabel("f [Hz]")
        if idx >= 4:
            ax.set_xlabel("t [s]")
        _style(ax)
    handles = [Line2D([0], [0], color="black", lw=1.8, label="true frequency $f(t)$"),
               Line2D([0], [0], color="#6A4C93", lw=1.5, label="estimated $\\hat f(t)$"),
               Line2D([0], [0], color="#5B677A", lw=0.9, ls=(0, (3, 2)),
                      label="event markers")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.995), fontsize=9.0)
    fig.suptitle("IBR power-imbalance ringdown: representative estimator tracking",
                 y=0.925, fontsize=11.0, fontweight="bold", color="#0E2841")
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{fname}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / f"{fname}.pdf")


def ringdown_all18(fname):
    panels = [(est, _series(est)) for est in ALL_ORDER]
    panels = [(est, item) for est, item in panels if item is not None]
    if len(panels) != 18:
        print(f"[warn] expected 18 ringdown signal files, found {len(panels)}")

    fig, axes = plt.subplots(3, 6, figsize=(15.6, 7.4))
    fig.subplots_adjust(left=0.045, right=0.995, top=0.88, bottom=0.075,
                        wspace=0.26, hspace=0.40)
    for idx, (est, item) in enumerate(panels):
        ax = axes.flat[idx]
        family = FAMTAG.get(est, "")
        color = FAMILY_COLORS.get(family, "#5B677A")
        ax.plot(item["t"], item["f_true"], color="black", lw=1.15, zorder=3)
        ax.plot(item["t"], item["f_hat"], color=color, lw=0.95, alpha=0.96, zorder=2)
        _event_lines(ax, annotate=(idx == 0))
        ax.set_ylim(55, 65)
        ax.set_title(SHORT.get(est, est), fontsize=7.8,
                     color=(FAIL_COLOR if item["bad"] else "black"), pad=1.5)
        status = "FLAG" if item["bad"] else "OK"
        label_color = FAIL_COLOR if item["bad"] else color
        ax.annotate(f"{status}  RMSE {item['rmse']:.2g}",
                    xy=(0.50, 0.045), xycoords="axes fraction",
                    ha="center", va="bottom", fontsize=5.8, color=label_color,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.82))
        if idx % 6 == 0:
            ax.set_ylabel("f [Hz]", fontsize=7.0)
        if idx >= 12:
            ax.set_xlabel("t [s]", fontsize=7.0)
        _style(ax)
        ax.tick_params(labelsize=5.9)

    for j in range(len(panels), 18):
        axes.flat[j].set_visible(False)

    handles = [
        Line2D([0], [0], color="black", lw=1.4, label="true frequency $f(t)$"),
        Line2D([0], [0], color="#156082", lw=1.2, label="estimate $\\hat f(t)$"),
        Line2D([0], [0], color=FAIL_COLOR, lw=1.2, label="title/label red = sanity flag"),
        Line2D([0], [0], color="#5B677A", lw=0.9, ls=(0, (3, 2)), label="event markers"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.985), fontsize=8.4)
    fig.suptitle(
        "IBR power-imbalance ringdown: all available estimators, grouped by family",
        y=0.945, fontsize=11.2, fontweight="bold", color="#0E2841",
    )
    fig.text(
        0.045, 0.022,
        "Source: artifacts/full_mc_benchmark/IBR_Power_Imbalance_Ringdown. "
        "Flag if invalid-rate >2%, bound-hit >2%, RMSE >2 Hz, or trace leaves 55-65 Hz.",
        fontsize=7.2, color="#5B677A", ha="left",
    )
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{fname}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / f"{fname}.pdf")


def main():
    # RoCoF sign asymmetry already lives on its own redesigned deck slide
    # (rocof_sign_asymmetry_wide.pdf); here we build the ringdown dashboards.
    ringdown_dashboard("ringdown_tracking_rep")
    ringdown_all18("ringdown_tracking_all18")


if __name__ == "__main__":
    main()
