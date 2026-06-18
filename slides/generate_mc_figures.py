"""Data-driven deck figures straight from artifacts/full_mc_benchmark.
Produces (paper style):
  event_responses_mc  - estimator tracking f_hat(t) vs f_true, shows divergence
  cost_risk_mc        - CPU vs trip-risk scatter (multi-event)
  balance_radar_mc    - per-estimator normalized radar
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
MC = REPO / "artifacts" / "full_mc_benchmark"
FIG = Path(__file__).with_name("figures")

INK = "#0E2841"; BLUE = "#156082"; GOLD = "#E97132"; GREEN = "#196B24"
VIOLET = "#A02B93"; RED = "#B23A48"; MUTED = "#5B677A"; GREY = "#8A98A4"

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9.0,
    "axes.titlesize": 9.8, "axes.titleweight": "bold", "axes.titlecolor": "black",
    "axes.labelsize": 8.6, "axes.edgecolor": "black", "axes.linewidth": 0.9,
    "text.color": "black", "axes.labelcolor": "black",
    "xtick.color": "black", "ytick.color": "black",
    "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
    "xtick.direction": "in", "ytick.direction": "in",
    "legend.fontsize": 7.0, "legend.frameon": False, "figure.dpi": 300,
    "savefig.dpi": 300, "lines.antialiased": True,
})

MET = "m1_rmse_hz"


def sig(scn, est, run=None):
    df = pd.read_csv(MC / scn / est / f"{scn}__{est}_signals.csv")
    if run is not None:
        df = df[df["run_idx"] == run]
    return df["t_s"].to_numpy(), df["f_true_hz"].to_numpy(), df["f_hat_hz"].to_numpy()


def worst_run(scn, est):
    """run_idx with the largest tracking error (to expose divergence)."""
    df = pd.read_csv(MC / scn / est / f"{scn}__{est}_signals.csv")
    err = df.assign(e=(df["f_hat_hz"] - df["f_true_hz"]).abs()).groupby("run_idx")["e"].max()
    return int(err.idxmax())


def summ(scn, est, col):
    df = pd.read_csv(MC / scn / est / f"{scn}__{est}_summary.csv")
    return float(np.nanmean(df[col]))


def style(ax):
    for s in ax.spines.values():
        s.set_color("black"); s.set_linewidth(0.9)
    ax.grid(True, color="#c7c7c7", lw=0.55, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(direction="in", length=3.2, width=0.8)


def track(ax, scn, items, title, run=0, ylim=None, clip=None):
    tt, ft, _ = sig(scn, items[0][0], run)
    ax.plot(tt, ft, lw=2.0, color="black", ls=(0, (4, 2)), label="true", zorder=5)
    for est, lab, c in items:
        t, _, fh = sig(scn, est, run)
        if clip is not None:
            fh = np.clip(fh, *clip)
        ax.plot(t, fh, lw=1.4, color=c, label=lab, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("time [s]"); ax.set_ylabel("$\\hat f$ [Hz]")
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(loc="best", handlelength=1.4, ncol=2)
    style(ax)


def fig_event_responses():
    fig, ax = plt.subplots(2, 2, figsize=(11.6, 6.4))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.085, wspace=0.24, hspace=0.40)

    # (a) Ramp: EKF / IpDFT diverge, RA-EKF stable -> pick the worst EKF seed
    r = worst_run("IEEE_Freq_Ramp", "EKF")
    track(ax[0, 0], "IEEE_Freq_Ramp",
          [("EKF", "EKF", RED), ("IPDFT", "IpDFT", GOLD), ("RA-EKF", "RA-EKF", GREEN)],
          "(a) Frequency ramp --- EKF/IpDFT diverge", run=r, ylim=(35, 70), clip=(35, 70))
    ax[0, 0].annotate("EKF, IpDFT leave the\ntrack (tens of Hz)", xy=(0.97, 0.30), xycoords="axes fraction",
                      ha="right", va="top", fontsize=7, color=RED, fontweight="bold")

    # (b) Multi-event: tracking the stacked sequence
    track(ax[0, 1], "IBR_Multi_Event",
          [("RA-EKF", "RA-EKF", GREEN), ("EKF", "EKF", RED), ("UKF", "UKF", BLUE)],
          "(b) Multi-event --- all degrade to $\\approx$1 Hz", run=0, ylim=(52, 63))

    # (c) Phase jump: EKF/RA-EKF recover, IpDFT struggles
    track(ax[1, 0], "NERC_Phase_Jump_60",
          [("RA-EKF", "RA-EKF", GREEN), ("EKF", "EKF", BLUE), ("IPDFT", "IpDFT", GOLD)],
          "(c) Phase jump --- state-space recover, IpDFT trips", run=0, ylim=(57, 63))

    # (d) Modulation: LKF diverges
    r2 = worst_run("IEEE_Modulation", "LKF")
    track(ax[1, 1], "IEEE_Modulation",
          [("LKF", "LKF", RED), ("RA-EKF", "RA-EKF", GREEN), ("SOGI-FLL", "SOGI-FLL", BLUE)],
          "(d) Modulation --- LKF diverges", run=r2, ylim=(57, 63))

    fig.suptitle("Event responses (full\\_mc\\_benchmark): intermittent divergence is the dominant failure mode",
                 fontsize=11, fontweight="bold", color=INK, y=0.965)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"event_responses_mc.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote event_responses_mc")


FAM = {"EKF": "Model", "UKF": "Model", "RA-EKF": "Model", "LKF": "Model", "LKF2": "Model",
       "PLL": "Loop", "SOGI-PLL": "Loop", "SOGI-FLL": "Loop", "Type-3 SOGI-PLL": "Loop",
       "IPDFT": "Window", "TFT": "Window", "ESPRIT": "Window", "Prony": "Window", "MUSIC": "Window",
       "Koopman (RK-DPMU)": "Data-driven", "RLS": "Adaptive", "ZCD": "Legacy", "TKEO": "Legacy"}
FAMC = {"Model": VIOLET, "Loop": BLUE, "Window": GOLD, "Data-driven": RED, "Adaptive": GREEN, "Legacy": MUTED}


def fig_cost_risk(scn="IBR_Multi_Event"):
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    seen = set()
    for est, fam in FAM.items():
        d = MC / scn / est
        if not d.exists():
            continue
        cpu = summ(scn, est, "m13_cpu_time_us")
        trip = summ(scn, est, "m5_trip_risk_s")
        rmse = summ(scn, est, "m1_rmse_hz")
        c = FAMC[fam]
        ax.scatter(cpu, trip * 1000, s=40 + 220 * min(rmse, 3) / 3, color=c, alpha=0.8,
                   edgecolor="black", lw=0.5, label=fam if fam not in seen else None, zorder=3)
        seen.add(fam)
        lab = est.replace(" (RK-DPMU)", "")
        ax.annotate(lab, (cpu, trip * 1000), fontsize=6.4, xytext=(3, 3),
                    textcoords="offset points", color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("median CPU cost [$\\mu$s/sample] (log)")
    ax.set_ylabel("trip-risk $T_{\\mathrm{trip}}$ [ms]")
    ax.set_title("Cost vs trip-risk under multi-event stress (marker $\\propto$ RMSE)")
    ax.legend(loc="upper left", title="family", title_fontsize=7)
    style(ax)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"cost_risk_mc.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote cost_risk_mc")


def fig_radar():
    ests = [("RA-EKF", VIOLET), ("EKF", RED), ("PLL", BLUE), ("SOGI-FLL", GREEN),
            ("IPDFT", GOLD), ("Koopman (RK-DPMU)", "#7A5C2E")]
    scens = ["IEEE_Mag_Step", "IEEE_Freq_Ramp", "IEEE_Modulation", "NERC_Phase_Jump_60", "IBR_Multi_Event"]
    # axes: accuracy, ramp robustness, modulation, phase-jump, multi-event (all = 1/(1+RMSE))
    labels = ["Step\naccuracy", "Ramp\nrobustness", "Modulation", "Phase-jump", "Multi-event"]
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    ang += ang[:1]
    fig, ax = plt.subplots(figsize=(6.2, 6.0), subplot_kw=dict(polar=True))
    for est, c in ests:
        vals = []
        for scn in scens:
            try:
                rmse = summ(scn, est, "m1_rmse_hz")
            except Exception:
                rmse = np.nan
            vals.append(1.0 / (1.0 + (rmse if np.isfinite(rmse) else 1e3)))
        vals += vals[:1]
        ax.plot(ang, vals, lw=1.8, color=c, label=est.replace(" (RK-DPMU)", ""))
        ax.fill(ang, vals, color=c, alpha=0.06)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["", "0.5", "", "1.0"], fontsize=6.5)
    ax.set_ylim(0, 1.0)
    ax.set_title("Robustness profile --- $1/(1+\\mathrm{RMSE})$ per scenario\n(larger is better)",
                 fontsize=9.5, pad=16)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=7)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"balance_radar_mc.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote balance_radar_mc")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    fig_event_responses()
    fig_cost_risk()
    fig_radar()
