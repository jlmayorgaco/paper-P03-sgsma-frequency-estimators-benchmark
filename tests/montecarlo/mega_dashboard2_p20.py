"""
Subplot (e): MC RMSE Distribution by Algorithmic Family — grouped boxplot.
Shows full N=30 distribution per estimator. LKF2 excluded per CLAUDE.md.
"""
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


_EXCLUDE = {"LKF2"}

_FAMILY_ORDER = ["Model-based", "Loop-based", "Window-based", "Adaptive", "Data-driven", "Unknown"]

_FAMILY_PALETTE = {
    "Model-based":  "#1565C0",
    "Loop-based":   "#2E7D32",
    "Window-based": "#E65100",
    "Adaptive":     "#6A1B9A",
    "Data-driven":  "#B71C1C",
    "Unknown":      "#546E7A",
}

_SHORT_EST = {
    "Koopman (RK-DPMU)": "Koopman",
    "Type-3 SOGI-PLL":   "T3-SOGI",
}


def md2_subplot_20(ax, data_bundle):
    df_raw = data_bundle.get("df_raw")
    if df_raw is None:
        ax.text(0.5, 0.5, "Raw MC data unavailable\n(df_raw missing)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=7, color="red")
        ax.set_title("(e) MC RMSE by Family (N=30)", fontweight="bold",
                     fontsize=8, loc="left", pad=3)
        return

    df = df_raw[~df_raw["estimator"].isin(_EXCLUDE)].copy()
    df["est_short"] = df["estimator"].map(lambda x: _SHORT_EST.get(x, x))

    families_present = [f for f in _FAMILY_ORDER if f in df["family"].values]
    ordered_ests = []
    family_boundaries = []
    tick_colors = []

    for fam in families_present:
        sub = df[df["family"] == fam]
        ests_in_fam = (
            sub.groupby("est_short")["m1_rmse_hz"]
            .median()
            .sort_values()
            .index.tolist()
        )
        family_boundaries.append((len(ordered_ests), len(ordered_ests) + len(ests_in_fam), fam))
        for e in ests_in_fam:
            ordered_ests.append(e)
            tick_colors.append(_FAMILY_PALETTE[fam])

    rmse_data = [
        df[df["est_short"] == e]["m1_rmse_hz"].dropna().values
        for e in ordered_ests
    ]
    positions = np.arange(1, len(ordered_ests) + 1)

    bp = ax.boxplot(
        rmse_data, positions=positions, vert=True,
        patch_artist=True, widths=0.58, showfliers=True,
        flierprops=dict(marker="o", markersize=1.8, linestyle="none",
                        markeredgewidth=0.4, alpha=0.45),
        medianprops=dict(color="white", linewidth=1.3),
        whiskerprops=dict(linewidth=0.65),
        capprops=dict(linewidth=0.65),
        boxprops=dict(linewidth=0.65),
    )

    for patch, est_short in zip(bp["boxes"], ordered_ests):
        fam = df[df["est_short"] == est_short]["family"].iloc[0]
        patch.set_facecolor(_FAMILY_PALETTE.get(fam, "#546E7A"))
        patch.set_alpha(0.75)

    # Family background bands + header label just above axes top
    for start, end, fam in family_boundaries:
        clr = _FAMILY_PALETTE.get(fam, "#eee")
        ax.axvspan(start + 0.5, end + 0.5, color=clr, alpha=0.06, zorder=0)
        mid = (start + end) / 2.0 + 1.0
        # y=1.02 is in AXES coordinates (0–1 range) — correct for get_xaxis_transform
        ax.text(mid, 1.02,
                fam.replace("-", "\n"), ha="center", va="bottom",
                fontsize=4.5, color=clr, fontweight="bold",
                clip_on=False, transform=ax.get_xaxis_transform())

    ax.set_xticks(positions)
    ax.set_xticklabels(ordered_ests, rotation=45, ha="right", fontsize=5.5)
    for tick, clr in zip(ax.get_xticklabels(), tick_colors):
        tick.set_color(clr)

    ax.set_yscale("log")
    ax.set_ylim(bottom=5e-4)
    ax.tick_params(axis="y", which="both", labelsize=6)
    ax.set_xlim(0.5, len(ordered_ests) + 0.5)

    ax.set_xlabel("Estimator", fontsize=7, labelpad=2)
    ax.set_ylabel("RMSE [Hz]", fontsize=7, labelpad=2)
    ax.set_title("(e) MC RMSE by Family (N=30)", fontweight="bold",
                 fontsize=8, loc="left", pad=3)

    ax.grid(True, axis="y", which="major", ls="-", alpha=0.20, zorder=0)
    ax.grid(True, axis="y", which="minor", ls=":", alpha=0.10, zorder=0)

    handles = [Patch(facecolor=_FAMILY_PALETTE[f], alpha=0.75, label=f)
               for f in families_present]
    ax.legend(handles=handles, loc="upper left", fontsize=5.0, frameon=True,
              framealpha=0.92, edgecolor="#ccc", handlelength=0.9,
              borderpad=0.4, labelspacing=0.22, ncol=2)
