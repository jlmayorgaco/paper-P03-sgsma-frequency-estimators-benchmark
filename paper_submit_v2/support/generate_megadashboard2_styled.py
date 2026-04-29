from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipelines.benchmark_definition import ESTIMATOR_FAMILIES
from plotting.benchmark.mega_dashboard2_p20 import md2_subplot_20
from plotting.benchmark.mega_dashboard2_time_utils import (
    aggregate_aligned_curves,
    compute_data_limits,
)


DEFAULT_SOURCE_DIR = REPO_ROOT / "tests" / "montecarlo" / "outputs"
DEFAULT_OUTPUT_DIR = PAPER_ROOT / "outputs"
DEFAULT_FIGURE_DIR = PAPER_ROOT / "Figures" / "Plots_And_Graphs"
FIG_BASENAME = "Fig2_Mega_Dashboard"


IEEE_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8.0,
    "axes.labelsize": 7.2,
    "axes.titlesize": 8.1,
    "xtick.labelsize": 6.6,
    "ytick.labelsize": 6.6,
    "legend.fontsize": 6.0,
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.16,
    "grid.linewidth": 0.45,
    "axes.linewidth": 0.8,
}


TRACK_COLORS = {
    "RA-EKF": "#8C1D40",
    "EKF": "#C75B5B",
    "UKF": "#9B8BC3",
    "SOGI-PLL": "#3F8F5A",
    "SOGI-FLL": "#66A64B",
    "PLL": "#9C7C6B",
    "IPDFT": "#4A86C5",
    "TFT": "#7A68C3",
    "RLS": "#A66AD8",
    "Koopman (RK-DPMU)": "#D55E3A",
    "Prony": "#7A5C49",
    "ESPRIT": "#6C5BA7",
}


TRACK_STYLES = {
    "RA-EKF": dict(ls="-", lw=1.45),
    "EKF": dict(ls="--", lw=1.20),
    "UKF": dict(ls="--", lw=1.10),
    "SOGI-PLL": dict(ls="-.", lw=1.20),
    "SOGI-FLL": dict(ls="-.", lw=1.05),
    "PLL": dict(ls=":", lw=1.10),
    "IPDFT": dict(ls="-", lw=1.15),
    "TFT": dict(ls=":", lw=1.10),
    "RLS": dict(ls="--", lw=1.05),
    "Koopman (RK-DPMU)": dict(ls="-", lw=1.10),
    "Prony": dict(ls=":", lw=1.10),
    "ESPRIT": dict(ls="--", lw=1.05),
}


SHORT_LABELS = {
    "Koopman (RK-DPMU)": "Koopman",
    "SOGI-PLL": "SOGI-PLL",
    "SOGI-FLL": "SOGI-FLL",
    "IPDFT": "IpDFT",
    "RA-EKF": "RA-EKF",
    "EKF": "EKF",
    "UKF": "UKF",
    "PLL": "PLL",
    "TFT": "TFT",
    "RLS": "RLS",
    "Prony": "Prony",
}


FAMILY_PALETTE = {
    "Model-based": "#2A6DBB",
    "Loop-based": "#59A14F",
    "Window-based": "#F28E2B",
    "Adaptive": "#9C6BD6",
    "Data-driven": "#C53B35",
    "Unknown": "#7A8A94",
}


TRACKING_SPECS = (
    dict(
        row=0,
        col=0,
        scenario="IEEE_Phase_Jump_60",
        estimators=["RA-EKF", "EKF", "SOGI-PLL", "IPDFT", "UKF"],
        panel_title="(a) Phase Jump Robustness (Scen. D)",
        align_col="t_jump_s",
        align_value=0.0,
        t_window=(-0.02, 0.18),
        event_label="60 deg jump",
        extra_vlines_ms=(),
        legend_loc="upper right",
        legend_ncol=2,
        show_ylabel=True,
        min_y_span=1.0,
        y_limits=(59.4, 63.0),
    ),
    dict(
        row=0,
        col=1,
        scenario="IEEE_Freq_Ramp",
        estimators=["RA-EKF", "EKF", "SOGI-PLL", "IPDFT", "TFT"],
        panel_title="(b) Ramp Tracking Lag (Scen. B)",
        align_col="t_start_s",
        align_value=0.0,
        t_window=(-0.10, 0.70),
        event_label="Ramp onset",
        extra_vlines_ms=(),
        legend_loc="upper left",
        legend_ncol=2,
        show_ylabel=False,
        min_y_span=0.8,
        y_limits=(59.8, 61.7),
    ),
    dict(
        row=1,
        col=0,
        scenario="IBR_Multi_Event",
        estimators=["RA-EKF", "EKF", "SOGI-PLL", "IPDFT"],
        panel_title="(c) Multi-Event Recovery",
        align_col="t_event_s",
        align_value=0.5,
        t_window=(-0.08, 0.90),
        event_label="Composite onset",
        extra_vlines_ms=(500.0,),
        legend_loc="upper right",
        legend_ncol=2,
        show_ylabel=True,
        min_y_span=2.0,
        y_limits=(57.4, 60.6),
    ),
    dict(
        row=1,
        col=1,
        scenario="IBR_Power_Imbalance_Ringdown",
        estimators=["RA-EKF", "EKF", "SOGI-PLL", "IPDFT", "UKF"],
        panel_title="(d) Ringdown Recovery",
        align_col="t_event_s",
        align_value=1.0,
        t_window=(-0.20, 0.80),
        event_label="Ringdown onset",
        extra_vlines_ms=(),
        legend_loc="upper right",
        legend_ncol=2,
        show_ylabel=False,
        min_y_span=1.2,
        y_limits=(58.9, 60.9),
    ),
)

HARMONIC_SCENARIOS = (
    "IBR_Harmonics_Small",
    "IBR_Harmonics_Medium",
    "IBR_Harmonics_Large",
)

SCORECARD_ESTIMATORS = (
    "Koopman (RK-DPMU)",
    "TFT",
    "IPDFT",
    "SOGI-PLL",
    "SOGI-FLL",
    "PLL",
    "EKF",
    "UKF",
    "RA-EKF",
)

SCORECARD_GROUPS = (
    ("Step", lambda s: s.startswith("IEEE_Mag_Step")),
    ("Ramp", lambda s: s.startswith("IEEE_Freq_Ramp")),
    ("Mod", lambda s: s in {"IEEE_Modulation", "IEEE_Modulation_AM", "IEEE_Modulation_FM"}),
    ("Ring", lambda s: s.startswith("IBR_Power_Imbalance_Ringdown")),
    ("Multi", lambda s: s == "IBR_Multi_Event"),
)

PARETO_LABELS_ACC = (
    "SOGI-FLL",
    "SOGI-PLL",
    "EKF",
    "RA-EKF",
    "TFT",
    "IPDFT",
    "UKF",
    "Koopman (RK-DPMU)",
)

PARETO_LABELS_TRIP = (
    "SOGI-PLL",
    "PLL",
    "EKF",
    "RA-EKF",
    "TFT",
    "IPDFT",
    "UKF",
    "Koopman (RK-DPMU)",
)

PARETO_OFFSETS_ACC = {
    "SOGI-FLL": (2, -10),
    "SOGI-PLL": (4, -8),
    "EKF": (-4, -10),
    "RA-EKF": (5, 3),
    "TFT": (6, 2),
    "IPDFT": (5, -3),
    "UKF": (5, 3),
    "Koopman (RK-DPMU)": (5, 2),
}

PARETO_OFFSETS_TRIP = {
    "SOGI-PLL": (4, 2),
    "PLL": (5, 8),
    "EKF": (-6, -10),
    "RA-EKF": (5, 2),
    "TFT": (5, 2),
    "IPDFT": (5, 2),
    "UKF": (5, -8),
    "Koopman (RK-DPMU)": (4, 6),
}


def _load_dashboard_inputs(source_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    csv_path = source_dir / "global_metrics_report.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {csv_path}")
    df_global = pd.read_csv(csv_path)

    json_path = source_dir / "benchmark_full_report.json"
    df_raw = None
    if json_path.exists():
        with open(json_path, encoding="utf-8") as fh:
            payload = json.load(fh)
        raw = payload.get("raw_run_records", [])
        if raw:
            df_raw = pd.DataFrame(raw)
            if "scenario" not in df_raw.columns and "scenario_name" in df_raw.columns:
                df_raw = df_raw.rename(columns={"scenario_name": "scenario"})
    return df_global, df_raw


def _estimator_color(estimator: str) -> str:
    family = ESTIMATOR_FAMILIES.get(estimator, "Unknown")
    return TRACK_COLORS.get(estimator, FAMILY_PALETTE.get(family, "#7A8A94"))


def _short_label(estimator: str) -> str:
    return SHORT_LABELS.get(estimator, estimator.replace(" (RK-DPMU)", "").replace("Type-3 ", "T3-"))


def _find_scenario_meta(source_dir: Path, scenario: str) -> dict:
    scenario_dir = source_dir / scenario
    for run_spec in sorted(scenario_dir.glob("*/run_spec.json")):
        with open(run_spec, encoding="utf-8") as fh:
            payload = json.load(fh)
        meta = payload.get("scenario_meta")
        if meta:
            return meta
    return {}


def _extract_thd_pct(meta: dict) -> float | None:
    params = meta.get("parameters", {})
    thd_pct = params.get("thd_pct")
    if thd_pct is not None:
        return float(thd_pct)
    description = str(meta.get("description", ""))
    match = re.search(r"THD[^0-9]*([0-9]+(?:\.[0-9]+)?)", description)
    if match:
        return float(match.group(1))
    h5 = float(params.get("h5_pct", 0.0))
    h7 = float(params.get("h7_pct", 0.0))
    h11 = float(params.get("h11_pct", 0.0))
    if h5 or h7 or h11:
        return 100.0 * float(np.sqrt(h5**2 + h7**2 + h11**2))
    return None


def _harmonic_signature(meta: dict) -> tuple[str, str]:
    params = meta.get("parameters", {})
    mc_space = meta.get("monte_carlo_space", {})
    t_event = params.get("t_event_s")
    if "h11_pct" in params:
        event = f"step @ {t_event:.1f} s" if t_event is not None else "step event"
        return event, "5th + 7th + 11th"
    if "ih75_pct" in mc_space:
        event = f"ramp @ {t_event:.1f} s" if t_event is not None else "ramp event"
        return event, "5th + 75 Hz IH"
    if "f_sub_hz" in params:
        event = f"step @ {t_event:.1f} s" if t_event is not None else "step event"
        return event, f"5th + {params['f_sub_hz']:.0f} Hz sub + IH"
    return "event", "mixed harmonics"


def _pareto_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.ones(len(x), dtype=bool)
    for i in range(len(x)):
        for j in range(len(x)):
            if i == j:
                continue
            dominates = (x[j] <= x[i] and y[j] <= y[i]) and (x[j] < x[i] or y[j] < y[i])
            if dominates:
                mask[i] = False
                break
    return mask


def _plot_scorecard_panel(ax, df_global: pd.DataFrame) -> None:
    df = df_global[df_global["estimator"].isin(SCORECARD_ESTIMATORS)].copy()
    row_labels = [_short_label(est) for est in SCORECARD_ESTIMATORS]
    col_labels = [name for name, _ in SCORECARD_GROUPS]

    cell_values = np.zeros((len(SCORECARD_ESTIMATORS), len(SCORECARD_GROUPS)), dtype=float)
    cell_text: list[list[str]] = [["" for _ in SCORECARD_GROUPS] for _ in SCORECARD_ESTIMATORS]

    for j, (_, matcher) in enumerate(SCORECARD_GROUPS):
        sub = df[df["scenario"].map(matcher)].copy()
        summary = sub.groupby("estimator")["m1_rmse_hz_mean"].mean().reindex(SCORECARD_ESTIMATORS)
        rank = summary.rank(method="dense", ascending=True)
        for i, estimator in enumerate(SCORECARD_ESTIMATORS):
            val = summary.iloc[i]
            rk = rank.iloc[i]
            if pd.isna(val) or pd.isna(rk):
                cell_values[i, j] = 0.5
                cell_text[i][j] = "-"
                continue
            passed = rk <= 3
            best = rk == 1
            cell_values[i, j] = 1.0 if passed else 0.0
            cell_text[i][j] = "P*" if best else ("P" if passed else "F")

    cmap = ListedColormap(["#c94f49", "#f0efe8", "#48b04a"])
    ax.imshow(cell_values, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_facecolor("white")

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=6.1)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, pad=1.5, length=0)

    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=5.9)
    ax.tick_params(axis="y", length=0)

    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            text = cell_text[i][j]
            if text == "-":
                color = "#666666"
                weight = "normal"
            elif text.startswith("P"):
                color = "white"
                weight = "bold"
            else:
                color = "white"
                weight = "bold"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=6.0, fontweight=weight)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("black")

    ax.set_title("(e) Disturbance Scorecard", fontsize=8.0, fontweight="bold", loc="left", pad=6)
    ax.text(
        0.0,
        -0.12,
        "P = top-3 RMSE within family, P* = best",
        transform=ax.transAxes,
        fontsize=5.0,
        color="#555555",
        ha="left",
        va="top",
    )


def _plot_harmonics_bar_panel(ax, df_global: pd.DataFrame, source_dir: Path) -> None:
    df_h = df_global[df_global["scenario"].isin(HARMONIC_SCENARIOS)].copy()
    top_estimators = (
        df_h.groupby("estimator")["m1_rmse_hz_mean"].mean().sort_values().head(5).index.tolist()
    )

    stats = (
        df_h[df_h["estimator"].isin(top_estimators)][["scenario", "estimator", "m1_rmse_hz_mean"]]
        .copy()
        .assign(m1_rmse_mhz=lambda d: d["m1_rmse_hz_mean"] * 1000.0)
    )

    scenario_labels = []
    for scenario in HARMONIC_SCENARIOS:
        meta = _find_scenario_meta(source_dir, scenario)
        thd_pct = _extract_thd_pct(meta)
        event_text, harmonic_text = _harmonic_signature(meta)
        thd_text = f"{thd_pct:.1f}% THD" if thd_pct is not None else scenario.replace("IBR_Harmonics_", "")
        scenario_labels.append(f"{thd_text}\n{harmonic_text}\n{event_text}")

    centers = np.arange(len(HARMONIC_SCENARIOS), dtype=float)
    bar_w = 0.13
    offsets = (np.arange(len(top_estimators)) - (len(top_estimators) - 1) / 2.0) * bar_w

    ax.set_facecolor("white")
    for idx, estimator in enumerate(top_estimators):
        sub = (
            stats[stats["estimator"] == estimator]
            .set_index("scenario")
            .reindex(HARMONIC_SCENARIOS)
        )
        heights = sub["m1_rmse_mhz"].to_numpy(dtype=float)
        bars = ax.bar(
            centers + offsets[idx],
            heights,
            width=bar_w * 0.92,
            color=_estimator_color(estimator),
            alpha=0.86,
            edgecolor="white",
            linewidth=0.55,
            label=_short_label(estimator),
            zorder=3,
        )
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 3.0,
                f"{bar.get_height():.0f}",
                ha="center",
                va="bottom",
                fontsize=4.8,
                color="#4d4d4d",
                rotation=90,
            )

    best_by_scenario = (
        stats.loc[stats.groupby("scenario")["m1_rmse_mhz"].idxmin()][["scenario", "estimator", "m1_rmse_mhz"]]
    )
    for i, scenario in enumerate(HARMONIC_SCENARIOS):
        row = best_by_scenario[best_by_scenario["scenario"] == scenario].iloc[0]
        ax.text(
            centers[i],
            row["m1_rmse_mhz"] + 18.0,
            f"best: {_short_label(row['estimator'])}",
            ha="center",
            va="bottom",
            fontsize=5.0,
            color="#7a1f1f",
            fontstyle="italic",
        )

    ax.set_xticks(centers)
    ax.set_xticklabels(scenario_labels, fontsize=5.1, linespacing=1.05)
    ax.set_ylabel("RMSE [mHz]", fontsize=7.0, labelpad=1.5)
    ax.set_title("(f) Harmonic Robustness by THD", fontsize=8.0, fontweight="bold", loc="left", pad=3)
    ax.tick_params(axis="y", labelsize=6.3)
    ax.tick_params(axis="x", pad=1)
    ax.grid(True, axis="y", alpha=0.18, lw=0.45, zorder=0)
    ax.set_ylim(0.0, float(stats["m1_rmse_mhz"].max()) * 1.42)
    legend = ax.legend(
        loc="upper left",
        ncol=3,
        fontsize=5.5,
        frameon=True,
        framealpha=0.93,
        edgecolor="#d0d0d0",
        handlelength=1.2,
        handletextpad=0.35,
        columnspacing=0.8,
        borderpad=0.25,
    )
    legend.get_frame().set_linewidth(0.7)


def _plot_clean_pareto_panel(
    ax,
    *,
    df_global: pd.DataFrame,
    y_mean_col: str,
    y_scale: float,
    y_std_col: str | None,
    panel_title: str,
    ylabel: str,
    label_estimators: tuple[str, ...],
    offsets: dict[str, tuple[float, float]],
    guide_y: float | None,
    guide_y_label: str | None,
    quadrant_good: str,
    quadrant_bad: str,
    is_trip: bool = False,
) -> None:
    grp_dict = {
        "y_mean": (y_mean_col, "mean"),
        "cpu_mean": ("m13_cpu_time_us_mean", "mean"),
    }
    if y_std_col is not None:
        grp_dict["y_std"] = (y_std_col, "mean")
    grp = df_global.groupby("estimator").agg(**grp_dict).dropna()

    ests = grp.index.tolist()
    cpu = grp["cpu_mean"].to_numpy(dtype=float)
    y = grp["y_mean"].to_numpy(dtype=float) * y_scale
    if is_trip:
        y = np.where(y > 0.0, y, 5e-4)
    pareto = _pareto_mask(cpu, y)

    ax.set_facecolor("white")
    ax.set_xscale("log")
    ax.set_yscale("log")

    for i, estimator in enumerate(ests):
        clr = _estimator_color(estimator)
        highlighted = estimator in label_estimators
        marker = "o"
        if is_trip and grp.loc[estimator, "y_mean"] <= 0.0:
            marker = "v"
        ax.scatter(
            cpu[i],
            y[i],
            s=28 if highlighted else 15,
            color=clr,
            alpha=0.92 if highlighted else 0.30,
            edgecolors="black" if pareto[i] else "none",
            linewidths=0.65 if pareto[i] else 0.0,
            marker=marker,
            zorder=4 if highlighted else 2,
        )

    ax.axvline(100.0, color="#C62828", ls="--", lw=0.8, alpha=0.72, zorder=1)
    if guide_y is not None:
        ax.axhline(guide_y, color="#1565C0", ls="--", lw=0.8, alpha=0.75, zorder=1)
        ax.text(
            ax.get_xlim()[0],
            guide_y * 1.10,
            guide_y_label or "",
            fontsize=5.0,
            color="#1565C0",
            va="bottom",
            ha="left",
        )

    x_lo, x_hi = np.nanmin(cpu), np.nanmax(cpu)
    y_lo, y_hi = np.nanmin(y), np.nanmax(y)
    ax.text(x_lo * 1.2, y_lo * 1.35, quadrant_good, fontsize=5.2, color="#168A33", ha="left", va="bottom")
    ax.text(x_hi / 1.55, y_hi / 1.15, quadrant_bad, fontsize=5.2, color="#C62828", ha="left", va="top")
    ax.text(100.0 * 1.03, y_lo * 1.05, "RT limit", fontsize=4.9, color="#C62828", ha="left", va="bottom")

    for estimator in label_estimators:
        if estimator not in grp.index:
            continue
        x0 = float(grp.loc[estimator, "cpu_mean"])
        y0 = float(grp.loc[estimator, "y_mean"]) * y_scale
        if is_trip and float(grp.loc[estimator, "y_mean"]) <= 0.0:
            y0 = 5e-4
        dx, dy = offsets.get(estimator, (4, 4))
        ax.annotate(
            _short_label(estimator),
            (x0, y0),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=5.4,
            color=_estimator_color(estimator),
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy >= 0 else "top",
            fontweight="bold" if estimator in ("RA-EKF", "EKF", "IPDFT", "TFT", "SOGI-PLL") else "normal",
            zorder=5,
        )

    ax.tick_params(axis="both", which="both", labelsize=6.1)
    ax.set_xlabel(r"CPU time / sample [$\mu$s]", fontsize=7.0, labelpad=1.5)
    ax.set_ylabel(ylabel, fontsize=7.0, labelpad=1.5)
    ax.set_title(panel_title, fontsize=8.0, fontweight="bold", loc="left", pad=3)
    ax.grid(True, which="major", ls="-", alpha=0.12, zorder=0)
    ax.grid(False, which="minor")


def _plot_tracking_panel(
    ax,
    *,
    base_results_dir: Path,
    scenario: str,
    estimators: list[str],
    panel_title: str,
    align_col: str | None,
    align_value: float,
    t_window: tuple[float, float],
    event_label: str,
    extra_vlines_ms: tuple[float, ...],
    legend_loc: str,
    legend_ncol: int,
    show_ylabel: bool,
    min_y_span: float,
    y_limits: tuple[float, float] | None,
) -> None:
    t_common, f_true, curves = aggregate_aligned_curves(
        base_results_dir=base_results_dir,
        scenario=scenario,
        estimators=estimators,
        align_col=align_col,
        align_value=align_value,
        t_window=t_window,
        n_points=1000,
    )

    t_ms = t_common * 1000.0
    ax.set_facecolor("white")
    ax.plot(t_ms, f_true, color="black", lw=1.55, ls="--", label="Ref", zorder=6)

    plotted = [f_true]
    for estimator in estimators:
        curve = curves.get(estimator)
        if curve is None:
            continue
        plotted.append(curve)
        style = TRACK_STYLES.get(estimator, {"ls": "-", "lw": 1.15})
        ax.plot(
            t_ms,
            curve,
            color=TRACK_COLORS.get(estimator, "#6c757d"),
            alpha=0.98,
            label=SHORT_LABELS.get(estimator, estimator),
            zorder=4,
            **style,
        )

    if y_limits is None:
        y_lo, y_hi = compute_data_limits(*plotted, min_span=min_y_span, pad_frac=0.06)
    else:
        y_lo, y_hi = y_limits

    ax.axvline(0.0, color="#8c8c8c", lw=0.9, ls=":", alpha=0.95, zorder=2)
    for x_ms in extra_vlines_ms:
        ax.axvline(x_ms, color="#b0b0b0", lw=0.75, ls="--", alpha=0.90, zorder=2)

    ax.text(
        0.02,
        0.94,
        event_label,
        transform=ax.transAxes,
        fontsize=6.0,
        color="#b03a2e",
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.88),
    )

    ax.set_xlim(t_ms[0], t_ms[-1])
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Time [ms]", fontsize=7.0, labelpad=1.5)
    if show_ylabel:
        ax.set_ylabel("f [Hz]", fontsize=7.0, labelpad=1.5)
    ax.set_title(panel_title, fontsize=8.0, fontweight="bold", loc="left", pad=3)
    ax.tick_params(labelsize=6.4)
    ax.grid(True, alpha=0.18, lw=0.45)

    legend = ax.legend(
        loc=legend_loc,
        ncol=legend_ncol,
        fontsize=5.8,
        frameon=True,
        framealpha=0.92,
        edgecolor="#d0d0d0",
        handlelength=1.5,
        handletextpad=0.40,
        columnspacing=0.8,
        borderpad=0.30,
        labelspacing=0.22,
    )
    legend.get_frame().set_linewidth(0.7)


def _shrink_axis_text(ax, factor: float = 0.78) -> None:
    title = ax.title
    if title is not None and title.get_fontsize():
        title.set_fontsize(title.get_fontsize() * factor)

    ax.xaxis.label.set_fontsize(ax.xaxis.label.get_fontsize() * factor)
    ax.yaxis.label.set_fontsize(ax.yaxis.label.get_fontsize() * factor)

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        if tick.get_fontsize():
            tick.set_fontsize(tick.get_fontsize() * factor)

    for txt in ax.texts:
        if txt.get_fontsize():
            txt.set_fontsize(txt.get_fontsize() * factor)

    leg = ax.get_legend()
    if leg is not None:
        for txt in leg.get_texts():
            txt.set_fontsize(txt.get_fontsize() * factor)
        leg.get_title().set_fontsize(leg.get_title().get_fontsize() * factor if leg.get_title() else 0)


def _save_outputs(fig: plt.Figure, output_dir: Path, figure_dir: Path, publish_final: bool) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    out_png = output_dir / f"{FIG_BASENAME}.png"
    out_pdf = output_dir / f"{FIG_BASENAME}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.010)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.010)

    fig_png = figure_dir / out_png.name
    fig_pdf = figure_dir / out_pdf.name
    fig_png.write_bytes(out_png.read_bytes())
    fig_pdf.write_bytes(out_pdf.read_bytes())

    if publish_final:
        final_png = figure_dir / "Fig2_Mega_Dashboard.png"
        final_pdf = figure_dir / "Fig2_Mega_Dashboard.pdf"
        final_out_png = output_dir / "Fig2_Mega_Dashboard.png"
        final_out_pdf = output_dir / "Fig2_Mega_Dashboard.pdf"
        for src, dst in (
            (out_png, final_out_png),
            (out_pdf, final_out_pdf),
            (out_png, final_png),
            (out_pdf, final_pdf),
        ):
            dst.write_bytes(src.read_bytes())

    return out_png, out_pdf


def build_figure(
    source_dir: Path,
    output_dir: Path,
    figure_dir: Path,
    publish_final: bool,
) -> tuple[Path, Path]:
    df_global, df_raw = _load_dashboard_inputs(source_dir)
    avg_rmse = df_global.groupby("estimator")["m1_rmse_hz_mean"].mean()
    avg_cpu = df_global.groupby("estimator")["m13_cpu_time_us_mean"].mean()
    avg_trip = df_global.groupby("estimator")["m5_trip_risk_s_mean"].mean()
    avg_mae = df_global.groupby("estimator")["m2_mae_hz_mean"].mean()
    common_estimators = avg_rmse.index.intersection(avg_cpu.index)

    data_bundle = {
        "df_global": df_global,
        "df_raw": df_raw,
        "avg_rmse": avg_rmse,
        "avg_mae": avg_mae,
        "avg_cpu": avg_cpu,
        "avg_trip": avg_trip,
        "common_estimators": common_estimators,
        "_ESTIMATOR_FAMILIES": dict(ESTIMATOR_FAMILIES),
        "_FAMILY_PALETTE": FAMILY_PALETTE,
        "BASE_RESULTS_DIR": source_dir,
    }

    with plt.rc_context(IEEE_RC):
        fig = plt.figure(figsize=(7.16, 8.55), constrained_layout=False)
        gs = gridspec.GridSpec(
            4,
            2,
            figure=fig,
            left=0.075,
            right=0.985,
            top=0.985,
            bottom=0.050,
            hspace=0.55,
            wspace=0.24,
        )

        axes = {
            (0, 0): fig.add_subplot(gs[0, 0]),
            (0, 1): fig.add_subplot(gs[0, 1]),
            (1, 0): fig.add_subplot(gs[1, 0]),
            (1, 1): fig.add_subplot(gs[1, 1]),
            (2, 0): fig.add_subplot(gs[2, 0]),
            (2, 1): fig.add_subplot(gs[2, 1]),
            (3, 0): fig.add_subplot(gs[3, 0]),
            (3, 1): fig.add_subplot(gs[3, 1]),
        }

        for spec in TRACKING_SPECS:
            _plot_tracking_panel(
                axes[(spec["row"], spec["col"])],
                base_results_dir=source_dir,
                scenario=spec["scenario"],
                estimators=spec["estimators"],
                panel_title=spec["panel_title"],
                align_col=spec["align_col"],
                align_value=spec["align_value"],
                t_window=spec["t_window"],
                event_label=spec["event_label"],
                extra_vlines_ms=spec["extra_vlines_ms"],
                legend_loc=spec["legend_loc"],
                legend_ncol=spec["legend_ncol"],
                show_ylabel=spec["show_ylabel"],
                min_y_span=spec["min_y_span"],
                y_limits=spec["y_limits"],
            )

        _plot_scorecard_panel(axes[(2, 0)], df_global)
        md2_subplot_20(axes[(2, 1)], data_bundle)
        axes[(2, 1)].set_title("(f) MC RMSE by Family (N=60)", fontsize=8.0, fontweight="bold", loc="left", pad=3)
        _shrink_axis_text(axes[(2, 1)], factor=0.78)
        _plot_clean_pareto_panel(
            axes[(3, 0)],
            df_global=df_global,
            y_mean_col="m2_mae_hz_mean",
            y_scale=1000.0,
            y_std_col="m2_mae_hz_std",
            panel_title="(g) Cost vs. Accuracy",
            ylabel="Mean MAE [mHz]",
            label_estimators=PARETO_LABELS_ACC,
            offsets=PARETO_OFFSETS_ACC,
            guide_y=None,
            guide_y_label=None,
            quadrant_good="Fast & Accurate",
            quadrant_bad="Slow & Inaccurate",
        )
        _plot_clean_pareto_panel(
            axes[(3, 1)],
            df_global=df_global,
            y_mean_col="m5_trip_risk_s_mean",
            y_scale=1.0,
            y_std_col="m5_trip_risk_s_std",
            panel_title="(h) Cost vs. Trip Risk",
            ylabel="Trip-risk [s]",
            label_estimators=PARETO_LABELS_TRIP,
            offsets=PARETO_OFFSETS_TRIP,
            guide_y=None,
            guide_y_label=None,
            quadrant_good="Efficient & Safe",
            quadrant_bad="Costly & Risky",
            is_trip=True,
        )

        for ax in axes.values():
            ax.set_facecolor("white")

        out_png, out_pdf = _save_outputs(fig, output_dir, figure_dir, publish_final)
        plt.close(fig)

    return out_png, out_pdf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a styled MegaDashboard2 for paper_submit_v2 from tests/montecarlo/outputs."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--publish-final", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_png, out_pdf = build_figure(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        publish_final=args.publish_final,
    )
    print(f"[OK] MegaDashboard2 PNG -> {out_png}")
    print(f"[OK] MegaDashboard2 PDF -> {out_pdf}")
    if args.publish_final:
        print("[OK] Fig2_Mega_Dashboard.* updated in outputs and Figures/Plots_And_Graphs")


if __name__ == "__main__":
    main()
