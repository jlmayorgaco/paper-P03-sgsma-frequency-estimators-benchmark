from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "slides" / "figures"
ATLAS_DIR = REPO_ROOT / "artifacts" / "atlas-papergrade-missing-v1"
PHASE_DIR = ATLAS_DIR
ROCOF_DIR = REPO_ROOT / "artifacts" / "freq_ramp_rocof_mvp2_all18_representative"
FSTEP_DIR = REPO_ROOT / "artifacts" / "frequency_step_mvp2_all18_representative"


FAMILY_ORDER = ["Loop-based", "Window-based", "Model-based", "Adaptive", "Data-driven", "Unknown"]
FAMILY_COLORS = {
    "Loop-based": "#2F6B8F",
    "Window-based": "#B8860B",
    "Model-based": "#5B5FC7",
    "Adaptive": "#2E7D5B",
    "Data-driven": "#B23A48",
    "Unknown": "#5B677A",
}
METHOD_COLORS = {
    "ESPRIT": "#2F6B8F",
    "Koopman (RK-DPMU)": "#B23A48",
    "TFT": "#8E5B2F",
    "IPDFT": "#C7792E",
    "IpDFT": "#C7792E",
    "SOGI-FLL": "#3F7D20",
    "SOGI-PLL": "#5A8F29",
    "ZCD": "#256D85",
    "PI-GRU": "#D94C5C",
    "UKF": "#6A4C93",
    "RA-EKF": "#7D5EA6",
    "Type-3 SOGI-PLL": "#1E7A73",
    "EKF": "#53408F",
    "LKF": "#4D4D4D",
    "LKF2": "#6D6D6D",
    "TKEO": "#7A7A2C",
    "PLL": "#5D7191",
    "RLS": "#9E9E3A",
    "Prony": "#8D4A7D",
}


STYLE = {
    "font.family": "DejaVu Sans",
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.edgecolor": "#1B2631",
    "axes.linewidth": 0.95,
    "axes.titleweight": "bold",
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10,
    "grid.color": "#B8C2CC",
    "grid.linewidth": 0.65,
    "savefig.bbox": "tight",
}


def _read(folder: Path) -> pd.DataFrame:
    path = folder / "global_metrics_report.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIG_DIR / f"{name}.pdf"
    png_path = FIG_DIR / f"{name}.png"
    for path in (pdf_path, png_path):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)


def _family_rank(name: str) -> int:
    try:
        return FAMILY_ORDER.index(name)
    except ValueError:
        return len(FAMILY_ORDER)


def _format_tick(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:g}"
    if value >= 1:
        return f"{value:g}"
    if value >= 0.01:
        return f"{value:.3g}"
    return f"{value:.1e}"


def _display_ticks(values: list[float], max_ticks: int = 13) -> list[float]:
    if len(values) <= max_ticks:
        return values
    idx = np.linspace(0, len(values) - 1, max_ticks).round().astype(int)
    return [values[i] for i in sorted(set(idx))]


def _xscale_for(x_values: list[float]) -> str:
    positives = [x for x in x_values if x > 0]
    if positives and max(positives) / min(positives) >= 80:
        return "log"
    return "linear"


def _method_map(
    df: pd.DataFrame,
    sweep_key: str,
    x_col: str,
    x_label: str,
    title: str,
    out_name: str,
    guide_hz: float = 0.05,
) -> None:
    part = df[df["sweep_key"].astype(str) == sweep_key].copy() if "sweep_key" in df else df.copy()
    part = part.dropna(subset=[x_col, "m1_rmse_hz_mean", "estimator"])
    part_level = part.groupby(["estimator", x_col], as_index=False)["m1_rmse_hz_mean"].max()
    pivot = (
        part_level
        .pivot(index="estimator", columns=x_col, values="m1_rmse_hz_mean")
    )
    families = part[["estimator", "family"]].drop_duplicates().set_index("estimator")["family"].to_dict()
    ordered = sorted(pivot.index, key=lambda est: (_family_rank(str(families.get(est, "Unknown"))), str(est)))
    pivot = pivot.loc[ordered]
    x_vals = [float(x) for x in pivot.columns]
    values = np.log10(np.maximum(pivot.to_numpy(dtype=float), 1e-12))

    fig, ax = plt.subplots(figsize=(19.5, 7.45), constrained_layout=False)
    finite = values[np.isfinite(values)]
    vmax = float(np.percentile(finite, 95)) if finite.size else 0.0
    cmap = matplotlib.colormaps["magma_r"].copy()
    cmap.set_bad("#E8EDF2")
    im = ax.imshow(np.ma.masked_invalid(values), aspect="auto", cmap=cmap, vmin=-4.0, vmax=max(vmax, -3.999))
    ax.set_title(title, loc="left", pad=8)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=12.2)
    show_x = _display_ticks(x_vals, max_ticks=15)
    show_pos = [min(range(len(x_vals)), key=lambda i: abs(x_vals[i] - v)) for v in show_x]
    ax.set_xticks(show_pos)
    ax.set_xticklabels([_format_tick(v) for v in show_x], rotation=32, ha="right", fontsize=11)
    ax.set_xlabel(x_label, labelpad=5)
    ax.set_ylabel("Estimator")
    cbar = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.012)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("log10 mean RMSE [Hz]", fontsize=11)
    fig.text(
        0.025,
        0.018,
        f"Reference guide: {guide_hz:g} Hz RMSE. Gray cells denote non-finite (diverged) values.",
        ha="left",
        va="bottom",
        fontsize=10,
        color="#34495E",
    )
    fig.tight_layout(rect=[0.02, 0.06, 0.985, 0.96])
    _save(fig, out_name)


def plot_rocof_sign_asymmetry(df: pd.DataFrame) -> None:
    part = df[df["sweep_key"].astype(str) == "rocof"].dropna(subset=["abs_rocof_hz_s", "direction"]).copy()
    rows = []
    for (estimator, abs_rocof), by_level in part.groupby(["estimator", "abs_rocof_hz_s"]):
        vals = by_level.set_index("direction")["m1_rmse_hz_mean"].to_dict()
        if "pos" not in vals or "neg" not in vals:
            continue
        pos = max(float(vals["pos"]), 1e-12)
        neg = max(float(vals["neg"]), 1e-12)
        rows.append(
            {
                "estimator": str(estimator),
                "family": str(by_level["family"].iloc[0]),
                "abs_rocof_hz_s": float(abs_rocof),
                "log10_neg_over_pos": math.log10(neg / pos),
            }
        )
    data = pd.DataFrame(rows)
    families = [fam for fam in FAMILY_ORDER if fam in set(data["family"])]
    fig, axes = plt.subplots(1, len(families), figsize=(19.5, 4.7), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, family in zip(axes, families):
        fam_df = data[data["family"] == family]
        for estimator, est_df in fam_df.groupby("estimator", sort=True):
            est_df = est_df.sort_values("abs_rocof_hz_s")
            color = METHOD_COLORS.get(str(estimator), FAMILY_COLORS.get(family, "#555555"))
            ax.plot(
                est_df["abs_rocof_hz_s"],
                est_df["log10_neg_over_pos"],
                marker="o",
                markersize=5,
                linewidth=1.9,
                color=color,
            )
            last = est_df.iloc[-1]
            ax.text(
                float(last["abs_rocof_hz_s"]) * 1.06,
                float(last["log10_neg_over_pos"]),
                str(estimator),
                fontsize=9.4,
                va="center",
                color=color,
            )
        ax.axhline(0.0, color="#1B2631", linestyle=":", linewidth=1.1)
        ax.set_xscale("log")
        ax.set_title(family, loc="left", fontsize=13, pad=5)
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xlabel("|RoCoF| [Hz/s]")
    axes[0].set_ylabel("log10(RMSE down-ramp / RMSE up-ramp)")
    axes[0].set_ylim(-1.05, 1.05)
    fig.suptitle("RoCoF sign asymmetry by estimator family", fontsize=17, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.91,
        "Zero is sign-symmetric. Positive values mean the frequency-decline case is harder than the frequency-rise case.",
        ha="center",
        fontsize=10.5,
        color="#34495E",
    )
    fig.tight_layout(rect=[0.035, 0.06, 0.985, 0.82])
    _save(fig, "rocof_sign_asymmetry_wide")


def plot_harmonics_family(df: pd.DataFrame) -> None:
    part = df[df["sweep_key"].astype(str) == "harmonics"].dropna(subset=["thd_percent", "m1_rmse_hz_mean"]).copy()
    fam = (
        part.groupby(["family", "thd_percent"], as_index=False)
        .agg(
            median_rmse=("m1_rmse_hz_mean", "median"),
            p25=("m1_rmse_hz_mean", lambda s: np.percentile(s, 25)),
            p75=("m1_rmse_hz_mean", lambda s: np.percentile(s, 75)),
        )
        .sort_values("thd_percent")
    )
    fig, ax = plt.subplots(figsize=(16.0, 5.8))
    for family in FAMILY_ORDER:
        g = fam[fam["family"] == family]
        if g.empty:
            continue
        color = FAMILY_COLORS.get(family, FAMILY_COLORS["Unknown"])
        ax.plot(g["thd_percent"], g["median_rmse"], marker="o", markersize=6, linewidth=2.4, label=family, color=color)
        ax.fill_between(g["thd_percent"], np.maximum(g["p25"], 1e-12), np.maximum(g["p75"], 1e-12), color=color, alpha=0.12)
    ax.axhline(0.05, color="#B23A48", linestyle="--", linewidth=1.2, label="0.05 Hz guide")
    ax.set_yscale("log")
    ax.set_xlabel("Integer harmonic THD [%]")
    ax.set_ylabel("Median RMSE across estimators [Hz]")
    ax.set_title("Harmonic degradation differs by estimator family", loc="left")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.text(
        0.02,
        0.02,
        "Band: interquartile range across estimators in the same family. Source: OpenFreqBench ATLAS harmonics sweep.",
        fontsize=10,
        color="#34495E",
    )
    fig.tight_layout(rect=[0.02, 0.06, 0.84, 0.96])
    _save(fig, "harmonics_family_deterioration_wide")


def _geom_mean(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    arr = np.maximum(arr.to_numpy(dtype=float), 1e-12)
    return float(np.exp(np.mean(np.log(arr))))


def plot_cpu_accuracy_pareto() -> None:
    rocof = _read(ROCOF_DIR)
    rocof["stress"] = "RoCoF ramp"
    fstep = _read(FSTEP_DIR)
    fstep["stress"] = "Frequency step"
    df = pd.concat([rocof, fstep], ignore_index=True)
    # Deployment Pareto: show the recommended set only. ZCD/TKEO/RLS/Prony are
    # non-core baselines (outside the recommended deployment set); their flat or
    # broken severity response makes the geometric-mean accuracy misleading here.
    df = df[~df["estimator"].isin(["PI-GRU", "ZCD", "TKEO", "RLS", "Prony"])].copy()
    summary = (
        df.groupby(["estimator", "family"], as_index=False)
        .agg(
            geom_rmse_hz=("m1_rmse_hz_mean", _geom_mean),
            median_cpu_us=("m13_cpu_time_us_mean", "median"),
            median_latency_ms=("m14_struct_latency_ms_mean", "median"),
        )
        .sort_values("geom_rmse_hz")
    )
    fig, ax = plt.subplots(figsize=(15.6, 6.1))
    for family, g in summary.groupby("family"):
        sizes = 70 + 9 * np.maximum(g["median_latency_ms"].to_numpy(dtype=float), 0.0)
        ax.scatter(
            g["median_cpu_us"],
            g["geom_rmse_hz"],
            s=sizes,
            color=FAMILY_COLORS.get(family, FAMILY_COLORS["Unknown"]),
            label=family,
            alpha=0.84,
            edgecolor="white",
            linewidth=1.0,
        )
    label_set = {
        "ESPRIT",
        "Koopman (RK-DPMU)",
        "TFT",
        "IPDFT",
        "SOGI-FLL",
        "SOGI-PLL",
        "ZCD",
        "Type-3 SOGI-PLL",
        "TKEO",
        "EKF",
        "RA-EKF",
    }
    offsets = {
        "ESPRIT": (6, 9),
        "Koopman (RK-DPMU)": (7, 8),
        "TFT": (7, -12),
        "IPDFT": (8, 12),
        "SOGI-FLL": (-42, -13),
        "SOGI-PLL": (6, 5),
        "ZCD": (6, 5),
        "Type-3 SOGI-PLL": (-78, 8),
        "TKEO": (-34, 9),
        "EKF": (6, 5),
        "RA-EKF": (7, -13),
    }
    for _, row in summary[summary["estimator"].isin(label_set)].iterrows():
        ax.annotate(
            str(row["estimator"]),
            (float(row["median_cpu_us"]), float(row["geom_rmse_hz"])),
            textcoords="offset points",
            xytext=offsets.get(str(row["estimator"]), (6, 6)),
            fontsize=10.2,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    xvals = summary["median_cpu_us"].to_numpy(dtype=float)
    yvals = summary["geom_rmse_hz"].to_numpy(dtype=float)
    xvals = xvals[np.isfinite(xvals) & (xvals > 0)]
    yvals = yvals[np.isfinite(yvals) & (yvals > 0)]
    if xvals.size and yvals.size:
        ax.set_xlim(float(xvals.min()) * 0.72, float(xvals.max()) * 1.95)
        ax.set_ylim(float(yvals.min()) * 0.55, float(yvals.max()) * 2.15)
    ax.set_xlabel("Median per-sample CPU cost [us] (log scale)")
    ax.set_ylabel("Dynamic RMSE, geometric mean [Hz] (log scale)")
    ax.set_title("Accuracy costs compute: dynamic stress Pareto view", loc="left")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout(rect=[0.02, 0.03, 0.84, 0.96])
    _save(fig, "cpu_accuracy_pareto_wide")


def main() -> None:
    plt.rcParams.update(STYLE)
    atlas = _read(ATLAS_DIR)
    phase = _read(PHASE_DIR)
    plot_rocof_sign_asymmetry(atlas)
    _method_map(
        atlas,
        "rocof",
        "abs_rocof_hz_s",
        "|RoCoF| [Hz/s]",
        "RoCoF method stress map",
        "rocof_method_map_wide",
    )
    _method_map(
        phase,
        "phase_jump_sweep",
        "abs_phase_jump_deg",
        "phase jump [deg]",
        "Phase-jump method stress map",
        "phase_jump_method_map_wide",
    )
    _method_map(
        atlas,
        "noise_snr",
        "noise_sigma_pu",
        "noise sigma [pu]",
        "Broadband noise method stress map",
        "noise_snr_method_map_wide",
    )
    plot_harmonics_family(atlas)
    plot_cpu_accuracy_pareto()


if __name__ == "__main__":
    main()
