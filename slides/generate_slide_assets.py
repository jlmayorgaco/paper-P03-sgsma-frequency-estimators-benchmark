from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SLIDE_DIR = REPO_ROOT / "slides"
FIG_DIR = SLIDE_DIR / "figures"

ROCOF_DIR = REPO_ROOT / "artifacts" / "freq_ramp_rocof_mvp2_all18_representative"
FSTEP_DIR = REPO_ROOT / "artifacts" / "frequency_step_mvp2_all18_representative"
HARM_DIR = REPO_ROOT / "artifacts" / "atlas-harmonics-fast14-preview-v2"
READINESS_DIR = REPO_ROOT / "artifacts" / "atlas-readiness-smoke"


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
    "SOGI-FLL": "#3F7D20",
    "SOGI-PLL": "#5A8F29",
    "ZCD": "#256D85",
    "PI-GRU": "#D94C5C",
    "UKF": "#6A4C93",
    "RA-EKF": "#7D5EA6",
    "Type-3 SOGI-PLL": "#1E7A73",
    "EKF": "#53408F",
    "LKF": "#4D4D4D",
    "TKEO": "#7A7A2C",
    "PLL": "#5D7191",
    "RLS": "#9E9E3A",
}


def _ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _read_csv(folder: Path, name: str) -> pd.DataFrame:
    path = folder / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _geom_mean(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    arr = np.maximum(arr.to_numpy(dtype=float), 1e-12)
    return float(np.exp(np.mean(np.log(arr))))


def _load_dynamic() -> pd.DataFrame:
    rocof = _read_csv(ROCOF_DIR, "global_metrics_report.csv")
    rocof["stress"] = "RoCoF ramp"
    rocof["x_value"] = rocof["abs_rocof_hz_s"]
    fstep = _read_csv(FSTEP_DIR, "global_metrics_report.csv")
    fstep["stress"] = "Frequency step"
    fstep["x_value"] = fstep["abs_step_hz"]
    return pd.concat([fstep, rocof], ignore_index=True)


def _dynamic_summary() -> pd.DataFrame:
    dyn = _load_dynamic()
    return (
        dyn.groupby(["estimator", "family"], as_index=False)
        .agg(
            geom_rmse_hz=("m1_rmse_hz_mean", _geom_mean),
            median_rmse_hz=("m1_rmse_hz_mean", "median"),
            worst_rmse_hz=("m1_rmse_hz_mean", "max"),
            geom_rfe_rms_hz_s=("m10_rfe_rms_hz_s_mean", _geom_mean),
            median_rfe_rms_hz_s=("m10_rfe_rms_hz_s_mean", "median"),
            median_cpu_us=("m13_cpu_time_us_mean", "median"),
            median_latency_ms=("m14_struct_latency_ms_mean", "median"),
            max_trip_risk_s=("m5_trip_risk_s_mean", "max"),
        )
        .sort_values("geom_rmse_hz")
        .reset_index(drop=True)
    )


def _savefig(name: str) -> None:
    for ext in ("pdf", "png"):
        plt.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=220)
    plt.close()


def _family_handles():
    import matplotlib.lines as mlines

    return [
        mlines.Line2D(
            [],
            [],
            color=color,
            marker="o",
            linestyle="None",
            markersize=7,
            label=family,
        )
        for family, color in FAMILY_COLORS.items()
        if family != "Unknown"
    ]


def plot_dynamic_rank() -> None:
    summary = _dynamic_summary().sort_values("geom_rmse_hz", ascending=True)
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    y = np.arange(len(summary))
    colors = [FAMILY_COLORS.get(f, FAMILY_COLORS["Unknown"]) for f in summary["family"]]
    ax.barh(y, summary["geom_rmse_hz"], color=colors, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(summary["estimator"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(summary["geom_rmse_hz"].min() / 1.4, summary["geom_rmse_hz"].max() * 3.0)
    ax.set_xlabel("Geometric mean RMSE across frequency-step and RoCoF sweeps [Hz]")
    ax.set_title("Dynamic stress ranking: accuracy is not the same as deployability")
    ax.grid(axis="x", which="both", alpha=0.25)
    for i, (_, row) in enumerate(summary.iterrows()):
        ax.text(
            row["geom_rmse_hz"] * 1.08,
            i,
            f"{row['geom_rmse_hz']:.3f} Hz | {row['median_cpu_us']:.0f} us",
            va="center",
            fontsize=7,
        )
    ax.legend(handles=_family_handles(), loc="upper right", fontsize=7, frameon=False)
    _savefig("dynamic_composite_rank")


def plot_dynamic_curves() -> None:
    dyn = _load_dynamic()
    selected = [
        "ESPRIT",
        "Koopman (RK-DPMU)",
        "IPDFT",
        "TFT",
        "SOGI-FLL",
        "SOGI-PLL",
        "Type-3 SOGI-PLL",
        "ZCD",
        "EKF",
        "TKEO",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    specs = [
        ("Frequency step", "Step magnitude [Hz]"),
        ("RoCoF ramp", "Absolute RoCoF [Hz/s]"),
    ]
    for ax, (stress, xlabel) in zip(axes, specs):
        sub = dyn[(dyn["stress"] == stress) & (dyn["estimator"].isin(selected))]
        agg = (
            sub.groupby(["estimator", "family", "x_value"], as_index=False)
            .agg(rmse=("m1_rmse_hz_mean", "mean"))
            .sort_values("x_value")
        )
        for est, g in agg.groupby("estimator", sort=False):
            color = METHOD_COLORS.get(est, "#555555")
            ax.plot(g["x_value"], g["rmse"], marker="o", linewidth=1.7, markersize=3.5, label=est, color=color)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_title(stress)
        ax.grid(True, which="both", alpha=0.22)
    axes[0].set_ylabel("Mean RMSE [Hz] (log scale)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7, frameon=False)
    fig.suptitle("Severity response curves reveal different failure modes", y=1.02)
    _savefig("dynamic_sensitivity_curves")


def plot_accuracy_rfe_tradeoff() -> None:
    summary = _dynamic_summary()
    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    for family, g in summary.groupby("family"):
        sizes = 35 + 35 * np.log10(np.maximum(g["median_cpu_us"], 1.0))
        ax.scatter(
            g["geom_rmse_hz"],
            g["geom_rfe_rms_hz_s"],
            s=sizes,
            color=FAMILY_COLORS.get(family, FAMILY_COLORS["Unknown"]),
            label=family,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.8,
        )
    labels = [
        "ESPRIT",
        "Koopman (RK-DPMU)",
        "TFT",
        "SOGI-FLL",
        "UKF",
        "EKF",
        "Type-3 SOGI-PLL",
        "ZCD",
        "PI-GRU",
        "LKF",
        "TKEO",
    ]
    offsets = {
        "ESPRIT": (-8, -14),
        "Koopman (RK-DPMU)": (4, 8),
        "TFT": (5, -8),
        "SOGI-FLL": (8, -15),
        "UKF": (8, -14),
        "EKF": (5, 6),
        "Type-3 SOGI-PLL": (5, 6),
        "ZCD": (5, 6),
        "PI-GRU": (8, 8),
        "LKF": (5, -10),
        "TKEO": (-45, 3),
    }
    for _, row in summary[summary["estimator"].isin(labels)].iterrows():
        dx, dy = offsets.get(row["estimator"], (5, 5))
        ax.annotate(
            row["estimator"],
            (row["geom_rmse_hz"], row["geom_rfe_rms_hz_s"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=7,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Dynamic frequency RMSE, geometric mean [Hz]")
    ax.set_ylabel("RoCoF error RMS, geometric mean [Hz/s]")
    ax.set_title("Frequency tracking and derivative tracking are separate objectives")
    ax.grid(True, which="both", alpha=0.24)
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    _savefig("accuracy_rfe_tradeoff")


def plot_cpu_accuracy_pareto() -> None:
    summary = _dynamic_summary()
    fig, ax = plt.subplots(figsize=(10.4, 5.2))
    for family, g in summary.groupby("family"):
        latency = np.maximum(g["median_latency_ms"].to_numpy(dtype=float), 0.0)
        sizes = 45 + 7 * latency
        ax.scatter(
            g["median_cpu_us"],
            g["geom_rmse_hz"],
            s=sizes,
            color=FAMILY_COLORS.get(family, FAMILY_COLORS["Unknown"]),
            label=family,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.8,
        )
    for _, row in summary.iterrows():
        if row["estimator"] in {
            "ESPRIT",
            "Koopman (RK-DPMU)",
            "TFT",
            "IPDFT",
            "SOGI-FLL",
            "SOGI-PLL",
            "ZCD",
            "PI-GRU",
            "Type-3 SOGI-PLL",
            "TKEO",
        }:
            ax.annotate(
                row["estimator"],
                (row["median_cpu_us"], row["geom_rmse_hz"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Median per-sample CPU cost [us] (log scale)")
    ax.set_ylabel("Dynamic RMSE, geometric mean [Hz] (log scale)")
    ax.set_title("Pareto view: PI-GRU is accurate but computationally expensive")
    ax.grid(True, which="both", alpha=0.24)
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    _savefig("cpu_accuracy_pareto")


def plot_harmonics_sensitivity() -> None:
    harm = _read_csv(HARM_DIR, "global_metrics_report.csv")
    selected = [
        "ZCD",
        "EKF",
        "UKF",
        "TFT",
        "RA-EKF",
        "SOGI-PLL",
        "SOGI-FLL",
        "IPDFT",
        "PLL",
        "RLS",
        "TKEO",
    ]
    fig, ax = plt.subplots(figsize=(10.4, 5.2))
    sub = harm[harm["estimator"].isin(selected)]
    agg = (
        sub.groupby(["estimator", "thd_percent"], as_index=False)
        .agg(rmse=("m1_rmse_hz_mean", "mean"))
        .sort_values("thd_percent")
    )
    for est, g in agg.groupby("estimator", sort=False):
        ax.plot(
            g["thd_percent"],
            g["rmse"],
            marker="o",
            linewidth=1.7,
            markersize=3.8,
            color=METHOD_COLORS.get(est, "#555555"),
            label=est,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Integer harmonic THD [%]")
    ax.set_ylabel("RMSE [Hz] (log scale)")
    ax.set_title("Integer-harmonic stress: low baseline error matters more than flatness")
    ax.grid(True, which="both", alpha=0.23)
    ax.legend(ncol=2, fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    _savefig("harmonics_sensitivity")


def plot_regime_summary() -> None:
    rocof = _read_csv(ROCOF_DIR, "rocof_hypothesis_tests.csv")
    rocof["stress"] = "RoCoF ramp"
    fstep = _read_csv(FSTEP_DIR, "frequency_step_hypothesis_tests.csv")
    fstep["stress"] = "Frequency step"
    data = pd.concat([fstep, rocof], ignore_index=True)
    regime_colors = {
        "Power-law-like": "#2F6B8F",
        "Monotone growth": "#8E5B2F",
        "Insensitive/flat": "#6A4C93",
        "Weak monotone/noise-floor-limited": "#3F7D20",
        "Sign-asymmetric": "#B23A48",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5), sharey=True)
    for ax, stress in zip(axes, ["Frequency step", "RoCoF ramp"]):
        sub = data[data["stress"] == stress]
        for regime, g in sub.groupby("primary_regime"):
            ax.scatter(
                g["power_slope_b"],
                g["max_sign_asymmetry_ratio"],
                color=regime_colors.get(regime, "#666666"),
                s=45,
                alpha=0.88,
                label=regime,
                edgecolor="white",
                linewidth=0.7,
            )
        for _, row in sub.iterrows():
            if row["estimator"] in {"LKF", "TKEO", "Type-3 SOGI-PLL", "EKF", "TFT", "ESPRIT"}:
                ax.annotate(
                    row["estimator"],
                    (row["power_slope_b"], row["max_sign_asymmetry_ratio"]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                )
        ax.axhline(2.0, color="#999999", linewidth=0.9, linestyle="--")
        ax.set_xlabel("Log-log severity slope")
        ax.set_title(stress)
        ax.grid(True, alpha=0.22)
    axes[0].set_ylabel("Max sign asymmetry ratio")
    handles, labels = axes[1].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7, frameon=False)
    fig.suptitle("Trend diagnostics: slope, flatness, and sign asymmetry", y=1.02)
    _savefig("regime_summary")


def plot_core_trip_risk_comparison() -> None:
    islanding = pd.DataFrame(
        [
            ("SRF-PLL", 0.0, "Loop-based"),
            ("RA-EKF", 0.0006, "Model-based"),
            ("EKF", 0.165, "Model-based"),
            ("UKF", 0.165, "Model-based"),
        ],
        columns=["estimator", "trip_s", "family"],
    )
    multievent = pd.DataFrame(
        [
            ("EKF", 0.118, "Model-based"),
            ("UKF", 0.119, "Model-based"),
            ("Koopman", 0.137, "Data-driven"),
            ("RA-EKF", 0.141, "Model-based"),
            ("IpDFT", 0.335, "Window-based"),
            ("SOGI-FLL", 0.422, "Loop-based"),
            ("SRF-PLL", 0.471, "Loop-based"),
        ],
        columns=["estimator", "trip_s", "family"],
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))
    for ax, df, title in [
        (axes[0], islanding, "Scenario D: composite islanding"),
        (axes[1], multievent, "Scenario E: IBR multi-event"),
    ]:
        df = df.sort_values("trip_s", ascending=True)
        colors = [FAMILY_COLORS.get(f, FAMILY_COLORS["Unknown"]) for f in df["family"]]
        bars = ax.barh(df["estimator"], df["trip_s"] * 1000, color=colors, alpha=0.9)
        ax.invert_yaxis()
        ax.set_xlabel("Trip-risk duration [ms]")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.24)
        xmax = max((df["trip_s"] * 1000).max() * 1.22, 1.0)
        ax.set_xlim(0, xmax)
        for bar, value_s in zip(bars, df["trip_s"]):
            value_ms = value_s * 1000
            label = "0" if value_ms == 0 else (f"{value_ms:.1f}" if value_ms < 10 else f"{value_ms:.0f}")
            ax.text(
                bar.get_width() + xmax * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{label} ms",
                va="center",
                fontsize=8,
            )
    axes[0].annotate(
        "EKF / RA-EKF = 275x",
        xy=(0.6, 1),
        xytext=(42, 1.7),
        arrowprops={"arrowstyle": "->", "color": "#B23A48", "lw": 1.2},
        color="#B23A48",
        fontsize=9,
        fontweight="bold",
    )
    fig.suptitle("Trip-risk is an event metric, not an average-error metric", y=1.02)
    _savefig("core_trip_risk_comparison")


def plot_metric_rank_shift() -> None:
    summary = _dynamic_summary().copy()
    summary["rmse_rank"] = summary["geom_rmse_hz"].rank(method="min", ascending=True)
    summary["rfe_rank"] = summary["geom_rfe_rms_hz_s"].rank(method="min", ascending=True)
    selected = [
        "ESPRIT",
        "Koopman (RK-DPMU)",
        "TFT",
        "IpDFT",
        "IPDFT",
        "SOGI-FLL",
        "SOGI-PLL",
        "UKF",
        "EKF",
        "RA-EKF",
        "PI-GRU",
    ]
    sub = summary[summary["estimator"].isin(selected)].copy()
    sub["display"] = sub["estimator"].replace({"IPDFT": "IpDFT"})
    sub = sub.sort_values("rmse_rank")
    fig, ax = plt.subplots(figsize=(10.6, 5.5))
    y = np.arange(len(sub))
    for i, (_, row) in enumerate(sub.iterrows()):
        color = FAMILY_COLORS.get(row["family"], FAMILY_COLORS["Unknown"])
        ax.plot([row["rmse_rank"], row["rfe_rank"]], [i, i], color="#CBD7DF", linewidth=2.5, zorder=1)
        ax.scatter(row["rmse_rank"], i, color=color, s=80, edgecolor="white", linewidth=0.8, zorder=2)
        ax.scatter(row["rfe_rank"], i, color=color, marker="s", s=78, edgecolor="white", linewidth=0.8, zorder=2)
        delta = int(row["rfe_rank"] - row["rmse_rank"])
        ax.text(
            max(row["rmse_rank"], row["rfe_rank"]) + 0.35,
            i,
            f"{delta:+d}",
            va="center",
            fontsize=8,
            color="#5B677A",
        )
    ax.set_yticks(y)
    ax.set_yticklabels(sub["display"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0.4, max(summary["rmse_rank"].max(), summary["rfe_rank"].max()) + 2.6)
    ax.set_xlabel("Rank among dynamic sweep estimators (lower is better)")
    ax.set_title("Metric choice changes the leaderboard: RMSE rank vs RoCoF-error rank")
    ax.grid(axis="x", alpha=0.22)
    ax.scatter([], [], color="#2F6B8F", marker="o", label="Frequency RMSE rank")
    ax.scatter([], [], color="#2F6B8F", marker="s", label="RoCoF-error rank")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    _savefig("metric_rank_shift")


def _tex_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def _format(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "--"
    value = float(value)
    if value == 0:
        return "0"
    if abs(value) < 0.01 or abs(value) >= 1000:
        return f"{value:.2e}"
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def generate_tables() -> None:
    summary = _dynamic_summary()
    rfe = summary.sort_values("geom_rfe_rms_hz_s")
    harm = _read_csv(HARM_DIR, "global_metrics_report.csv")
    hp = harm.pivot_table(
        index=["estimator", "family"],
        columns="thd_percent",
        values="m1_rmse_hz_mean",
        aggfunc="mean",
    )
    hp["rmse1"] = hp.get(1.0)
    hp["rmse30"] = hp.get(30.0)
    hp["ratio30_1"] = hp["rmse30"] / hp["rmse1"]
    hp = hp.reset_index().sort_values("rmse30")
    rocof_tests = _read_csv(ROCOF_DIR, "rocof_hypothesis_tests.csv").sort_values("max_sign_asymmetry_ratio", ascending=False)
    fstep_tests = _read_csv(FSTEP_DIR, "frequency_step_hypothesis_tests.csv").sort_values("max_sign_asymmetry_ratio", ascending=False)

    readiness = json.loads((READINESS_DIR / "atlas_readiness_report.json").read_text(encoding="utf-8"))
    ready_summary = readiness.get("summary", {})
    issues = readiness.get("issues", [])

    lines: list[str] = []
    lines.append("% Auto-generated by slides/generate_slide_assets.py")
    lines.append(r"\newcommand{\CurrentAtlasStatus}{" + _tex_escape(readiness.get("status", "unknown")) + "}")
    lines.append(r"\newcommand{\CurrentAtlasPolicy}{" + _tex_escape(readiness.get("settings", {}).get("policy", "unknown")) + "}")
    lines.append(r"\newcommand{\CurrentAtlasRuns}{" + _tex_escape(readiness.get("settings", {}).get("n_mc_runs", "unknown")) + "}")
    lines.append(r"\newcommand{\CurrentAtlasEstimators}{" + _tex_escape(len(ready_summary.get("estimators_present", []))) + "}")
    lines.append(r"\newcommand{\CanonicalEstimatorCount}{" + _tex_escape(len(ready_summary.get("canonical_estimators", []))) + "}")
    lines.append(r"\newcommand{\ReadinessIssueCount}{" + _tex_escape(len(issues)) + "}")

    def table_env(name: str, df: pd.DataFrame, columns: list[tuple[str, str, str]], rows: int) -> None:
        lines.append("")
        lines.append(rf"\newcommand{{\{name}}}{{%")
        col_spec = "".join("l" if kind == "text" else "r" for _, _, kind in columns)
        lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
        lines.append(r"\toprule")
        header = " & ".join(label for _, label, _ in columns) + r" \\"
        lines.append(header)
        lines.append(r"\midrule")
        for _, row in df.head(rows).iterrows():
            cells = []
            for col, _, kind in columns:
                value = row[col]
                if kind == "text":
                    if col == "estimator" and value == "IPDFT":
                        value = "IpDFT"
                    cells.append(_tex_escape(value))
                elif kind == "int":
                    cells.append(str(int(round(float(value)))))
                else:
                    cells.append(_format(float(value), 3))
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}%")
        lines.append("}")

    table_env(
        "DynamicTopTable",
        summary,
        [
            ("estimator", "Estimator", "text"),
            ("family", "Family", "text"),
            ("geom_rmse_hz", "Geo. RMSE", "float"),
            ("worst_rmse_hz", "Worst", "float"),
            ("median_cpu_us", "CPU us", "int"),
            ("median_latency_ms", "Latency ms", "float"),
        ],
        7,
    )
    table_env(
        "RfeTopTable",
        rfe,
        [
            ("estimator", "Estimator", "text"),
            ("family", "Family", "text"),
            ("geom_rfe_rms_hz_s", "Geo. RFE", "float"),
            ("geom_rmse_hz", "Geo. RMSE", "float"),
            ("median_cpu_us", "CPU us", "int"),
            ("median_latency_ms", "Latency ms", "float"),
        ],
        7,
    )
    table_env(
        "HarmonicsEndpointTable",
        hp,
        [
            ("estimator", "Estimator", "text"),
            ("family", "Family", "text"),
            ("rmse1", "1\\% THD", "float"),
            ("rmse30", "30\\% THD", "float"),
            ("ratio30_1", "Ratio 30/1", "float"),
        ],
        7,
    )

    # Custom compact diagnostic table.
    lines.append("")
    lines.append(r"\newcommand{\RegimeDiagnosticTable}{%")
    lines.append(r"\begin{tabular}{llrrr}")
    lines.append(r"\toprule")
    lines.append(r"Stress & Estimator & Slope & Ratio & Sign asym. \\")
    lines.append(r"\midrule")
    rows = [
        ("RoCoF", rocof_tests.iloc[0]),
        ("RoCoF", rocof_tests.iloc[1]),
        ("RoCoF", rocof_tests[rocof_tests["estimator"] == "Type-3 SOGI-PLL"].iloc[0]),
        ("Step", fstep_tests.iloc[0]),
        ("Step", fstep_tests[fstep_tests["estimator"] == "TKEO"].iloc[0]),
        ("Step", fstep_tests[fstep_tests["estimator"] == "EKF"].iloc[0]),
    ]
    for stress, row in rows:
        lines.append(
            " & ".join(
                [
                    _tex_escape(stress),
                    _tex_escape(row["estimator"]),
                    _format(row["power_slope_b"], 3),
                    _format(row["total_ratio_high_low"], 1),
                    _format(row["max_sign_asymmetry_ratio"], 2),
                ]
            )
            + r" \\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append("}")

    (SLIDE_DIR / "generated_results.tex").write_text("\n".join(lines) + "\n", encoding="ascii")


def write_summary_json() -> None:
    summary = _dynamic_summary()
    out = {
        "dynamic_top": summary.head(8).to_dict(orient="records"),
        "dynamic_rfe_top": summary.sort_values("geom_rfe_rms_hz_s").head(8).to_dict(orient="records"),
        "sources": {
            "rocof": str(ROCOF_DIR.relative_to(REPO_ROOT)),
            "frequency_step": str(FSTEP_DIR.relative_to(REPO_ROOT)),
            "harmonics": str(HARM_DIR.relative_to(REPO_ROOT)),
            "readiness": str(READINESS_DIR.relative_to(REPO_ROOT)),
        },
    }
    (SLIDE_DIR / "results_summary.json").write_text(json.dumps(out, indent=2), encoding="ascii")


def main() -> None:
    _ensure_dirs()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#FAFBFC",
            "axes.facecolor": "white",
            "axes.edgecolor": "#D9E1E8",
            "axes.labelcolor": "#1F2933",
            "axes.titlecolor": "#123047",
            "text.color": "#1F2933",
            "xtick.color": "#5B677A",
            "ytick.color": "#5B677A",
            "grid.color": "#D9E1E8",
            "grid.linewidth": 0.7,
            "axes.titlesize": 12,
            "figure.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
        }
    )
    plot_dynamic_rank()
    plot_dynamic_curves()
    plot_accuracy_rfe_tradeoff()
    plot_cpu_accuracy_pareto()
    plot_harmonics_sensitivity()
    plot_regime_summary()
    plot_core_trip_risk_comparison()
    plot_metric_rank_shift()
    generate_tables()
    write_summary_json()
    print(f"Wrote slide figures and generated tables to {SLIDE_DIR}")


if __name__ == "__main__":
    main()
