from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_SPECS: list[dict[str, Any]] = [
    {"col": "m1_rmse_hz_mean", "label": "RMSE [Hz]", "slug": "rmse"},
    {"col": "m3_max_peak_hz_mean", "label": "Peak Error [Hz]", "slug": "peak_error"},
    {"col": "m5_trip_risk_s_mean", "label": "Trip-Risk [s]", "slug": "trip_risk"},
    {"col": "m8_settling_time_s_mean", "label": "Settling Time [s]", "slug": "settling_time"},
]

HEATMAP_METRICS: list[dict[str, Any]] = [
    {"col": "m1_rmse_hz_mean", "label": "RMSE [Hz]", "slug": "rmse"},
    {"col": "m5_trip_risk_s_mean", "label": "Trip-Risk [s]", "slug": "trip_risk"},
]


def _load_manifest_rows(manifest: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family_idx, family in enumerate(manifest.get("families", [])):
        scenarios = family.get("scenarios", [])
        sweep_values = [float(scenario["sweep_value"]) for scenario in scenarios]
        x_min = min(sweep_values) if sweep_values else math.nan
        x_max = max(sweep_values) if sweep_values else math.nan
        anchor_name = str(family.get("anchor_scenario", ""))
        for scenario_idx, scenario in enumerate(scenarios):
            rows.append(
                {
                    "scenario": scenario["scenario_name"],
                    "family_order": family_idx,
                    "scenario_order": scenario_idx,
                    "sweep_family_key": family["key"],
                    "sweep_family_display": family["display_name"],
                    "sweep_label": family["sweep_label"],
                    "sweep_unit": family["unit"],
                    "sweep_xscale": family.get("xscale", "linear"),
                    "sweep_value": float(scenario["sweep_value"]),
                    "sweep_value_label": str(scenario.get("sweep_value_label", scenario["sweep_value"])),
                    "anchor_scenario": bool(scenario.get("is_anchor", False)),
                    "family_anchor_scenario": anchor_name,
                    "family_description": family.get("description", ""),
                    "family_x_min": x_min,
                    "family_x_max": x_max,
                }
            )
    return pd.DataFrame(rows)


def _family_slug(text: str) -> str:
    return text.lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def _estimator_colors(labels: list[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = matplotlib.colormaps["tab20"]
    return {label: cmap(idx % cmap.N) for idx, label in enumerate(labels)}


def _merge_report(output_dir: Path, manifest_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report_path = output_dir / "global_metrics_report.csv"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing global metrics report: {report_path}")

    df_report = pd.read_csv(report_path)
    df_manifest = _load_manifest_rows(manifest)
    if df_report.empty or df_manifest.empty:
        raise ValueError("The ramp/mag sweep report needs both manifest rows and metric rows.")

    df = df_report.merge(df_manifest, on="scenario", how="inner")
    if df.empty:
        raise ValueError("No overlap between sweep manifest and global_metrics_report.csv.")
    df["sweep_value"] = pd.to_numeric(df["sweep_value"], errors="coerce")
    return df, df_manifest, manifest


def _positive_for_log(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(arr)
    positive = arr[finite & (arr > 0.0)]
    if positive.size == 0:
        arr[finite] = 1e-9
        return arr
    floor = max(float(np.min(positive)) * 0.5, 1e-9)
    arr[finite & (arr <= 0.0)] = floor
    return arr


def _scenario_tick_stride(n_points: int) -> int:
    if n_points <= 10:
        return 1
    if n_points <= 20:
        return 2
    return 4


def _plot_metric_dashboard(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    families = (
        df[["family_order", "sweep_family_key", "sweep_family_display", "sweep_label", "sweep_unit", "sweep_xscale"]]
        .drop_duplicates()
        .sort_values("family_order")
        .to_dict(orient="records")
    )
    estimators = sorted(df["estimator"].unique().tolist())
    colors = _estimator_colors(estimators)

    fig, axes = plt.subplots(len(families), len(METRIC_SPECS), figsize=(18.0, 8.6), squeeze=False)
    legend_handles: list[Any] = []
    legend_labels: list[str] = []

    for row_idx, family in enumerate(families):
        df_family = df[df["sweep_family_key"] == family["sweep_family_key"]].copy()
        df_family = df_family.sort_values(["sweep_value", "estimator"])
        for col_idx, metric in enumerate(METRIC_SPECS):
            ax = axes[row_idx][col_idx]
            metric_col = metric["col"]
            if metric_col not in df_family.columns:
                ax.set_visible(False)
                continue

            for estimator in estimators:
                df_est = df_family[df_family["estimator"] == estimator]
                if df_est.empty:
                    continue
                y = _positive_for_log(df_est[metric_col].to_numpy(dtype=float))
                line = ax.plot(
                    df_est["sweep_value"].to_numpy(dtype=float),
                    y,
                    color=colors[estimator],
                    linewidth=1.05,
                    marker="o",
                    markersize=2.2,
                    alpha=0.88,
                    label=estimator,
                )[0]
                if estimator not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(estimator)

            if str(family["sweep_xscale"]) == "log":
                ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.24)
            if row_idx == 0:
                ax.set_title(metric["label"], loc="left", fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{family['sweep_family_display']}\nmetric value")
            if row_idx == len(families) - 1:
                ax.set_xlabel(f"{family['sweep_label']} [{family['sweep_unit']}]")

    fig.suptitle("IEEE ramps and magnitude steps: all-estimator performance dashboard", fontsize=14, y=0.992)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(0.995, 0.5),
        frameon=False,
        fontsize=8,
        ncol=1,
    )
    fig.tight_layout(rect=[0.03, 0.03, 0.86, 0.95])

    png_path = out_dir / "ramp_mag_metric_dashboard.png"
    pdf_path = out_dir / "ramp_mag_metric_dashboard.pdf"
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [png_path, pdf_path]


def _plot_heatmaps(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    families = (
        df[["family_order", "sweep_family_key", "sweep_family_display"]]
        .drop_duplicates()
        .sort_values("family_order")
        .to_dict(orient="records")
    )
    estimators = sorted(df["estimator"].unique().tolist())
    fig, axes = plt.subplots(len(families), len(HEATMAP_METRICS), figsize=(18.0, 9.0), squeeze=False)

    for row_idx, family in enumerate(families):
        df_family = df[df["sweep_family_key"] == family["sweep_family_key"]].copy()
        df_family = df_family.sort_values(["sweep_value", "scenario_order", "estimator"])
        scenario_meta = (
            df_family[["scenario", "scenario_order", "sweep_value_label"]]
            .drop_duplicates()
            .sort_values("scenario_order")
        )
        tick_stride = _scenario_tick_stride(len(scenario_meta))
        tick_positions = np.arange(len(scenario_meta))
        tick_labels = scenario_meta["sweep_value_label"].tolist()

        for col_idx, metric in enumerate(HEATMAP_METRICS):
            ax = axes[row_idx][col_idx]
            pivot = df_family.pivot_table(
                index="estimator",
                columns="scenario",
                values=metric["col"],
                aggfunc="mean",
            )
            if pivot.empty:
                ax.set_visible(False)
                continue
            pivot = pivot.reindex(index=estimators, columns=scenario_meta["scenario"].tolist())
            values = _positive_for_log(pivot.to_numpy(dtype=float))
            image_values = np.log10(values)
            finite = image_values[np.isfinite(image_values)]
            vmin = float(np.nanmin(finite)) if finite.size else -9.0
            vmax = float(np.nanmax(finite)) if finite.size else 1.0
            im = ax.imshow(image_values, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(f"{family['sweep_family_display']}: {metric['label']}", loc="left", fontweight="bold")
            ax.set_yticks(np.arange(len(estimators)))
            ax.set_yticklabels(estimators, fontsize=7)
            ax.set_xticks(tick_positions[::tick_stride])
            ax.set_xticklabels(tick_labels[::tick_stride], rotation=35, ha="right", fontsize=7)
            ax.set_xlabel("sweep level")
            if col_idx == 0:
                ax.set_ylabel("estimator")
            cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_label("log10(metric)")

    fig.tight_layout()
    png_path = out_dir / "ramp_mag_heatmaps.png"
    pdf_path = out_dir / "ramp_mag_heatmaps.pdf"
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [png_path, pdf_path]


def _build_rank_long(df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in metric_cols:
        for (_, _, _, _), df_group in df.groupby(
            ["sweep_family_key", "sweep_family_display", "scenario", "sweep_value"],
            sort=False,
        ):
            df_group = df_group[["estimator", metric, "sweep_family_key", "sweep_family_display", "scenario", "sweep_value"]].copy()
            df_group[metric] = pd.to_numeric(df_group[metric], errors="coerce")
            df_group = df_group.dropna(subset=[metric])
            if df_group.empty:
                continue
            df_group["rank"] = df_group[metric].rank(method="min", ascending=True)
            for row in df_group.itertuples(index=False):
                rows.append(
                    {
                        "sweep_family_key": row.sweep_family_key,
                        "sweep_family_display": row.sweep_family_display,
                        "scenario": row.scenario,
                        "sweep_value": float(row.sweep_value),
                        "estimator": row.estimator,
                        "metric_col": metric,
                        "rank": float(row.rank),
                    }
                )
    return pd.DataFrame(rows)


def _plot_rank_trajectories(df: pd.DataFrame, out_dir: Path) -> tuple[list[Path], pd.DataFrame]:
    metric_lookup = {spec["col"]: spec["label"] for spec in HEATMAP_METRICS}
    rank_long = _build_rank_long(df, [spec["col"] for spec in HEATMAP_METRICS])
    if rank_long.empty:
        return [], pd.DataFrame()

    families = (
        rank_long[["sweep_family_key", "sweep_family_display"]]
        .drop_duplicates()
        .to_dict(orient="records")
    )
    estimators = sorted(rank_long["estimator"].unique().tolist())
    colors = _estimator_colors(estimators)
    fig, axes = plt.subplots(len(families), len(HEATMAP_METRICS), figsize=(18.0, 9.0), squeeze=False)
    legend_handles: list[Any] = []
    legend_labels: list[str] = []

    for row_idx, family in enumerate(families):
        for col_idx, metric in enumerate(HEATMAP_METRICS):
            ax = axes[row_idx][col_idx]
            df_metric = rank_long[
                (rank_long["sweep_family_key"] == family["sweep_family_key"])
                & (rank_long["metric_col"] == metric["col"])
            ].copy()
            df_metric = df_metric.sort_values(["sweep_value", "estimator"])
            if df_metric.empty:
                ax.set_visible(False)
                continue
            for estimator in estimators:
                df_est = df_metric[df_metric["estimator"] == estimator]
                if df_est.empty:
                    continue
                line = ax.plot(
                    df_est["sweep_value"].to_numpy(dtype=float),
                    df_est["rank"].to_numpy(dtype=float),
                    color=colors[estimator],
                    linewidth=1.0,
                    marker="o",
                    markersize=2.0,
                    alpha=0.88,
                    label=estimator,
                )[0]
                if estimator not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(estimator)
            ax.invert_yaxis()
            ax.grid(True, alpha=0.24)
            ax.set_title(f"{family['sweep_family_display']}: {metric_lookup[metric['col']]}", loc="left", fontweight="bold")
            ax.set_ylabel("rank (1 = best)")
            ax.set_xlabel("sweep value")

    fig.suptitle("Rank trajectories across stress level", fontsize=14, y=0.992)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(0.995, 0.5),
        frameon=False,
        fontsize=8,
        ncol=1,
    )
    fig.tight_layout(rect=[0.03, 0.03, 0.86, 0.95])

    png_path = out_dir / "ramp_mag_rank_trajectories.png"
    pdf_path = out_dir / "ramp_mag_rank_trajectories.pdf"
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [png_path, pdf_path], rank_long


def _plot_sensitivity_bars(regression_df: pd.DataFrame, out_dir: Path) -> list[Path]:
    metric_map = {
        "m1_rmse_hz_mean": "RMSE [Hz]",
        "m5_trip_risk_s_mean": "Trip-Risk [s]",
    }
    df_sens = regression_df[regression_df["metric_col"].isin(metric_map)].copy()
    if df_sens.empty:
        return []

    families = (
        df_sens[["sweep_family_key", "sweep_family_display"]]
        .drop_duplicates()
        .to_dict(orient="records")
    )
    fig, axes = plt.subplots(len(families), len(metric_map), figsize=(18.0, 9.0), squeeze=False)

    for row_idx, family in enumerate(families):
        for col_idx, metric_col in enumerate(metric_map):
            ax = axes[row_idx][col_idx]
            df_metric = df_sens[
                (df_sens["sweep_family_key"] == family["sweep_family_key"])
                & (df_sens["metric_col"] == metric_col)
            ].copy()
            df_metric = df_metric.sort_values("pct_change_per_report_step", ascending=False)
            if df_metric.empty:
                ax.set_visible(False)
                continue
            y = np.arange(len(df_metric))
            vals = pd.to_numeric(df_metric["pct_change_per_report_step"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            colors = ["#b2182b" if val >= 0.0 else "#2166ac" for val in vals]
            ax.barh(y, vals, color=colors, alpha=0.9)
            ax.set_yticks(y)
            ax.set_yticklabels(df_metric["estimator"].tolist(), fontsize=7)
            ax.invert_yaxis()
            ax.axvline(0.0, color="#333", linewidth=0.8)
            ax.grid(True, axis="x", alpha=0.24)
            ax.set_title(f"{family['sweep_family_display']}: {metric_map[metric_col]}", loc="left", fontweight="bold")
            ax.set_xlabel("% change per reporting step")

    fig.tight_layout()
    png_path = out_dir / "ramp_mag_sensitivity_bars.png"
    pdf_path = out_dir / "ramp_mag_sensitivity_bars.pdf"
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [png_path, pdf_path]


def _family_overview(df_manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family_key, df_family in df_manifest.groupby("sweep_family_key", sort=False):
        df_family = df_family.sort_values("scenario_order")
        rows.append(
            {
                "family_key": family_key,
                "family_display": str(df_family["sweep_family_display"].iloc[0]),
                "scenario_count": int(df_family["scenario"].nunique()),
                "sweep_label": str(df_family["sweep_label"].iloc[0]),
                "sweep_unit": str(df_family["sweep_unit"].iloc[0]),
                "sweep_xscale": str(df_family["sweep_xscale"].iloc[0]),
                "min_sweep_value": float(df_family["sweep_value"].min()),
                "max_sweep_value": float(df_family["sweep_value"].max()),
                "anchor_scenario": str(df_family["family_anchor_scenario"].iloc[0]),
                "description": str(df_family["family_description"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def _build_average_rank_summary(rank_long: pd.DataFrame) -> pd.DataFrame:
    if rank_long.empty:
        return pd.DataFrame()
    metric_lookup = {spec["col"]: spec["label"] for spec in HEATMAP_METRICS}
    df_avg = (
        rank_long.groupby(["sweep_family_display", "estimator", "metric_col"], as_index=False)["rank"]
        .mean()
        .rename(columns={"rank": "avg_rank"})
    )
    df_avg["metric_label"] = df_avg["metric_col"].map(metric_lookup)
    return df_avg.sort_values(["sweep_family_display", "metric_label", "avg_rank", "estimator"])


def _build_endpoint_winners(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    families = df["sweep_family_display"].drop_duplicates().tolist()
    for family_display in families:
        df_family = df[df["sweep_family_display"] == family_display].copy()
        x_min = float(df_family["sweep_value"].min())
        x_max = float(df_family["sweep_value"].max())
        for metric in METRIC_SPECS:
            metric_col = metric["col"]
            if metric_col not in df_family.columns:
                continue
            low = df_family[df_family["sweep_value"] == x_min][["estimator", metric_col]].dropna()
            high = df_family[df_family["sweep_value"] == x_max][["estimator", metric_col]].dropna()
            if low.empty or high.empty:
                continue
            low_best = low.sort_values(metric_col, ascending=True).iloc[0]
            high_best = high.sort_values(metric_col, ascending=True).iloc[0]
            rows.append(
                {
                    "family_display": family_display,
                    "metric_label": metric["label"],
                    "low_stress_value": x_min,
                    "low_stress_winner": str(low_best["estimator"]),
                    "low_stress_metric": float(low_best[metric_col]),
                    "high_stress_value": x_max,
                    "high_stress_winner": str(high_best["estimator"]),
                    "high_stress_metric": float(high_best[metric_col]),
                }
            )
    return pd.DataFrame(rows)


def _write_report_bundle(
    *,
    out_dir: Path,
    manifest: dict[str, Any],
    df: pd.DataFrame,
    df_manifest: pd.DataFrame,
    regression_df: pd.DataFrame | None,
    average_rank: pd.DataFrame,
    endpoint_winners: pd.DataFrame,
    figure_paths: list[Path],
) -> dict[str, str]:
    report_dir = out_dir / "report_bundle"
    report_dir.mkdir(parents=True, exist_ok=True)

    overview = _family_overview(df_manifest)
    overview_csv = report_dir / "family_overview.csv"
    overview.to_csv(overview_csv, index=False)

    avg_rank_csv = report_dir / "average_rank_summary.csv"
    if not average_rank.empty:
        average_rank.to_csv(avg_rank_csv, index=False)
    else:
        avg_rank_csv.write_text("no_data\n", encoding="utf-8")

    endpoint_csv = report_dir / "endpoint_winners.csv"
    if not endpoint_winners.empty:
        endpoint_winners.to_csv(endpoint_csv, index=False)
    else:
        endpoint_csv.write_text("no_data\n", encoding="utf-8")

    sens_csv = report_dir / "sensitivity_summary.csv"
    if regression_df is not None and not regression_df.empty:
        regression_df.to_csv(sens_csv, index=False)
    else:
        sens_csv.write_text("no_data\n", encoding="utf-8")

    figure_manifest = report_dir / "figure_manifest.json"
    figure_manifest.write_text(
        json.dumps({"generated_figures": [str(path) for path in figure_paths]}, indent=2),
        encoding="utf-8",
    )

    estimators_loaded = sorted(df["estimator"].unique().tolist())
    md_lines = [
        "# IEEE Ramp + Magnitude Step Sweep Report",
        "",
        "## Run Summary",
        f"- Benchmark identity: `{manifest.get('benchmark_identity', 'unknown')}`",
        f"- Families: `{len(manifest.get('families', []))}`",
        f"- Scenarios: `{df['scenario'].nunique()}`",
        f"- Estimators: `{len(estimators_loaded)}`",
        f"- MC runs per scenario: `{manifest.get('run_configuration', {}).get('n_mc_runs', 'see JSON') if isinstance(manifest, dict) else 'see JSON'}`",
        "",
        "## Families",
    ]
    for row in overview.itertuples(index=False):
        md_lines.extend(
            [
                f"### {row.family_display}",
                f"- Sweep variable: `{row.sweep_label}` [{row.sweep_unit}]",
                f"- Range: `{row.min_sweep_value:.4g}` to `{row.max_sweep_value:.4g}`",
                f"- Scale: `{row.sweep_xscale}`",
                f"- Anchor scenario: `{row.anchor_scenario}`",
                f"- Scenarios: `{row.scenario_count}`",
                f"- Description: {row.description}",
                "",
            ]
        )

    md_lines.extend(["## Figure Set", ""])
    for path in figure_paths:
        md_lines.append(f"- `{path.name}`")
    md_lines.append("")

    if not endpoint_winners.empty:
        md_lines.extend(["## Endpoint Winners", ""])
        for row in endpoint_winners.itertuples(index=False):
            md_lines.append(
                f"- {row.family_display} | {row.metric_label}: low-stress winner `{row.low_stress_winner}` ({row.low_stress_metric:.4g}), high-stress winner `{row.high_stress_winner}` ({row.high_stress_metric:.4g})."
            )
        md_lines.append("")

    if not average_rank.empty:
        md_lines.extend(["## Average-Rank Leaders", ""])
        for (family_display, metric_label), df_group in average_rank.groupby(["sweep_family_display", "metric_label"], sort=False):
            top3 = df_group.nsmallest(3, "avg_rank")
            summary = ", ".join(
                f"{row.estimator} (rank {row.avg_rank:.2f})" for row in top3.itertuples(index=False)
            )
            md_lines.append(f"- {family_display} | {metric_label}: {summary}")
        md_lines.append("")

    if regression_df is not None and not regression_df.empty:
        rmse_trip = regression_df[regression_df["metric_col"].isin(["m1_rmse_hz_mean", "m5_trip_risk_s_mean"])].copy()
        if not rmse_trip.empty:
            md_lines.extend(["## Sensitivity Highlights", ""])
            for (family_display, metric_label), df_group in rmse_trip.groupby(["sweep_family_display", "metric_label"], sort=False):
                df_group = df_group.sort_values("pct_change_per_report_step", ascending=False)
                worst = df_group.iloc[0]
                best = df_group.iloc[-1]
                md_lines.append(
                    f"- {family_display} | {metric_label}: steepest degradation `{worst['estimator']}` ({worst['pct_change_per_report_step']:.2f}% per {worst['report_step_label']}), most stable `{best['estimator']}` ({best['pct_change_per_report_step']:.2f}% per {best['report_step_label']})."
                )
            md_lines.append("")

    md_path = report_dir / "ieee_ramp_mag_sweep_report.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    html_lines = ["<html><body><pre>"]
    html_lines.append("\n".join(md_lines).replace("&", "&amp;").replace("<", "&lt;"))
    html_lines.append("</pre></body></html>")
    html_path = report_dir / "ieee_ramp_mag_sweep_report.html"
    html_path.write_text("".join(html_lines), encoding="utf-8")

    manifest_path = report_dir / "report_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "markdown": str(md_path),
                "html": str(html_path),
                "family_overview_csv": str(overview_csv),
                "average_rank_csv": str(avg_rank_csv),
                "endpoint_winners_csv": str(endpoint_csv),
                "sensitivity_csv": str(sens_csv),
                "figure_manifest": str(figure_manifest),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "markdown": str(md_path),
        "html": str(html_path),
        "family_overview_csv": str(overview_csv),
        "average_rank_csv": str(avg_rank_csv),
        "endpoint_winners_csv": str(endpoint_csv),
        "sensitivity_csv": str(sens_csv),
        "manifest": str(manifest_path),
    }


def build_ieee_ramp_mag_sweep_report(
    output_dir: Path,
    manifest_path: Path | None = None,
    regression_csv_path: Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    manifest_path = Path(manifest_path or (output_dir / "scenario_sweep_manifest.json"))
    df, df_manifest, manifest = _merge_report(output_dir, manifest_path)

    figures_dir = output_dir / "ramp_mag_report_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated_figures: list[Path] = []
    generated_figures.extend(_plot_metric_dashboard(df, figures_dir))
    generated_figures.extend(_plot_heatmaps(df, figures_dir))
    rank_figures, rank_long = _plot_rank_trajectories(df, figures_dir)
    generated_figures.extend(rank_figures)

    regression_df: pd.DataFrame | None = None
    if regression_csv_path is not None:
        regression_csv_path = Path(regression_csv_path)
        if regression_csv_path.exists():
            regression_df = pd.read_csv(regression_csv_path)
            generated_figures.extend(_plot_sensitivity_bars(regression_df, figures_dir))

    average_rank = _build_average_rank_summary(rank_long)
    endpoint_winners = _build_endpoint_winners(df)
    report_paths = _write_report_bundle(
        out_dir=output_dir,
        manifest=manifest,
        df=df,
        df_manifest=df_manifest,
        regression_df=regression_df,
        average_rank=average_rank,
        endpoint_winners=endpoint_winners,
        figure_paths=generated_figures,
    )

    plot_manifest = figures_dir / "plot_manifest.json"
    plot_manifest.write_text(
        json.dumps({"generated_files": [str(path) for path in generated_figures]}, indent=2),
        encoding="utf-8",
    )

    return {
        "figures_dir": str(figures_dir),
        "figure_manifest": str(plot_manifest),
        "generated_figures": [str(path) for path in generated_figures],
        **report_paths,
    }
