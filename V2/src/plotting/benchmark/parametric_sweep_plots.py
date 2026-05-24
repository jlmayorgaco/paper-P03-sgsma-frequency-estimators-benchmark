from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_manifest_rows(manifest: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family in manifest.get("families", []):
        for scenario in family.get("scenarios", []):
            rows.append(
                {
                    "scenario": scenario["scenario_name"],
                    "sweep_family_key": family["key"],
                    "sweep_family_display": family["display_name"],
                    "sweep_label": family["sweep_label"],
                    "sweep_unit": family["unit"],
                    "sweep_xscale": family.get("xscale", "linear"),
                    "sweep_value": float(scenario["sweep_value"]),
                    "anchor_scenario": bool(scenario.get("is_anchor", False)),
                }
            )
    return pd.DataFrame(rows)


def _family_slug(text: str) -> str:
    return (
        text.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def _estimator_colors(labels: list[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = matplotlib.colormaps["tab20"]
    return {
        label: cmap(idx % cmap.N)
        for idx, label in enumerate(labels)
    }


def _metric_spec() -> list[tuple[str, str]]:
    return [
        ("m1_rmse_hz_mean", "RMSE [Hz]"),
        ("m5_trip_risk_s_mean", "Trip-Risk [s]"),
    ]


def generate_parametric_sweep_figures(output_dir: Path, manifest_path: Path | None = None) -> list[Path]:
    output_dir = Path(output_dir)
    manifest_path = Path(manifest_path or (output_dir / "scenario_sweep_manifest.json"))
    report_path = output_dir / "global_metrics_report.csv"
    if not manifest_path.exists() or not report_path.exists():
        return []

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    df_report = pd.read_csv(report_path)
    df_manifest = _load_manifest_rows(manifest)
    if df_manifest.empty or df_report.empty:
        return []

    df = df_report.merge(df_manifest, on="scenario", how="inner")
    if df.empty:
        return []

    plots_dir = output_dir / "parametric_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    estimators = sorted(df["estimator"].unique().tolist())
    colors = _estimator_colors(estimators)
    outputs: list[Path] = []

    for family_key, df_family in df.groupby("sweep_family_key", sort=False):
        df_family = df_family.sort_values(["sweep_value", "estimator"]).copy()
        first = df_family.iloc[0]
        sweep_label = str(first["sweep_label"])
        sweep_unit = str(first["sweep_unit"])
        xscale = str(first["sweep_xscale"])
        family_display = str(first["sweep_family_display"])

        fig, axes = plt.subplots(2, 1, figsize=(11.4, 7.2), sharex=True)
        legend_handles = []
        legend_labels = []

        for ax, (metric_col, metric_title) in zip(axes, _metric_spec()):
            if metric_col not in df_family.columns:
                ax.set_visible(False)
                continue

            for estimator in estimators:
                df_est = df_family[df_family["estimator"] == estimator]
                if df_est.empty:
                    continue
                line = ax.plot(
                    df_est["sweep_value"].to_numpy(dtype=float),
                    df_est[metric_col].to_numpy(dtype=float),
                    color=colors[estimator],
                    linewidth=1.1,
                    marker="o",
                    markersize=2.3,
                    alpha=0.9,
                    label=estimator,
                )[0]
                if estimator not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(estimator)

            ax.set_title(metric_title, loc="left", fontweight="bold")
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.24)
            if xscale == "log":
                ax.set_xscale("log")

        axes[-1].set_xlabel(f"{sweep_label} [{sweep_unit}]")
        fig.suptitle(
            f"{family_display}: estimator degradation across the parametric sweep",
            fontsize=12,
            y=0.988,
        )
        fig.legend(
            legend_handles,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(0.995, 0.5),
            frameon=False,
            ncol=1,
            fontsize=8,
        )
        fig.tight_layout(rect=[0.03, 0.03, 0.86, 0.95])

        stem = f"{_family_slug(family_key)}_degradation"
        png_path = plots_dir / f"{stem}.png"
        pdf_path = plots_dir / f"{stem}.pdf"
        fig.savefig(png_path, dpi=220)
        fig.savefig(pdf_path)
        plt.close(fig)
        outputs.extend([png_path, pdf_path])

    if outputs:
        manifest_out = plots_dir / "plot_manifest.json"
        manifest_out.write_text(
            json.dumps({"generated_files": [str(path) for path in outputs]}, indent=2),
            encoding="utf-8",
        )
        outputs.append(manifest_out)

    return outputs
