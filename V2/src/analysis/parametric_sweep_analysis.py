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


REGRESSION_METRICS: list[tuple[str, str]] = [
    ("m1_rmse_hz_mean", "RMSE [Hz]"),
    ("m3_max_peak_hz_mean", "Peak Error [Hz]"),
    ("m5_trip_risk_s_mean", "Trip-Risk [s]"),
    ("m8_settling_time_s_mean", "Settling Time [s]"),
]

REPORT_STEP_BY_FAMILY: dict[str, tuple[float, str]] = {
    "ibr_harmonics": (0.01, "1%"),
    "oob_interference": (0.01, "1%"),
    "single_tone_noise": (0.01, "1%"),
    "interharmonic": (0.01, "1%"),
    "ieee_mag_step": (0.01, "1%"),
    "phase_jump": (10.0, "10 deg"),
    "ieee_freq_ramp": (1.0, "1 Hz/s"),
    "ibr_ringdown_timescale": (0.1, "0.1x"),
}


def _load_manifest_rows(manifest: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family in manifest.get("families", []):
        report_step, report_step_label = REPORT_STEP_BY_FAMILY.get(
            str(family["key"]),
            (1.0, f"1 {family.get('unit', '')}".strip()),
        )
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
                    "report_step_value": float(report_step),
                    "report_step_label": report_step_label,
                }
            )
    return pd.DataFrame(rows)


def _linear_regression(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if x.size < 2:
        return {"slope": math.nan, "intercept": math.nan, "r2": math.nan}
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0.0:
        r2 = 1.0
    else:
        r2 = 1.0 - (ss_res / ss_tot)
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2)}


def _rankdata(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype=float)


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return math.nan
    x_rank = _rankdata(x)
    y_rank = _rankdata(y)
    x_std = float(np.std(x_rank))
    y_std = float(np.std(y_rank))
    if x_std <= 0.0 or y_std <= 0.0:
        return 0.0
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _log_regression(x: np.ndarray, y: np.ndarray) -> dict[str, float | bool]:
    positive = y[y > 0.0]
    if x.size < 2 or positive.size < 3:
        return {
            "available": False,
            "offset": math.nan,
            "slope": math.nan,
            "intercept": math.nan,
            "r2": math.nan,
        }

    offset = 0.0
    if np.any(y <= 0.0):
        offset = max(float(np.min(positive)) * 0.5, 1e-12)
    y_log = np.log(y + offset)
    linear = _linear_regression(x, y_log)
    return {
        "available": True,
        "offset": float(offset),
        "slope": float(linear["slope"]),
        "intercept": float(linear["intercept"]),
        "r2": float(linear["r2"]),
    }


def _trend_direction(slope: float, tolerance: float = 1e-12) -> str:
    if not math.isfinite(slope):
        return "unknown"
    if slope > tolerance:
        return "increasing"
    if slope < -tolerance:
        return "decreasing"
    return "flat"


def _statement(row: pd.Series) -> str:
    estimator = str(row["estimator"])
    metric = str(row["metric_label"])
    family = str(row["sweep_family_display"])
    direction = str(row["trend_direction"])
    step_label = str(row["report_step_label"])
    baseline = float(row["baseline_value"])
    end_mult = float(row["end_to_baseline_ratio"])
    rho = float(row["spearman_rho"])
    linear_r2 = float(row["linear_r2"])

    if pd.notna(row["pct_change_per_report_step"]):
        pct = float(row["pct_change_per_report_step"])
        verb = "increases" if pct >= 0 else "decreases"
        return (
            f"{estimator}: {metric} {verb} {abs(pct):.2f}% per {step_label} in {family} "
            f"(trend={direction}, R2={linear_r2:.3f}, rho={rho:.3f}, baseline={baseline:.4g}, x{end_mult:.2f} end/base)."
        )

    abs_step = float(row["absolute_change_per_report_step"])
    return (
        f"{estimator}: {metric} changes {abs_step:.4g} per {step_label} in {family} "
        f"(trend={direction}, R2={linear_r2:.3f}, rho={rho:.3f}, baseline={baseline:.4g}, x{end_mult:.2f} end/base)."
    )


def _heatmap_family_slug(text: str) -> str:
    return text.lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def _build_heatmap(df_summary: pd.DataFrame, metric_col: str, metric_label: str, out_dir: Path) -> list[Path]:
    df_metric = df_summary[df_summary["metric_col"] == metric_col].copy()
    if df_metric.empty:
        return []

    pivot = df_metric.pivot_table(
        index="estimator",
        columns="sweep_family_display",
        values="pct_change_per_report_step",
        aggfunc="mean",
    )
    if pivot.empty:
        return []

    filled = pivot.fillna(0.0)
    vmax = float(np.nanpercentile(np.abs(filled.to_numpy(dtype=float)), 95))
    vmax = max(vmax, 1.0)

    fig_w = max(8.8, 1.1 * len(pivot.columns))
    fig_h = max(5.4, 0.36 * len(pivot.index))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(filled.to_numpy(dtype=float), cmap="coolwarm", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns.tolist(), rotation=30, ha="right")
    ax.set_yticklabels(pivot.index.tolist())
    ax.set_title(f"{metric_label}: percent change per reporting step")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("% per reporting step")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = filled.iat[i, j]
            ax.text(
                j,
                i,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )

    fig.tight_layout()
    stem = f"{_heatmap_family_slug(metric_col)}_sensitivity_heatmap"
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [png_path, pdf_path]


def build_parametric_regression_reports(output_dir: Path, manifest_path: Path | None = None) -> dict[str, str]:
    output_dir = Path(output_dir)
    manifest_path = Path(manifest_path or (output_dir / "scenario_sweep_manifest.json"))
    report_path = output_dir / "global_metrics_report.csv"
    results_dir = output_dir / "parametric_regression"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not report_path.exists():
        raise FileNotFoundError(f"Missing global metrics report: {report_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    df_manifest = _load_manifest_rows(manifest)
    df_report = pd.read_csv(report_path)
    df = df_report.merge(df_manifest, on="scenario", how="inner")
    if df.empty:
        raise ValueError("No overlap between the parametric manifest and the global metrics report.")

    summary_rows: list[dict[str, Any]] = []
    for (family_key, family_display, estimator), df_group in df.groupby(
        ["sweep_family_key", "sweep_family_display", "estimator"],
        sort=False,
    ):
        df_group = df_group.sort_values("sweep_value").copy()
        x = df_group["sweep_value"].to_numpy(dtype=float)
        report_step_value = float(df_group["report_step_value"].iloc[0])
        report_step_label = str(df_group["report_step_label"].iloc[0])
        sweep_label = str(df_group["sweep_label"].iloc[0])
        sweep_unit = str(df_group["sweep_unit"].iloc[0])

        for metric_col, metric_label in REGRESSION_METRICS:
            if metric_col not in df_group.columns:
                continue
            y = pd.to_numeric(df_group[metric_col], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            x_valid = x[valid]
            y_valid = y[valid]
            if x_valid.size < 2:
                continue

            linear = _linear_regression(x_valid, y_valid)
            log_reg = _log_regression(x_valid, y_valid)
            baseline = float(y_valid[0])
            end_value = float(y_valid[-1])
            if abs(baseline) > 1e-12:
                ratio = end_value / baseline
                total_pct = ((end_value / baseline) - 1.0) * 100.0
            else:
                ratio = math.nan
                total_pct = math.nan

            if log_reg["available"]:
                pct_step = (math.exp(float(log_reg["slope"]) * report_step_value) - 1.0) * 100.0
            else:
                pct_step = math.nan

            abs_step = float(linear["slope"]) * report_step_value
            summary_rows.append(
                {
                    "sweep_family_key": family_key,
                    "sweep_family_display": family_display,
                    "estimator": estimator,
                    "metric_col": metric_col,
                    "metric_label": metric_label,
                    "sweep_label": sweep_label,
                    "sweep_unit": sweep_unit,
                    "n_points": int(x_valid.size),
                    "x_min": float(np.min(x_valid)),
                    "x_max": float(np.max(x_valid)),
                    "report_step_value": report_step_value,
                    "report_step_label": report_step_label,
                    "baseline_value": baseline,
                    "end_value": end_value,
                    "end_to_baseline_ratio": ratio,
                    "total_change_pct": total_pct,
                    "linear_slope_per_unit": float(linear["slope"]),
                    "linear_intercept": float(linear["intercept"]),
                    "linear_r2": float(linear["r2"]),
                    "absolute_change_per_report_step": abs_step,
                    "spearman_rho": _spearman_rho(x_valid, y_valid),
                    "trend_direction": _trend_direction(float(linear["slope"])),
                    "log_regression_available": bool(log_reg["available"]),
                    "log_offset_used": float(log_reg["offset"]) if math.isfinite(float(log_reg["offset"])) else math.nan,
                    "log_slope_per_unit": float(log_reg["slope"]) if log_reg["available"] else math.nan,
                    "log_r2": float(log_reg["r2"]) if log_reg["available"] else math.nan,
                    "pct_change_per_report_step": pct_step,
                }
            )

    df_summary = pd.DataFrame(summary_rows)
    if df_summary.empty:
        raise ValueError("No parametric regression rows could be computed.")

    df_summary["statement"] = df_summary.apply(_statement, axis=1)

    csv_path = results_dir / "parametric_regression_summary.csv"
    json_path = results_dir / "parametric_regression_summary.json"
    md_path = results_dir / "parametric_regression_report.md"
    df_summary.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(df_summary.to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    md_lines = [
        "# Parametric Sweep Regression Report",
        "",
        "## Interpretation",
        "- `pct_change_per_report_step` comes from a log-linear fit of the metric versus the sweep variable.",
        "- `absolute_change_per_report_step` is the additive slope over the same reporting step.",
        "- `end_to_baseline_ratio` compares the last sweep point against the first.",
        "- Use `R2` and `spearman_rho` to judge whether the fitted statement is trustworthy.",
        "",
    ]

    for family_display, df_family in df_summary.groupby("sweep_family_display", sort=False):
        md_lines.append(f"## {family_display}")
        md_lines.append("")
        for metric_label, df_metric in df_family.groupby("metric_label", sort=False):
            md_lines.append(f"### {metric_label}")
            md_lines.append("")
            df_metric = df_metric.sort_values(
                by=["pct_change_per_report_step", "absolute_change_per_report_step"],
                ascending=False,
                na_position="last",
            )
            for _, row in df_metric.iterrows():
                md_lines.append(f"- {row['statement']}")
            md_lines.append("")

    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    metric_pivot_paths: list[Path] = []
    for metric_col, metric_label in REGRESSION_METRICS:
        pivot = df_summary[df_summary["metric_col"] == metric_col].pivot_table(
            index="estimator",
            columns="sweep_family_display",
            values="pct_change_per_report_step",
            aggfunc="mean",
        )
        if pivot.empty:
            continue
        pivot_path = results_dir / f"{_heatmap_family_slug(metric_col)}_pct_per_step_pivot.csv"
        pivot.to_csv(pivot_path)
        metric_pivot_paths.append(pivot_path)

    heatmap_paths: list[Path] = []
    for metric_col, metric_label in REGRESSION_METRICS[:2]:
        heatmap_paths.extend(_build_heatmap(df_summary, metric_col, metric_label, results_dir))

    manifest_out = results_dir / "regression_manifest.json"
    manifest_payload = {
        "csv": str(csv_path),
        "json": str(json_path),
        "markdown": str(md_path),
        "pivot_csvs": [str(path) for path in metric_pivot_paths],
        "heatmaps": [str(path) for path in heatmap_paths],
    }
    manifest_out.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "markdown": str(md_path),
        "manifest": str(manifest_out),
    }
