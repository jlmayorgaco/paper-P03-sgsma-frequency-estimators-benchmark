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


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _read_report(input_json: Path) -> dict[str, Any]:
    if not input_json.exists():
        raise FileNotFoundError(f"Benchmark report not found: {input_json}")
    return json.loads(input_json.read_text(encoding="utf-8"))


def _raw_df(report: dict[str, Any]) -> pd.DataFrame:
    rows = report.get("raw_run_records", [])
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _agg_df(report: dict[str, Any]) -> pd.DataFrame:
    rows = report.get("aggregated_metrics", [])
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _metric_mean_col(df: pd.DataFrame, metric: str) -> str | None:
    candidates = [f"{metric}_mean", metric]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _save_bar(
    df: pd.DataFrame,
    value_col: str,
    output_path: Path,
    title: str,
    ylabel: str,
    top_n: int = 20,
) -> None:
    if df.empty or value_col not in df.columns:
        return
    work = df[["estimator", value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = (
        work.dropna()
        .groupby("estimator", as_index=False)[value_col]
        .mean()
        .sort_values(value_col, ascending=True)
        .head(top_n)
    )
    if work.empty:
        return
    fig_w = max(7.0, min(12.0, 0.45 * len(work) + 4.0))
    fig, ax = plt.subplots(figsize=(fig_w, 4.8), constrained_layout=True)
    ax.bar(work["estimator"], work[value_col], color="#2F6B8F")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Estimator")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _bootstrap_mean_ci(values: np.ndarray, *, seed: int = 12345, iters: int = 1000) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan, math.nan
    mean = float(np.mean(values))
    if len(values) == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    means = np.empty(iters, dtype=float)
    for idx in range(iters):
        means[idx] = float(np.mean(rng.choice(values, size=len(values), replace=True)))
    lo, hi = np.quantile(means, [0.025, 0.975])
    return mean, float(lo), float(hi)


def _estimator_metric_summary(raw: pd.DataFrame, metric: str) -> pd.DataFrame:
    if raw.empty or metric not in raw.columns:
        return pd.DataFrame()
    work = raw[["estimator", "family", metric]].copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    rows: list[dict[str, Any]] = []
    for (estimator, family), block in work.dropna().groupby(["estimator", "family"]):
        values = block[metric].to_numpy(dtype=float)
        mean, lo, hi = _bootstrap_mean_ci(values)
        rows.append(
            {
                "estimator": str(estimator),
                "family": str(family),
                "metric": metric,
                "n": int(len(values)),
                "mean": mean,
                "ci95_lo": lo,
                "ci95_hi": hi,
            }
        )
    return pd.DataFrame(rows).sort_values("mean") if rows else pd.DataFrame()


def _save_ci_bar(summary: pd.DataFrame, output_path: Path, title: str, ylabel: str, top_n: int = 20) -> None:
    if summary.empty:
        return
    work = summary.sort_values("mean", ascending=True).head(top_n).copy()
    lower = np.maximum(0.0, work["mean"].to_numpy(dtype=float) - work["ci95_lo"].to_numpy(dtype=float))
    upper = np.maximum(0.0, work["ci95_hi"].to_numpy(dtype=float) - work["mean"].to_numpy(dtype=float))
    fig_w = max(7.0, min(12.0, 0.45 * len(work) + 4.0))
    fig, ax = plt.subplots(figsize=(fig_w, 4.8), constrained_layout=True)
    ax.bar(
        work["estimator"],
        work["mean"],
        yerr=np.vstack([lower, upper]),
        capsize=3,
        color="#2F6B8F",
        ecolor="#263238",
        linewidth=0.8,
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Estimator")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _save_pareto(df: pd.DataFrame, output_path: Path) -> None:
    rmse_col = _metric_mean_col(df, "m1_rmse_hz")
    cpu_col = _metric_mean_col(df, "m13_cpu_time_us")
    if df.empty or rmse_col is None or cpu_col is None:
        return
    work = df[["estimator", "family", rmse_col, cpu_col]].copy()
    work[rmse_col] = pd.to_numeric(work[rmse_col], errors="coerce")
    work[cpu_col] = pd.to_numeric(work[cpu_col], errors="coerce")
    work = work.dropna().groupby(["estimator", "family"], as_index=False)[[rmse_col, cpu_col]].mean()
    if work.empty:
        return

    families = list(dict.fromkeys(work["family"].astype(str)))
    palette = {
        "Model-based": "#1565C0",
        "Loop-based": "#2E7D32",
        "Window-based": "#E65100",
        "Adaptive": "#6A1B9A",
        "Data-driven": "#B71C1C",
        "Custom": "#455A64",
    }
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    for family in families:
        block = work[work["family"].astype(str) == family]
        ax.scatter(
            block[cpu_col],
            block[rmse_col],
            s=52,
            label=family,
            color=palette.get(family, "#757575"),
            alpha=0.9,
        )
        for _, row in block.iterrows():
            ax.annotate(
                str(row["estimator"]),
                (float(row[cpu_col]), float(row[rmse_col])),
                textcoords="offset points",
                xytext=(4, 3),
                fontsize=7,
            )
    ax.set_title("RMSE vs CPU Tradeoff")
    ax.set_xlabel("CPU time (us/sample)")
    ax.set_ylabel("RMSE (Hz)")
    ax.grid(alpha=0.25)
    if len(families) > 1:
        ax.legend(fontsize=8)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _save_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    rmse_col = _metric_mean_col(df, "m1_rmse_hz")
    if df.empty or rmse_col is None:
        return
    work = df[["scenario", "estimator", rmse_col]].copy()
    work[rmse_col] = pd.to_numeric(work[rmse_col], errors="coerce")
    pivot = work.pivot_table(index="scenario", columns="estimator", values=rmse_col, aggfunc="mean")
    if pivot.empty:
        return
    values = pivot.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return
    vmax = float(np.nanpercentile(finite, 95))
    vmax = vmax if vmax > 0 else float(np.nanmax(finite) or 1.0)
    fig_w = max(6.5, min(13.0, 0.52 * len(pivot.columns) + 4.0))
    fig_h = max(4.8, min(12.0, 0.34 * len(pivot.index) + 2.0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    im = ax.imshow(values, aspect="auto", cmap="viridis_r", vmin=0.0, vmax=vmax)
    ax.set_title("Scenario x Estimator RMSE Heatmap")
    ax.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index, fontsize=8)
    fig.colorbar(im, ax=ax, label="RMSE (Hz)")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _save_family_boxplot(raw: pd.DataFrame, output_path: Path) -> None:
    if raw.empty or "family" not in raw.columns or "m1_rmse_hz" not in raw.columns:
        return
    work = raw[["family", "m1_rmse_hz"]].copy()
    work["m1_rmse_hz"] = pd.to_numeric(work["m1_rmse_hz"], errors="coerce")
    work = work.dropna()
    groups = [(name, block["m1_rmse_hz"].to_numpy(dtype=float)) for name, block in work.groupby("family")]
    groups = [(name, arr) for name, arr in groups if len(arr) > 0]
    if not groups:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    ax.boxplot([arr for _, arr in groups], labels=[name for name, _ in groups], showfliers=False)
    ax.set_title("RMSE Distribution by Estimator Family")
    ax.set_ylabel("RMSE (Hz)")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _save_signal_traces(report: dict[str, Any], output_dir: Path, limit: int = 6) -> list[str]:
    per_pair = report.get("artifacts", {}).get("per_pair", [])
    out: list[str] = []
    for item in per_pair[:limit]:
        signals_path = str(item.get("signals_csv", "") or "")
        if not signals_path:
            continue
        path = Path(signals_path)
        if not path.exists():
            continue
        signals = pd.read_csv(path)
        needed = {"t_s", "f_true_hz", "f_hat_hz"}
        if not needed.issubset(signals.columns):
            continue
        scenario = str(item.get("scenario", "scenario"))
        estimator = str(item.get("estimator", "estimator")).replace("/", "_")
        first_run = signals["run_idx"].min() if "run_idx" in signals.columns else None
        if first_run is not None:
            signals = signals[signals["run_idx"] == first_run]
        if len(signals) > 3000:
            signals = signals.iloc[:: max(1, len(signals) // 3000), :]
        fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
        ax.plot(signals["t_s"], signals["f_true_hz"], label="true", color="#111111", linewidth=1.0)
        ax.plot(signals["t_s"], signals["f_hat_hz"], label="estimate", color="#B71C1C", linewidth=0.9)
        ax.set_title(f"{scenario} - {estimator}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        out_path = output_dir / f"trace_{scenario}__{estimator}.png"
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        out.append(str(out_path))
    return out


def _winner_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rmse_col = _metric_mean_col(df, "m1_rmse_hz")
    if df.empty or rmse_col is None:
        return []
    work = df[["scenario", "estimator", "family", rmse_col]].copy()
    work[rmse_col] = pd.to_numeric(work[rmse_col], errors="coerce")
    rows: list[dict[str, Any]] = []
    for scenario, block in work.dropna().groupby("scenario"):
        row = block.loc[block[rmse_col].idxmin()]
        rows.append(
            {
                "scenario": str(scenario),
                "estimator": str(row["estimator"]),
                "family": str(row["family"]),
                "rmse_hz": float(row[rmse_col]),
            }
        )
    return rows


def _fastest_rows(df: pd.DataFrame, top_n: int = 10) -> list[dict[str, Any]]:
    cpu_col = _metric_mean_col(df, "m13_cpu_time_us")
    if df.empty or cpu_col is None:
        return []
    work = df[["estimator", "family", cpu_col]].copy()
    work[cpu_col] = pd.to_numeric(work[cpu_col], errors="coerce")
    work = (
        work.dropna()
        .groupby(["estimator", "family"], as_index=False)[cpu_col]
        .mean()
        .sort_values(cpu_col)
        .head(top_n)
    )
    return [
        {"estimator": str(row["estimator"]), "family": str(row["family"]), "cpu_time_us": float(row[cpu_col])}
        for _, row in work.iterrows()
    ]


def build_report_outputs(input_json: Path, output_dir: Path | None = None) -> dict[str, Any]:
    report = _read_report(input_json)
    raw = _raw_df(report)
    agg = _agg_df(report)
    if output_dir is None:
        output_dir = input_json.parent / "report"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    raw_out = output_dir / "raw_run_records.csv"
    agg_out = output_dir / "aggregated_metrics.csv"
    if not raw.empty:
        raw.to_csv(raw_out, index=False)
    if not agg.empty:
        agg.to_csv(agg_out, index=False)

    rmse_summary = _estimator_metric_summary(raw, "m1_rmse_hz")
    cpu_summary = _estimator_metric_summary(raw, "m13_cpu_time_us")
    statistical_tables: list[str] = []
    if not rmse_summary.empty:
        path = output_dir / "estimator_rmse_ci.csv"
        rmse_summary.to_csv(path, index=False)
        statistical_tables.append(str(path))
    if not cpu_summary.empty:
        path = output_dir / "estimator_cpu_ci.csv"
        cpu_summary.to_csv(path, index=False)
        statistical_tables.append(str(path))

    rmse_col = _metric_mean_col(agg, "m1_rmse_hz")
    cpu_col = _metric_mean_col(agg, "m13_cpu_time_us")
    if not rmse_summary.empty:
        _save_ci_bar(rmse_summary, plots_dir / "rmse_by_estimator_ci.png", "Mean RMSE by Estimator (95% CI)", "RMSE (Hz)")
    elif rmse_col:
        _save_bar(agg, rmse_col, plots_dir / "rmse_by_estimator.png", "Mean RMSE by Estimator", "RMSE (Hz)")
    if not cpu_summary.empty:
        _save_ci_bar(cpu_summary, plots_dir / "cpu_by_estimator_ci.png", "Mean CPU Time by Estimator (95% CI)", "us/sample")
    elif cpu_col:
        _save_bar(agg, cpu_col, plots_dir / "cpu_by_estimator.png", "Mean CPU Time by Estimator", "us/sample")
    _save_pareto(agg, plots_dir / "pareto_rmse_cpu.png")
    _save_heatmap(agg, plots_dir / "scenario_rmse_heatmap.png")
    _save_family_boxplot(raw, plots_dir / "family_rmse_boxplot.png")
    trace_paths = _save_signal_traces(report, plots_dir)

    plot_paths = sorted(str(path) for path in plots_dir.glob("*.png"))
    winners = _winner_rows(agg)
    fastest = _fastest_rows(agg)
    summary = {
        "source_report": str(input_json),
        "output_dir": str(output_dir),
        "n_raw_records": int(len(raw)),
        "n_aggregated_rows": int(len(agg)),
        "scenario_winners_by_rmse": winners,
        "fastest_estimators": fastest,
        "plots": plot_paths,
        "trace_plots": trace_paths,
        "statistical_tables": statistical_tables,
        "confidence_interval_method": "nonparametric bootstrap of per-run estimator means; fixed seed=12345",
    }

    summary_json = output_dir / "analysis_summary.json"
    summary_json.write_text(json.dumps(_json_safe(summary), indent=2, allow_nan=False), encoding="utf-8")

    lines = [
        "# OpenFreqBench Analysis Summary",
        "",
        f"- Source report: `{input_json}`",
        f"- Raw records: `{len(raw)}`",
        f"- Aggregated rows: `{len(agg)}`",
        f"- Plots: `{len(plot_paths)}`",
        "",
        "## RMSE Winners",
    ]
    if winners:
        lines.extend(["", "| scenario | estimator | family | rmse_hz |", "|---|---|---|---:|"])
        for row in winners:
            lines.append(
                f"| {row['scenario']} | {row['estimator']} | {row['family']} | {row['rmse_hz']:.6g} |"
            )
    else:
        lines.append("")
        lines.append("No RMSE winner table available.")

    lines.extend(["", "## Fastest Estimators"])
    if fastest:
        lines.extend(["", "| estimator | family | cpu_time_us |", "|---|---|---:|"])
        for row in fastest:
            lines.append(f"| {row['estimator']} | {row['family']} | {row['cpu_time_us']:.6g} |")
    else:
        lines.append("")
        lines.append("No CPU ranking available.")

    lines.extend(["", "## Plots"])
    for path in plot_paths:
        lines.append(f"- `{path}`")
    if statistical_tables:
        lines.extend(["", "## Statistical Tables"])
        for path in statistical_tables:
            lines.append(f"- `{path}`")
    summary_md = output_dir / "analysis_summary.md"
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
        "raw_csv": str(raw_out) if raw_out.exists() else "",
        "aggregated_csv": str(agg_out) if agg_out.exists() else "",
        "statistical_tables": statistical_tables,
        "plots": plot_paths,
    }
