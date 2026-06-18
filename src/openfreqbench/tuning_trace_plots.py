from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .registry import METRIC_LABELS, load_estimators, scenario_registry


IEEE_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#E69F00",
    "#000000",
)


@dataclass(frozen=True)
class TraceSpec:
    objective: str
    scenario: str
    estimator: str
    params: dict[str, Any]
    run_spec_path: Path


def _safe_token(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def _configure_ieee_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.05,
            "grid.linewidth": 0.35,
            "grid.alpha": 0.28,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _read_specs(
    run_root: Path,
    *,
    objectives: Iterable[str] | None,
    scenarios: Iterable[str] | None,
    estimators: Iterable[str] | None,
) -> list[TraceSpec]:
    selected = run_root / "selected_replay"
    if not selected.exists():
        raise FileNotFoundError(f"selected_replay directory not found: {selected}")

    objective_filter = set(objectives or [])
    scenario_filter = set(scenarios or [])
    estimator_filter = set(estimators or [])
    specs: list[TraceSpec] = []
    for path in sorted(selected.glob("*/*/*/run_spec.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        objective = str(payload.get("objective_metric") or path.parents[2].name)
        scenario = str(payload.get("scenario") or path.parents[1].name)
        estimator = str(payload.get("estimator") or path.parent.name)
        if objective_filter and objective not in objective_filter:
            continue
        if scenario_filter and scenario not in scenario_filter:
            continue
        if estimator_filter and estimator not in estimator_filter:
            continue
        params = payload.get("best_params") or payload.get("estimator_params") or payload.get("params") or {}
        if not isinstance(params, dict):
            raise TypeError(f"Estimator params in {path} must be an object.")
        specs.append(
            TraceSpec(
                objective=objective,
                scenario=scenario,
                estimator=estimator,
                params=dict(params),
                run_spec_path=path,
            )
        )
    if not specs:
        raise ValueError(f"No replay run_spec.json files matched under {selected}")
    return specs


def _estimate_trace(estimator_cls: type, params: dict[str, Any], t: np.ndarray, v: np.ndarray) -> np.ndarray:
    estimator = estimator_cls(**params)
    estimate = getattr(estimator, "estimate", None)
    if callable(estimate):
        return np.asarray(estimate(t, v), dtype=float)
    out = np.empty(len(v), dtype=float)
    for idx, (ti, zi) in enumerate(zip(t, v)):
        out[idx] = float(estimator.step(float(zi), float(ti)))
    return out


def _window_mask(t: np.ndarray, start_s: float, window_s: float | None) -> np.ndarray:
    if window_s is None or window_s <= 0:
        return t >= start_s
    return (t >= start_s) & (t <= start_s + window_s)


def _finite_rmse_mhz(error_hz: np.ndarray) -> float:
    values = np.asarray(error_hz, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.inf
    return float(np.sqrt(np.mean(values**2)) * 1000.0)


def _finite_mae_mhz(error_hz: np.ndarray) -> float:
    values = np.asarray(error_hz, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.inf
    return float(np.mean(np.abs(values)) * 1000.0)


def _finite_max_abs_mhz(error_hz: np.ndarray) -> float:
    values = np.asarray(error_hz, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.inf
    return float(np.max(np.abs(values)) * 1000.0)


def _axis_limit_mhz(values: Iterable[np.ndarray], floor_mhz: float, cap_mhz: float | None) -> tuple[float, bool]:
    finite: list[np.ndarray] = []
    for arr in values:
        clean = np.asarray(arr, dtype=float)
        clean = clean[np.isfinite(clean)]
        if len(clean):
            finite.append(np.abs(clean))
    peak = float(np.max(np.concatenate(finite))) if finite else 0.0
    limit = max(float(floor_mhz), peak * 1.15)
    clipped = False
    if cap_mhz is not None and cap_mhz > 0 and limit > cap_mhz:
        limit = float(cap_mhz)
        clipped = True
    return limit, clipped


def _save_figure(fig: plt.Figure, stem: Path, formats: Iterable[str], dpi: int) -> list[str]:
    out: list[str] = []
    stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fmt_clean = fmt.lower().lstrip(".")
        path = stem.with_suffix(f".{fmt_clean}")
        save_kwargs: dict[str, Any] = {"bbox_inches": "tight", "facecolor": "white"}
        if fmt_clean == "png":
            save_kwargs["dpi"] = int(dpi)
        fig.savefig(path, **save_kwargs)
        out.append(str(path))
    plt.close(fig)
    return out


def _plot_ground_truth(
    scenario: str,
    data: Any,
    out_stem: Path,
    *,
    start_s: float,
    window_s: float | None,
    formats: Iterable[str],
    dpi: int,
) -> list[str]:
    mask = _window_mask(np.asarray(data.t), start_s, window_s)
    t_ms = np.asarray(data.t)[mask] * 1000.0
    v = np.asarray(data.v)[mask]
    f_true = np.asarray(data.f_true)[mask]
    nominal_hz = float(np.median(f_true[np.isfinite(f_true)])) if np.any(np.isfinite(f_true)) else 60.0

    fig, axes = plt.subplots(2, 1, figsize=(7.16, 2.65), sharex=True, constrained_layout=True)
    axes[0].plot(t_ms, v, color="#111111", linewidth=0.75)
    axes[0].set_ylabel("Voltage (p.u.)")
    axes[0].text(0.01, 0.90, "(a)", transform=axes[0].transAxes, weight="bold")
    axes[0].grid(True)

    axes[1].plot(t_ms, (f_true - nominal_hz) * 1000.0, color="#111111", linewidth=0.9)
    axes[1].set_ylabel(r"$f_{\mathrm{true}} - f_0$ (mHz)")
    axes[1].set_xlabel("Time (ms)")
    axes[1].text(0.01, 0.90, "(b)", transform=axes[1].transAxes, weight="bold")
    axes[1].grid(True)
    fig.suptitle(scenario.replace("_", " "), y=1.02)
    return _save_figure(fig, out_stem, formats, dpi)


def _plot_trace_group(
    objective: str,
    scenario: str,
    traces: dict[str, np.ndarray],
    data: Any,
    out_stem: Path,
    *,
    start_s: float,
    window_s: float | None,
    min_error_mhz: float,
    max_error_mhz: float | None,
    formats: Iterable[str],
    dpi: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    mask = _window_mask(np.asarray(data.t), start_s, window_s)
    t_ms = np.asarray(data.t)[mask] * 1000.0
    v = np.asarray(data.v)[mask]
    f_true = np.asarray(data.f_true)[mask]
    nominal_hz = float(np.median(f_true[np.isfinite(f_true)])) if np.any(np.isfinite(f_true)) else 60.0
    label = METRIC_LABELS.get(objective, objective).replace("CPU time", "CPU")

    deviations = [(f_true - nominal_hz) * 1000.0]
    errors: list[np.ndarray] = []
    summary_rows: list[dict[str, Any]] = []
    for estimator, f_hat_full in traces.items():
        f_hat = np.asarray(f_hat_full)[mask]
        deviations.append((f_hat - nominal_hz) * 1000.0)
        error_hz = f_hat - f_true
        error_mhz = error_hz * 1000.0
        errors.append(error_mhz)
        summary_rows.append(
            {
                "objective": objective,
                "objective_label": METRIC_LABELS.get(objective, objective),
                "scenario": scenario,
                "estimator": estimator,
                "trace_rmse_mhz": _finite_rmse_mhz(error_hz),
                "trace_mae_mhz": _finite_mae_mhz(error_hz),
                "trace_max_abs_error_mhz": _finite_max_abs_mhz(error_hz),
            }
        )
    deviation_ylim, deviation_clipped = _axis_limit_mhz(deviations, min_error_mhz, max_error_mhz)
    error_ylim, error_clipped = _axis_limit_mhz(errors, min_error_mhz, max_error_mhz)

    fig, axes = plt.subplots(3, 1, figsize=(7.16, 4.85), sharex=True, constrained_layout=True)
    axes[0].plot(t_ms, v, color="#111111", linewidth=0.75)
    axes[0].set_ylabel("Voltage (p.u.)")
    axes[0].text(0.01, 0.88, "(a)", transform=axes[0].transAxes, weight="bold")
    axes[0].grid(True)

    axes[1].plot(t_ms, (f_true - nominal_hz) * 1000.0, color="#111111", linewidth=0.95, label="truth")
    for idx, (estimator, f_hat_full) in enumerate(traces.items()):
        color = IEEE_COLORS[idx % len(IEEE_COLORS)]
        axes[1].plot(t_ms, (np.asarray(f_hat_full)[mask] - nominal_hz) * 1000.0, color=color, label=estimator)
    axes[1].set_ylabel(r"$\hat{f} - f_0$ (mHz)")
    axes[1].set_ylim(-deviation_ylim, deviation_ylim)
    axes[1].text(0.01, 0.88, "(b)", transform=axes[1].transAxes, weight="bold")
    axes[1].legend(loc="upper right", ncol=min(4, len(traces) + 1), frameon=True, framealpha=0.9)
    axes[1].grid(True)

    for idx, (estimator, f_hat_full) in enumerate(traces.items()):
        row = next(item for item in summary_rows if item["estimator"] == estimator)
        color = IEEE_COLORS[idx % len(IEEE_COLORS)]
        rmse = row["trace_rmse_mhz"]
        rmse_label = "inf" if not math.isfinite(rmse) else f"{rmse:.3g}"
        axes[2].plot(
            t_ms,
            (np.asarray(f_hat_full)[mask] - f_true) * 1000.0,
            color=color,
            label=f"{estimator} ({rmse_label} mHz)",
        )
    axes[2].set_ylabel("Error (mHz)")
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_ylim(-error_ylim, error_ylim)
    axes[2].text(0.01, 0.88, "(c)", transform=axes[2].transAxes, weight="bold")
    axes[2].legend(loc="upper right", ncol=min(3, len(traces)), frameon=True, framealpha=0.9)
    axes[2].grid(True)
    axes[2].text(
        0.99,
        0.05,
        f"y-lim: +/-{error_ylim:g} mHz" + (" (clipped)" if error_clipped else ""),
        transform=axes[2].transAxes,
        ha="right",
        va="bottom",
        color="#444444",
        fontsize=7.5,
    )

    if deviation_clipped:
        axes[1].text(
            0.99,
            0.05,
            "clipped",
            transform=axes[1].transAxes,
            ha="right",
            va="bottom",
            color="#444444",
            fontsize=7.5,
        )
    fig.suptitle(f"{label}-tuned policy", y=1.015)
    paths = _save_figure(fig, out_stem, formats, dpi)
    for row in summary_rows:
        row["deviation_ylim_mhz"] = deviation_ylim
        row["error_ylim_mhz"] = error_ylim
        row["deviation_clipped"] = deviation_clipped
        row["error_clipped"] = error_clipped
    return paths, summary_rows


def _plot_rmse_summary(summary: pd.DataFrame, out_stem: Path, *, formats: Iterable[str], dpi: int) -> list[str]:
    objectives = list(dict.fromkeys(summary["objective"].astype(str).tolist()))
    scenarios = list(dict.fromkeys(summary["scenario"].astype(str).tolist()))
    estimators = list(dict.fromkeys(summary["estimator"].astype(str).tolist()))
    groups: list[tuple[str, str]] = [
        (objective, scenario)
        for objective in objectives
        for scenario in scenarios
        if not summary[(summary["objective"] == objective) & (summary["scenario"] == scenario)].empty
    ]
    width = min(0.22, 0.8 / max(1, len(estimators)))
    x = np.arange(len(groups), dtype=float)

    fig_w = max(3.5, min(7.16, 1.25 * max(1, len(groups)) + 0.4 * len(estimators)))
    fig, ax = plt.subplots(figsize=(fig_w, 2.65), constrained_layout=True)
    for idx, estimator in enumerate(estimators):
        values = []
        for objective, scenario in groups:
            row = summary[
                (summary["objective"] == objective)
                & (summary["scenario"] == scenario)
                & (summary["estimator"] == estimator)
            ]
            value = float(row["trace_rmse_mhz"].iloc[0]) if not row.empty else math.nan
            values.append(max(value, 1e-6) if math.isfinite(value) else math.nan)
        offset = (idx - (len(estimators) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width * 0.92, label=estimator, color=IEEE_COLORS[idx % len(IEEE_COLORS)])

    if len(scenarios) == 1:
        labels = [METRIC_LABELS.get(objective, objective).replace("CPU time", "CPU") + "-tuned" for objective, _ in groups]
    else:
        labels = [
            METRIC_LABELS.get(objective, objective).replace("CPU time", "CPU") + "\n" + scenario.replace("_", " ")
            for objective, scenario in groups
        ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set_ylabel("Trace RMSE (mHz, log)")
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", ncol=min(4, len(estimators)), frameon=False)
    return _save_figure(fig, out_stem, formats, dpi)


def build_tuning_trace_plots(
    run_root: Path,
    *,
    output_dir: Path | None = None,
    objectives: Iterable[str] | None = None,
    scenarios: Iterable[str] | None = None,
    estimators: Iterable[str] | None = None,
    window_start_s: float = 0.0,
    window_s: float | None = 0.2,
    min_error_mhz: float = 10.0,
    max_error_mhz: float | None = 50.0,
    formats: Iterable[str] = ("png", "pdf", "svg"),
    dpi: int = 600,
) -> dict[str, Any]:
    """Build IEEE-style diagnostic traces from full_mc_tuning_matrix replay specs."""
    _configure_ieee_style()
    root = Path(run_root).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve() if output_dir else root / "plots" / "ieee"
    out_dir.mkdir(parents=True, exist_ok=True)
    format_list = tuple(dict.fromkeys(fmt.lower().lstrip(".") for fmt in formats))

    specs = _read_specs(root, objectives=objectives, scenarios=scenarios, estimators=estimators)
    scenario_names = list(dict.fromkeys(spec.scenario for spec in specs))
    objective_names = list(dict.fromkeys(spec.objective for spec in specs))
    scenario_map = scenario_registry()
    estimator_map = load_estimators(sorted(set(spec.estimator for spec in specs)))

    missing_scenarios = sorted(set(scenario_names) - set(scenario_map))
    if missing_scenarios:
        raise ValueError(f"Unknown scenario(s) in replay specs: {missing_scenarios}")

    scenario_data = {name: scenario_map[name].run() for name in scenario_names}
    plots: dict[str, list[str]] = {}
    summary_rows: list[dict[str, Any]] = []

    single_scenario = len(scenario_names) == 1
    for scenario in scenario_names:
        stem_name = "ground_truth_ieee_mhz" if single_scenario else f"ground_truth_{_safe_token(scenario)}_ieee_mhz"
        plots[f"ground_truth:{scenario}"] = _plot_ground_truth(
            scenario,
            scenario_data[scenario],
            out_dir / stem_name,
            start_s=window_start_s,
            window_s=window_s,
            formats=format_list,
            dpi=dpi,
        )

    for objective in objective_names:
        for scenario in scenario_names:
            group_specs = [spec for spec in specs if spec.objective == objective and spec.scenario == scenario]
            if not group_specs:
                continue
            data = scenario_data[scenario]
            traces = {
                spec.estimator: _estimate_trace(estimator_map[spec.estimator], spec.params, data.t, data.v)
                for spec in group_specs
            }
            stem_name = f"traces_{_safe_token(objective)}_ieee_mhz"
            if not single_scenario:
                stem_name = f"traces_{_safe_token(objective)}__{_safe_token(scenario)}_ieee_mhz"
            paths, rows = _plot_trace_group(
                objective,
                scenario,
                traces,
                data,
                out_dir / stem_name,
                start_s=window_start_s,
                window_s=window_s,
                min_error_mhz=min_error_mhz,
                max_error_mhz=max_error_mhz,
                formats=format_list,
                dpi=dpi,
            )
            plots[f"trace:{objective}:{scenario}"] = paths
            for row in rows:
                row["run_spec"] = str(
                    next(
                        spec.run_spec_path
                        for spec in group_specs
                        if spec.estimator == row["estimator"]
                    )
                )
            summary_rows.extend(rows)

    summary = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "trace_summary_ieee_mhz.csv"
    summary.to_csv(summary_csv, index=False)
    if not summary.empty:
        plots["trace_rmse_summary"] = _plot_rmse_summary(
            summary,
            out_dir / "trace_rmse_summary_ieee_mhz",
            formats=format_list,
            dpi=dpi,
        )

    manifest = {
        "status": "pass",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(root),
        "output_dir": str(out_dir),
        "style": "ieee-trace-mhz-v1",
        "dpi": int(dpi),
        "formats": list(format_list),
        "window_start_s": float(window_start_s),
        "window_s": None if window_s is None else float(window_s),
        "min_error_mhz": float(min_error_mhz),
        "max_error_mhz": None if max_error_mhz is None else float(max_error_mhz),
        "n_trace_specs": int(len(specs)),
        "objectives": objective_names,
        "scenarios": scenario_names,
        "estimators": list(dict.fromkeys(spec.estimator for spec in specs)),
        "summary_csv": str(summary_csv),
        "plots": plots,
    }
    manifest_path = out_dir / "ieee_plot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest
