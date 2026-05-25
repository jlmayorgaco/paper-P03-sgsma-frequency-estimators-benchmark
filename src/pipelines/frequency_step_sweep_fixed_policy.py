from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BENCHMARK_INCLUDE_EXPERIMENTAL", "0")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in (str(SRC), str(ROOT)):
    while p in sys.path:
        sys.path.remove(p)
sys.path.insert(0, str(SRC))
sys.path.insert(1, str(ROOT))

from analysis.monte_carlo_engine import MonteCarloEngine, MonteCarloResult
from pipelines.benchmark_definition import ESTIMATOR_FAMILIES, load_active_estimators
import pipelines.full_mc_benchmark as benchmark
from scenarios.ieee_freq_step import IEEEFreqStepScenario


OUTPUT_SUBDIR = os.getenv("FREQSTEP_OUTPUT_SUBDIR", "frequency_step_mvp2_all18_representative")
OUTPUT_DIR = ROOT / "artifacts" / OUTPUT_SUBDIR

MANIFEST_NAME = "experiment_manifest.json"
GLOBAL_CSV_NAME = "global_metrics_report.csv"
RMSE_EST_CSV_NAME = "rmse_by_estimator.csv"
RMSE_FAM_CSV_NAME = "rmse_by_family.csv"
TIMING_CSV_NAME = "timing_profile.csv"
TUNING_CONTINUITY_CSV_NAME = "tuning_parameter_continuity.csv"
PLOT_NAME = "rmse_deterioration_by_family"
MULTIPAGE_PDF_NAME = "metrics_dashboard_multipage.pdf"
METHOD_MAP_PDF_NAME = "frequency_step_method_map.pdf"
METHOD_MAP_PNG_NAME = "frequency_step_method_map.png"
SIGN_DIAGNOSTIC_PDF_NAME = "frequency_step_sign_asymmetry.pdf"
SIGN_DIAGNOSTIC_PNG_NAME = "frequency_step_sign_asymmetry.png"
HYPOTHESIS_CSV_NAME = "frequency_step_hypothesis_tests.csv"
HYPOTHESIS_JSON_NAME = "frequency_step_hypothesis_tests.json"
HYPOTHESIS_MD_NAME = "frequency_step_hypothesis_tests.md"
LEGEND_CSV_NAME = "rmse_plot_method_legend.csv"

FAST_ESTIMATORS = (
    "ZCD,IPDFT,TFT,RLS,PLL,SOGI-PLL,SOGI-FLL,Type-3 SOGI-PLL,"
    "LKF,LKF2,EKF,UKF,RA-EKF,TKEO"
)
WINDOW_ESTIMATORS = "Prony,ESPRIT"
DATA_DRIVEN_ESTIMATORS = "Koopman (RK-DPMU),PI-GRU"
JOURNAL_ESTIMATORS = f"{FAST_ESTIMATORS},{WINDOW_ESTIMATORS},{DATA_DRIVEN_ESTIMATORS}"

STEP_LEVELS_HZ: tuple[float, ...] = (
    0.05,
    0.10,
    0.20,
    0.50,
    1.00,
    2.00,
    3.00,
    5.00,
)

METRIC_COLUMNS = [
    "m1_rmse_hz",
    "m2_mae_hz",
    "m3_max_peak_hz",
    "m4_std_error_hz",
    "m5_trip_risk_s",
    "m5_trip_risk_resolution_s",
    "m6_max_contig_trip_s",
    "m7_pcb_hz",
    "m8_settling_time_s",
    "m9_rfe_max_hz_s",
    "m10_rfe_rms_hz_s",
    "m11_rnaf_db",
    "m12_isi_pu",
    "m13_cpu_time_us",
    "m14_struct_latency_ms",
    "m15_pcb_compliant",
    "m16_heatmap_pass",
    "m17_hw_class",
    "m18_mem_peak_kb",
    "m19_mem_mean_kb",
    "m20_runtime_jitter_us",
    "m21_startup_valid_samples",
    "m22_invalid_output_rate",
    "m23_memory_key_count",
    "m24_pre_event_rmse_hz",
    "m25_post_1cy_rmse_hz",
    "m26_post_3cy_rmse_hz",
    "m27_post_100ms_rmse_hz",
    "m28_post_event_peak_hz",
    "m29_late_event_rmse_hz",
    "m30_event_settling_time_s",
    "m31_freq_bound_hit_rate",
    "m32_freq_lower_bound_hit_rate",
    "m33_freq_upper_bound_hit_rate",
]

METHODOLOGY_TEXT = (
    "Frequency-step atlas. The swept variable is the true step magnitude; noise, "
    "event time, and pre-event frequency stay controlled. Fixed-policy mode uses "
    "the estimator default parameter set across every step size and both signs, "
    "so the curves show transient tracking behavior rather than per-point retuning."
)


@dataclass(frozen=True)
class SweepScenario:
    step_hz: float
    abs_step_hz: float
    direction: str
    scenario_cls: type
    scenario_name: str


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        value = int(raw)
    except ValueError:
        value = int(default)
    return max(int(minimum), value)


def _env_float(name: str, default: float, minimum: float | None = None) -> float:
    raw = os.getenv(name)
    if raw is None:
        value = float(default)
    else:
        try:
            value = float(raw)
        except ValueError:
            value = float(default)
    if minimum is not None and value < minimum:
        return float(minimum)
    return float(value)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_csv(name: str) -> list[str]:
    raw = os.getenv(name)
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_float_csv(name: str) -> list[float]:
    out: list[float] = []
    for item in _env_csv(name):
        try:
            out.append(float(item))
        except ValueError:
            continue
    return out


def _sanitize_token(value: float) -> str:
    token = f"{abs(float(value)):g}".replace(".", "p")
    return token.replace("-", "m")


def _duration_s() -> float:
    return _env_float("FREQSTEP_DURATION_S", 1.50, minimum=0.3)


def _t_step_s() -> float:
    return _env_float("FREQSTEP_T_STEP_S", 0.50, minimum=0.0)


def _noise_bounds() -> tuple[float, float]:
    lo = _env_float("FREQSTEP_NOISE_LOW", 0.0005, minimum=0.0)
    hi = _env_float("FREQSTEP_NOISE_HIGH", 0.0020, minimum=lo)
    return lo, hi


def _apply_stratified_overrides(cls: type, params: dict[str, Any], run_idx: int, n_runs: int, base_seed: int) -> dict[str, Any]:
    del cls, base_seed
    if not _env_bool("FREQSTEP_MC_STRATIFIED_COVARIATES", True):
        return params
    n = max(1, int(n_runs))
    u = (int(run_idx) + 0.5) / n
    phase_u = (0.61803398875 * (int(run_idx) + 1)) % 1.0
    time_u = (0.41421356237 * (int(run_idx) + 1)) % 1.0
    noise_lo, noise_hi = _noise_bounds()
    out = dict(params)
    out["phase_rad"] = float(2.0 * math.pi * phase_u)
    out["noise_sigma"] = float(noise_lo + (noise_hi - noise_lo) * u)
    t0 = _t_step_s()
    out["t_step_s"] = float(t0 - 0.025 + 0.050 * time_u)
    out["seed"] = int(out.get("seed", 0))
    return out


def _create_frequency_step_variant(step_hz: float) -> SweepScenario:
    step = float(step_hz)
    abs_step = abs(step)
    direction = "pos" if step >= 0.0 else "neg"
    token = _sanitize_token(step)
    scenario_name = f"Sweep_FreqStep_{direction}_{token}Hz"
    class_name = f"SweepFreqStep{direction.title()}{token}"
    freq_pre = 60.0
    attrs = {
        "SCENARIO_NAME": scenario_name,
        "DEFAULT_PARAMS": {
            **IEEEFreqStepScenario.DEFAULT_PARAMS,
            "duration_s": _duration_s(),
            "freq_pre_hz": freq_pre,
            "freq_post_hz": freq_pre + step,
            "t_step_s": _t_step_s(),
            "noise_sigma": 0.001,
        },
        "MONTE_CARLO_SPACE": {
            "phase_rad": {"kind": "uniform", "low": 0.0, "high": 2.0 * math.pi},
            "noise_sigma": {"kind": "uniform", "low": _noise_bounds()[0], "high": _noise_bounds()[1]},
        },
        "FREQSTEP_SWEEP_VALUE": step,
        "FREQSTEP_ABS_VALUE": abs_step,
        "FREQSTEP_DIRECTION": direction,
        "get_name": classmethod(lambda cls: cls.SCENARIO_NAME),
        "apply_run_index_overrides": classmethod(_apply_stratified_overrides),
    }
    new_cls = type(class_name, (IEEEFreqStepScenario,), attrs)
    new_cls.__module__ = __name__
    globals()[class_name] = new_cls
    return SweepScenario(
        step_hz=step,
        abs_step_hz=abs_step,
        direction=direction,
        scenario_cls=new_cls,
        scenario_name=scenario_name,
    )


def _build_abs_step_levels() -> list[float]:
    configured = _env_float_csv("FREQSTEP_LEVELS_HZ")
    values = configured if configured else list(STEP_LEVELS_HZ)
    min_v = _env_float("FREQSTEP_MIN_HZ", min(STEP_LEVELS_HZ), minimum=0.0)
    max_v = _env_float("FREQSTEP_MAX_HZ", max(STEP_LEVELS_HZ), minimum=min_v)
    return sorted({round(abs(float(v)), 9) for v in values if min_v <= abs(float(v)) <= max_v and abs(float(v)) > 0.0})


def _build_signed_step_values() -> list[float]:
    directions_raw = [x.lower() for x in (_env_csv("FREQSTEP_SWEEP_DIRECTIONS") or ["pos", "neg"])]
    include_pos = any(x in {"pos", "+", "positive", "up"} for x in directions_raw)
    include_neg = any(x in {"neg", "-", "negative", "down"} for x in directions_raw)
    levels = _build_abs_step_levels()
    values: list[float] = []
    if include_pos:
        values.extend(levels)
    if include_neg:
        values.extend([-v for v in levels])
    return values


def _build_scenarios() -> list[SweepScenario]:
    scenarios = [_create_frequency_step_variant(v) for v in _build_signed_step_values()]
    include_names = set(_env_csv("FREQSTEP_SWEEP_INCLUDE_SCENARIOS"))
    if include_names:
        scenarios = [sc for sc in scenarios if sc.scenario_name in include_names]
    return scenarios


def _select_estimators() -> dict[str, type]:
    estimators = load_active_estimators()
    estimator_set = os.getenv("FREQSTEP_ESTIMATOR_SET", "journal").strip().lower()
    if estimator_set in {"fast", "fast14"}:
        default_include = FAST_ESTIMATORS
    elif estimator_set in {"data-driven", "datadriven", "data"}:
        default_include = DATA_DRIVEN_ESTIMATORS
    elif estimator_set in {"journal", "all18", "canonical"}:
        default_include = JOURNAL_ESTIMATORS
    elif estimator_set in {"active", "all"}:
        default_include = ""
    else:
        default_include = JOURNAL_ESTIMATORS

    include_raw = os.getenv("FREQSTEP_SWEEP_INCLUDE_ESTIMATORS", default_include).strip()
    exclude_raw = os.getenv("FREQSTEP_SWEEP_EXCLUDE_ESTIMATORS", "").strip()
    by_lower = {label.lower(): label for label in estimators}

    if include_raw:
        selected: set[str] = set()
        for item in [x.strip() for x in include_raw.split(",") if x.strip()]:
            hit = by_lower.get(item.lower())
            if hit:
                selected.add(hit)
        if not selected:
            raise ValueError("FREQSTEP_SWEEP_INCLUDE_ESTIMATORS did not match any estimator.")
        estimators = {k: v for k, v in estimators.items() if k in selected}

    if exclude_raw:
        excluded: set[str] = set()
        for item in [x.strip() for x in exclude_raw.split(",") if x.strip()]:
            hit = by_lower.get(item.lower())
            if hit:
                excluded.add(hit)
        estimators = {k: v for k, v in estimators.items() if k not in excluded}

    if not estimators:
        raise ValueError("Estimator filter removed all estimators.")
    return estimators


def _aggregate_summary(summary_df: pd.DataFrame) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for metric in METRIC_COLUMNS:
        if metric not in summary_df.columns:
            continue
        series = pd.to_numeric(summary_df[metric], errors="coerce")
        valid = series.dropna().astype(float)
        if valid.empty:
            continue
        row[f"{metric}_mean"] = float(valid.mean())
        row[f"{metric}_median"] = float(valid.median())
        row[f"{metric}_p10"] = float(np.percentile(valid, 10))
        row[f"{metric}_p90"] = float(np.percentile(valid, 90))
        row[f"{metric}_std"] = float(valid.std(ddof=1)) if len(valid) > 1 else 0.0
    return row


def _run_engine_local(engine: MonteCarloEngine) -> MonteCarloResult:
    summary_rows: list[dict[str, Any]] = []
    signal_dfs: list[pd.DataFrame] = []
    for run_idx in range(engine.n_runs):
        row, signal_df = engine.run_once(run_idx)
        summary_rows.append(row)
        if not signal_df.empty:
            signal_dfs.append(signal_df)
    summary_df = pd.DataFrame(summary_rows).sort_values(by="run_idx").reset_index(drop=True)
    signals_df = (
        pd.concat(signal_dfs, ignore_index=True).sort_values(by=["run_idx", "t_s"]).reset_index(drop=True)
        if signal_dfs
        else pd.DataFrame()
    )
    estimator_name = None
    if engine.estimator_cls is not None:
        estimator_name = getattr(engine.estimator_cls, "name", engine.estimator_cls.__name__)
    return MonteCarloResult(
        scenario_name=engine.scenario_cls.get_name(),
        estimator_name=estimator_name,
        summary_df=summary_df,
        signals_df=signals_df,
        meta={
            "n_runs": engine.n_runs,
            "base_seed": engine.base_seed,
            "estimator_params": dict(engine.estimator_params or {}),
            "execution_mode": "local_sequential",
        },
    )


def _enforce_standardized_step(est_name: str) -> bool:
    vectorized = set(_env_csv("FREQSTEP_VECTORIZE_ESTIMATORS") or ["PI-GRU", "Koopman (RK-DPMU)"])
    return est_name not in vectorized


def _env_key_for_estimator(est_name: str) -> str:
    return (
        est_name.upper()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )


def _estimator_n_mc_runs(est_name: str, default: int) -> int:
    key = _env_key_for_estimator(est_name)
    specific = os.getenv(f"FREQSTEP_{key}_N_MC_RUNS")
    if specific is not None:
        return _env_int(f"FREQSTEP_{key}_N_MC_RUNS", default, minimum=1)
    if ESTIMATOR_FAMILIES.get(est_name) == "Data-driven":
        return _env_int("FREQSTEP_DATA_DRIVEN_N_MC_RUNS", default, minimum=1)
    return int(default)


def _estimator_n_cost_reps(est_name: str, default: int) -> int:
    key = _env_key_for_estimator(est_name)
    specific = os.getenv(f"FREQSTEP_{key}_N_COST_REPS")
    if specific is not None:
        return _env_int(f"FREQSTEP_{key}_N_COST_REPS", default, minimum=1)
    if ESTIMATOR_FAMILIES.get(est_name) == "Data-driven":
        return _env_int("FREQSTEP_DATA_DRIVEN_N_COST_REPS", default, minimum=1)
    return int(default)


def _can_reuse_existing_run(
    summary_csv: Path,
    run_spec_path: Path,
    *,
    requested_n_mc_runs: int,
    requested_tuning_policy: str,
) -> bool:
    if not summary_csv.exists() or not run_spec_path.exists():
        return False
    try:
        spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if int(spec.get("n_mc_runs", -1)) != int(requested_n_mc_runs):
        return False
    return str(spec.get("tuning_policy", "")).lower() == requested_tuning_policy


def _tuning_policy() -> str:
    return os.getenv("FREQSTEP_TUNING_POLICY", "fixed_policy").strip().lower()


def _estimator_params(est_cls: type) -> dict[str, Any]:
    if hasattr(est_cls, "default_params"):
        return dict(est_cls.default_params())
    return {}


def _line_style(direction: str) -> str:
    return "-" if direction == "pos" else "--"


def _shade_step_regions(ax: plt.Axes, x_lo: float, x_hi: float, *, labels: bool = True) -> None:
    regions = [
        ("Small", 0.05, 0.20, "#66BB6A"),
        ("Nominal", 0.20, 1.00, "#FDD835"),
        ("Severe", 1.00, 3.00, "#FFB74D"),
        ("Extreme", 3.00, 5.00, "#EF5350"),
    ]
    for idx, (label, lo, hi, color) in enumerate(regions):
        band_lo = max(float(lo), x_lo)
        band_hi = min(float(hi), x_hi)
        if band_hi <= band_lo:
            continue
        ax.axvspan(band_lo, band_hi, color=color, alpha=0.075 if idx < 3 else 0.055, zorder=0)
        if labels:
            x_mid = math.sqrt(max(band_lo, 1e-9) * max(band_hi, 1e-9))
            ax.text(
                x_mid,
                0.985 - 0.04 * (idx % 2),
                label,
                transform=ax.get_xaxis_transform(),
                va="top",
                ha="center",
                fontsize=6.5,
                color="#263238",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.65),
            )


def _add_methodology_text(fig: plt.Figure) -> None:
    fig.text(
        0.5,
        0.965,
        METHODOLOGY_TEXT,
        ha="center",
        va="top",
        fontsize=7.2,
        color="#263238",
        wrap=True,
    )


def _plot_metric_by_family_page(
    *,
    df_global: pd.DataFrame,
    metric_col: str,
    metric_label: str,
    title_prefix: str,
    yscale: str,
    guide_line: float | None,
    strict_line: float | None,
) -> tuple[plt.Figure, dict[str, tuple[float, float, float, float]]]:
    if df_global.empty or metric_col not in df_global.columns:
        fig, _ = plt.subplots(1, 1, figsize=(8, 4))
        return fig, {}

    band_cols = [c for c in [metric_col.replace("_mean", "_p10"), metric_col.replace("_mean", "_p90")] if c in df_global.columns]
    df_metric = df_global[
        ["scenario", "abs_step_hz", "direction", "estimator", "family", metric_col] + band_cols
    ].copy()
    df_metric = df_metric.rename(columns={metric_col: "metric_value"})
    df_metric = df_metric.dropna(subset=["metric_value"])
    if df_metric.empty:
        fig, _ = plt.subplots(1, 1, figsize=(8, 4))
        return fig, {}

    families = ["Loop-based", "Model-based", "Window-based", "Adaptive", "Data-driven"]
    panels = ["Reference Step"] + families
    ticks = sorted(df_metric["abs_step_hz"].dropna().astype(float).unique().tolist())
    ncols = 2
    nrows = int(math.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12.0, 3.65 * nrows), sharex=False)
    axes_arr = np.atleast_1d(axes).ravel()
    cmap = matplotlib.colormaps["tab20"]
    est_labels = sorted(df_metric["estimator"].unique().tolist())
    color_map = {label: cmap(i % cmap.N) for i, label in enumerate(est_labels)}

    for idx, panel in enumerate(panels):
        ax = axes_arr[idx]
        if panel == "Reference Step":
            for step, ls, label in [(1.0, "-", "+1 Hz"), (-1.0, "--", "-1 Hz")]:
                sc = IEEEFreqStepScenario.run(
                    duration_s=_duration_s(),
                    freq_pre_hz=60.0,
                    freq_post_hz=60.0 + step,
                    t_step_s=_t_step_s(),
                    noise_sigma=0.0,
                    seed=0,
                )
                ax.plot(sc.t, sc.f_true, color="#111111", linestyle=ls, linewidth=1.35, label=label)
            ax.set_title("Reference Step", loc="left", fontweight="bold")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Frequency [Hz]")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(loc="best", fontsize=7, frameon=True)
            continue

        df_family = df_metric[df_metric["family"] == panel].copy()
        if df_family.empty:
            ax.set_title(f"{panel} (no data)", loc="left", fontweight="bold")
            ax.set_xscale("log")
            ax.grid(True, which="both", alpha=0.25)
            continue

        p10_col = metric_col.replace("_mean", "_p10")
        p90_col = metric_col.replace("_mean", "_p90")
        for (estimator, direction), df_est in df_family.sort_values(["estimator", "direction", "abs_step_hz"]).groupby(["estimator", "direction"], sort=True):
            x_vals = df_est["abs_step_hz"].to_numpy(dtype=float)
            y_vals = df_est["metric_value"].to_numpy(dtype=float)
            if yscale == "log":
                y_vals = np.maximum(y_vals, 1e-12)
            if p10_col in df_est.columns and p90_col in df_est.columns and len(df_est) > 1:
                lo = df_est[p10_col].to_numpy(dtype=float)
                hi = df_est[p90_col].to_numpy(dtype=float)
                if yscale == "log":
                    lo = np.maximum(lo, 1e-12)
                    hi = np.maximum(hi, 1e-12)
                ax.fill_between(x_vals, lo, hi, alpha=0.10, color=color_map[str(estimator)], linewidth=0)
            label = f"{estimator} {'+' if direction == 'pos' else '-'}"
            ax.plot(
                x_vals,
                y_vals,
                marker="o",
                markersize=2.8,
                linewidth=1.0,
                alpha=0.90,
                color=color_map[str(estimator)],
                linestyle=_line_style(str(direction)),
                label=label,
            )

        if ticks:
            _shade_step_regions(ax, min(ticks), max(ticks), labels=True)
        if guide_line is not None:
            ax.axhline(guide_line, color="#303F9F", linestyle="--", linewidth=0.95, label=f"Guide {guide_line:g}")
        if strict_line is not None:
            ax.axhline(strict_line, color="#00897B", linestyle="--", linewidth=0.95, label=f"Strict {strict_line:g}")
        ax.axvline(1.0, color="#7B1FA2", linestyle=":", linewidth=0.9, label="1 Hz ref")
        ax.set_xscale("log")
        if yscale == "log":
            ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.set_title(panel, loc="left", fontweight="bold")
        ax.set_ylabel(metric_label)
        if ticks:
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x:g}" if x in ticks else ""))
            for tick in ax.get_xticklabels():
                tick.set_rotation(70)
                tick.set_ha("right")
                tick.set_fontsize(6)
        ax.legend(loc="best", fontsize=5.8, frameon=True, ncol=1)

    for j in range(len(panels), len(axes_arr)):
        axes_arr[j].set_visible(False)
    for ax in axes_arr[1:len(panels)]:
        ax.set_xlabel("|Frequency step| [Hz]")
    fig.suptitle(f"{title_prefix}: by estimator family (fixed policy)", fontsize=13, y=0.995)
    _add_methodology_text(fig)
    fig.text(
        0.5,
        0.006,
        "Solid lines are upward steps; dashed lines are downward steps. Thresholds are interpretation guides, not formal compliance claims.",
        ha="center",
        va="bottom",
        fontsize=7,
        color="#37474F",
    )
    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.92])
    return fig, color_map


def _add_derived_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    for base_col, derived_prefix, scale in [
        ("m15_pcb_compliant", "m15_pass_rate_pct", 100.0),
        ("m16_heatmap_pass", "m16_heatmap_pass_rate_pct", 100.0),
        ("m22_invalid_output_rate", "m22_invalid_output_rate_pct", 100.0),
    ]:
        for suffix in ["mean", "median", "p10", "p90", "std"]:
            col = f"{base_col}_{suffix}"
            if col in df_out.columns:
                df_out[f"{derived_prefix}_{suffix}"] = pd.to_numeric(df_out[col], errors="coerce") * scale
    return df_out


def _save_rmse_by_family_plot(df_global: pd.DataFrame, out_dir: Path) -> tuple[list[Path], dict[str, tuple[float, float, float, float]]]:
    fig, color_map = _plot_metric_by_family_page(
        df_global=df_global,
        metric_col="m1_rmse_hz_mean",
        metric_label="RMSE [Hz]",
        title_prefix="Frequency-step RMSE",
        yscale="log",
        guide_line=_env_float("FREQSTEP_LIMIT_RMSE_GUIDE", 0.05, minimum=0.0),
        strict_line=_env_float("FREQSTEP_LIMIT_RMSE_STRICT", 0.01, minimum=0.0),
    )
    png_path = out_dir / f"{PLOT_NAME}.png"
    pdf_path = out_dir / f"{PLOT_NAME}.pdf"
    fig.savefig(png_path, dpi=240)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [png_path, pdf_path], color_map


def _save_method_map(df_global: pd.DataFrame, out_dir: Path) -> list[Path]:
    if df_global.empty or "m1_rmse_hz_mean" not in df_global.columns:
        return []
    df = df_global.copy()
    pivot = df.pivot_table(index="estimator", columns="abs_step_hz", values="m1_rmse_hz_mean", aggfunc="mean")
    if pivot.empty:
        return []
    families = df[["estimator", "family"]].drop_duplicates().set_index("estimator")["family"].to_dict()
    family_order = {name: i for i, name in enumerate(["Loop-based", "Model-based", "Window-based", "Adaptive", "Data-driven"])}
    ordered = sorted(pivot.index, key=lambda est: (family_order.get(families.get(est, ""), 99), str(est)))
    pivot = pivot.loc[ordered]
    x_vals = [float(x) for x in pivot.columns]
    values = np.log10(np.maximum(pivot.to_numpy(dtype=float), 1e-12))

    fig, axes = plt.subplots(2, 1, figsize=(13.0, 8.2), gridspec_kw={"height_ratios": [3.2, 1.15]})
    ax = axes[0]
    finite = values[np.isfinite(values)]
    vmax = float(np.percentile(finite, 95)) if finite.size else 0.0
    im = ax.imshow(values, aspect="auto", cmap="magma_r", vmin=-4.0, vmax=max(vmax, -4.0 + 1e-6))
    ax.set_title("Frequency-Step Method Stress Map", loc="left", fontweight="bold")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_xticks(np.arange(len(x_vals)))
    ax.set_xticklabels([f"{v:g}" for v in x_vals], rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("|Frequency step| [Hz]")
    ax.set_ylabel("Estimator")
    cbar = fig.colorbar(im, ax=ax, fraction=0.024, pad=0.012)
    cbar.set_label("log10 mean RMSE [Hz]")

    ax2 = axes[1]
    guide = _env_float("FREQSTEP_LIMIT_RMSE_GUIDE", 0.05, minimum=0.0)
    rows = []
    for est, df_est in df.sort_values("abs_step_hz").groupby("estimator", sort=False):
        reduced = df_est.groupby("abs_step_hz", as_index=False)["m1_rmse_hz_mean"].mean()
        fail = reduced[reduced["m1_rmse_hz_mean"] > guide]
        critical = float(fail.iloc[0]["abs_step_hz"]) if not fail.empty else float("nan")
        rows.append((est, families.get(est, ""), critical, float(reduced["abs_step_hz"].max())))
    summary = pd.DataFrame(rows, columns=["estimator", "family", "critical_step_hz", "max_step_hz"])
    summary = summary.sort_values(["family", "critical_step_hz"], na_position="last")
    y = np.arange(len(summary))
    x = summary["critical_step_hz"].fillna(summary["max_step_hz"] * 1.05).to_numpy(dtype=float)
    ax2.scatter(x, y, s=24, color="#263238")
    for i, row in enumerate(summary.itertuples(index=False)):
        ax2.text(float(x[i]) * 1.03, i, str(row.estimator), va="center", fontsize=6.4)
    ax2.axvline(guide, color="#303F9F", linestyle="--", linewidth=0.9, label=f"RMSE guide {guide:g} Hz")
    ax2.set_xscale("log")
    ax2.set_xlim(min(x_vals), max(x_vals) * 1.4)
    ax2.set_yticks([])
    ax2.set_xlabel("First |step| where mean RMSE exceeds guide [Hz]")
    ax2.set_title("Critical Frequency-Step Summary", loc="left", fontweight="bold")
    _shade_step_regions(ax2, min(x_vals), max(x_vals), labels=True)
    ax2.grid(True, which="both", alpha=0.22)
    ax2.legend(loc="best", fontsize=6.4, frameon=True)

    fig.suptitle("Frequency-Step Fixed-Policy Method Atlas", fontsize=13, y=0.995)
    _add_methodology_text(fig)
    fig.tight_layout(rect=[0.06, 0.04, 0.98, 0.93])
    png_path = out_dir / METHOD_MAP_PNG_NAME
    pdf_path = out_dir / METHOD_MAP_PDF_NAME
    fig.savefig(png_path, dpi=240)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [png_path, pdf_path]


def _save_sign_asymmetry_diagnostic(df_global: pd.DataFrame, out_dir: Path) -> list[Path]:
    if df_global.empty or "m1_rmse_hz_mean" not in df_global.columns:
        return []
    rows = []
    for (est, abs_step), part in df_global.groupby(["estimator", "abs_step_hz"]):
        vals = part.set_index("direction")["m1_rmse_hz_mean"].to_dict()
        if "pos" not in vals or "neg" not in vals:
            continue
        pos = max(float(vals["pos"]), 1e-12)
        neg = max(float(vals["neg"]), 1e-12)
        rows.append(
            {
                "estimator": est,
                "family": str(part["family"].iloc[0]),
                "abs_step_hz": float(abs_step),
                "signed_log10_ratio": float(math.log10(pos / neg)),
                "ratio_max_over_min": float(max(pos, neg) / min(pos, neg)),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    families = ["Loop-based", "Model-based", "Window-based", "Adaptive", "Data-driven"]
    ncols = 2
    nrows = int(math.ceil(len(families) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12.0, 3.35 * nrows), sharex=True)
    axes_arr = np.atleast_1d(axes).ravel()
    cmap = matplotlib.colormaps["tab20"]
    estimators = sorted(df["estimator"].unique())
    colors = {est: cmap(i % cmap.N) for i, est in enumerate(estimators)}
    for idx, family in enumerate(families):
        ax = axes_arr[idx]
        df_family = df[df["family"] == family]
        for est, df_est in df_family.groupby("estimator", sort=True):
            ax.plot(
                df_est["abs_step_hz"],
                df_est["signed_log10_ratio"],
                marker="o",
                markersize=2.8,
                linewidth=1.0,
                color=colors[str(est)],
                label=str(est),
            )
        ax.axhline(0.0, color="#212121", linestyle=":", linewidth=0.9)
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.set_title(family, loc="left", fontweight="bold")
        ax.set_ylabel("log10(RMSE + / RMSE -)")
        ax.set_xlabel("|Frequency step| [Hz]")
        if not df_family.empty:
            ax.legend(loc="best", fontsize=6.2, frameon=True)
    for j in range(len(families), len(axes_arr)):
        axes_arr[j].set_visible(False)
    fig.suptitle("Upward/Downward Step Asymmetry", fontsize=13, y=0.995)
    fig.text(
        0.5,
        0.965,
        "Values near zero mean sign-symmetric tracking. Large positive or negative values indicate directional bias.",
        ha="center",
        va="top",
        fontsize=7.3,
        color="#263238",
        wrap=True,
    )
    fig.tight_layout(rect=[0.04, 0.04, 0.98, 0.92])
    png_path = out_dir / SIGN_DIAGNOSTIC_PNG_NAME
    pdf_path = out_dir / SIGN_DIAGNOSTIC_PDF_NAME
    fig.savefig(png_path, dpi=240)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [png_path, pdf_path]


def _linear_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yy = np.asarray(y_true, dtype=float)
    pp = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(yy) & np.isfinite(pp)
    if np.count_nonzero(ok) < 2:
        return float("nan")
    yy = yy[ok]
    pp = pp[ok]
    sst = float(np.sum((yy - np.mean(yy)) ** 2))
    if sst <= 0.0:
        return float("nan")
    return float(1.0 - np.sum((yy - pp) ** 2) / sst)


def _classify_step_regime(df_est: pd.DataFrame) -> dict[str, Any]:
    reduced = (
        df_est.groupby("abs_step_hz", as_index=False)["m1_rmse_hz_mean"]
        .mean()
        .sort_values("abs_step_hz")
    )
    x = reduced["abs_step_hz"].to_numpy(dtype=float)
    y = np.maximum(reduced["m1_rmse_hz_mean"].to_numpy(dtype=float), 1e-12)
    out: dict[str, Any] = {
        "n_points": int(len(x)),
        "primary_regime": "Erratic",
        "claim_status": "weak",
        "power_slope_b": float("nan"),
        "power_r2_loglog": float("nan"),
        "total_ratio_high_low": float("nan"),
        "monotone_fraction": float("nan"),
        "large_reversals": 0,
        "max_sign_asymmetry_ratio": float("nan"),
    }
    if len(x) < 4:
        out["interpretation"] = "Too few step levels for automatic classification."
        return out

    lx = np.log10(x)
    ly = np.log10(y)
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    r2 = _linear_r2(ly, pred)
    ratio = float(y[-1] / max(y[0], 1e-12))
    diffs = np.diff(ly)
    monotone_fraction = float(np.mean(diffs >= -0.08)) if len(diffs) else float("nan")
    large_reversals = int(np.sum(diffs < -0.25))

    asym = []
    for _abs, part in df_est.groupby("abs_step_hz"):
        vals = part.set_index("direction")["m1_rmse_hz_mean"].to_dict()
        if "pos" in vals and "neg" in vals:
            pos = max(float(vals["pos"]), 1e-12)
            neg = max(float(vals["neg"]), 1e-12)
            asym.append(max(pos, neg) / min(pos, neg))
    max_asym = float(max(asym)) if asym else float("nan")

    out.update(
        {
            "power_slope_b": float(slope),
            "power_r2_loglog": float(r2),
            "total_ratio_high_low": ratio,
            "monotone_fraction": monotone_fraction,
            "large_reversals": large_reversals,
            "max_sign_asymmetry_ratio": max_asym,
        }
    )
    if math.isfinite(max_asym) and max_asym >= 3.0:
        regime = "Sign-asymmetric"
        status = "diagnostic"
        interp = "Upward and downward frequency steps produce materially different RMSE."
    elif ratio <= 1.35 and abs(slope) <= 0.12:
        regime = "Insensitive/flat"
        status = "claimable"
        interp = "Frequency RMSE is practically flat across the swept step range."
    elif slope > 0.20 and r2 >= 0.85 and large_reversals <= 1:
        regime = "Power-law-like"
        status = "claimable"
        interp = "Log-log RMSE trend is close to a power law in the non-saturated range."
    elif slope > 0.10 and monotone_fraction >= 0.75:
        regime = "Monotone growth"
        status = "claimable"
        interp = "RMSE grows mostly monotonically with frequency-step magnitude."
    elif ratio >= 2.0 and abs(float(np.mean(diffs[-3:]))) < 0.12:
        regime = "Saturation/plateau"
        status = "claimable"
        interp = "High-step error appears to approach a plateau or estimator rail."
    else:
        regime = "Erratic"
        status = "do_not_interpret"
        interp = "Curve has reversals or weak model support; repeat with stronger MC before claiming a law."
    out["primary_regime"] = regime
    out["claim_status"] = status
    out["interpretation"] = interp
    return out


def _build_tuning_continuity(df_global: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for estimator, df_est in df_global.groupby("estimator", sort=True):
        params_json = df_est.get("best_params_json")
        unique_sets = int(params_json.nunique()) if params_json is not None else 0
        rows.append(
            {
                "estimator": estimator,
                "unique_parameter_sets": unique_sets,
                "parameter_switches": max(0, unique_sets - 1),
                "tuning_policy": _tuning_policy(),
            }
        )
    return pd.DataFrame(rows)


def _save_frequency_step_hypothesis_tests(df_global: pd.DataFrame, out_dir: Path) -> tuple[Path, Path, Path]:
    rows = []
    continuity = _build_tuning_continuity(df_global)
    continuity_map = continuity.set_index("estimator").to_dict(orient="index") if not continuity.empty else {}
    for estimator, df_est in df_global.groupby("estimator", sort=True):
        result = _classify_step_regime(df_est)
        result["estimator"] = estimator
        result["family"] = str(df_est["family"].iloc[0])
        result.update(continuity_map.get(estimator, {}))
        rows.append(result)
    df_tests = pd.DataFrame(rows)
    csv_path = out_dir / HYPOTHESIS_CSV_NAME
    json_path = out_dir / HYPOTHESIS_JSON_NAME
    md_path = out_dir / HYPOTHESIS_MD_NAME
    df_tests.to_csv(csv_path, index=False)
    counts = df_tests["primary_regime"].value_counts().to_dict() if "primary_regime" in df_tests.columns else {}
    payload = {
        "artifact": str(out_dir),
        "metric": "m1_rmse_hz",
        "x_axis": "abs_step_hz",
        "regime_counts": counts,
        "results": benchmark._to_builtin(df_tests.to_dict(orient="records")),
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Frequency-Step Hypothesis Tests",
        "",
        f"- Artifact: `{out_dir}`",
        "- Metric: `m1_rmse_hz`",
        "- X axis: `abs_step_hz`",
        "",
        "These automatic screens classify observed trend shape. They do not smooth curves and do not create compliance claims.",
        "",
        "## Regime Counts",
        "",
    ]
    for regime, count in counts.items():
        lines.append(f"- `{regime}`: {count}")
    lines.extend(["", "## Per-Estimator Classification", ""])
    for _, row in df_tests.sort_values(["primary_regime", "estimator"]).iterrows():
        lines.append(
            f"- `{row['estimator']}`: `{row['primary_regime']}`; "
            f"b={float(row.get('power_slope_b', float('nan'))):.3g}, "
            f"R2_loglog={float(row.get('power_r2_loglog', float('nan'))):.3g}, "
            f"asym={float(row.get('max_sign_asymmetry_ratio', float('nan'))):.3g}, "
            f"status=`{row.get('claim_status', '')}`."
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def _save_multipage_metrics_dashboard(df_global: pd.DataFrame, out_dir: Path) -> Path:
    pdf_path = out_dir / MULTIPAGE_PDF_NAME
    df_plot = _add_derived_metric_columns(df_global)
    pages = [
        ("m1_rmse_hz_mean", "RMSE [Hz]", "RMSE", "log", 0.05, 0.01),
        ("m3_max_peak_hz_mean", "FE max per test [Hz]", "Peak error", "log", 0.5, 0.01),
        ("m27_post_100ms_rmse_hz_mean", "Post-step 100 ms RMSE [Hz]", "Early post-step RMSE", "log", 0.05, 0.01),
        ("m29_late_event_rmse_hz_mean", "Late post-step RMSE [Hz]", "Late-window RMSE", "log", 0.05, 0.01),
        ("m30_event_settling_time_s_mean", "Event settling time [s]", "Settling time", "linear", 0.1, 0.02),
        ("m5_trip_risk_s_mean", "Trip-risk time [s]", "Trip-risk time", "linear", 0.1, 0.02),
        ("m15_pass_rate_pct_mean", "Pass rate [%]", "Pass rate", "linear", 95.0, 99.0),
        ("m13_cpu_time_us_mean", "CPU time [us/pass]", "CPU cost", "log", None, None),
    ]
    with PdfPages(pdf_path) as pdf:
        for metric_col, metric_label, title_prefix, yscale, guide, strict in pages:
            if metric_col not in df_plot.columns:
                continue
            fig, _ = _plot_metric_by_family_page(
                df_global=df_plot,
                metric_col=metric_col,
                metric_label=metric_label,
                title_prefix=title_prefix,
                yscale=yscale,
                guide_line=guide,
                strict_line=strict,
            )
            pdf.savefig(fig)
            plt.close(fig)
        fig = _make_summary_page(df_global)
        pdf.savefig(fig)
        plt.close(fig)
    return pdf_path


def _make_summary_page(df_global: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11.0, 8.0))
    ax.axis("off")
    lines = [
        "Frequency-Step Fixed-Policy Atlas",
        "",
        METHODOLOGY_TEXT,
        "",
        f"Estimator count: {df_global['estimator'].nunique() if not df_global.empty else 0}",
        f"Step levels: {df_global['abs_step_hz'].nunique() if 'abs_step_hz' in df_global else 0}",
        f"Directions: {', '.join(sorted(df_global['direction'].unique())) if 'direction' in df_global else ''}",
        f"MC runs per pair: {int(df_global['n_mc_runs'].median()) if 'n_mc_runs' in df_global and not df_global.empty else 0}",
        "",
        "Interpretation:",
        "- Solid/dashed sign split checks upward versus downward step bias.",
        "- RMSE curves test transient tracking under abrupt frequency changes.",
        "- Settling-time pages show whether low error arrives quickly enough.",
        "- Guide lines support reading the atlas; they are not formal IEEE/IEC compliance claims.",
    ]
    ax.text(0.04, 0.95, "\n".join(lines), va="top", ha="left", fontsize=11, color="#263238", wrap=True)
    fig.tight_layout()
    return fig


def _save_summary_tables(df_global: pd.DataFrame, out_dir: Path) -> tuple[Path, Path, Path, Path]:
    global_csv = out_dir / GLOBAL_CSV_NAME
    df_global.to_csv(global_csv, index=False)
    keep_cols = [
        "scenario", "step_hz", "abs_step_hz", "direction", "estimator", "family", "n_mc_runs",
        "m1_rmse_hz_mean", "m1_rmse_hz_median", "m1_rmse_hz_p10", "m1_rmse_hz_p90", "m1_rmse_hz_std",
    ]
    rmse_cols = [col for col in keep_cols if col in df_global.columns]
    rmse_est_csv = out_dir / RMSE_EST_CSV_NAME
    df_global[rmse_cols].to_csv(rmse_est_csv, index=False)
    df_family = (
        df_global.groupby(["abs_step_hz", "direction", "family"], as_index=False)
        .agg(
            family_rmse_mean=("m1_rmse_hz_mean", "mean"),
            family_rmse_std=("m1_rmse_hz_mean", "std"),
            family_rmse_min=("m1_rmse_hz_mean", "min"),
            family_rmse_max=("m1_rmse_hz_mean", "max"),
        )
        .sort_values(["abs_step_hz", "direction", "family"])
    )
    rmse_family_csv = out_dir / RMSE_FAM_CSV_NAME
    df_family.to_csv(rmse_family_csv, index=False)
    continuity = _build_tuning_continuity(df_global)
    continuity_csv = out_dir / TUNING_CONTINUITY_CSV_NAME
    continuity.to_csv(continuity_csv, index=False)
    return global_csv, rmse_est_csv, rmse_family_csv, continuity_csv


def _write_readme(out_dir: Path) -> Path:
    path = out_dir / "README.md"
    text = f"""# Frequency-Step Fixed-Policy Atlas

This directory contains report-level artifacts for the IEEE frequency-step sweep.

Primary PDF:

```text
{MULTIPAGE_PDF_NAME}
```

Key companion files:

```text
{GLOBAL_CSV_NAME}
{RMSE_EST_CSV_NAME}
{RMSE_FAM_CSV_NAME}
{HYPOTHESIS_CSV_NAME}
{TUNING_CONTINUITY_CSV_NAME}
{SIGN_DIAGNOSTIC_PDF_NAME}
```

Per-run folders such as `Sweep_FreqStep_*` are generated simulation outputs and
should not be committed unless a specific archival run requires them.
"""
    path.write_text(text, encoding="utf-8")
    return path


def _write_manifest(out_dir: Path, scenarios: list[SweepScenario], estimators: dict[str, type], settings: dict[str, Any]) -> Path:
    payload = {
        "experiment": "frequency_step_fixed_policy_atlas",
        "status": "representative",
        "output_subdir": OUTPUT_SUBDIR,
        "pipeline_entrypoint": "src/pipelines/frequency_step_sweep_fixed_policy.py",
        "tuning_policy": _tuning_policy(),
        "methodology": METHODOLOGY_TEXT,
        "step_levels_hz": sorted({float(sc.abs_step_hz) for sc in scenarios}),
        "directions": sorted({sc.direction for sc in scenarios}),
        "signed_step_values_hz": [float(sc.step_hz) for sc in scenarios],
        "scenario_contract": {
            "duration_s": _duration_s(),
            "t_step_s_nominal": _t_step_s(),
            "freq_pre_hz": 60.0,
            "freq_post_hz": "60 + step_hz",
            "noise_sigma_uniform": list(_noise_bounds()),
            "phase_rad_stratified": [0.0, "2*pi"],
        },
        "settings": settings,
        "estimators": list(estimators.keys()),
        "families": {label: ESTIMATOR_FAMILIES.get(label, "Unknown") for label in estimators},
        "artifacts": {
            "global_metrics": GLOBAL_CSV_NAME,
            "rmse_by_estimator": RMSE_EST_CSV_NAME,
            "rmse_by_family": RMSE_FAM_CSV_NAME,
            "timing_profile": TIMING_CSV_NAME,
            "metrics_multipage_pdf": MULTIPAGE_PDF_NAME,
            "rmse_plot_pdf": f"{PLOT_NAME}.pdf",
            "method_map_pdf": METHOD_MAP_PDF_NAME,
            "sign_asymmetry_pdf": SIGN_DIAGNOSTIC_PDF_NAME,
            "hypothesis_tests_csv": HYPOTHESIS_CSV_NAME,
            "tuning_parameter_continuity_csv": TUNING_CONTINUITY_CSV_NAME,
        },
        "artifact_policy": "Commit report-level PDFs, PNGs, aggregate CSV/JSON/MD, README, and manifest. Do not commit Sweep_FreqStep_* per-run simulation folders.",
    }
    path = out_dir / MANIFEST_NAME
    path.write_text(json.dumps(benchmark._to_builtin(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> None:
    n_mc_runs = _env_int("FREQSTEP_SWEEP_N_MC_RUNS", _env_int("BENCHMARK_N_MC_RUNS", 1, minimum=1), minimum=1)
    base_seed = _env_int("FREQSTEP_SWEEP_BASE_SEED", 12345, minimum=0)
    resume_run = _env_bool("FREQSTEP_SWEEP_RESUME", True)
    n_cost_reps = _env_int("FREQSTEP_SWEEP_N_COST_REPS", 1, minimum=1)
    capture_signals = _env_bool("FREQSTEP_CAPTURE_SIGNALS", False)

    settings = {
        "n_mc_runs": n_mc_runs,
        "base_seed": base_seed,
        "resume_run": resume_run,
        "n_cost_reps": n_cost_reps,
        "capture_signals": capture_signals,
        "estimator_set": os.getenv("FREQSTEP_ESTIMATOR_SET", "journal"),
    }

    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scenarios = _build_scenarios()
    estimators = _select_estimators()

    print(f"Running fixed-policy frequency-step sweep: {len(scenarios)} scenarios x {len(estimators)} estimators")
    print(f"  MC runs per pair: {n_mc_runs}")
    print(f"  CPU timing reps per pair: {n_cost_reps}")
    print(f"  Tuning policy: {_tuning_policy()}")
    print(f"  Output dir: {OUTPUT_DIR}")

    rows_agg: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    for sc in scenarios:
        print(f"\nScenario {sc.scenario_name} (step={sc.step_hz:+g} Hz)")
        sc_dir = OUTPUT_DIR / sc.scenario_name
        sc_dir.mkdir(parents=True, exist_ok=True)
        for est_name, est_cls in estimators.items():
            est_n_mc_runs = _estimator_n_mc_runs(est_name, n_mc_runs)
            est_n_cost_reps = _estimator_n_cost_reps(est_name, n_cost_reps)
            out_dir = sc_dir / est_name
            out_dir.mkdir(parents=True, exist_ok=True)
            summary_csv = out_dir / f"{sc.scenario_name}__{est_name}_summary.csv"
            run_spec_path = out_dir / "run_spec.json"
            best_params = _estimator_params(est_cls)

            if resume_run and _can_reuse_existing_run(
                summary_csv,
                run_spec_path,
                requested_n_mc_runs=est_n_mc_runs,
                requested_tuning_policy=_tuning_policy(),
            ):
                summary_df = pd.read_csv(summary_csv)
                timing = {}
            else:
                print(f"  - {est_name}", flush=True)
                run_start = time.time()
                engine = MonteCarloEngine(
                    scenario_cls=sc.scenario_cls,
                    estimator_cls=est_cls,
                    estimator_params=best_params,
                    n_runs=est_n_mc_runs,
                    base_seed=base_seed,
                    n_cost_reps=est_n_cost_reps,
                    enforce_standardized_step=_enforce_standardized_step(est_name),
                    capture_signals=capture_signals,
                )
                result = _run_engine_local(engine)
                result.summary_df.to_csv(summary_csv, index=False)
                if capture_signals and not result.signals_df.empty:
                    result.signals_df.to_csv(out_dir / f"{sc.scenario_name}__{est_name}_signals.csv", index=False)
                summary_df = result.summary_df
                timing = {"total_elapsed_s": time.time() - run_start}
                run_spec = {
                    "scenario": sc.scenario_name,
                    "step_hz": float(sc.step_hz),
                    "abs_step_hz": float(sc.abs_step_hz),
                    "direction": sc.direction,
                    "estimator": est_name,
                    "family": ESTIMATOR_FAMILIES.get(est_name, "Unknown"),
                    "best_params": benchmark._to_builtin(best_params),
                    "tuning_policy": _tuning_policy(),
                    "n_mc_runs": int(est_n_mc_runs),
                    "n_cost_reps": int(est_n_cost_reps),
                    "enforce_standardized_step": _enforce_standardized_step(est_name),
                    "base_seed": int(base_seed),
                    "timing": benchmark._to_builtin(timing),
                }
                run_spec_path.write_text(json.dumps(benchmark._to_builtin(run_spec), indent=2, ensure_ascii=False), encoding="utf-8")

            agg = _aggregate_summary(summary_df)
            rows_agg.append(
                {
                    "scenario": sc.scenario_name,
                    "step_hz": float(sc.step_hz),
                    "abs_step_hz": float(sc.abs_step_hz),
                    "direction": sc.direction,
                    "estimator": est_name,
                    "family": ESTIMATOR_FAMILIES.get(est_name, "Unknown"),
                    "n_mc_runs": int(len(summary_df)),
                    "tuning_policy": _tuning_policy(),
                    "best_params_json": json.dumps(benchmark._to_builtin(best_params), sort_keys=True, ensure_ascii=False),
                    **agg,
                }
            )
            timing_rows.append(
                {
                    "scenario": sc.scenario_name,
                    "step_hz": float(sc.step_hz),
                    "abs_step_hz": float(sc.abs_step_hz),
                    "direction": sc.direction,
                    "estimator": est_name,
                    "family": ESTIMATOR_FAMILIES.get(est_name, "Unknown"),
                    "n_mc_runs": int(len(summary_df)),
                    "n_cost_reps": int(est_n_cost_reps),
                    "total_elapsed_s": timing.get("total_elapsed_s") if isinstance(timing, dict) else None,
                }
            )

    df_global = pd.DataFrame(rows_agg).sort_values(["abs_step_hz", "direction", "family", "estimator"])
    global_csv, rmse_est_csv, rmse_family_csv, continuity_csv = _save_summary_tables(df_global, OUTPUT_DIR)
    timing_csv = OUTPUT_DIR / TIMING_CSV_NAME
    pd.DataFrame(timing_rows).to_csv(timing_csv, index=False)
    plot_paths, color_map = _save_rmse_by_family_plot(df_global, OUTPUT_DIR)
    plot_paths += _save_method_map(df_global, OUTPUT_DIR)
    plot_paths += _save_sign_asymmetry_diagnostic(df_global, OUTPUT_DIR)
    hypothesis_paths = _save_frequency_step_hypothesis_tests(df_global, OUTPUT_DIR)
    multipage_pdf = _save_multipage_metrics_dashboard(df_global, OUTPUT_DIR)
    legend_path = OUTPUT_DIR / LEGEND_CSV_NAME
    pd.DataFrame(
        [
            {"estimator": est, "hex_color": matplotlib.colors.to_hex(rgba), "family": ESTIMATOR_FAMILIES.get(est, "Unknown")}
            for est, rgba in sorted(color_map.items())
        ]
    ).to_csv(legend_path, index=False)
    readme_path = _write_readme(OUTPUT_DIR)
    manifest_path = _write_manifest(OUTPUT_DIR, scenarios, estimators, settings)

    elapsed = (time.time() - t0) / 60.0
    print("\nArtifacts:")
    for path in [
        global_csv,
        rmse_est_csv,
        rmse_family_csv,
        timing_csv,
        continuity_csv,
        *plot_paths,
        *hypothesis_paths,
        multipage_pdf,
        legend_path,
        readme_path,
        manifest_path,
    ]:
        print(f"  - {path.relative_to(ROOT)}")
    print(f"\n[DONE] Frequency-step sweep completed in {elapsed:.1f} min.")


if __name__ == "__main__":
    main()
