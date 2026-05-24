from __future__ import annotations

import json
import math
import os
import sys
import time
import inspect
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import optuna
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
from scenarios.ieee_freq_ramp import IEEEFreqRampScenario


FAST_ESTIMATORS = (
    "ZCD,IPDFT,TFT,RLS,PLL,SOGI-PLL,SOGI-FLL,Type-3 SOGI-PLL,"
    "LKF,LKF2,EKF,UKF,RA-EKF,TKEO"
)
DATA_DRIVEN_ESTIMATORS = "Koopman (RK-DPMU),PI-GRU"
JOURNAL_ESTIMATORS = f"{FAST_ESTIMATORS},{DATA_DRIVEN_ESTIMATORS}"

OUTPUT_SUBDIR = os.getenv("FREQRAMP_OUTPUT_SUBDIR", "freq_ramp_rocof_v1_fixed_policy_fast14")
OUTPUT_DIR = ROOT / "artifacts" / OUTPUT_SUBDIR

MANIFEST_NAME = "experiment_manifest.json"
GLOBAL_CSV_NAME = "global_metrics_report.csv"
RMSE_EST_CSV_NAME = "rmse_by_estimator.csv"
RMSE_FAM_CSV_NAME = "rmse_by_family.csv"
PLOT_NAME = "rmse_deterioration_by_family"
MULTIPAGE_PDF_NAME = "metrics_dashboard_multipage.pdf"
METHOD_MAP_PDF_NAME = "rocof_method_map.pdf"
METHOD_MAP_PNG_NAME = "rocof_method_map.png"
SIGN_DIAGNOSTIC_PDF_NAME = "rocof_sign_asymmetry.pdf"
SIGN_DIAGNOSTIC_PNG_NAME = "rocof_sign_asymmetry.png"
HYPOTHESIS_CSV_NAME = "rocof_hypothesis_tests.csv"
HYPOTHESIS_JSON_NAME = "rocof_hypothesis_tests.json"
HYPOTHESIS_MD_NAME = "rocof_hypothesis_tests.md"
TUNING_CONTINUITY_CSV_NAME = "tuning_parameter_continuity.csv"
TIMING_CSV_NAME = "timing_profile.csv"

ROCOF_LEVELS_HZ_S: tuple[float, ...] = (
    0.10, 0.15, 0.20, 0.25, 0.35,
    0.50, 0.75, 1.00, 1.50, 2.00,
    3.00, 4.00, 5.00, 7.50, 10.00,
    15.00, 20.00, 30.00, 40.00, 50.00,
)

ROCOF_REGIONS: tuple[tuple[str, float, float, str, str], ...] = (
    ("Low", 0.10, 1.00, "#66BB6A", "slow grid frequency drift"),
    ("Standard-like", 1.00, 3.00, "#DCE775", "typical PMU dynamic tracking reference"),
    ("IBR stress", 3.00, 10.00, "#FDD835", "low-inertia and IBR ride-through stress"),
    ("Severe", 10.00, 30.00, "#FFB74D", "fast controls and protection stress"),
    ("Extreme", 30.00, 50.00, "#EF5350", "laboratory stress within the configured estimator frequency range"),
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
    "Frequency-ramp RoCoF atlas. The swept variable is the true linear ramp rate; "
    "amplitude and noise remain controlled nuisance variables. Fixed-policy mode uses one "
    "parameter set per estimator across every RoCoF level and both ramp signs, so curve shape "
    "reflects dynamic tracking degradation rather than per-point retuning. Scenario-derived "
    "frequency bounds prevent artificial rail saturation, and TKEO tuning penalizes noise-only "
    "bias by comparing paired noisy and clean evaluation signals."
)


@dataclass(frozen=True)
class SweepScenario:
    rocof_hz_s: float
    abs_rocof_hz_s: float
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


def _ramp_duration_s() -> float:
    return _env_float("FREQRAMP_RAMP_DURATION_S", 0.40, minimum=0.02)


def _t_start_s() -> float:
    return _env_float("FREQRAMP_T_START_S", 0.30, minimum=0.0)


def _duration_s() -> float:
    return _env_float("FREQRAMP_DURATION_S", _t_start_s() + _ramp_duration_s() + 0.50, minimum=0.3)


def _noise_bounds() -> tuple[float, float]:
    lo = _env_float("FREQRAMP_NOISE_LOW", 0.0005, minimum=0.0)
    hi = _env_float("FREQRAMP_NOISE_HIGH", 0.0020, minimum=lo)
    return lo, hi


def _scenario_frequency_bounds(scenarios: list[SweepScenario]) -> tuple[float, float]:
    values: list[float] = []
    for sc in scenarios:
        params = sc.scenario_cls.get_default_params()
        f_nom = float(params.get("freq_nom_hz", 60.0))
        f_cap = float(params.get("freq_cap_hz", f_nom))
        values.extend([f_nom, f_cap])
    if not values:
        values = [60.0]
    margin = _env_float("FREQRAMP_FREQ_BOUND_MARGIN_HZ", 10.0, minimum=0.0)
    f_min = max(0.0, min(values) - margin)
    f_max = max(values) + margin
    return float(f_min), float(f_max)


def _accepted_init_params(est_cls: type) -> set[str]:
    try:
        sig = inspect.signature(est_cls.__init__)
    except (TypeError, ValueError):
        return set()
    accepted: set[str] = set()
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            accepted.add("**kwargs")
        else:
            accepted.add(name)
    return accepted


def _apply_rocof_frequency_bounds(
    est_name: str,
    est_cls: type,
    params: dict[str, Any],
    frequency_bounds: tuple[float, float] | None,
) -> dict[str, Any]:
    if frequency_bounds is None or not _env_bool("FREQRAMP_FORCE_SCENARIO_FREQ_BOUNDS", True):
        return dict(params)

    f_min, f_max = frequency_bounds
    out = dict(params)
    accepted = _accepted_init_params(est_cls)
    accepts_any = "**kwargs" in accepted
    bound_keys = (("f_min_hz", "f_max_hz"), ("freq_min_hz", "freq_max_hz"))

    for lo_key, hi_key in bound_keys:
        force_sogi_fll_keys = est_name == "SOGI-FLL" and lo_key == "f_min_hz" and hi_key == "f_max_hz"
        should_apply = (
            accepts_any
            or lo_key in accepted
            or hi_key in accepted
            or lo_key in out
            or hi_key in out
            or force_sogi_fll_keys
        )
        if not should_apply:
            continue
        if accepts_any or lo_key in accepted or lo_key in out or force_sogi_fll_keys:
            out[lo_key] = float(f_min)
        if accepts_any or hi_key in accepted or hi_key in out or force_sogi_fll_keys:
            out[hi_key] = float(f_max)
    return out


def _apply_stratified_overrides(
    cls: type,
    params: dict[str, Any],
    run_idx: int,
    n_runs: int,
    base_seed: int,
) -> dict[str, Any]:
    del cls, base_seed
    if not _env_bool("FREQRAMP_MC_STRATIFIED_COVARIATES", True):
        return params
    n = max(1, int(n_runs))
    u = (int(run_idx) + 0.5) / n
    noise_lo, noise_hi = _noise_bounds()
    # Incommensurate progressions avoid aligning phase, noise, and event time.
    phase_u = (0.61803398875 * (int(run_idx) + 1)) % 1.0
    time_u = (0.41421356237 * (int(run_idx) + 1)) % 1.0
    params = dict(params)
    params["phase_rad"] = float(2.0 * math.pi * phase_u)
    params["noise_sigma"] = float(noise_lo + (noise_hi - noise_lo) * u)
    t0 = _t_start_s()
    params["t_start_s"] = float(t0 - 0.025 + 0.050 * time_u)
    params["seed"] = int(params.get("seed", 0))
    return params


def _create_rocof_variant(rocof_hz_s: float) -> SweepScenario:
    rocof = float(rocof_hz_s)
    abs_rocof = abs(rocof)
    direction = "pos" if rocof >= 0.0 else "neg"
    token = _sanitize_token(rocof)
    scenario_name = f"Sweep_FreqRamp_RoCoF_{direction}_{token}Hzs"
    class_name = f"SweepFreqRampRoCoF{direction.title()}{token}"

    t_start = _t_start_s()
    ramp_duration = _ramp_duration_s()
    freq_cap = 60.0 + rocof * ramp_duration
    attrs = {
        "SCENARIO_NAME": scenario_name,
        "DEFAULT_PARAMS": {
            **IEEEFreqRampScenario.DEFAULT_PARAMS,
            "duration_s": _duration_s(),
            "rocof_hz_s": rocof,
            "t_start_s": t_start,
            "freq_cap_hz": freq_cap,
            "noise_sigma": 0.001,
        },
        "MONTE_CARLO_SPACE": {
            "phase_rad": {"kind": "uniform", "low": 0.0, "high": 2.0 * math.pi},
            "noise_sigma": {"kind": "uniform", "low": _noise_bounds()[0], "high": _noise_bounds()[1]},
        },
        "ROCOF_SWEEP_VALUE": rocof,
        "ROCOF_ABS_VALUE": abs_rocof,
        "ROCOF_DIRECTION": direction,
        "get_name": classmethod(lambda cls: cls.SCENARIO_NAME),
        "apply_run_index_overrides": classmethod(_apply_stratified_overrides),
    }
    new_cls = type(class_name, (IEEEFreqRampScenario,), attrs)
    new_cls.__module__ = __name__
    globals()[class_name] = new_cls
    return SweepScenario(
        rocof_hz_s=rocof,
        abs_rocof_hz_s=abs_rocof,
        direction=direction,
        scenario_cls=new_cls,
        scenario_name=scenario_name,
    )


def _build_abs_rocof_levels() -> list[float]:
    configured = _env_float_csv("FREQRAMP_LEVELS_HZ_S")
    values = configured if configured else list(ROCOF_LEVELS_HZ_S)
    min_v = _env_float("FREQRAMP_MIN_HZ_S", min(ROCOF_LEVELS_HZ_S), minimum=0.0)
    max_v = _env_float("FREQRAMP_MAX_HZ_S", max(ROCOF_LEVELS_HZ_S), minimum=min_v)
    levels = sorted({round(abs(float(v)), 9) for v in values if min_v <= abs(float(v)) <= max_v and abs(float(v)) > 0.0})
    return levels


def _build_signed_rocof_values() -> list[float]:
    directions_raw = [x.lower() for x in (_env_csv("FREQRAMP_SWEEP_DIRECTIONS") or ["pos", "neg"])]
    include_pos = any(x in {"pos", "+", "positive", "up"} for x in directions_raw)
    include_neg = any(x in {"neg", "-", "negative", "down"} for x in directions_raw)
    levels = _build_abs_rocof_levels()
    values: list[float] = []
    if include_pos:
        values.extend(levels)
    if include_neg:
        values.extend([-v for v in levels])
    return values


def _build_scenarios() -> list[SweepScenario]:
    scenarios = [_create_rocof_variant(v) for v in _build_signed_rocof_values()]
    include_names = set(_env_csv("FREQRAMP_SWEEP_INCLUDE_SCENARIOS"))
    if include_names:
        scenarios = [sc for sc in scenarios if sc.scenario_name in include_names]
    return scenarios


def _select_estimators() -> dict[str, type]:
    estimators = load_active_estimators()
    estimator_set = os.getenv("FREQRAMP_ESTIMATOR_SET", "journal").strip().lower()
    if estimator_set in {"fast", "fast14"}:
        default_include = FAST_ESTIMATORS
    elif estimator_set in {"data-driven", "datadriven", "data"}:
        default_include = DATA_DRIVEN_ESTIMATORS
    elif estimator_set in {"journal", "journal16", "all-fast-data"}:
        default_include = JOURNAL_ESTIMATORS
    elif estimator_set in {"active", "all"}:
        default_include = ""
    else:
        default_include = JOURNAL_ESTIMATORS
    include_raw = os.getenv("FREQRAMP_SWEEP_INCLUDE_ESTIMATORS", default_include).strip()
    exclude_raw = os.getenv("FREQRAMP_SWEEP_EXCLUDE_ESTIMATORS", "").strip()
    by_lower = {label.lower(): label for label in estimators}

    if include_raw:
        selected: set[str] = set()
        for item in [x.strip() for x in include_raw.split(",") if x.strip()]:
            hit = by_lower.get(item.lower())
            if hit:
                selected.add(hit)
        if not selected:
            raise ValueError("FREQRAMP_SWEEP_INCLUDE_ESTIMATORS did not match any estimator.")
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


def _score_estimator_on_scenario(est_cls: type, params: dict[str, Any], sc: Any, eval_start: int) -> tuple[float, float]:
    est = est_cls(**params)
    f_hat = benchmark._run_estimator(est, sc.v)
    f_true = np.asarray(sc.f_true, dtype=float)
    f_hat = np.asarray(f_hat, dtype=float)
    if len(f_hat) != len(f_true):
        return 1e9, 1e9
    err = f_hat[eval_start:] - f_true[eval_start:]
    if len(err) < 4 or not np.all(np.isfinite(err)):
        return 1e9, 1e9
    rmse = float(np.sqrt(np.mean(err ** 2)))
    dt = float(sc.t[1] - sc.t[0]) if len(sc.t) > 1 else 1e-4
    rfe = np.gradient(f_hat[eval_start:], dt) - np.gradient(f_true[eval_start:], dt)
    rfe_rms = float(np.sqrt(np.mean(np.clip(rfe, -500.0, 500.0) ** 2)))
    if not np.isfinite(rmse) or not np.isfinite(rfe_rms):
        return 1e9, 1e9
    return rmse, rfe_rms


def _clean_counterpart(sc: Any) -> Any | None:
    params = getattr(sc, "meta", {}).get("parameters", {}) if hasattr(sc, "meta") else {}
    if not isinstance(params, dict):
        return None
    clean_params = dict(params)
    clean_params["noise_sigma"] = 0.0
    try:
        return IEEEFreqRampScenario.run(**clean_params)
    except Exception:
        return None


def _evaluate_standard_params_score(est_cls: type, params: dict[str, Any], scenarios_eval: list[Any], eval_start: int) -> float:
    try:
        rmses: list[float] = []
        rfes: list[float] = []
        for sc in scenarios_eval:
            rmse, rfe_rms = _score_estimator_on_scenario(est_cls, params, sc, eval_start)
            if rmse >= 1e9 or rfe_rms >= 1e9:
                return 1e9
            rmses.append(rmse)
            rfes.append(rfe_rms)
        if not rmses:
            return 1e9
        rfe_weight = _env_float("FREQRAMP_TUNE_RFE_WEIGHT", 0.001, minimum=0.0)
        return float(np.mean(rmses) + rfe_weight * np.mean(rfes))
    except Exception:
        return 1e9


def _evaluate_tkeo_rocof_score(est_cls: type, params: dict[str, Any], scenarios_eval: list[Any], eval_start: int) -> float:
    try:
        noisy_rmses: list[float] = []
        clean_rmses: list[float] = []
        noise_gaps: list[float] = []
        rfe_values: list[float] = []
        by_rocof: dict[float, list[float]] = {}

        for sc in scenarios_eval:
            noisy_rmse, noisy_rfe = _score_estimator_on_scenario(est_cls, params, sc, eval_start)
            if noisy_rmse >= 1e9 or noisy_rfe >= 1e9:
                return 1e9
            clean_sc = _clean_counterpart(sc)
            if clean_sc is None:
                return 1e9
            clean_rmse, clean_rfe = _score_estimator_on_scenario(est_cls, params, clean_sc, eval_start)
            if clean_rmse >= 1e9 or clean_rfe >= 1e9:
                return 1e9

            noisy_rmses.append(noisy_rmse)
            clean_rmses.append(clean_rmse)
            noise_gaps.append(abs(noisy_rmse - clean_rmse))
            rfe_values.append(noisy_rfe)
            params_meta = getattr(sc, "meta", {}).get("parameters", {}) if hasattr(sc, "meta") else {}
            rocof = abs(float(params_meta.get("rocof_hz_s", 0.0))) if isinstance(params_meta, dict) else 0.0
            by_rocof.setdefault(rocof, []).append(noisy_rmse)

        if not noisy_rmses:
            return 1e9

        means = [float(np.mean(by_rocof[k])) for k in sorted(by_rocof)]
        reversal_tol = _env_float("FREQRAMP_TKEO_REVERSAL_TOL_HZ", 0.05, minimum=0.0)
        reversal_penalty = 0.0
        for prev, cur in zip(means, means[1:]):
            if cur < prev - reversal_tol:
                reversal_penalty += (prev - cur) + reversal_tol

        clean_weight = _env_float("FREQRAMP_TKEO_CLEAN_WEIGHT", 0.25, minimum=0.0)
        noise_bias_weight = _env_float("FREQRAMP_TKEO_NOISE_BIAS_WEIGHT", 1.0, minimum=0.0)
        reversal_weight = _env_float("FREQRAMP_TKEO_REVERSAL_WEIGHT", 2.0, minimum=0.0)
        max_weight = _env_float("FREQRAMP_TKEO_MAX_WEIGHT", 0.10, minimum=0.0)
        rfe_weight = _env_float("FREQRAMP_TUNE_RFE_WEIGHT", 0.001, minimum=0.0)

        return float(
            np.mean(noisy_rmses)
            + clean_weight * np.mean(clean_rmses)
            + noise_bias_weight * np.mean(noise_gaps)
            + reversal_weight * reversal_penalty
            + max_weight * max(noisy_rmses)
            + rfe_weight * np.mean(rfe_values)
        )
    except Exception:
        return 1e9


def _evaluate_params_score(
    est_name: str,
    est_cls: type,
    params: dict[str, Any],
    scenarios_eval: list[Any],
    eval_start: int,
) -> float:
    if est_name == "TKEO" and _env_bool("FREQRAMP_TKEO_ROBUST_OBJECTIVE", True):
        return _evaluate_tkeo_rocof_score(est_cls, params, scenarios_eval, eval_start)
    return _evaluate_standard_params_score(est_cls, params, scenarios_eval, eval_start)


def _tkeo_rocof_candidates() -> list[dict[str, Any]]:
    input_values = _env_float_csv("FREQRAMP_TKEO_INPUT_SMOOTHING_GRID") or [0.04, 0.08, 0.12, 0.20, 0.35, 0.50, 1.00]
    output_values = _env_float_csv("FREQRAMP_TKEO_OUTPUT_SMOOTHING_GRID") or [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
    noise_values = _env_float_csv("FREQRAMP_TKEO_NOISE_POWER_GRID") or [0.0, 1e-7, 3e-7, 1e-6, 3e-6]
    derivative_factors = _env_float_csv("FREQRAMP_TKEO_DERIVATIVE_NOISE_FACTOR_GRID") or [1.0, 2.0]
    return [
        {
            "input_smoothing": float(inp),
            "output_smoothing": float(out),
            "noise_power": float(noise_power),
            "derivative_noise_factor": float(derivative_factor),
        }
        for inp, out, noise_power, derivative_factor in product(input_values, output_values, noise_values, derivative_factors)
        if (
            0.0 < float(inp) <= 1.0
            and 0.0 < float(out) <= 1.0
            and float(noise_power) >= 0.0
            and float(derivative_factor) >= 0.0
        )
    ]


def _small_grid_candidates(est_name: str) -> list[dict[str, Any]]:
    if est_name == "TKEO" and _env_bool("FREQRAMP_TKEO_ROBUST_GRID", True):
        return _tkeo_rocof_candidates()
    if est_name == "Koopman (RK-DPMU)" and _env_bool("FREQRAMP_KOOPMAN_ROCOF_FAST_GRID", True):
        n_cycles = _env_float_csv("FREQRAMP_KOOPMAN_N_CYCLES_GRID") or [0.5, 0.75, 1.0, 1.5, 2.0]
        return [{"n_cycles": float(c)} for c in n_cycles if float(c) > 0.0]
    if est_name == "Prony":
        orders = [2, 4, 6, 8, 10]
        n_cycles = [0.5, 1.0, 2.0, 4.0]
        return [{"order": o, "n_cycles": c} for o, c in product(orders, n_cycles)]
    if est_name == "ESPRIT":
        return [{"n_cycles": c} for c in [0.5, 1.0, 2.0, 4.0]]
    if est_name == "MUSIC":
        gains = [1e-4, 1e-3, 1e-2]
        orders = [2, 4, 6, 8]
        return [{"gain": g, "subspace_order": o} for g, o in product(gains, orders)]
    return []


def _tune_estimator_on_scenarios(
    est_name: str,
    est_cls: type,
    scenarios_eval: list[Any],
    *,
    n_trials: int,
    tune_eval_runs: int,
    mode: str,
    frequency_bounds: tuple[float, float] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    defaults: dict[str, Any] = est_cls.default_params() if hasattr(est_cls, "default_params") else {}
    defaults = _apply_rocof_frequency_bounds(est_name, est_cls, defaults, frequency_bounds)
    tuning_meta: dict[str, Any] = {
        "mode": mode,
        "policy": _tuning_policy(),
        "objective": "minimize_rmse_plus_weighted_rfe",
        "n_trials_requested": int(n_trials),
        "n_trials_executed": 0,
        "tune_eval_runs": int(max(1, tune_eval_runs)),
        "n_eval_scenarios": int(len(scenarios_eval)),
        "sampler_mode_effective": None,
        "best_objective": None,
        "frequency_bounds_hz": list(frequency_bounds) if frequency_bounds is not None else None,
    }
    if est_name == "TKEO" and _env_bool("FREQRAMP_TKEO_ROBUST_OBJECTIVE", True):
        tuning_meta["objective"] = "minimize_noisy_rmse_plus_clean_noise_bias_reversal_rfe_penalty"
    if est_name not in benchmark.SEARCH_SPACES:
        tuning_meta["reason"] = "no_search_space"
        return defaults, tuning_meta
    if n_trials <= 0:
        tuning_meta["reason"] = "n_trials<=0"
        return defaults, tuning_meta
    if not scenarios_eval:
        tuning_meta["reason"] = "no_eval_scenarios"
        return defaults, tuning_meta

    fs_dsp = 1.0 / float(scenarios_eval[0].t[1] - scenarios_eval[0].t[0])
    eval_start = int(0.15 * fs_dsp)

    grid_candidates = _small_grid_candidates(est_name)
    if grid_candidates:
        best_params = defaults
        best_loss = 1e9
        for cand in grid_candidates:
            params = _apply_rocof_frequency_bounds(est_name, est_cls, {**defaults, **cand}, frequency_bounds)
            loss = _evaluate_params_score(est_name, est_cls, params, scenarios_eval, eval_start)
            if loss < best_loss:
                best_loss = loss
                best_params = params
        tuning_meta["mode"] = f"{mode}_manual_grid"
        tuning_meta["sampler_mode_effective"] = "manual_grid"
        tuning_meta["n_trials_executed"] = len(grid_candidates)
        tuning_meta["best_objective"] = float(best_loss)
        tuning_meta["grid_candidates"] = len(grid_candidates)
        if best_loss >= 1e9:
            tuning_meta["reason"] = "all_trials_failed"
            return defaults, tuning_meta
        return best_params, tuning_meta

    space_fn = benchmark.SEARCH_SPACES[est_name]
    if not benchmark._grid_space_for_estimator(space_fn, n_trials=2):
        tuning_meta["reason"] = "empty_search_space"
        return defaults, tuning_meta

    def objective(trial: optuna.Trial) -> float:
        suggested = space_fn(trial)
        params = _apply_rocof_frequency_bounds(est_name, est_cls, {**defaults, **suggested}, frequency_bounds)
        return _evaluate_params_score(est_name, est_cls, params, scenarios_eval, eval_start)

    study, n_trials_exec, sampler_mode_effective = benchmark._build_optuna_study(
        space_fn=space_fn,
        n_trials=int(n_trials),
    )
    tuning_meta["n_trials_executed"] = int(n_trials_exec)
    tuning_meta["sampler_mode_effective"] = sampler_mode_effective
    study.optimize(objective, n_trials=n_trials_exec)
    if study.best_value >= 1e9:
        tuning_meta["reason"] = "all_trials_failed"
        return defaults, tuning_meta

    best_suggested = space_fn(study.best_trial)
    tuning_meta["best_objective"] = float(study.best_value)
    return _apply_rocof_frequency_bounds(est_name, est_cls, {**defaults, **best_suggested}, frequency_bounds), tuning_meta


def _tuning_policy() -> str:
    return os.getenv("FREQRAMP_TUNING_POLICY", "fixed_policy").strip().lower()


def _fixed_policy_enabled() -> bool:
    return _tuning_policy() == "fixed_policy"


def _enforce_standardized_step(est_name: str) -> bool:
    vectorized = set(_env_csv("FREQRAMP_VECTORIZE_ESTIMATORS") or ["PI-GRU", "Koopman (RK-DPMU)"])
    return est_name not in vectorized


def _env_key_for_estimator(est_name: str) -> str:
    return (
        est_name.upper()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
    )


def _estimator_n_mc_runs(est_name: str, default: int) -> int:
    key = _env_key_for_estimator(est_name)
    specific = os.getenv(f"FREQRAMP_{key}_N_MC_RUNS")
    if specific is not None:
        return _env_int(f"FREQRAMP_{key}_N_MC_RUNS", default, minimum=1)
    if ESTIMATOR_FAMILIES.get(est_name) == "Data-driven":
        return _env_int("FREQRAMP_DATA_DRIVEN_N_MC_RUNS", default, minimum=1)
    return int(default)


def _estimator_n_cost_reps(est_name: str, default: int) -> int:
    key = _env_key_for_estimator(est_name)
    specific = os.getenv(f"FREQRAMP_{key}_N_COST_REPS")
    if specific is not None:
        return _env_int(f"FREQRAMP_{key}_N_COST_REPS", default, minimum=1)
    if ESTIMATOR_FAMILIES.get(est_name) == "Data-driven":
        return _env_int("FREQRAMP_DATA_DRIVEN_N_COST_REPS", default, minimum=1)
    return int(default)


def _fixed_policy_train_levels() -> list[float]:
    configured = _env_float_csv("FREQRAMP_FIXED_POLICY_TRAIN_LEVELS")
    return sorted({abs(float(v)) for v in (configured if configured else [0.25, 1.0, 3.0, 10.0, 30.0, 50.0])})


def _build_fixed_policy_eval_scenarios(scenarios: list[SweepScenario], base_seed: int) -> list[Any]:
    train_levels = _fixed_policy_train_levels()
    eval_runs = _env_int("FREQRAMP_FIXED_POLICY_EVAL_RUNS_PER_LEVEL", 2, minimum=1)
    selected = [
        sc for sc in scenarios
        if any(abs(sc.abs_rocof_hz_s - target) <= max(1e-9, target * 1e-6) for target in train_levels)
    ]
    if not selected:
        selected = scenarios[:]
    eval_scenarios: list[Any] = []
    for idx, sc in enumerate(selected):
        for run_idx in range(eval_runs):
            eval_scenarios.append(sc.scenario_cls.run(seed=base_seed + 1000 * idx + run_idx))
    return eval_scenarios


def _can_reuse_existing_run(
    summary_csv: Path,
    run_spec_path: Path,
    *,
    requested_n_mc_runs: int,
    requested_tune_trials: int,
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
    if str(spec.get("tuning_policy", "")).lower() != requested_tuning_policy:
        return False
    tune_meta = spec.get("tuning_meta", {}) if isinstance(spec.get("tuning_meta", {}), dict) else {}
    try:
        trials_saved = int(tune_meta.get("n_trials_requested", -1))
    except Exception:
        trials_saved = -1
    return trials_saved == int(requested_tune_trials)


def _shade_rocof_regions(ax: plt.Axes, x_lo: float, x_hi: float, *, labels: bool = True) -> None:
    for idx, (label, lo, hi, color, _desc) in enumerate(ROCOF_REGIONS):
        band_lo = max(float(lo), x_lo)
        band_hi = min(float(hi), x_hi)
        if band_hi <= band_lo:
            continue
        ax.axvspan(band_lo, band_hi, color=color, alpha=0.07 if idx < 4 else 0.055, zorder=0)
        if labels:
            x_mid = math.sqrt(max(band_lo, 1e-9) * max(band_hi, 1e-9))
            ax.text(
                x_mid,
                0.985 - 0.04 * (idx % 3),
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


def _line_style(direction: str) -> str:
    return "-" if direction == "pos" else "--"


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

    df_metric = df_global[
        ["scenario", "abs_rocof_hz_s", "direction", "estimator", "family", metric_col]
        + [c for c in [metric_col.replace("_mean", "_p10"), metric_col.replace("_mean", "_p90")] if c in df_global.columns]
    ].copy()
    df_metric = df_metric.rename(columns={metric_col: "metric_value"})
    df_metric = df_metric.dropna(subset=["metric_value"])
    if df_metric.empty:
        fig, _ = plt.subplots(1, 1, figsize=(8, 4))
        return fig, {}

    families = ["Loop-based", "Model-based", "Window-based", "Adaptive", "Data-driven"]
    panels = ["Reference Ramp"] + families
    ticks = sorted(df_metric["abs_rocof_hz_s"].dropna().astype(float).unique().tolist())
    ncols = 2
    nrows = int(math.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12.0, 3.65 * nrows), sharex=False)
    axes_arr = np.atleast_1d(axes).ravel()
    cmap = matplotlib.colormaps["tab20"]
    est_labels = sorted(df_metric["estimator"].unique().tolist())
    color_map = {label: cmap(i % cmap.N) for i, label in enumerate(est_labels)}

    for idx, panel in enumerate(panels):
        ax = axes_arr[idx]
        if panel == "Reference Ramp":
            for rocof, ls, label in [(3.0, "-", "+3 Hz/s"), (-3.0, "--", "-3 Hz/s")]:
                sc = IEEEFreqRampScenario.run(
                    duration_s=_duration_s(),
                    rocof_hz_s=rocof,
                    t_start_s=_t_start_s(),
                    freq_cap_hz=60.0 + rocof * _ramp_duration_s(),
                    noise_sigma=0.0,
                    seed=0,
                )
                ax.plot(sc.t, sc.f_true, color="#111111", linestyle=ls, linewidth=1.35, label=label)
            ax.set_title("Reference Ramp", loc="left", fontweight="bold")
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

        for (estimator, direction), df_est in df_family.sort_values(["estimator", "direction", "abs_rocof_hz_s"]).groupby(["estimator", "direction"], sort=True):
            x_vals = df_est["abs_rocof_hz_s"].to_numpy(dtype=float)
            y_vals = df_est["metric_value"].to_numpy(dtype=float)
            if yscale == "log":
                y_vals = np.maximum(y_vals, 1e-12)
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
            _shade_rocof_regions(ax, min(ticks), max(ticks), labels=True)
        if guide_line is not None:
            ax.axhline(guide_line, color="#303F9F", linestyle="--", linewidth=0.95, label=f"Guide {guide_line:g}")
        if strict_line is not None:
            ax.axhline(strict_line, color="#00897B", linestyle="--", linewidth=0.95, label=f"Strict {strict_line:g}")
        ax.axvline(3.0, color="#7B1FA2", linestyle=":", linewidth=0.9, label="3 Hz/s ref")
        ax.axvline(5.0, color="#E65100", linestyle=":", linewidth=0.9, label="5 Hz/s ref")
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
        ax.set_xlabel("|RoCoF| [Hz/s]")
    fig.suptitle(f"{title_prefix}: by estimator family (fixed policy)", fontsize=13, y=0.995)
    _add_methodology_text(fig)
    fig.text(
        0.5,
        0.006,
        "Solid lines are positive ramps; dashed lines are negative ramps. Thresholds are interpretation guides, not formal compliance claims.",
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
        title_prefix="RMSE",
        yscale="log",
        guide_line=_env_float("FREQRAMP_LIMIT_RMSE_GUIDE", 0.05, minimum=0.0),
        strict_line=_env_float("FREQRAMP_LIMIT_RMSE_STRICT", 0.01, minimum=0.0),
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
    pivot = df.pivot_table(index="estimator", columns="abs_rocof_hz_s", values="m1_rmse_hz_mean", aggfunc="mean")
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
    ax.set_title("RoCoF Method Stress Map", loc="left", fontweight="bold")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_xticks(np.arange(len(x_vals)))
    ax.set_xticklabels([f"{v:g}" for v in x_vals], rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("|RoCoF| [Hz/s]")
    ax.set_ylabel("Estimator")
    cbar = fig.colorbar(im, ax=ax, fraction=0.024, pad=0.012)
    cbar.set_label("log10 mean RMSE [Hz]")

    ax2 = axes[1]
    guide = _env_float("FREQRAMP_LIMIT_RMSE_GUIDE", 0.05, minimum=0.0)
    rows = []
    for est, df_est in df.sort_values("abs_rocof_hz_s").groupby("estimator", sort=False):
        reduced = df_est.groupby("abs_rocof_hz_s", as_index=False)["m1_rmse_hz_mean"].mean()
        fail = reduced[reduced["m1_rmse_hz_mean"] > guide]
        critical = float(fail.iloc[0]["abs_rocof_hz_s"]) if not fail.empty else float("nan")
        rows.append((est, families.get(est, ""), critical, float(reduced["abs_rocof_hz_s"].max())))
    summary = pd.DataFrame(rows, columns=["estimator", "family", "critical_rocof", "max_rocof"])
    summary = summary.sort_values(["family", "critical_rocof"], na_position="last")
    y = np.arange(len(summary))
    x = summary["critical_rocof"].fillna(summary["max_rocof"] * 1.05).to_numpy(dtype=float)
    ax2.scatter(x, y, s=24, color="#263238")
    for i, row in enumerate(summary.itertuples(index=False)):
        ax2.text(float(x[i]) * 1.03, i, str(row.estimator), va="center", fontsize=6.4)
    ax2.axvline(guide, color="#303F9F", linestyle="--", linewidth=0.9, label=f"RMSE guide {guide:g} Hz")
    ax2.set_xscale("log")
    ax2.set_xlim(min(x_vals), max(x_vals) * 1.4)
    ax2.set_yticks([])
    ax2.set_xlabel("First |RoCoF| where mean RMSE exceeds guide [Hz/s]")
    ax2.set_title("Critical RoCoF Summary", loc="left", fontweight="bold")
    _shade_rocof_regions(ax2, min(x_vals), max(x_vals), labels=True)
    ax2.grid(True, which="both", alpha=0.22)
    ax2.legend(loc="best", fontsize=6.4, frameon=True)

    fig.suptitle("RoCoF Fixed-Policy Method Atlas", fontsize=13, y=0.995)
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
    for (est, abs_rocof), part in df_global.groupby(["estimator", "abs_rocof_hz_s"]):
        vals = part.set_index("direction")["m1_rmse_hz_mean"].to_dict()
        if "pos" not in vals or "neg" not in vals:
            continue
        pos = max(float(vals["pos"]), 1e-12)
        neg = max(float(vals["neg"]), 1e-12)
        rows.append(
            {
                "estimator": est,
                "family": str(part["family"].iloc[0]),
                "abs_rocof_hz_s": float(abs_rocof),
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
                df_est["abs_rocof_hz_s"],
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
        ax.set_xlabel("|RoCoF| [Hz/s]")
        if not df_family.empty:
            ax.legend(loc="best", fontsize=6.2, frameon=True)
    for j in range(len(families), len(axes_arr)):
        axes_arr[j].set_visible(False)
    fig.suptitle("Positive/Negative Ramp Asymmetry", fontsize=13, y=0.995)
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


def _classify_rocof_regime(df_est: pd.DataFrame) -> dict[str, Any]:
    reduced = (
        df_est.groupby("abs_rocof_hz_s", as_index=False)["m1_rmse_hz_mean"]
        .mean()
        .sort_values("abs_rocof_hz_s")
    )
    x = reduced["abs_rocof_hz_s"].to_numpy(dtype=float)
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
        out["interpretation"] = "Too few RoCoF levels for automatic classification."
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
    for _abs, part in df_est.groupby("abs_rocof_hz_s"):
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
        interp = "Positive and negative ramps produce materially different RMSE; inspect estimator model symmetry and operating bounds."
    elif ratio <= 1.35 and abs(slope) <= 0.12:
        regime = "Insensitive/flat"
        status = "claimable"
        interp = "Frequency RMSE is practically flat across the swept RoCoF range."
    elif slope < -0.15 and ratio < 0.85:
        regime = "SNR-improving"
        status = "diagnostic"
        interp = "Observed error decreases with RoCoF; likely slope/SNR or window-selection effect."
    elif slope > 0.20 and r2 >= 0.85 and large_reversals <= 1:
        regime = "Power-law-like"
        status = "claimable"
        interp = "Log-log RMSE trend is close to a power law in the non-saturated range."
    elif slope > 0.10 and monotone_fraction >= 0.75:
        regime = "Monotone growth"
        status = "claimable"
        interp = "RMSE grows mostly monotonically with RoCoF, but not as a strict power law."
    elif monotone_fraction >= 0.90 and large_reversals == 0 and ratio > 1.35:
        regime = "Weak monotone/noise-floor-limited"
        status = "claimable_with_caveat"
        interp = "RMSE is monotone but weakly shaped; interpret as noise-floor-limited or weakly RoCoF-sensitive, not erratic."
    elif ratio >= 2.0 and abs(float(np.mean(diffs[-3:]))) < 0.12:
        regime = "Saturation/plateau"
        status = "claimable"
        interp = "High-RoCoF error appears to approach a plateau or estimator rail."
    else:
        regime = "Erratic"
        status = "do_not_interpret"
        interp = "Curve has reversals or weak model support; repeat with stronger MC/tuning before claiming a law."
    out["primary_regime"] = regime
    out["claim_status"] = status
    out["interpretation"] = interp
    return out


def _save_rocof_hypothesis_tests(df_global: pd.DataFrame, out_dir: Path) -> tuple[Path, Path, Path]:
    rows = []
    continuity = _build_tuning_continuity(df_global)
    continuity_map = continuity.set_index("estimator").to_dict(orient="index") if not continuity.empty else {}
    for estimator, df_est in df_global.groupby("estimator", sort=True):
        result = _classify_rocof_regime(df_est)
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
        "x_axis": "abs_rocof_hz_s",
        "regime_counts": counts,
        "hypotheses": {
            "Insensitive/flat": "RMSE ratio and log-log slope remain small.",
            "SNR-improving": "RMSE falls as RoCoF increases.",
            "Power-law-like": "Positive log-log slope with high R2.",
            "Monotone growth": "Mostly monotone growth but weaker power-law evidence.",
            "Weak monotone/noise-floor-limited": "Monotone response with weak slope or low model support; claim trend only with caveat.",
            "Saturation/plateau": "High-RoCoF response approaches a plateau or rail.",
            "Sign-asymmetric": "Positive and negative ramps differ strongly.",
            "Erratic": "Large reversals or non-monotone behavior; do not interpret without repeat run.",
        },
    }
    json_path.write_text(json.dumps(benchmark._to_builtin(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# RoCoF Hypothesis Tests",
        "",
        f"- Artifact: `{out_dir}`",
        "- Metric: `m1_rmse_hz`",
        "- X axis: `abs_rocof_hz_s`",
        "",
        "These automatic screens classify observed trend shape. They do not smooth curves and do not create compliance claims. "
        "`Weak monotone/noise-floor-limited` is not an erratic failure: it marks a monotone but weak trend whose low-RoCoF region is dominated by estimator/noise floor.",
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


def _save_multipage_metrics_dashboard(df_global: pd.DataFrame, out_dir: Path) -> Path:
    pdf_path = out_dir / MULTIPAGE_PDF_NAME
    df_plot = _add_derived_metric_columns(df_global)
    pages = [
        ("m1_rmse_hz_mean", "RMSE [Hz]", "RMSE", "log", 0.05, 0.01),
        ("m10_rfe_rms_hz_s_mean", "RFE RMS [Hz/s]", "ROCOF error RMS", "log", 3.0, 0.4),
        ("m9_rfe_max_hz_s_mean", "RFE max [Hz/s]", "ROCOF error max", "log", 3.0, 0.4),
        ("m27_post_100ms_rmse_hz_mean", "Ramp-start 100 ms RMSE [Hz]", "Early-ramp RMSE", "log", 0.05, 0.01),
        ("m29_late_event_rmse_hz_mean", "Late ramp/hold RMSE [Hz]", "Late-window RMSE", "log", 0.05, 0.01),
        ("m30_event_settling_time_s_mean", "Event settling time [s]", "Settling time", "linear", 0.1, 0.02),
        ("m3_max_peak_hz_mean", "FE max per test [Hz]", "FE max per test", "log", 0.5, 0.01),
        ("m15_pass_rate_pct_mean", "Pass rate [%]", "Pass rate", "linear", 95.0, 99.0),
        ("m13_cpu_time_us_mean", "CPU time [us/pass]", "CPU cost", "log", None, None),
        ("m22_invalid_output_rate_pct_mean", "Invalid output rate [%]", "Invalid outputs", "log", 1.0, 0.1),
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
        "RoCoF Fixed-Policy Atlas",
        "",
        METHODOLOGY_TEXT,
        "",
        f"Estimator count: {df_global['estimator'].nunique() if not df_global.empty else 0}",
        f"RoCoF levels: {df_global['abs_rocof_hz_s'].nunique() if 'abs_rocof_hz_s' in df_global else 0}",
        f"Directions: {', '.join(sorted(df_global['direction'].unique())) if 'direction' in df_global else ''}",
        f"MC runs per pair: {int(df_global['n_mc_runs'].median()) if 'n_mc_runs' in df_global and not df_global.empty else 0}",
        "",
        "Interpretation:",
        "- Solid/dashed sign split checks whether an estimator is directionally biased.",
        "- RMSE curves test dynamic frequency tracking under sustained ramps.",
        "- RFE pages test whether a method can recover derivative information, not just frequency.",
        "- Guide lines support reading the atlas; they are not formal IEEE/IEC compliance claims.",
    ]
    ax.text(0.04, 0.95, "\n".join(lines), va="top", ha="left", fontsize=11, color="#263238", wrap=True)
    fig.tight_layout()
    return fig


def _write_manifest(out_dir: Path, scenarios: list[SweepScenario], estimators: dict[str, type], settings: dict[str, Any]) -> Path:
    payload = {
        "experiment": "rocof_sweep_fixed_policy_atlas",
        "status": "draft",
        "output_subdir": OUTPUT_SUBDIR,
        "pipeline_entrypoint": "src/pipelines/rocof_sweep_fixed_policy.py",
        "tuning_policy": _tuning_policy(),
        "methodology": METHODOLOGY_TEXT,
        "rocof_levels_hz_s": sorted({float(sc.abs_rocof_hz_s) for sc in scenarios}),
        "directions": sorted({sc.direction for sc in scenarios}),
        "signed_rocof_values_hz_s": [float(sc.rocof_hz_s) for sc in scenarios],
        "scenario_contract": {
            "duration_s": _duration_s(),
            "t_start_s_nominal": _t_start_s(),
            "ramp_duration_s": _ramp_duration_s(),
            "freq_cap_hz": "60 + rocof_hz_s * ramp_duration_s",
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
        "artifact_policy": "Commit report-level PDFs, PNGs, aggregate CSV/JSON/MD, README, and manifest. Do not commit Sweep_FreqRamp_RoCoF_* per-run simulation folders.",
    }
    path = out_dir / MANIFEST_NAME
    path.write_text(json.dumps(benchmark._to_builtin(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _write_readme(out_dir: Path) -> Path:
    path = out_dir / "README.md"
    text = f"""# RoCoF Fixed-Policy Atlas

This directory contains report-level artifacts for the frequency-ramp/RoCoF sweep.

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

Per-run folders such as `Sweep_FreqRamp_RoCoF_*` are generated simulation outputs
and should not be committed unless a specific archival run requires them.
"""
    path.write_text(text, encoding="utf-8")
    return path


def _save_summary_tables(df_global: pd.DataFrame, out_dir: Path) -> tuple[Path, Path, Path, Path]:
    global_csv = out_dir / GLOBAL_CSV_NAME
    df_global.to_csv(global_csv, index=False)
    keep_cols = [
        "scenario", "rocof_hz_s", "abs_rocof_hz_s", "direction", "estimator", "family", "n_mc_runs",
        "m1_rmse_hz_mean", "m1_rmse_hz_median", "m1_rmse_hz_p10", "m1_rmse_hz_p90", "m1_rmse_hz_std",
    ]
    rmse_cols = [col for col in keep_cols if col in df_global.columns]
    rmse_est_csv = out_dir / RMSE_EST_CSV_NAME
    df_global[rmse_cols].to_csv(rmse_est_csv, index=False)
    df_family = (
        df_global.groupby(["abs_rocof_hz_s", "direction", "family"], as_index=False)
        .agg(
            family_rmse_mean=("m1_rmse_hz_mean", "mean"),
            family_rmse_std=("m1_rmse_hz_mean", "std"),
            family_rmse_min=("m1_rmse_hz_mean", "min"),
            family_rmse_max=("m1_rmse_hz_mean", "max"),
        )
        .sort_values(["abs_rocof_hz_s", "direction", "family"])
    )
    rmse_family_csv = out_dir / RMSE_FAM_CSV_NAME
    df_family.to_csv(rmse_family_csv, index=False)
    continuity = _build_tuning_continuity(df_global)
    continuity_csv = out_dir / TUNING_CONTINUITY_CSV_NAME
    continuity.to_csv(continuity_csv, index=False)
    return global_csv, rmse_est_csv, rmse_family_csv, continuity_csv


def main() -> None:
    n_mc_runs = _env_int("FREQRAMP_SWEEP_N_MC_RUNS", _env_int("BENCHMARK_N_MC_RUNS", 20, minimum=1), minimum=1)
    base_seed = _env_int("FREQRAMP_SWEEP_BASE_SEED", 12345, minimum=0)
    tuning_base_seed = _env_int("FREQRAMP_TUNING_BASE_SEED", base_seed + 200000, minimum=0)
    resume_run = _env_bool("FREQRAMP_SWEEP_RESUME", True)
    n_cost_reps = _env_int("FREQRAMP_SWEEP_N_COST_REPS", 1, minimum=1)
    tune_trials = _env_int("FREQRAMP_SWEEP_TUNE_TRIALS", 80, minimum=0)
    tune_eval_runs = _env_int("FREQRAMP_SWEEP_TUNE_EVAL_RUNS", 2, minimum=1)
    capture_signals = _env_bool("FREQRAMP_CAPTURE_SIGNALS", False)

    settings = {
        "n_mc_runs": n_mc_runs,
        "base_seed": base_seed,
        "tuning_base_seed": tuning_base_seed,
        "resume_run": resume_run,
        "n_cost_reps": n_cost_reps,
        "tune_trials": tune_trials,
        "tune_eval_runs": tune_eval_runs,
        "capture_signals": capture_signals,
        "estimator_set": os.getenv("FREQRAMP_ESTIMATOR_SET", "journal"),
    }

    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scenarios = _build_scenarios()
    estimators = _select_estimators()
    frequency_bounds = _scenario_frequency_bounds(scenarios)
    settings["frequency_bounds_hz"] = list(frequency_bounds)
    settings["force_scenario_frequency_bounds"] = _env_bool("FREQRAMP_FORCE_SCENARIO_FREQ_BOUNDS", True)
    settings["tkeo_robust_objective"] = _env_bool("FREQRAMP_TKEO_ROBUST_OBJECTIVE", True)
    settings["tkeo_robust_grid"] = _env_bool("FREQRAMP_TKEO_ROBUST_GRID", True)
    print(f"Running fixed-policy RoCoF sweep: {len(scenarios)} scenarios x {len(estimators)} estimators")
    print(f"  MC runs per pair: {n_mc_runs}")
    print(f"  CPU timing reps per pair: {n_cost_reps}")
    print(f"  Tuning policy: {_tuning_policy()}")
    print(f"  Tuning trials: {tune_trials}")
    print(f"  Scenario frequency bounds: {frequency_bounds[0]:g}..{frequency_bounds[1]:g} Hz")
    print(f"  Output dir: {OUTPUT_DIR}")

    fixed_policy_params: dict[str, dict[str, Any]] = {}
    fixed_policy_meta: dict[str, dict[str, Any]] = {}
    if _fixed_policy_enabled():
        eval_scenarios = _build_fixed_policy_eval_scenarios(scenarios, tuning_base_seed)
        print(f"  Fixed-policy tuning scenarios: {len(eval_scenarios)}")
        for est_name, est_cls in estimators.items():
            print(f"  [tune fixed] {est_name}", flush=True)
            params, meta = _tune_estimator_on_scenarios(
                est_name,
                est_cls,
                eval_scenarios,
                n_trials=tune_trials,
                tune_eval_runs=tune_eval_runs,
                mode="fixed_policy_global",
                frequency_bounds=frequency_bounds,
            )
            fixed_policy_params[est_name] = params
            fixed_policy_meta[est_name] = meta

    rows_agg: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    for sc in scenarios:
        print(f"\nScenario {sc.scenario_name} (RoCoF={sc.rocof_hz_s:+g} Hz/s)")
        sc_dir = OUTPUT_DIR / sc.scenario_name
        sc_dir.mkdir(parents=True, exist_ok=True)
        for est_name, est_cls in estimators.items():
            est_n_mc_runs = _estimator_n_mc_runs(est_name, n_mc_runs)
            est_n_cost_reps = _estimator_n_cost_reps(est_name, n_cost_reps)
            out_dir = sc_dir / est_name
            out_dir.mkdir(parents=True, exist_ok=True)
            summary_csv = out_dir / f"{sc.scenario_name}__{est_name}_summary.csv"
            run_spec_path = out_dir / "run_spec.json"
            best_params: dict[str, Any]
            tuning_meta: dict[str, Any]
            if _fixed_policy_enabled():
                best_params = dict(fixed_policy_params.get(est_name, {}))
                tuning_meta = dict(fixed_policy_meta.get(est_name, {}))
            else:
                eval_scenarios = [sc.scenario_cls.run(seed=tuning_base_seed + i) for i in range(max(1, tune_eval_runs))]
                best_params, tuning_meta = _tune_estimator_on_scenarios(
                    est_name,
                    est_cls,
                    eval_scenarios,
                    n_trials=tune_trials,
                    tune_eval_runs=tune_eval_runs,
                    mode="per_scenario_oracle",
                    frequency_bounds=frequency_bounds,
                )
            best_params = _apply_rocof_frequency_bounds(est_name, est_cls, best_params, frequency_bounds)

            if resume_run and _can_reuse_existing_run(
                summary_csv,
                run_spec_path,
                requested_n_mc_runs=est_n_mc_runs,
                requested_tune_trials=tune_trials,
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
                    "rocof_hz_s": float(sc.rocof_hz_s),
                    "abs_rocof_hz_s": float(sc.abs_rocof_hz_s),
                    "direction": sc.direction,
                    "estimator": est_name,
                    "family": ESTIMATOR_FAMILIES.get(est_name, "Unknown"),
                    "best_params": benchmark._to_builtin(best_params),
                    "tuning_meta": benchmark._to_builtin(tuning_meta),
                    "tuning_policy": _tuning_policy(),
                    "frequency_bounds_hz": list(frequency_bounds),
                    "n_mc_runs": int(est_n_mc_runs),
                    "n_cost_reps": int(est_n_cost_reps),
                    "enforce_standardized_step": _enforce_standardized_step(est_name),
                    "base_seed": int(base_seed),
                    "timing": benchmark._to_builtin(timing),
                }
                run_spec_path.write_text(json.dumps(benchmark._to_builtin(run_spec), indent=2, ensure_ascii=False), encoding="utf-8")

            agg = _aggregate_summary(summary_df)
            best_params_json = json.dumps(benchmark._to_builtin(best_params), sort_keys=True, ensure_ascii=False)
            rows_agg.append(
                {
                    "scenario": sc.scenario_name,
                    "rocof_hz_s": float(sc.rocof_hz_s),
                    "abs_rocof_hz_s": float(sc.abs_rocof_hz_s),
                    "direction": sc.direction,
                    "estimator": est_name,
                    "family": ESTIMATOR_FAMILIES.get(est_name, "Unknown"),
                    "n_mc_runs": int(len(summary_df)),
                    "tuning_policy": _tuning_policy(),
                    "best_params_json": best_params_json,
                    **agg,
                }
            )
            timing_rows.append(
                {
                    "scenario": sc.scenario_name,
                    "rocof_hz_s": float(sc.rocof_hz_s),
                    "abs_rocof_hz_s": float(sc.abs_rocof_hz_s),
                    "direction": sc.direction,
                    "estimator": est_name,
                    "family": ESTIMATOR_FAMILIES.get(est_name, "Unknown"),
                    "n_mc_runs": int(len(summary_df)),
                    "tune_trials": int(tuning_meta.get("n_trials_requested", tune_trials) or 0),
                    "n_cost_reps": int(est_n_cost_reps),
                    "total_elapsed_s": timing.get("total_elapsed_s") if isinstance(timing, dict) else None,
                }
            )

    df_global = pd.DataFrame(rows_agg).sort_values(["abs_rocof_hz_s", "direction", "family", "estimator"])
    global_csv, rmse_est_csv, rmse_family_csv, continuity_csv = _save_summary_tables(df_global, OUTPUT_DIR)
    timing_csv = OUTPUT_DIR / TIMING_CSV_NAME
    pd.DataFrame(timing_rows).to_csv(timing_csv, index=False)
    plot_paths, color_map = _save_rmse_by_family_plot(df_global, OUTPUT_DIR)
    plot_paths += _save_method_map(df_global, OUTPUT_DIR)
    plot_paths += _save_sign_asymmetry_diagnostic(df_global, OUTPUT_DIR)
    hypothesis_paths = _save_rocof_hypothesis_tests(df_global, OUTPUT_DIR)
    multipage_pdf = _save_multipage_metrics_dashboard(df_global, OUTPUT_DIR)
    legend_path = OUTPUT_DIR / "rmse_plot_method_legend.csv"
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
    print(f"\n[DONE] RoCoF sweep completed in {elapsed:.1f} min.")


if __name__ == "__main__":
    main()
