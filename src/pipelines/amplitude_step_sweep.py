from __future__ import annotations

import json
import math
import os
import sys
import time
import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import optuna
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter

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
from scenarios.ieee_mag_step import IEEEMagStepScenario


OUTPUT_SUBDIR = os.getenv("ASTEP_OUTPUT_SUBDIR", "amplitude_step_sweep")
OUTPUT_DIR = ROOT / "artifacts" / OUTPUT_SUBDIR
MANIFEST_NAME = "scenario_manifest.json"
GLOBAL_CSV_NAME = "global_metrics_report.csv"
RMSE_EST_CSV_NAME = "rmse_by_estimator.csv"
RMSE_FAM_CSV_NAME = "rmse_by_family.csv"
PLOT_NAME = "rmse_deterioration_by_family"
MULTIPAGE_PDF_NAME = "metrics_dashboard_multipage.pdf"
LEGEND_MAP_CSV_NAME = "rmse_plot_method_legend.csv"
SUMMARY_MAP_PDF_NAME = "amplitude_step_method_map.pdf"
SUMMARY_MAP_PNG_NAME = "amplitude_step_method_map.png"

# Journal-style stress-atlas schedule: log-spaced enough to reveal knees,
# while preserving standard-like and protection-relevant anchor points.
STEP_LEVELS_PERCENT: tuple[float, ...] = (
    1.0, 2.0, 3.0, 5.0, 7.5,
    10.0, 15.0, 20.0, 25.0, 35.0,
    50.0, 75.0, 100.0, 150.0, 200.0,
    250.0, 350.0, 500.0, 750.0, 1000.0,
)

AMPLITUDE_STEP_REGIONS: tuple[tuple[str, float, float, str, str], ...] = (
    ("Voltage PMU", 1.0, 10.0, "#66BB6A", "standard-like amplitude perturbation"),
    ("Grid stress", 10.0, 25.0, "#DCE775", "traditional grid voltage/current stress"),
    ("IBR normal", 25.0, 100.0, "#FDD835", "IBR ride-through and controller interaction"),
    ("IBR stress", 100.0, 500.0, "#FFB74D", "fault-current and fast-tracking stress"),
    ("Mega stress", 500.0, 1000.0, "#EF5350", "faults, islanding, protection transients"),
)

SLOW_ESTIMATORS = {"Prony", "ESPRIT", "Koopman (RK-DPMU)", "PI-GRU", "MUSIC"}
FIXED_MODEL_ESTIMATORS = {"PI-GRU"}
PIPELINE_METHOD_VERSION = "amplitude_step_oracle_v2_2026_05_03"
ORACLE_LABEL = "per_scenario_practical_lower_bound"
UKF_ORACLE_LABEL = "ukf_per_scenario_oracle"
KALMAN_ORACLE_ESTIMATORS = {"EKF", "UKF", "LKF", "LKF2", "RA-EKF"}
DEFAULT_TUNING_SEED_OFFSET = 1_000_000

METHODOLOGY_TEXT = (
    "This atlas treats the disturbance as an amplitude-only step: the true frequency remains nominal, "
    "so reported frequency and ROCOF errors quantify AM-to-FM coupling, numerical robustness, and "
    "tracking stability rather than formal PMU magnitude-step compliance. The 1-1000% schedule combines "
    "standard-like voltage perturbations, grid and IBR ride-through stress, and high-amplitude current/fault "
    "surrogates; horizontal lines are engineering guide thresholds, not IEEE compliance limits."
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
]


@dataclass(frozen=True)
class SweepScenario:
    step_percent: float
    scenario_cls: type
    scenario_name: str


@dataclass(frozen=True)
class EstimatorRunConfig:
    n_mc_runs: int
    tune_trials: int
    tune_eval_runs: int
    n_cost_reps: int
    tier: str


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, value)


def _env_float(name: str, default: float, minimum: float | None = None) -> float:
    raw = os.getenv(name)
    if raw is None:
        value = default
    else:
        try:
            value = float(raw)
        except ValueError:
            value = default
    if minimum is not None and value < minimum:
        return minimum
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_int_alias(primary: str, legacy: str, default: int, minimum: int = 0) -> int:
    if os.getenv(primary) is not None:
        return _env_int(primary, default, minimum=minimum)
    return _env_int(legacy, default, minimum=minimum)


def _env_float_alias(primary: str, legacy: str, default: float, minimum: float | None = None) -> float:
    if os.getenv(primary) is not None:
        return _env_float(primary, default, minimum=minimum)
    return _env_float(legacy, default, minimum=minimum)


def _env_bool_alias(primary: str, legacy: str, default: bool) -> bool:
    if os.getenv(primary) is not None:
        return _env_bool(primary, default)
    return _env_bool(legacy, default)


def _env_csv(name: str) -> list[str]:
    raw = os.getenv(name)
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _csv_set(name: str) -> set[str]:
    return {item.strip() for item in _env_csv(name) if item.strip()}


def _tuning_base_seed(base_seed: int) -> int:
    offset = _env_int("ASTEP_TUNE_SEED_OFFSET", DEFAULT_TUNING_SEED_OFFSET, minimum=0)
    return int(base_seed) + int(offset)


def _sanitize_token(value: float) -> str:
    token = f"{value:g}".replace(".", "p")
    return token.replace("-", "m")


def _create_step_variant(step_percent: float) -> SweepScenario:
    token = _sanitize_token(step_percent)
    scenario_name = f"Sweep_AmplitudeStep_{token}pct"
    class_name = f"SweepAmplitudeStep{token}"
    step_pu = float(step_percent) / 100.0

    attrs = {
        "SCENARIO_NAME": scenario_name,
        "DEFAULT_PARAMS": {
            **IEEEMagStepScenario.DEFAULT_PARAMS,
            "duration_s": 1.8,
            "amp_pre_pu": 1.0,
            "amp_post_pu": 1.0 + step_pu,
            "t_step_s": 0.50,
            "noise_sigma": 0.001,
        },
        "MONTE_CARLO_SPACE": {
            "phase_rad": {"kind": "uniform", "low": 0.0, "high": 2.0 * math.pi},
            "noise_sigma": {"kind": "uniform", "low": 0.0005, "high": 0.0020},
            "t_step_s": {"kind": "uniform", "low": 0.45, "high": 0.55},
        },
        "STEP_PERCENT": float(step_percent),
        "get_name": classmethod(lambda cls: cls.SCENARIO_NAME),
    }

    new_cls = type(class_name, (IEEEMagStepScenario,), attrs)
    new_cls.__module__ = __name__
    globals()[class_name] = new_cls
    return SweepScenario(step_percent=float(step_percent), scenario_cls=new_cls, scenario_name=scenario_name)


def _build_step_levels() -> list[float]:
    return sorted({round(float(v), 6) for v in STEP_LEVELS_PERCENT if 0.0 < float(v) <= 1000.0})


def _build_scenarios() -> list[SweepScenario]:
    scenarios = [_create_step_variant(v) for v in _build_step_levels()]
    include_names = set(_env_csv("ASTEP_SWEEP_INCLUDE_SCENARIOS") or _env_csv("VSTEP_SWEEP_INCLUDE_SCENARIOS"))
    if include_names:
        scenarios = [sc for sc in scenarios if sc.scenario_name in include_names]
    return scenarios


def _run_config_for_estimator(
    est_name: str,
    *,
    fast_n_mc_runs: int,
    fast_tune_trials: int,
    fast_tune_eval_runs: int,
    fast_n_cost_reps: int,
) -> EstimatorRunConfig:
    if est_name not in SLOW_ESTIMATORS:
        return EstimatorRunConfig(
            n_mc_runs=fast_n_mc_runs,
            tune_trials=fast_tune_trials,
            tune_eval_runs=fast_tune_eval_runs,
            n_cost_reps=fast_n_cost_reps,
            tier="fast",
        )

    default_trials = 0 if est_name in FIXED_MODEL_ESTIMATORS else 20
    if est_name == "Prony":
        default_trials = 25
    elif est_name == "ESPRIT":
        default_trials = 5
    elif est_name == "MUSIC":
        default_trials = 25

    return EstimatorRunConfig(
        n_mc_runs=_env_int(f"ASTEP_SLOW_{_sanitize_env_key(est_name)}_N_MC_RUNS", _env_int("ASTEP_SLOW_N_MC_RUNS", 15, minimum=1), minimum=1),
        tune_trials=_env_int(f"ASTEP_SLOW_{_sanitize_env_key(est_name)}_TUNE_TRIALS", _env_int("ASTEP_SLOW_TUNE_TRIALS", default_trials, minimum=0), minimum=0),
        tune_eval_runs=_env_int(f"ASTEP_SLOW_{_sanitize_env_key(est_name)}_TUNE_EVAL_RUNS", _env_int("ASTEP_SLOW_TUNE_EVAL_RUNS", 5, minimum=1), minimum=1),
        n_cost_reps=_env_int(f"ASTEP_SLOW_{_sanitize_env_key(est_name)}_N_COST_REPS", _env_int("ASTEP_SLOW_N_COST_REPS", 5, minimum=1), minimum=1),
        tier="slow",
    )


def _sanitize_env_key(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.upper()).strip("_")


def _select_estimators() -> dict[str, type]:
    estimators = load_active_estimators()
    include_raw = (os.getenv("ASTEP_SWEEP_INCLUDE_ESTIMATORS") or os.getenv("VSTEP_SWEEP_INCLUDE_ESTIMATORS") or "").strip()
    exclude_raw = (os.getenv("ASTEP_SWEEP_EXCLUDE_ESTIMATORS") or os.getenv("VSTEP_SWEEP_EXCLUDE_ESTIMATORS") or "").strip()
    by_lower = {label.lower(): label for label in estimators}

    def _norm(text: str) -> str:
        return "".join(ch for ch in text.lower() if ch.isalnum())

    if include_raw:
        selected: set[str] = set()
        for item in [x.strip() for x in include_raw.split(",") if x.strip()]:
            hit = by_lower.get(item.lower())
            if not hit:
                item_norm = _norm(item)
                for label in estimators:
                    label_norm = _norm(label)
                    if item_norm and (item_norm in label_norm or label_norm in item_norm):
                        hit = label
                        break
            if hit:
                selected.add(hit)
        if not selected:
            raise ValueError("ASTEP_SWEEP_INCLUDE_ESTIMATORS did not match any estimator.")
        estimators = {k: v for k, v in estimators.items() if k in selected}

    if exclude_raw:
        excluded: set[str] = set()
        for item in [x.strip() for x in exclude_raw.split(",") if x.strip()]:
            item_norm = _norm(item)
            for label in estimators:
                label_norm = _norm(label)
                if item.lower() == label.lower() or (item_norm and (item_norm in label_norm or label_norm in item_norm)):
                    excluded.add(label)
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
        valid = series.dropna()
        if valid.empty:
            continue
        valid_f = valid.astype(float)
        row[f"{metric}_mean"] = float(valid_f.mean())
        row[f"{metric}_median"] = float(valid_f.median())
        row[f"{metric}_p10"] = float(np.percentile(valid_f, 10))
        row[f"{metric}_p90"] = float(np.percentile(valid_f, 90))
        row[f"{metric}_std"] = float(valid_f.std(ddof=1)) if len(valid_f) > 1 else 0.0
    return row


def _run_engine_local(engine: MonteCarloEngine) -> MonteCarloResult:
    summary_rows: list[dict[str, Any]] = []
    signal_dfs: list[pd.DataFrame] = []
    for run_idx in range(engine.n_runs):
        row, signal_df = engine.run_once(run_idx)
        summary_rows.append(row)
        if engine.capture_signals and not signal_df.empty:
            signal_dfs.append(signal_df)
    summary_df = pd.DataFrame(summary_rows).sort_values(by="run_idx").reset_index(drop=True)
    if signal_dfs:
        signals_df = pd.concat(signal_dfs, ignore_index=True).sort_values(by=["run_idx", "t_s"]).reset_index(drop=True)
    else:
        signals_df = pd.DataFrame()
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


def _can_reuse_existing_run(
    summary_csv: Path,
    run_spec_path: Path,
    *,
    scenario_name: str,
    estimator_name: str,
    step_percent: float,
    requested_n_mc_runs: int,
    requested_tune_trials: int,
    requested_tune_eval_runs: int,
    requested_n_cost_reps: int,
    requested_base_seed: int,
    requested_tuning_base_seed: int,
) -> bool:
    if not summary_csv.exists() or not run_spec_path.exists():
        return False
    try:
        spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if spec.get("scenario") != scenario_name or spec.get("estimator") != estimator_name:
        return False
    if spec.get("pipeline_method_version") != PIPELINE_METHOD_VERSION:
        return False
    try:
        if abs(float(spec.get("step_percent", float("nan"))) - float(step_percent)) > 1e-12:
            return False
    except Exception:
        return False
    try:
        n_mc_saved = int(spec.get("n_mc_runs", -1))
    except Exception:
        return False
    if n_mc_saved != int(requested_n_mc_runs):
        return False
    try:
        n_cost_saved = int(spec.get("n_cost_reps", -1))
    except Exception:
        n_cost_saved = -1
    if n_cost_saved != int(requested_n_cost_reps):
        return False
    try:
        seed_saved = int(spec.get("base_seed", -1))
    except Exception:
        seed_saved = -1
    if seed_saved != int(requested_base_seed):
        return False
    try:
        tuning_seed_saved = int(spec.get("tuning_base_seed", -1))
    except Exception:
        tuning_seed_saved = -1
    if tuning_seed_saved != int(requested_tuning_base_seed):
        return False
    tune_meta = spec.get("tuning_meta", {}) if isinstance(spec.get("tuning_meta", {}), dict) else {}
    try:
        trials_saved = int(tune_meta.get("n_trials_requested", -1))
    except Exception:
        trials_saved = -1
    if trials_saved != int(requested_tune_trials):
        return False
    try:
        eval_saved = int(tune_meta.get("tune_eval_runs", -1))
    except Exception:
        eval_saved = -1
    if eval_saved != int(requested_tune_eval_runs):
        return False
    if _oracle_enabled(estimator_name):
        expected_mode = UKF_ORACLE_LABEL if estimator_name == "UKF" else ORACLE_LABEL
        if tune_meta.get("mode") != expected_mode:
            return False
        expected_effective = _oracle_trial_count(estimator_name, int(requested_tune_trials))
        try:
            effective_saved = int(tune_meta.get("n_trials_effective_requested", -1))
        except Exception:
            effective_saved = -1
        if effective_saved != int(expected_effective):
            return False
    return True


def _tuning_objective_mode() -> str:
    return os.getenv("ASTEP_TUNE_OBJECTIVE", "rmse_p90_oracle").strip().lower()


def _evaluate_params_rmse(
    est_cls: type,
    params: dict[str, Any],
    scenarios_eval: list[Any],
    eval_start: int,
    *,
    objective_mode: str | None = None,
) -> float:
    try:
        rmses: list[float] = []
        peaks: list[float] = []
        rfe_maxes: list[float] = []
        fail_count = 0
        fail_peak_hz = _env_float_alias("ASTEP_TUNE_FAIL_PEAK_HZ", "VSTEP_TUNE_FAIL_PEAK_HZ", 10.0, minimum=0.0)
        fail_rfe_hz_s = _env_float_alias("ASTEP_TUNE_FAIL_RFE_HZ_S", "VSTEP_TUNE_FAIL_RFE_HZ_S", 100.0, minimum=0.0)
        for sc in scenarios_eval:
            est = est_cls(**params)
            f_hat = benchmark._run_estimator(est, sc.v)
            error = f_hat[eval_start:] - sc.f_true[eval_start:]
            rmse = float(np.sqrt(np.mean(error ** 2)))
            peak = float(np.max(np.abs(error)))
            dt = float(sc.t[1] - sc.t[0])
            rfe_max = float(np.max(np.abs(np.diff(error) / dt))) if len(error) > 1 else 0.0
            if not np.isfinite(rmse):
                return 1e6
            if (peak > fail_peak_hz) or (rfe_max > fail_rfe_hz_s):
                fail_count += 1
            rmses.append(rmse)
            peaks.append(peak)
            rfe_maxes.append(rfe_max)
        if not rmses:
            return 1e6
        rmse_med = float(np.median(rmses))
        rmse_p90 = float(np.quantile(rmses, 0.90))
        peak_med = float(np.median(peaks)) if peaks else 0.0
        rfe_p90 = float(np.quantile(rfe_maxes, 0.90)) if rfe_maxes else 0.0
        mode = (objective_mode or _tuning_objective_mode()).strip().lower()
        if mode in {"legacy", "legacy_robust", "robust"}:
            score = (
                rmse_med
                + (0.60 * rmse_p90)
                + (0.05 * peak_med)
                + (0.001 * rfe_p90)
                + (25.0 * float(fail_count))
            )
        elif mode in {"rmse_mean", "mean_rmse"}:
            score = float(np.mean(rmses)) + (5.0 * float(fail_count))
        else:
            # Oracle objective: mainly minimize frequency RMSE while keeping
            # enough tail pressure to avoid brittle seed-specific optima.
            score = (
                rmse_med
                + (0.25 * rmse_p90)
                + (0.02 * peak_med)
                + (5.0 * float(fail_count))
            )
        return float(score) if np.isfinite(score) else 1e6
    except Exception:
        return 1e6


def _small_grid_candidates(est_name: str) -> list[dict[str, Any]]:
    if est_name == "Prony":
        orders = [2, 4, 6, 8, 10]
        n_cycles = [0.5, 1.0, 2.0, 4.0, 8.0]
        return [{"order": o, "n_cycles": c} for o, c in product(orders, n_cycles)]
    if est_name == "ESPRIT":
        n_cycles = [0.5, 1.0, 2.0, 4.0, 8.0]
        return [{"n_cycles": c} for c in n_cycles]
    if est_name == "MUSIC":
        gains = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
        orders = [2, 4, 6, 8, 10]
        return [{"gain": g, "subspace_order": o} for g, o in product(gains, orders)]
    return []


def _oracle_enabled(est_name: str) -> bool:
    if not _env_bool("ASTEP_ORACLE_TUNING", True):
        return False
    if est_name == "UKF" and not _env_bool("ASTEP_UKF_ORACLE_TUNING", True):
        return False
    include = _csv_set("ASTEP_ORACLE_ESTIMATORS")
    if not include:
        include = set(KALMAN_ORACLE_ESTIMATORS)
    include_norm = {_normalize_estimator_token(x) for x in include}
    return _normalize_estimator_token(est_name) in include_norm


def _ukf_oracle_enabled() -> bool:
    return _oracle_enabled("UKF")


def _normalize_estimator_token(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _oracle_trial_count(est_name: str, requested_trials: int) -> int:
    key = _sanitize_env_key(est_name)
    default_floor = _env_int("ASTEP_ORACLE_TRIALS", max(300, int(requested_trials)), minimum=1)
    if est_name == "UKF":
        return _env_int("ASTEP_UKF_ORACLE_TRIALS", max(default_floor, int(requested_trials)), minimum=1)
    return _env_int(f"ASTEP_{key}_ORACLE_TRIALS", max(default_floor, int(requested_trials)), minimum=1)


def _ukf_oracle_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "q_dc": trial.suggest_float("q_dc", 1e-12, 1e-1, log=True),
        "q_alpha": trial.suggest_float("q_alpha", 1e-12, 1e1, log=True),
        "q_beta": trial.suggest_float("q_beta", 1e-12, 1e1, log=True),
        "q_omega": trial.suggest_float("q_omega", 1e-12, 1e2, log=True),
        "r_meas": trial.suggest_float("r_meas", 1e-8, 1e3, log=True),
        "output_smoothing": trial.suggest_float("output_smoothing", 1e-5, 0.7, log=True),
        "alpha_ut": trial.suggest_float("alpha_ut", 0.05, 1.0, log=True),
        "beta_ut": trial.suggest_float("beta_ut", 1.0, 4.0),
        "kappa_ut": trial.suggest_float("kappa_ut", -1.0, 3.0),
        "p_dc": trial.suggest_float("p_dc", 1e-4, 1e3, log=True),
        "p_alpha": trial.suggest_float("p_alpha", 1e-4, 1e3, log=True),
        "p_beta": trial.suggest_float("p_beta", 1e-4, 1e3, log=True),
        "p_omega_hz": trial.suggest_float("p_omega_hz", 0.01, 25.0, log=True),
    }


def _ekf_oracle_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "q_dc": trial.suggest_float("q_dc", 1e-12, 1e-1, log=True),
        "q_alpha": trial.suggest_float("q_alpha", 1e-12, 1e1, log=True),
        "q_beta": trial.suggest_float("q_beta", 1e-12, 1e1, log=True),
        "q_omega": trial.suggest_float("q_omega", 1e-12, 1e2, log=True),
        "r_meas": trial.suggest_float("r_meas", 1e-8, 1e3, log=True),
        "output_smoothing": trial.suggest_float("output_smoothing", 1e-5, 0.7, log=True),
        "p_dc": trial.suggest_float("p_dc", 1e-4, 1e3, log=True),
        "p_alpha": trial.suggest_float("p_alpha", 1e-4, 1e3, log=True),
        "p_beta": trial.suggest_float("p_beta", 1e-4, 1e3, log=True),
        "p_omega_hz": trial.suggest_float("p_omega_hz", 0.01, 25.0, log=True),
    }


def _lkf_oracle_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "q": trial.suggest_float("q", 1e-12, 1e1, log=True),
        "r": trial.suggest_float("r", 1e-8, 1e3, log=True),
        "rho": trial.suggest_float("rho", 0.88, 1.0),
        "output_smoothing": trial.suggest_float("output_smoothing", 1e-5, 0.7, log=True),
        "phase_lag_samples": trial.suggest_int("phase_lag_samples", 1, 120),
        "p_x1": trial.suggest_float("p_x1", 1e-4, 1e4, log=True),
        "p_x2": trial.suggest_float("p_x2", 1e-4, 1e4, log=True),
    }


def _lkf2_oracle_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "q_dc": trial.suggest_float("q_dc", 1e-12, 1e0, log=True),
        "q_vc": trial.suggest_float("q_vc", 1e-12, 1e2, log=True),
        "q_vs": trial.suggest_float("q_vs", 1e-12, 1e2, log=True),
        "r": trial.suggest_float("r", 1e-8, 1e3, log=True),
        "beta": trial.suggest_float("beta", 1e-2, 2e3, log=True),
        "lpf_mu": trial.suggest_float("lpf_mu", 0.01, 1.0),
        "p0": trial.suggest_float("p0", 1e-4, 1e5, log=True),
        "x0_init": trial.suggest_float("x0_init", -0.25, 0.25),
        "x1_init": trial.suggest_float("x1_init", -2.0, 2.0),
        "x2_init": trial.suggest_float("x2_init", -2.0, 2.0),
    }


def _ra_ekf_oracle_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "q_theta": trial.suggest_float("q_theta", 1e-12, 1e-1, log=True),
        "q_omega": trial.suggest_float("q_omega", 1e-12, 1e1, log=True),
        "q_A": trial.suggest_float("q_A", 1e-12, 1e0, log=True),
        "q_rocof": trial.suggest_float("q_rocof", 1e-10, 1e2, log=True),
        "r_meas": trial.suggest_float("r_meas", 1e-8, 1e3, log=True),
        "sigma_v": trial.suggest_float("sigma_v", 1e-4, 20.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.5, 100.0, log=True),
        "deriv_lpf_alpha": trial.suggest_float("deriv_lpf_alpha", 0.001, 0.9),
        "tau_rocof": trial.suggest_float("tau_rocof", 0.005, 2.0, log=True),
        "freq_min_hz": trial.suggest_float("freq_min_hz", 20.0, 55.0),
        "freq_max_hz": trial.suggest_float("freq_max_hz", 65.0, 120.0),
        "amp_min": trial.suggest_float("amp_min", 1e-4, 0.25, log=True),
        "amp_max": trial.suggest_float("amp_max", 2.0, 25.0, log=True),
        "rocof_limit_hz_s": trial.suggest_float("rocof_limit_hz_s", 1.0, 200.0, log=True),
        "p_theta": trial.suggest_float("p_theta", 1e-4, 10.0, log=True),
        "p_omega_hz": trial.suggest_float("p_omega_hz", 0.01, 25.0, log=True),
        "p_amp": trial.suggest_float("p_amp", 1e-4, 25.0, log=True),
        "p_rocof_hz_s": trial.suggest_float("p_rocof_hz_s", 0.01, 100.0, log=True),
    }


ORACLE_SEARCH_SPACES: dict[str, Any] = {
    "EKF": _ekf_oracle_space,
    "UKF": _ukf_oracle_space,
    "LKF": _lkf_oracle_space,
    "LKF2": _lkf2_oracle_space,
    "RA-EKF": _ra_ekf_oracle_space,
}


def _tune_estimator_for_scenario(est_name: str, est_cls: type, scenario_cls: type, *, n_trials: int, tune_eval_runs: int, base_seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    defaults: dict[str, Any] = est_cls.default_params() if hasattr(est_cls, "default_params") else {}
    tuning_meta: dict[str, Any] = {
        "mode": "per_scenario_best_tuning",
        "objective": _tuning_objective_mode(),
        "n_trials_requested": int(n_trials),
        "n_trials_executed": 0,
        "tune_eval_runs": int(max(1, tune_eval_runs)),
        "tuning_base_seed": int(base_seed),
        "sampler_mode_effective": None,
        "best_objective": None,
    }
    is_oracle = _oracle_enabled(est_name) and est_name in ORACLE_SEARCH_SPACES
    if is_oracle:
        tuning_meta["mode"] = UKF_ORACLE_LABEL if est_name == "UKF" else ORACLE_LABEL
        tuning_meta["interpretation"] = "per-scenario practical lower-bound tuning; not deployment-transferable"
        tuning_meta["search_space"] = f"{est_name}_scenario_oracle_space"
        tuning_meta["n_trials_requested_base"] = int(n_trials)
        n_trials = max(int(n_trials), _oracle_trial_count(est_name, int(n_trials)))
        tuning_meta["n_trials_effective_requested"] = int(n_trials)

    if est_name not in benchmark.SEARCH_SPACES:
        tuning_meta["reason"] = "no_search_space"
        return defaults, tuning_meta
    if n_trials <= 0:
        tuning_meta["reason"] = "n_trials<=0"
        return defaults, tuning_meta

    space_fn = ORACLE_SEARCH_SPACES[est_name] if is_oracle else benchmark.SEARCH_SPACES[est_name]
    if not benchmark._grid_space_for_estimator(space_fn, n_trials=2):
        tuning_meta["reason"] = "empty_search_space"
        return defaults, tuning_meta

    scenarios_eval = [scenario_cls.run(seed=base_seed + i) for i in range(max(1, tune_eval_runs))]
    fs_dsp = 1.0 / float(scenarios_eval[0].t[1] - scenarios_eval[0].t[0])
    eval_start = int(0.15 * fs_dsp)

    grid_candidates = _small_grid_candidates(est_name)
    if grid_candidates:
        best_params = defaults
        best_loss = 1e6
        for cand in grid_candidates:
            params = {**defaults, **cand}
            loss = _evaluate_params_rmse(est_cls, params, scenarios_eval, eval_start, objective_mode=tuning_meta["objective"])
            if loss < best_loss:
                best_loss = loss
                best_params = params
        tuning_meta["mode"] = "discrete_small_grid"
        tuning_meta["sampler_mode_effective"] = "manual_grid"
        tuning_meta["n_trials_executed"] = len(grid_candidates)
        tuning_meta["best_objective"] = float(best_loss)
        tuning_meta["grid_candidates"] = len(grid_candidates)
        if best_loss >= 1e6:
            tuning_meta["reason"] = "all_trials_failed"
            return defaults, tuning_meta
        return best_params, tuning_meta

    def objective(trial: optuna.Trial) -> float:
        suggested = space_fn(trial)
        params = {**defaults, **suggested}
        return _evaluate_params_rmse(est_cls, params, scenarios_eval, eval_start, objective_mode=tuning_meta["objective"])

    study, n_trials_exec, sampler_mode_effective = benchmark._build_optuna_study(space_fn=space_fn, n_trials=int(n_trials))
    tuning_meta["n_trials_executed"] = int(n_trials_exec)
    tuning_meta["sampler_mode_effective"] = sampler_mode_effective
    study.optimize(objective, n_trials=n_trials_exec)
    if study.best_value >= 1e6:
        tuning_meta["reason"] = "all_trials_failed"
        return defaults, tuning_meta

    best_suggested = space_fn(study.best_trial)
    tuning_meta["best_objective"] = float(study.best_value)
    return {**defaults, **best_suggested}, tuning_meta


def _plot_metric_by_family_page(*, df_global: pd.DataFrame, metric_col: str, metric_label: str, title_prefix: str, yscale: str, ieee_line: float | None, iec_line: float | None) -> tuple[plt.Figure, dict[str, tuple[float, float, float, float]]]:
    if df_global.empty or metric_col not in df_global.columns:
        fig, _ = plt.subplots(1, 1, figsize=(8, 4))
        return fig, {}

    central_col = metric_col.replace("_mean", "_median") if metric_col.endswith("_mean") else metric_col
    if central_col not in df_global.columns:
        central_col = metric_col
    p10_col = metric_col.replace("_mean", "_p10") if metric_col.endswith("_mean") else ""
    p90_col = metric_col.replace("_mean", "_p90") if metric_col.endswith("_mean") else ""
    cols = ["scenario", "step_percent", "estimator", "family", metric_col]
    if central_col not in cols:
        cols.append(central_col)
    if p10_col and p10_col in df_global.columns:
        cols.append(p10_col)
    if p90_col and p90_col in df_global.columns:
        cols.append(p90_col)
    df_metric = df_global[cols].copy()
    df_metric = df_metric.rename(columns={central_col: "metric_value"}).dropna(subset=["metric_value"])
    if p10_col and p10_col in df_metric.columns:
        df_metric = df_metric.rename(columns={p10_col: "metric_p10"})
    else:
        df_metric["metric_p10"] = df_metric["metric_value"]
    if p90_col and p90_col in df_metric.columns:
        df_metric = df_metric.rename(columns={p90_col: "metric_p90"})
    else:
        df_metric["metric_p90"] = df_metric["metric_value"]
    if df_metric.empty:
        fig, _ = plt.subplots(1, 1, figsize=(8, 4))
        return fig, {}

    family_order = ["Loop-based", "Model-based", "Window-based", "Adaptive", "Data-driven"]
    present_families = set(df_metric["family"].dropna().astype(str))
    families = [family for family in family_order if family in present_families]
    panels = ["Reference Step", "Reference Frequency"] + families
    step_ticks = sorted(df_metric["step_percent"].dropna().astype(float).unique().tolist())
    tick_major = [0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]

    ncols = 2
    nrows = int(math.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12.0, 3.6 * nrows), sharex=False)
    axes_arr = np.atleast_1d(axes).ravel()

    cmap = matplotlib.colormaps["tab20"]
    est_labels = sorted(df_metric["estimator"].unique().tolist())
    color_map = {label: cmap(i % cmap.N) for i, label in enumerate(est_labels)}

    ref_step = _env_float_alias("ASTEP_REFERENCE_STEP_PCT", "VSTEP_REFERENCE_STEP_PCT", 50.0, minimum=0.0)

    for idx, panel in enumerate(panels):
        ax = axes_arr[idx]
        if panel == "Reference Step":
            t_step_s = 0.50
            amp_pre = 1.0
            amp_post = 1.0 + (ref_step / 100.0)
            sc = IEEEMagStepScenario.run(
                duration_s=1.8,
                amp_pre_pu=amp_pre,
                amp_post_pu=amp_post,
                t_step_s=t_step_s,
                noise_sigma=0.0,
                seed=0,
            )
            ax.plot(sc.t, sc.v, color="#111111", linewidth=1.2, label=f"Reference step=+{ref_step:g}%")
            ax.axvline(t_step_s, color="#455A64", linestyle="--", linewidth=1.0, label="Step instant (t=0.50 s)")
            ax.axhline(amp_pre, color="#455A64", linestyle="--", linewidth=1.0, label=f"|x| envelope = {amp_pre:g} pu")
            ax.axhline(amp_post, color="#455A64", linestyle="--", linewidth=1.0, label=f"|x| envelope = {amp_post:g} pu")
            ax.axhline(-amp_pre, color="#455A64", linestyle="--", linewidth=0.8)
            ax.axhline(-amp_post, color="#455A64", linestyle="--", linewidth=0.8)
            ax.annotate(
                f"+{ref_step:g}%",
                xy=(t_step_s + 0.001, amp_post),
                xytext=(t_step_s + 0.02, amp_post + 0.1),
                textcoords="data",
                fontsize=9,
                color="#1B5E20",
                arrowprops=dict(arrowstyle="->", lw=0.9, color="#1B5E20"),
            )
            ax.set_xlim(0.45, 0.55)
            ylim = max(1.25 * amp_post, 1.25)
            ax.set_ylim(-ylim, ylim)
            ax.set_title("Reference Amplitude Step", loc="left", fontweight="bold")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Input signal x(t) [pu]")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(loc="best", fontsize=7, frameon=True)
            continue
        if panel == "Reference Frequency":
            t_step_s = 0.50
            amp_pre = 1.0
            amp_post = 1.0 + (ref_step / 100.0)
            sc = IEEEMagStepScenario.run(
                duration_s=1.8,
                amp_pre_pu=amp_pre,
                amp_post_pu=amp_post,
                t_step_s=t_step_s,
                noise_sigma=0.0,
                seed=0,
            )
            ax.plot(sc.t, sc.f_true, color="#111111", linewidth=1.2, label="Reference f(t)")
            ax.axvline(t_step_s, color="#455A64", linestyle="--", linewidth=1.0, label="Step instant (t=0.50 s)")
            ax.set_xlim(0.40, 0.60)
            f_min = float(np.min(sc.f_true))
            f_max = float(np.max(sc.f_true))
            pad = 0.05 if abs(f_max - f_min) < 1e-9 else 0.12 * abs(f_max - f_min)
            ax.set_ylim(f_min - pad, f_max + pad)
            ax.set_title("Reference Frequency f(t)", loc="left", fontweight="bold")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Frequency [Hz]")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(loc="best", fontsize=7, frameon=True)
            continue

        family = panel
        df_family = df_metric[df_metric["family"] == family].sort_values(["estimator", "step_percent"])
        x_lo = max(min(step_ticks), 1e-6) if step_ticks else 0.25
        x_hi = max(step_ticks) if step_ticks else 1000.0
        if df_family.empty:
            ax.grid(True, which="both", alpha=0.25)
            ax.set_title(f"{family} (no data)", loc="left", fontweight="bold")
            ax.set_ylabel(metric_label)
            ax.set_xlabel("Signal amplitude step [%]")
            continue

        for estimator, df_est in df_family.groupby("estimator", sort=True):
            y_vals = df_est["metric_value"].to_numpy(dtype=float)
            y_p10 = df_est["metric_p10"].to_numpy(dtype=float)
            y_p90 = df_est["metric_p90"].to_numpy(dtype=float)
            if yscale == "log":
                y_vals = np.maximum(y_vals, 1e-9)
                y_p10 = np.maximum(y_p10, 1e-9)
                y_p90 = np.maximum(y_p90, 1e-9)
            x_vals = df_est["step_percent"].to_numpy(dtype=float)
            if len(x_vals) >= 2:
                ax.fill_between(
                    x_vals,
                    y_p10,
                    y_p90,
                    alpha=0.10,
                    color=color_map[str(estimator)],
                    linewidth=0,
                )
            ax.plot(
                x_vals,
                y_vals,
                marker="o",
                markersize=2.8,
                linewidth=1.0,
                alpha=0.9,
                color=color_map[str(estimator)],
                label=str(estimator),
            )

        if step_ticks:
            _shade_amplitude_regions(ax, x_lo, x_hi, include_labels=True)

        if ieee_line is not None:
            y_ieee = max(ieee_line, 1e-9) if yscale == "log" else ieee_line
            ax.axhline(y_ieee, color="#303F9F", linestyle="--", linewidth=1.0, label=f"Guide threshold ({ieee_line:g})")
        if iec_line is not None:
            y_iec = max(iec_line, 1e-9) if yscale == "log" else iec_line
            ax.axhline(y_iec, color="#00897B", linestyle="--", linewidth=1.0, label=f"Strict guide ({iec_line:g})")

        if yscale == "log":
            ax.set_yscale("log")
        ax.set_xscale("log")
        if x_hi <= x_lo:
            x_lo_plot = max(x_lo / 1.25, 1e-6)
            x_hi_plot = x_hi * 1.25
        else:
            x_lo_plot = x_lo
            x_hi_plot = x_hi
        ax.set_xlim(x_lo_plot, x_hi_plot)
        local_ticks = [v for v in tick_major if x_lo_plot <= v <= x_hi_plot]
        if local_ticks:
            ax.xaxis.set_major_locator(FixedLocator(local_ticks))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}"))
        ax.grid(True, which="both", alpha=0.25)
        ax.set_title(f"{family}", loc="left", fontweight="bold")
        ax.set_ylabel(metric_label)
        ax.set_xlabel("Signal amplitude step [%]")
        ax.tick_params(axis="x", labelrotation=0, labelsize=7)

        if len(df_family["step_percent"].unique()) >= 4:
            fam_mean = (
                df_family.groupby("step_percent", as_index=False)["metric_value"]
                .mean()
                .sort_values("step_percent")
            )
            x_vals = fam_mean["step_percent"].to_numpy(dtype=float)
            y_vals = fam_mean["metric_value"].to_numpy(dtype=float)
            y_for_grad = np.log10(np.maximum(y_vals, 1e-9)) if yscale == "log" else y_vals
            dx = np.diff(np.log10(np.maximum(x_vals, 1e-9)))
            dy = np.diff(y_for_grad)
            slope = np.divide(dy, np.maximum(dx, 1e-12))
            k_idx = int(np.argmax(slope)) + 1
            x_k = float(x_vals[k_idx])
            y_k = float(max(y_vals[k_idx], 1e-9) if yscale == "log" else y_vals[k_idx])
            ax.annotate(
                "Knee",
                xy=(x_k, y_k),
                xytext=(8, 10),
                textcoords="offset points",
                fontsize=6.4,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#424242"),
                color="#424242",
            )

        if ieee_line is not None:
            fam_best = (
                df_family.groupby("step_percent", as_index=False)["metric_value"]
                .min()
                .sort_values("step_percent")
            )
            cross = fam_best[fam_best["metric_value"] > float(ieee_line)]
            if not cross.empty:
                x_cross = float(cross.iloc[0]["step_percent"])
                y_cross = float(cross.iloc[0]["metric_value"])
                y_cross = max(y_cross, 1e-9) if yscale == "log" else y_cross
                ax.scatter([x_cross], [y_cross], s=14, color="#212121", zorder=7)
                ax.annotate(
                    f"Step*={x_cross:g}%",
                    xy=(x_cross, y_cross),
                    xytext=(6, -14),
                    textcoords="offset points",
                    fontsize=6.2,
                    color="#212121",
                )
        ax.legend(loc="best", fontsize=6.4, frameon=True, ncol=1)

    for j in range(len(panels), len(axes_arr)):
        axes_arr[j].set_visible(False)

    fig.suptitle(f"{title_prefix}: by estimator family", fontsize=13, y=0.995)
    _add_methodology_text(fig)
    fig.text(
        0.5,
        0.006,
        "Central curves use MC medians when available, otherwise means; shaded bands mark p10-p90 variability when available. Horizontal lines are guide thresholds, not formal step-test compliance limits.",
        ha="center",
        va="bottom",
        fontsize=7,
        color="#37474F",
    )
    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.92])
    return fig, color_map


def _shade_amplitude_regions(ax: plt.Axes, x_lo: float, x_hi: float, *, include_labels: bool = True) -> None:
    for idx, (label, lo, hi, color, _description) in enumerate(AMPLITUDE_STEP_REGIONS):
        band_lo = max(float(lo), x_lo)
        band_hi = min(float(hi), x_hi)
        if band_hi <= band_lo:
            continue
        ax.axvspan(band_lo, band_hi, color=color, alpha=0.070 if idx < 4 else 0.055, zorder=0)
        if include_labels:
            x_mid = float(np.sqrt(max(band_lo, 1e-9) * max(band_hi, 1e-9)))
            y_pos = 0.985 - (0.04 * (idx % 3))
            ax.text(
                x_mid,
                y_pos,
                label,
                transform=ax.get_xaxis_transform(),
                va="top",
                ha="center",
                fontsize=6.5,
                color="#263238",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.66),
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


def _save_method_summary_map(df_global: pd.DataFrame, out_dir: Path) -> tuple[list[Path], dict[str, tuple[float, float, float, float]]]:
    if df_global.empty:
        return [], {}
    df = _add_derived_metric_columns(df_global)
    value_col = "m15_pass_rate_pct_mean" if "m15_pass_rate_pct_mean" in df.columns else "m1_rmse_hz_mean"
    value_label = "Pass rate [%]" if value_col == "m15_pass_rate_pct_mean" else "RMSE [Hz]"
    higher_is_better = value_col == "m15_pass_rate_pct_mean"
    pivot = df.pivot_table(index="estimator", columns="step_percent", values=value_col, aggfunc="mean")
    if pivot.empty:
        return [], {}
    families = df[["estimator", "family"]].drop_duplicates().set_index("estimator")["family"].to_dict()
    family_order = {name: i for i, name in enumerate(["Loop-based", "Model-based", "Window-based", "Adaptive", "Data-driven"])}
    ordered_index = sorted(pivot.index, key=lambda est: (family_order.get(families.get(est, ""), 99), str(est)))
    pivot = pivot.loc[ordered_index]
    steps = [float(x) for x in pivot.columns.to_list()]
    values = pivot.to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(13.0, 8.4), gridspec_kw={"height_ratios": [3.2, 1.25]})
    ax = axes[0]
    cmap = "viridis" if higher_is_better else "magma_r"
    if higher_is_better:
        im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=0.0, vmax=100.0)
    else:
        finite = values[np.isfinite(values)]
        vmax = float(np.percentile(finite, 95)) if finite.size else 1.0
        im = ax.imshow(np.log10(np.maximum(values, 1e-9)), aspect="auto", cmap=cmap, vmin=-4.0, vmax=np.log10(max(vmax, 1e-9)))
    ax.set_title("Method Stress Map", loc="left", fontweight="bold")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_xticks(np.arange(len(steps)))
    ax.set_xticklabels([f"{v:g}" for v in steps], rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Amplitude step [%]")
    ax.set_ylabel("Estimator")
    for row_idx, est in enumerate(pivot.index):
        family = families.get(est, "")
        ax.text(-0.8, row_idx, family, ha="right", va="center", fontsize=5.8, color="#455A64")
    cbar = fig.colorbar(im, ax=ax, fraction=0.024, pad=0.012)
    cbar.set_label(value_label if higher_is_better else "log10 RMSE [Hz]")

    ax2 = axes[1]
    summary_rows = []
    guide = _env_float_alias("ASTEP_LIMIT_RMSE_GUIDE", "VSTEP_LIMIT_RMSE_IEEE", 0.05, minimum=0.0)
    strict = _env_float_alias("ASTEP_LIMIT_RMSE_STRICT", "VSTEP_LIMIT_RMSE_IEC", 0.01, minimum=0.0)
    for est, df_est in df.sort_values("step_percent").groupby("estimator", sort=False):
        rmse = pd.to_numeric(df_est.get("m1_rmse_hz_mean"), errors="coerce")
        steps_est = pd.to_numeric(df_est["step_percent"], errors="coerce")
        fail = df_est.loc[rmse > guide, "step_percent"] if rmse is not None else pd.Series(dtype=float)
        critical = float(fail.iloc[0]) if not fail.empty else float("nan")
        median_pass = float(pd.to_numeric(df_est.get("m15_pass_rate_pct_mean"), errors="coerce").median()) if "m15_pass_rate_pct_mean" in df_est.columns else float("nan")
        summary_rows.append((est, families.get(est, ""), critical, median_pass, float(steps_est.max())))
    summary = pd.DataFrame(summary_rows, columns=["estimator", "family", "critical_step", "median_pass_rate", "max_step"])
    summary = summary.sort_values(["family", "critical_step", "median_pass_rate"], na_position="last", ascending=[True, True, False])
    y = np.arange(len(summary))
    x = summary["critical_step"].fillna(summary["max_step"] * 1.05).to_numpy(dtype=float)
    ax2.scatter(x, y, s=24, color="#263238")
    for i, row in summary.iterrows():
        label = f"{row['estimator']}"
        ax2.text(float(x[list(summary.index).index(i)]) * 1.03, list(summary.index).index(i), label, va="center", fontsize=6.4)
    ax2.axvline(guide, color="#303F9F", linestyle="--", linewidth=0.9, label=f"RMSE guide {guide:g} Hz")
    ax2.axvline(strict, color="#00897B", linestyle="--", linewidth=0.9, label=f"Strict guide {strict:g} Hz")
    ax2.set_xscale("log")
    ax2.set_xlim(min(steps), max(steps) * 1.25)
    ax2.set_yticks([])
    ax2.set_xlabel("First amplitude step where mean RMSE exceeds guide [%]")
    ax2.set_title("Critical Step Summary", loc="left", fontweight="bold")
    _shade_amplitude_regions(ax2, min(steps), max(steps), include_labels=True)
    ax2.grid(True, which="both", alpha=0.22)
    ax2.legend(loc="best", fontsize=6.4, frameon=True)

    fig.suptitle("Amplitude-Step Method Atlas", fontsize=13, y=0.995)
    _add_methodology_text(fig)
    fig.tight_layout(rect=[0.06, 0.04, 0.98, 0.93])
    png_path = out_dir / SUMMARY_MAP_PNG_NAME
    pdf_path = out_dir / SUMMARY_MAP_PDF_NAME
    fig.savefig(png_path, dpi=240)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [png_path, pdf_path], {}


def _plot_rmse_by_family(df_rmse: pd.DataFrame, out_dir: Path) -> tuple[list[Path], dict[str, tuple[float, float, float, float]]]:
    if df_rmse.empty:
        return [], {}
    fig, color_map = _plot_metric_by_family_page(
        df_global=df_rmse,
        metric_col="m1_rmse_hz_mean",
        metric_label="RMSE [Hz]",
        title_prefix="RMSE",
        yscale="log",
        ieee_line=_env_float_alias("ASTEP_LIMIT_RMSE_GUIDE", "VSTEP_LIMIT_RMSE_IEEE", 0.05, minimum=0.0),
        iec_line=_env_float_alias("ASTEP_LIMIT_RMSE_STRICT", "VSTEP_LIMIT_RMSE_IEC", 0.01, minimum=0.0),
    )
    png_path = out_dir / f"{PLOT_NAME}.png"
    pdf_path = out_dir / f"{PLOT_NAME}.pdf"
    fig.savefig(png_path, dpi=240)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [png_path, pdf_path], color_map


def _save_multipage_metrics_dashboard(df_global: pd.DataFrame, out_dir: Path) -> Path:
    pdf_path = out_dir / MULTIPAGE_PDF_NAME
    df_plot = _add_derived_metric_columns(df_global)
    pages = [
        ("m1_rmse_hz_mean", "RMSE [Hz]", "RMSE", "log", _env_float_alias("ASTEP_LIMIT_RMSE_GUIDE", "VSTEP_LIMIT_RMSE_IEEE", 0.05, minimum=0.0), _env_float_alias("ASTEP_LIMIT_RMSE_STRICT", "VSTEP_LIMIT_RMSE_IEC", 0.01, minimum=0.0)),
        ("m3_max_peak_hz_mean", "FE max per test [Hz]", "FE max per test", "log", _env_float_alias("ASTEP_LIMIT_FE_MAX_GUIDE", "VSTEP_LIMIT_FE_MAX_IEEE", 0.5, minimum=0.0), _env_float_alias("ASTEP_LIMIT_FE_MAX_STRICT", "VSTEP_LIMIT_FE_MAX_IEC", 0.01, minimum=0.0)),
        ("m9_rfe_max_hz_s_mean", "RFE max [Hz/s]", "RFE", "log", _env_float_alias("ASTEP_LIMIT_RFE_GUIDE", "VSTEP_LIMIT_RFE_IEEE", 3.0, minimum=0.0), _env_float_alias("ASTEP_LIMIT_RFE_STRICT", "VSTEP_LIMIT_RFE_IEC", 0.4, minimum=0.0)),
        ("m5_trip_risk_s_mean", "Time out of band [s]", "Time out of band", "linear", _env_float_alias("ASTEP_LIMIT_TOB_GUIDE", "VSTEP_LIMIT_TOB_IEEE", 0.1, minimum=0.0), _env_float_alias("ASTEP_LIMIT_TOB_STRICT", "VSTEP_LIMIT_TOB_IEC", 0.02, minimum=0.0)),
        ("m15_pass_rate_pct_mean", "Pass rate [%]", "Pass rate", "linear", 95.0, 99.0),
        ("m13_cpu_time_us_mean", "CPU time [us/pass]", "CPU cost", "log", None, None),
        ("m20_runtime_jitter_us_mean", "Runtime jitter [us]", "Runtime jitter", "log", None, None),
        ("m22_invalid_output_rate_pct_mean", "Invalid output rate [%]", "Invalid outputs", "log", 1.0, 0.1),
        ("m21_startup_valid_samples_mean", "Startup valid samples [samples]", "Startup validity", "linear", None, None),
    ]
    with PdfPages(pdf_path) as pdf:
        for metric_col, metric_label, title_prefix, yscale, ieee_line, iec_line in pages:
            if metric_col not in df_plot.columns:
                continue
            fig, _ = _plot_metric_by_family_page(
                df_global=df_plot,
                metric_col=metric_col,
                metric_label=metric_label,
                title_prefix=title_prefix,
                yscale=yscale,
                ieee_line=ieee_line,
                iec_line=iec_line,
            )
            pdf.savefig(fig)
            plt.close(fig)
    return pdf_path


def _write_manifest(
    out_dir: Path,
    scenarios: list[SweepScenario],
    estimators: dict[str, type],
    *,
    fast_n_mc_runs: int,
    fast_n_cost_reps: int,
    fast_tune_trials: int,
    fast_tune_eval_runs: int,
    run_configs: dict[str, EstimatorRunConfig],
    base_seed: int,
    tuning_base_seed: int,
) -> Path:
    payload = {
        "benchmark_identity": "amplitude_step_sweep_active_pipeline",
        "pipeline_method_version": PIPELINE_METHOD_VERSION,
        "description": "Dedicated amplitude-step stress atlas over IEEE_Mag_Step for AM-to-FM coupling, IBR ride-through stress, and high-amplitude current/fault surrogates.",
        "output_dir": str(out_dir),
        "n_scenarios": len(scenarios),
        "n_estimators": len(estimators),
        "fast_tier": {
            "n_mc_runs": int(fast_n_mc_runs),
            "n_cost_reps": int(fast_n_cost_reps),
            "n_trials_requested": int(fast_tune_trials),
            "tune_eval_runs": int(fast_tune_eval_runs),
        },
        "slow_tier_defaults": {
            "estimators": sorted(SLOW_ESTIMATORS),
            "n_mc_runs": _env_int("ASTEP_SLOW_N_MC_RUNS", 15, minimum=1),
            "n_cost_reps": _env_int("ASTEP_SLOW_N_COST_REPS", 5, minimum=1),
            "tune_trials": _env_int("ASTEP_SLOW_TUNE_TRIALS", 20, minimum=0),
            "tune_eval_runs": _env_int("ASTEP_SLOW_TUNE_EVAL_RUNS", 5, minimum=1),
        },
        "estimator_run_configs": {
            label: {
                "tier": cfg.tier,
                "n_mc_runs": int(cfg.n_mc_runs),
                "n_cost_reps": int(cfg.n_cost_reps),
                "n_trials_requested": int(cfg.tune_trials),
                "tune_eval_runs": int(cfg.tune_eval_runs),
            }
            for label, cfg in run_configs.items()
        },
        "base_seed": int(base_seed),
        "tuning_base_seed": int(tuning_base_seed),
        "tuning": {
            "objective": _tuning_objective_mode(),
            "policy": "tiered_by_estimator_runtime",
            "oracle_tuning_enabled": _env_bool("ASTEP_ORACLE_TUNING", True),
            "oracle_estimators": sorted(_csv_set("ASTEP_ORACLE_ESTIMATORS") or KALMAN_ORACLE_ESTIMATORS),
            "oracle_trials_floor": _env_int("ASTEP_ORACLE_TRIALS", 300, minimum=1),
            "ukf_oracle_tuning_enabled": _ukf_oracle_enabled(),
            "ukf_oracle_trials": _env_int("ASTEP_UKF_ORACLE_TRIALS", _env_int("ASTEP_ORACLE_TRIALS", 300, minimum=1), minimum=1),
            "oracle_interpretation": "per-scenario practical lower-bound tuning; not deployment-transferable",
            "seed_policy": "tuning seeds are offset from final MC seeds to prevent seed reuse leakage",
        },
        "legacy_summary": {
            "n_mc_runs": "mixed_by_estimator",
            "n_cost_reps": "mixed_by_estimator",
            "n_trials_requested": "mixed_by_estimator",
            "tune_eval_runs": "mixed_by_estimator",
            "objective": _tuning_objective_mode(),
        },
        "step_levels_percent": [float(s.step_percent) for s in scenarios],
        "scenario_contract": {
            "duration_s": 1.8,
            "amp_pre_pu": 1.0,
            "amp_post_pu": "1.0 + step_percent / 100",
            "t_step_s_nominal": 0.50,
            "t_step_s_mc_uniform": [0.45, 0.55],
            "noise_sigma_mc_uniform": [0.0005, 0.0020],
            "phase_rad_mc_uniform": [0.0, "2*pi"],
            "frequency_hz": "nominal and constant; any FE is AM-to-FM coupling",
            "methodology_note": METHODOLOGY_TEXT,
            "regions": [
                {"label": label, "low_pct": lo, "high_pct": hi, "description": description}
                for label, lo, hi, _color, description in AMPLITUDE_STEP_REGIONS
            ],
        },
        "estimators": list(estimators.keys()),
        "families": {label: ESTIMATOR_FAMILIES.get(label, "Unknown") for label in estimators},
        "artifacts": {
            "global_metrics": GLOBAL_CSV_NAME,
            "rmse_by_estimator": RMSE_EST_CSV_NAME,
            "rmse_by_family": RMSE_FAM_CSV_NAME,
            "timing_profile": "timing_profile.csv",
            "plot_png": f"{PLOT_NAME}.png",
            "plot_pdf": f"{PLOT_NAME}.pdf",
            "metrics_multipage_pdf": MULTIPAGE_PDF_NAME,
            "summary_map_png": SUMMARY_MAP_PNG_NAME,
            "summary_map_pdf": SUMMARY_MAP_PDF_NAME,
        },
    }
    path = out_dir / MANIFEST_NAME
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Amplitude-step sweep runner and stress-atlas report generator.")
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate plot/image/pdf artifacts from existing global_metrics_report.csv only.",
    )
    args = parser.parse_args()

    n_mc_runs = _env_int_alias("ASTEP_SWEEP_N_MC_RUNS", "VSTEP_SWEEP_N_MC_RUNS", _env_int("BENCHMARK_N_MC_RUNS", 20, minimum=1), minimum=1)
    base_seed = _env_int_alias("ASTEP_SWEEP_BASE_SEED", "VSTEP_SWEEP_BASE_SEED", 12345, minimum=0)
    resume_run = _env_bool_alias("ASTEP_SWEEP_RESUME", "VSTEP_SWEEP_RESUME", True)
    n_cost_reps = _env_int_alias("ASTEP_SWEEP_N_COST_REPS", "VSTEP_SWEEP_N_COST_REPS", 1, minimum=1)
    tune_trials = _env_int_alias("ASTEP_SWEEP_TUNE_TRIALS", "VSTEP_SWEEP_TUNE_TRIALS", 60, minimum=0)
    tune_eval_runs = _env_int_alias("ASTEP_SWEEP_TUNE_EVAL_RUNS", "VSTEP_SWEEP_TUNE_EVAL_RUNS", 2, minimum=1)
    tuning_base_seed = _tuning_base_seed(base_seed)

    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.plots_only:
        global_csv = OUTPUT_DIR / GLOBAL_CSV_NAME
        if not global_csv.exists():
            raise FileNotFoundError(f"Missing required input for --plots-only: {global_csv}")
        df_global = pd.read_csv(global_csv)
        keep_cols = [
            "scenario", "step_percent", "estimator", "family", "n_mc_runs",
            "m1_rmse_hz_mean", "m1_rmse_hz_median", "m1_rmse_hz_p10", "m1_rmse_hz_p90", "m1_rmse_hz_std",
        ]
        rmse_cols = [col for col in keep_cols if col in df_global.columns]
        df_rmse = df_global[rmse_cols].copy()
        generated_plots, color_map = _plot_rmse_by_family(df_rmse=df_rmse, out_dir=OUTPUT_DIR)
        multipage_pdf_path = _save_multipage_metrics_dashboard(df_global=df_global, out_dir=OUTPUT_DIR)
        summary_map_paths, _ = _save_method_summary_map(df_global=df_global, out_dir=OUTPUT_DIR)
        legend_rows = [
            {
                "estimator": estimator,
                "hex_color": matplotlib.colors.to_hex(color_map[estimator]),
                "family": ESTIMATOR_FAMILIES.get(estimator, "Unknown"),
            }
            for estimator in sorted(color_map)
        ]
        legend_map_path = OUTPUT_DIR / LEGEND_MAP_CSV_NAME
        pd.DataFrame(legend_rows).to_csv(legend_map_path, index=False)

        print("Artifacts regenerated (--plots-only):")
        for path in generated_plots:
            print(f"  - {path.relative_to(ROOT)}")
        for path in summary_map_paths:
            print(f"  - {path.relative_to(ROOT)}")
        print(f"  - {multipage_pdf_path.relative_to(ROOT)}")
        print(f"  - {legend_map_path.relative_to(ROOT)}")
        elapsed = (time.time() - t0) / 60.0
        print(f"\n[DONE] Plot regeneration completed in {elapsed:.2f} min.")
        return

    scenarios = _build_scenarios()
    estimators = _select_estimators()
    if not scenarios:
        raise ValueError("Scenario filter removed all amplitude-step scenarios.")
    run_configs = {
        est_name: _run_config_for_estimator(
            est_name,
            fast_n_mc_runs=n_mc_runs,
            fast_tune_trials=tune_trials,
            fast_tune_eval_runs=tune_eval_runs,
            fast_n_cost_reps=n_cost_reps,
        )
        for est_name in estimators
    }

    print(f"Running Amplitude-Step sweep: {len(scenarios)} scenarios x {len(estimators)} estimators")
    print(f"  Fast tier MC runs per pair: {n_mc_runs}")
    print(f"  Fast tier CPU timing reps per pair: {n_cost_reps}")
    print(f"  Fast tier tuning trials per pair: {tune_trials}")
    print(f"  Fast tier tuning eval runs: {tune_eval_runs}")
    print(f"  MC base seed: {base_seed}; tuning base seed: {tuning_base_seed}")
    print(f"  Tuning objective: {_tuning_objective_mode()}; oracle tuning: {_env_bool('ASTEP_ORACLE_TUNING', True)}")
    print("  Slow tier configs:")
    for est_name, cfg in run_configs.items():
        if cfg.tier == "slow":
            print(f"    - {est_name}: MC={cfg.n_mc_runs}, tune_trials={cfg.tune_trials}, tune_eval_runs={cfg.tune_eval_runs}, cost_reps={cfg.n_cost_reps}")
    print(f"  Output dir: {OUTPUT_DIR}")

    rows_agg: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    for sc in scenarios:
        print(f"\nScenario {sc.scenario_name} (Amplitude step={sc.step_percent:g}%)")
        sc_dir = OUTPUT_DIR / sc.scenario_name
        sc_dir.mkdir(parents=True, exist_ok=True)

        for est_name, est_cls in estimators.items():
            cfg = run_configs[est_name]
            out_dir = sc_dir / est_name
            out_dir.mkdir(parents=True, exist_ok=True)
            summary_csv = out_dir / f"{sc.scenario_name}__{est_name}_summary.csv"
            run_spec_path = out_dir / "run_spec.json"

            if resume_run and _can_reuse_existing_run(
                summary_csv, run_spec_path,
                scenario_name=sc.scenario_name,
                estimator_name=est_name,
                step_percent=sc.step_percent,
                requested_n_mc_runs=cfg.n_mc_runs,
                requested_tune_trials=cfg.tune_trials,
                requested_tune_eval_runs=cfg.tune_eval_runs,
                requested_n_cost_reps=cfg.n_cost_reps,
                requested_base_seed=base_seed,
                requested_tuning_base_seed=tuning_base_seed,
            ):
                summary_df = pd.read_csv(summary_csv)
                tune_elapsed_s = float("nan")
                mc_elapsed_s = float("nan")
                total_elapsed_s = float("nan")
            else:
                print(f"  - {est_name} [{cfg.tier}: MC={cfg.n_mc_runs}, tune={cfg.tune_trials}x{cfg.tune_eval_runs}, cost={cfg.n_cost_reps}]", flush=True)
                t_estimator_start = time.perf_counter()
                t_tune_start = time.perf_counter()
                best_params, tuning_meta = _tune_estimator_for_scenario(
                    est_name=est_name,
                    est_cls=est_cls,
                    scenario_cls=sc.scenario_cls,
                    n_trials=cfg.tune_trials,
                    tune_eval_runs=cfg.tune_eval_runs,
                    base_seed=tuning_base_seed,
                )
                tune_elapsed_s = float(time.perf_counter() - t_tune_start)
                engine = MonteCarloEngine(
                    scenario_cls=sc.scenario_cls,
                    estimator_cls=est_cls,
                    estimator_params=best_params,
                    n_runs=cfg.n_mc_runs,
                    base_seed=base_seed,
                    n_cost_reps=cfg.n_cost_reps,
                    capture_signals=False,
                )
                t_mc_start = time.perf_counter()
                result = _run_engine_local(engine)
                mc_elapsed_s = float(time.perf_counter() - t_mc_start)
                total_elapsed_s = float(time.perf_counter() - t_estimator_start)
                result.summary_df.to_csv(summary_csv, index=False)
                summary_df = result.summary_df
                run_spec = {
                    "pipeline_method_version": PIPELINE_METHOD_VERSION,
                    "scenario": sc.scenario_name,
                    "step_percent": float(sc.step_percent),
                    "estimator": est_name,
                    "family": ESTIMATOR_FAMILIES.get(est_name, "Unknown"),
                    "best_params": benchmark._to_builtin(best_params),
                    "tuning_meta": benchmark._to_builtin(tuning_meta),
                    "run_tier": cfg.tier,
                    "n_mc_runs": int(cfg.n_mc_runs),
                    "n_cost_reps": int(cfg.n_cost_reps),
                    "base_seed": int(base_seed),
                    "tuning_base_seed": int(tuning_base_seed),
                    "timing": {
                        "tuning_elapsed_s": tune_elapsed_s,
                        "mc_eval_elapsed_s": mc_elapsed_s,
                        "total_elapsed_s": total_elapsed_s,
                    },
                }
                run_spec_path.write_text(json.dumps(benchmark._to_builtin(run_spec), indent=2, ensure_ascii=False), encoding="utf-8")

            agg = _aggregate_summary(summary_df)
            rows_agg.append(
                {
                    "scenario": sc.scenario_name,
                    "step_percent": float(sc.step_percent),
                    "estimator": est_name,
                    "family": ESTIMATOR_FAMILIES.get(est_name, "Unknown"),
                    "run_tier": cfg.tier,
                    "n_mc_runs": int(len(summary_df)),
                    **agg,
                }
            )
            timing_rows.append(
                {
                    "scenario": sc.scenario_name,
                    "step_percent": float(sc.step_percent),
                    "estimator": est_name,
                    "family": ESTIMATOR_FAMILIES.get(est_name, "Unknown"),
                    "run_tier": cfg.tier,
                    "n_mc_runs": int(len(summary_df)),
                    "tune_trials": int(cfg.tune_trials),
                    "tune_eval_runs": int(cfg.tune_eval_runs),
                    "n_cost_reps": int(cfg.n_cost_reps),
                    "base_seed": int(base_seed),
                    "tuning_base_seed": int(tuning_base_seed),
                    "pipeline_method_version": PIPELINE_METHOD_VERSION,
                    "tuning_elapsed_s": tune_elapsed_s,
                    "mc_eval_elapsed_s": mc_elapsed_s,
                    "total_elapsed_s": total_elapsed_s,
                }
            )

    df_global = pd.DataFrame(rows_agg).sort_values(["step_percent", "family", "estimator"])
    if "n_mc_runs" in df_global.columns and df_global["n_mc_runs"].nunique() > 1:
        print(f"[WARN] Mixed n_mc_runs detected in aggregated output: {sorted(df_global['n_mc_runs'].unique().tolist())}")
    global_csv = OUTPUT_DIR / GLOBAL_CSV_NAME
    df_global.to_csv(global_csv, index=False)

    keep_cols = [
        "scenario", "step_percent", "estimator", "family", "n_mc_runs",
        "m1_rmse_hz_mean", "m1_rmse_hz_median", "m1_rmse_hz_p10", "m1_rmse_hz_p90", "m1_rmse_hz_std",
    ]
    rmse_cols = [col for col in keep_cols if col in df_global.columns]
    df_rmse = df_global[rmse_cols].copy()
    rmse_est_csv = OUTPUT_DIR / RMSE_EST_CSV_NAME
    df_rmse.to_csv(rmse_est_csv, index=False)

    df_rmse_family = (
        df_rmse.groupby(["step_percent", "family"], as_index=False).agg(
            family_rmse_mean=("m1_rmse_hz_mean", "mean"),
            family_rmse_median=("m1_rmse_hz_median", "median") if "m1_rmse_hz_median" in df_rmse.columns else ("m1_rmse_hz_mean", "median"),
            family_rmse_p10=("m1_rmse_hz_p10", "median") if "m1_rmse_hz_p10" in df_rmse.columns else ("m1_rmse_hz_mean", "min"),
            family_rmse_p90=("m1_rmse_hz_p90", "median") if "m1_rmse_hz_p90" in df_rmse.columns else ("m1_rmse_hz_mean", "max"),
            family_rmse_std=("m1_rmse_hz_mean", "std"),
            family_rmse_min=("m1_rmse_hz_mean", "min"),
            family_rmse_max=("m1_rmse_hz_mean", "max"),
        ).sort_values(["step_percent", "family"])
    )
    rmse_family_csv = OUTPUT_DIR / RMSE_FAM_CSV_NAME
    df_rmse_family.to_csv(rmse_family_csv, index=False)
    timing_csv = OUTPUT_DIR / "timing_profile.csv"
    pd.DataFrame(timing_rows).to_csv(timing_csv, index=False)

    generated_plots, color_map = _plot_rmse_by_family(df_rmse=df_rmse, out_dir=OUTPUT_DIR)
    multipage_pdf_path = _save_multipage_metrics_dashboard(df_global=df_global, out_dir=OUTPUT_DIR)
    summary_map_paths, _ = _save_method_summary_map(df_global=df_global, out_dir=OUTPUT_DIR)
    legend_rows = [
        {
            "estimator": estimator,
            "hex_color": matplotlib.colors.to_hex(color_map[estimator]),
            "family": ESTIMATOR_FAMILIES.get(estimator, "Unknown"),
        }
        for estimator in sorted(color_map)
    ]
    legend_map_path = OUTPUT_DIR / LEGEND_MAP_CSV_NAME
    pd.DataFrame(legend_rows).to_csv(legend_map_path, index=False)
    manifest_path = _write_manifest(
        OUTPUT_DIR,
        scenarios,
        estimators,
        fast_n_mc_runs=n_mc_runs,
        fast_n_cost_reps=n_cost_reps,
        fast_tune_trials=tune_trials,
        fast_tune_eval_runs=tune_eval_runs,
        run_configs=run_configs,
        base_seed=base_seed,
        tuning_base_seed=tuning_base_seed,
    )

    elapsed = (time.time() - t0) / 60.0
    print("\nArtifacts:")
    print(f"  - {global_csv.relative_to(ROOT)}")
    print(f"  - {rmse_est_csv.relative_to(ROOT)}")
    print(f"  - {rmse_family_csv.relative_to(ROOT)}")
    print(f"  - {timing_csv.relative_to(ROOT)}")
    for path in generated_plots:
        print(f"  - {path.relative_to(ROOT)}")
    for path in summary_map_paths:
        print(f"  - {path.relative_to(ROOT)}")
    print(f"  - {multipage_pdf_path.relative_to(ROOT)}")
    print(f"  - {legend_map_path.relative_to(ROOT)}")
    print(f"  - {manifest_path.relative_to(ROOT)}")
    print(f"\n[DONE] Amplitude-step sweep completed in {elapsed:.1f} min.")


if __name__ == "__main__":
    main()
