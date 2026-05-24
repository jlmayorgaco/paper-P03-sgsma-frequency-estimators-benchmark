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
PHASE_DISPERSION_PDF_NAME = "amplitude_step_phase_dispersion.pdf"
PHASE_DISPERSION_PNG_NAME = "amplitude_step_phase_dispersion.png"
HYPOTHESIS_CSV_NAME = "deterioration_hypothesis_tests.csv"
HYPOTHESIS_JSON_NAME = "deterioration_hypothesis_tests.json"
HYPOTHESIS_MD_NAME = "deterioration_hypothesis_tests.md"
TUNING_CONTINUITY_CSV_NAME = "tuning_parameter_continuity.csv"

# Journal-style stress-atlas schedule: log-spaced enough to reveal knees,
# while preserving standard-like and protection-relevant anchor points.
STEP_LEVELS_PERCENT: tuple[float, ...] = (
    1.0, 2.0, 3.0, 5.0, 7.5,
    10.0, 15.0, 20.0, 25.0, 35.0,
    50.0, 75.0, 100.0, 150.0, 200.0,
    250.0, 300.0, 350.0, 400.0, 500.0,
    750.0, 1000.0,
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
PIPELINE_METHOD_VERSION = "amplitude_step_v14_fixed_policy_oracle_audit_2026_05_17"
ORACLE_LABEL = "per_scenario_practical_lower_bound"
UKF_ORACLE_LABEL = "ukf_per_scenario_oracle"
FIXED_POLICY_LABEL = "global_fixed_policy"
FIXED_POLICY_REUSED_LABEL = "global_fixed_policy_reused"
KALMAN_ORACLE_ESTIMATORS = {"EKF", "UKF", "LKF", "LKF2", "RA-EKF"}
AMPLITUDE_ORACLE_ESTIMATORS = KALMAN_ORACLE_ESTIMATORS | {
    "SOGI-PLL",
    "SOGI-FLL",
    "Type-3 SOGI-PLL",
    "IPDFT",
    "RLS",
}
TRACKING_GUARD_ESTIMATORS = AMPLITUDE_ORACLE_ESTIMATORS | {"PLL", "RLS", "TKEO"}
STABILITY_ORACLE_ESTIMATORS = AMPLITUDE_ORACLE_ESTIMATORS
DEFAULT_TUNING_SEED_OFFSET = 1_000_000
PHASE_DISPERSION_ESTIMATORS = ("RLS", "EKF", "UKF", "LKF2", "RA-EKF", "Type-3 SOGI-PLL")
PHASE_DISPERSION_STEPS = (3.0, 5.0, 10.0, 100.0, 500.0, 1000.0)

METHODOLOGY_TEXT = (
    "This atlas treats the disturbance as an amplitude-only step: the true frequency remains nominal, "
    "so reported frequency and ROCOF errors quantify magnitude-step cross-sensitivity, numerical robustness, and "
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


@dataclass(frozen=True)
class TuningSignal:
    t: np.ndarray
    v: np.ndarray
    f_true: np.ndarray


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


def _env_float_csv(name: str) -> list[float]:
    values: list[float] = []
    for item in _env_csv(name):
        try:
            values.append(float(item))
        except ValueError:
            print(f"[WARN] Ignoring invalid float in {name}: {item!r}")
    return values


def _csv_set(name: str) -> set[str]:
    return {item.strip() for item in _env_csv(name) if item.strip()}


def _tuning_policy() -> str:
    raw = os.getenv("ASTEP_TUNING_POLICY", os.getenv("VSTEP_TUNING_POLICY", "per_step_oracle"))
    policy = str(raw).strip().lower().replace("-", "_")
    aliases = {
        "oracle": "per_step_oracle",
        "per_scenario": "per_step_oracle",
        "per_scenario_oracle": "per_step_oracle",
        "per_step": "per_step_oracle",
        "fixed": "fixed_policy",
        "global": "fixed_policy",
        "global_fixed": "fixed_policy",
        "fixed_policy": "fixed_policy",
    }
    return aliases.get(policy, policy)


def _fixed_policy_enabled() -> bool:
    return _tuning_policy() == "fixed_policy"


def _tuning_base_seed(base_seed: int) -> int:
    offset = _env_int("ASTEP_TUNE_SEED_OFFSET", DEFAULT_TUNING_SEED_OFFSET, minimum=0)
    return int(base_seed) + int(offset)


def _sanitize_token(value: float) -> str:
    token = f"{value:g}".replace(".", "p")
    return token.replace("-", "m")


def _coprime_stride(n: int, preferred: int) -> int:
    n = max(1, int(n))
    if n <= 1:
        return 1
    stride = max(1, int(preferred) % n)
    for _ in range(n):
        if math.gcd(stride, n) == 1:
            return stride
        stride = (stride + 1) % n
        if stride == 0:
            stride = 1
    return 1


def _lhs_unit(run_idx: int, n_runs: int, *, stride: int = 1) -> float:
    n = max(1, int(n_runs))
    if n == 1:
        return 0.5
    permuted = (int(run_idx) * _coprime_stride(n, stride)) % n
    return (float(permuted) + 0.5) / float(n)


def _stratified_uniform(run_idx: int, n_runs: int, low: float, high: float, *, stride: int) -> float:
    u = _lhs_unit(run_idx, n_runs, stride=stride)
    return float(low) + u * (float(high) - float(low))


def _apply_amplitude_step_run_overrides(
    cls: type,
    *,
    params: dict[str, Any],
    run_idx: int,
    n_runs: int,
    base_seed: int,
) -> dict[str, Any]:
    params = dict(params)
    if _env_bool_alias("ASTEP_PHASE_STRATIFIED", "VSTEP_PHASE_STRATIFIED", True):
        if _env_bool_alias("ASTEP_MC_STRATIFIED_COVARIATES", "VSTEP_MC_STRATIFIED_COVARIATES", True):
            default_bins = int(max(1, n_runs))
        else:
            default_bins = max(8, min(32, int(max(1, n_runs))))
        bins = _env_int_alias("ASTEP_PHASE_BINS", "VSTEP_PHASE_BINS", default_bins, minimum=1)
        params["phase_rad"] = float(2.0 * math.pi * _lhs_unit(run_idx, bins, stride=1))
    if _env_bool_alias("ASTEP_MC_STRATIFIED_COVARIATES", "VSTEP_MC_STRATIFIED_COVARIATES", True):
        noise_sigma = _stratified_uniform(run_idx, n_runs, 0.0005, 0.0020, stride=17)
        noise_mode = os.getenv("ASTEP_NOISE_MODE", os.getenv("VSTEP_NOISE_MODE", "fixed_absolute")).strip().lower()
        if noise_mode in {"none", "noise_free", "no_noise"}:
            noise_sigma = 0.0
        elif noise_mode in {"fixed_snr", "post_fixed_snr", "amplitude_scaled"}:
            amp_post = float(getattr(cls, "DEFAULT_PARAMS", {}).get("amp_post_pu", params.get("amp_post_pu", 1.0)))
            noise_sigma = noise_sigma * max(amp_post, 1e-12)
        params["noise_sigma"] = noise_sigma
        params["t_step_s"] = _stratified_uniform(run_idx, n_runs, 0.45, 0.55, stride=37)
    return params


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
        "apply_run_index_overrides": classmethod(_apply_amplitude_step_run_overrides),
    }

    new_cls = type(class_name, (IEEEMagStepScenario,), attrs)
    new_cls.__module__ = __name__
    globals()[class_name] = new_cls
    return SweepScenario(step_percent=float(step_percent), scenario_cls=new_cls, scenario_name=scenario_name)


def _build_step_levels() -> list[float]:
    levels = {round(float(v), 6) for v in STEP_LEVELS_PERCENT if 0.0 < float(v) <= 1000.0}
    for value in _env_float_csv("ASTEP_SWEEP_EXTRA_STEPS") + _env_float_csv("VSTEP_SWEEP_EXTRA_STEPS"):
        if 0.0 < float(value) <= 1000.0:
            levels.add(round(float(value), 6))

    include = {
        round(float(v), 6)
        for v in (_env_float_csv("ASTEP_SWEEP_INCLUDE_STEPS") + _env_float_csv("VSTEP_SWEEP_INCLUDE_STEPS"))
        if 0.0 < float(v) <= 1000.0
    }
    if include:
        levels = include
    return sorted(levels)


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


def _use_vectorized_engine(est_name: str, cfg: EstimatorRunConfig) -> bool:
    key = _sanitize_env_key(est_name)
    specific = os.getenv(f"ASTEP_{key}_VECTORIZED_ENGINE")
    if specific is not None:
        return _env_bool(f"ASTEP_{key}_VECTORIZED_ENGINE", False)
    if _env_bool("ASTEP_FORCE_VECTORIZED_ENGINE", False):
        return True
    return cfg.tier == "slow" and _env_bool("ASTEP_SLOW_VECTORIZED_ENGINE", False)


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


def _build_tuning_scenarios(scenario_cls: type, *, base_seed: int, n_runs: int) -> list[Any]:
    sampler = MonteCarloEngine(
        scenario_cls=scenario_cls,
        estimator_cls=None,
        n_runs=max(1, int(n_runs)),
        base_seed=int(base_seed),
    )
    return [
        scenario_cls.run(**sampler.sample_params(run_idx))
        for run_idx in range(max(1, int(n_runs)))
    ]


def _fixed_policy_training_steps() -> list[float]:
    configured = _env_float_csv("ASTEP_FIXED_POLICY_TRAIN_STEPS") + _env_float_csv("VSTEP_FIXED_POLICY_TRAIN_STEPS")
    if configured:
        steps = [float(v) for v in configured if 0.0 < float(v) <= 1000.0]
    else:
        steps = [1.0, 5.0, 25.0, 100.0, 300.0, 1000.0]
    return sorted(set(round(v, 6) for v in steps))


def _fixed_policy_safety_step_percent() -> float:
    steps = _fixed_policy_training_steps() + _build_step_levels()
    return float(max(steps)) if steps else 1000.0


def _build_fixed_policy_tuning_scenarios(*, base_seed: int, n_runs_per_step: int) -> list[Any]:
    scenarios: list[Any] = []
    for step in _fixed_policy_training_steps():
        sc = _create_step_variant(step)
        scenarios.extend(
            _build_tuning_scenarios(
                sc.scenario_cls,
                base_seed=int(base_seed),
                n_runs=max(1, int(n_runs_per_step)),
            )
        )
    return scenarios


def _build_tracking_guard_scenarios(*, n_runs: int) -> list[TuningSignal]:
    """Small C0-continuous +/-1 Hz step set used only to reject hold-like tuning."""
    fs = 10_000.0
    dt = 1.0 / fs
    duration_s = 1.5
    step_time_s = 0.5
    t = np.arange(0.0, duration_s, dt, dtype=float)
    pre = t < step_time_s
    out: list[TuningSignal] = []
    n = max(1, int(n_runs))

    for run_idx in range(n):
        direction = 1.0 if (run_idx % 2 == 0) else -1.0
        phase = 2.0 * math.pi * ((run_idx % max(1, n)) + 0.5) / max(1, n)
        f_pre = 60.0
        f_post = 60.0 + direction
        f_true = np.where(pre, f_pre, f_post).astype(float)

        phi = np.empty_like(t)
        phi[pre] = 2.0 * math.pi * f_pre * t[pre] + phase
        phi_at_step = 2.0 * math.pi * f_pre * step_time_s + phase
        phi[~pre] = phi_at_step + 2.0 * math.pi * f_post * (t[~pre] - step_time_s)
        v = np.sin(phi)
        out.append(TuningSignal(t=t, v=v, f_true=f_true))

    return out


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
    expected_eval_runs = int(requested_tune_eval_runs)
    if _fixed_policy_enabled():
        expected_eval_runs = _env_int(
            "ASTEP_FIXED_POLICY_EVAL_RUNS_PER_STEP",
            max(2, min(6, int(max(1, requested_tune_eval_runs)))),
            minimum=1,
        )
    if eval_saved != int(expected_eval_runs):
        return False
    if _fixed_policy_enabled():
        if tune_meta.get("mode") != FIXED_POLICY_REUSED_LABEL:
            return False
        if tune_meta.get("tuning_policy") != "fixed_policy":
            return False
    elif _oracle_enabled(estimator_name):
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
    expected_guard_enabled = _tracking_guard_enabled(estimator_name)
    saved_guard = tune_meta.get("tracking_guard", {})
    saved_guard_enabled = bool(saved_guard.get("enabled", False)) if isinstance(saved_guard, dict) else False
    if saved_guard_enabled != bool(expected_guard_enabled):
        return False
    if expected_guard_enabled and isinstance(saved_guard, dict):
        try:
            saved_guard_runs = int(saved_guard.get("runs", -1))
        except Exception:
            saved_guard_runs = -1
        expected_guard_runs = _env_int_alias(
            "ASTEP_TRACKING_GUARD_RUNS",
            "VSTEP_TRACKING_GUARD_RUNS",
            max(4, min(8, int(max(1, requested_tune_eval_runs)))),
            minimum=1,
        )
        if saved_guard_runs != int(expected_guard_runs):
            return False
    return True


def _tuning_objective_mode() -> str:
    return os.getenv("ASTEP_TUNE_OBJECTIVE", "stability_oracle").strip().lower()


def _min_tracking_output_alpha() -> float:
    # y[k] = (1-alpha)y[k-1] + alpha*x[k], so tau ~= dt/alpha.
    # At 10 kHz, alpha=1e-3 is a ~100 ms time constant; lower values
    # effectively turn an estimator into a nominal-hold baseline.
    return _env_float_alias("ASTEP_MIN_TRACKING_OUTPUT_ALPHA", "VSTEP_MIN_TRACKING_OUTPUT_ALPHA", 1e-3, minimum=1e-6)


def _min_sogi_fll_gamma() -> float:
    # Very small FLL gains win amplitude-only tuning by refusing to track
    # frequency.  Keep the lower bound in a range that still follows a
    # 1 Hz step within the benchmark window.
    return _env_float_alias("ASTEP_MIN_SOGI_FLL_GAMMA", "VSTEP_MIN_SOGI_FLL_GAMMA", 5.0, minimum=1e-3)


def _fixed_frequency_bounds() -> tuple[float, float]:
    """Non-tunable frequency safety rails for amplitude-only stress tests."""
    f_min = _env_float("ASTEP_FIXED_F_MIN_HZ", 40.0, minimum=1e-6)
    f_max = _env_float("ASTEP_FIXED_F_MAX_HZ", 80.0, minimum=1e-6)
    if f_max <= f_min:
        f_min, f_max = 40.0, 80.0
    return float(f_min), float(f_max)


def _fixed_frequency_bound_params(*, min_key: str = "f_min_hz", max_key: str = "f_max_hz") -> dict[str, float]:
    f_min, f_max = _fixed_frequency_bounds()
    return {min_key: f_min, max_key: f_max}


def _fixed_freq_dev_limit_hz() -> float:
    return _env_float("ASTEP_FIXED_FREQ_DEV_LIMIT_HZ", 10.0, minimum=0.1)


def _fixed_rocof_limit_hz_s() -> float:
    return _env_float("ASTEP_FIXED_ROCOF_LIMIT_HZ_S", 20.0, minimum=0.1)


def _freq_bound_hit_rates(f_hat: np.ndarray, params: dict[str, Any]) -> tuple[float, float, float]:
    f_min = params.get("f_min_hz", params.get("freq_min_hz"))
    f_max = params.get("f_max_hz", params.get("freq_max_hz"))
    if f_min is None and f_max is None:
        return 0.0, 0.0, 0.0
    y = np.asarray(f_hat, dtype=float)
    finite = np.isfinite(y)
    if not np.any(finite):
        return 1.0, 1.0 if f_min is not None else 0.0, 1.0 if f_max is not None else 0.0
    y = y[finite]
    tol = _env_float("ASTEP_FREQ_BOUND_HIT_TOL_HZ", 0.02, minimum=0.0)
    lower = np.zeros(len(y), dtype=bool)
    upper = np.zeros(len(y), dtype=bool)
    if f_min is not None:
        lower = np.abs(y - float(f_min)) <= tol
    if f_max is not None:
        upper = np.abs(y - float(f_max)) <= tol
    any_hit = lower | upper
    return float(np.mean(any_hit)), float(np.mean(lower)), float(np.mean(upper))


def _tracking_guard_enabled(est_name: str) -> bool:
    if not _env_bool_alias("ASTEP_TRACKING_GUARD", "VSTEP_TRACKING_GUARD", True):
        return False
    include = _csv_set("ASTEP_TRACKING_GUARD_ESTIMATORS")
    if include:
        include_norm = {_normalize_estimator_token(x) for x in include}
    else:
        include_norm = {_normalize_estimator_token(x) for x in TRACKING_GUARD_ESTIMATORS}
    return _normalize_estimator_token(est_name) in include_norm


def _failed_eval_profile(reason: str = "evaluation_failed") -> dict[str, Any]:
    return {
        "score": 1e6,
        "amplitude_stability_score": 1e6,
        "stability_score": 1e6,
        "rmse_mean": float("nan"),
        "rmse_median": float("nan"),
        "rmse_p90": float("nan"),
        "rmse_p95": float("nan"),
        "rmse_cvar90": float("nan"),
        "peak_median": float("nan"),
        "peak_p90": float("nan"),
        "peak_p95": float("nan"),
        "rfe_p90": float("nan"),
        "bound_hit_rate_mean": float("nan"),
        "bound_hit_rate_p90": float("nan"),
        "fail_count": -1,
        "fail_rate": float("nan"),
        "guard_score": float("nan"),
        "guard_fail_count": -1,
        "guard_late_fail_count": -1,
        "reason": reason,
    }


def _evaluate_params_profile(
    est_cls: type,
    params: dict[str, Any],
    scenarios_eval: list[Any],
    eval_start: int,
    *,
    objective_mode: str | None = None,
    tracking_guard_scenarios: list[TuningSignal] | None = None,
    tracking_guard_weight: float = 0.0,
) -> dict[str, Any]:
    try:
        rmses: list[float] = []
        peaks: list[float] = []
        rfe_maxes: list[float] = []
        bound_hit_rates: list[float] = []
        fail_count = 0
        fail_peak_hz = _env_float_alias("ASTEP_TUNE_FAIL_PEAK_HZ", "VSTEP_TUNE_FAIL_PEAK_HZ", 10.0, minimum=0.0)
        fail_rfe_hz_s = _env_float_alias("ASTEP_TUNE_FAIL_RFE_HZ_S", "VSTEP_TUNE_FAIL_RFE_HZ_S", 100.0, minimum=0.0)
        fail_rmse_hz = _env_float_alias("ASTEP_TUNE_FAIL_RMSE_HZ", "VSTEP_TUNE_FAIL_RMSE_HZ", 0.05, minimum=0.0)
        for sc in scenarios_eval:
            est = est_cls(**params)
            f_hat = benchmark._run_estimator(est, sc.v)
            error = f_hat[eval_start:] - sc.f_true[eval_start:]
            rmse = float(np.sqrt(np.mean(error ** 2)))
            peak = float(np.max(np.abs(error)))
            dt = float(sc.t[1] - sc.t[0])
            rfe_max = float(np.max(np.abs(np.diff(error) / dt))) if len(error) > 1 else 0.0
            bound_hit_rate, _lower_hit_rate, _upper_hit_rate = _freq_bound_hit_rates(f_hat[eval_start:], params)
            if not np.isfinite(rmse):
                return _failed_eval_profile("nonfinite_rmse")
            if (rmse > fail_rmse_hz) or (peak > fail_peak_hz) or (rfe_max > fail_rfe_hz_s):
                fail_count += 1
            rmses.append(rmse)
            peaks.append(peak)
            rfe_maxes.append(rfe_max)
            bound_hit_rates.append(bound_hit_rate)
        if not rmses:
            return _failed_eval_profile("no_eval_scenarios")
        fail_rate = float(fail_count) / float(max(1, len(rmses)))
        rmse_med = float(np.median(rmses))
        rmse_mean = float(np.mean(rmses))
        rmse_p90 = float(np.quantile(rmses, 0.90))
        rmse_p95 = float(np.quantile(rmses, 0.95))
        rmse_tail = [x for x in rmses if x >= rmse_p90]
        rmse_cvar90 = float(np.mean(rmse_tail)) if rmse_tail else rmse_p90
        peak_med = float(np.median(peaks)) if peaks else 0.0
        peak_p90 = float(np.quantile(peaks, 0.90)) if peaks else 0.0
        peak_p95 = float(np.quantile(peaks, 0.95)) if peaks else 0.0
        rfe_p90 = float(np.quantile(rfe_maxes, 0.90)) if rfe_maxes else 0.0
        bound_hit_rate_mean = float(np.mean(bound_hit_rates)) if bound_hit_rates else 0.0
        bound_hit_rate_p90 = float(np.quantile(bound_hit_rates, 0.90)) if bound_hit_rates else 0.0
        bound_hit_penalty = _env_float("ASTEP_BOUND_HIT_WEIGHT", 4.0, minimum=0.0) * bound_hit_rate_p90
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
        elif mode in {"stability", "stability_oracle", "stable", "stable_oracle"}:
            # RMSE is the scientific endpoint for this atlas.  RFE is kept as a
            # small tie-breaker so discontinuous frequency chatter is not free,
            # but it cannot dominate the optimizer and hide brittle RMSE tails.
            # Failure pressure is normalized by rate: in severe stress regions
            # almost every candidate can exceed the strict guide threshold, and
            # "19 failures vs. 20 failures" must not beat a far lower RMSE tail.
            fail_weight = _env_float_alias("ASTEP_STABILITY_FAIL_WEIGHT", "VSTEP_STABILITY_FAIL_WEIGHT", 8.0, minimum=0.0)
            score = (
                rmse_med
                + (0.55 * rmse_p90)
                + (0.35 * rmse_p95)
                + (0.35 * rmse_cvar90)
                + (0.04 * peak_p90)
                + (0.02 * peak_p95)
                + (0.0002 * rfe_p90)
                + (fail_weight * fail_rate)
                + bound_hit_penalty
            )
        elif mode in {"tail", "tail_oracle", "cvar", "cvar_oracle"}:
            score = (
                rmse_med
                + (0.35 * rmse_p90)
                + (0.25 * rmse_p95)
                + (0.15 * rmse_cvar90)
                + (0.02 * peak_med)
                + (0.05 * peak_p90)
                + (0.03 * peak_p95)
                + (0.002 * rfe_p90)
                + (25.0 * float(fail_count))
            )
        else:
            # Oracle objective: mainly minimize frequency RMSE while keeping
            # enough tail pressure to avoid brittle seed-specific optima.
            score = (
                rmse_med
                + (0.25 * rmse_p90)
                + (0.02 * peak_med)
                + (5.0 * float(fail_count))
            )
        amplitude_stability_score = (
            rmse_med
            + (0.55 * rmse_p90)
            + (0.35 * rmse_p95)
            + (0.35 * rmse_cvar90)
            + (0.04 * peak_p90)
            + (0.02 * peak_p95)
            + (_env_float_alias("ASTEP_STABILITY_FAIL_WEIGHT", "VSTEP_STABILITY_FAIL_WEIGHT", 8.0, minimum=0.0) * fail_rate)
            + bound_hit_penalty
        )
        stability_score = amplitude_stability_score
        guard_score = float("nan")
        guard_fail_count = 0
        guard_late_fail_count = 0
        if tracking_guard_scenarios and tracking_guard_weight > 0.0:
            guard_rmses: list[float] = []
            guard_late_rmses: list[float] = []
            guard_fail_rmse = _env_float_alias("ASTEP_TRACKING_GUARD_FAIL_RMSE_HZ", "VSTEP_TRACKING_GUARD_FAIL_RMSE_HZ", 0.5, minimum=0.0)
            guard_fail_late_rmse = _env_float_alias("ASTEP_TRACKING_GUARD_FAIL_LATE_RMSE_HZ", "VSTEP_TRACKING_GUARD_FAIL_LATE_RMSE_HZ", 0.40, minimum=0.0)
            guard_late_start_s = _env_float_alias("ASTEP_TRACKING_GUARD_LATE_START_S", "VSTEP_TRACKING_GUARD_LATE_START_S", 1.0, minimum=0.0)
            for sc_guard in tracking_guard_scenarios:
                est_guard = est_cls(**params)
                f_guard = benchmark._run_estimator(est_guard, sc_guard.v)
                err_guard = f_guard[eval_start:] - sc_guard.f_true[eval_start:]
                guard_rmse = float(np.sqrt(np.mean(err_guard ** 2)))
                if len(sc_guard.t) > 1:
                    guard_dt = float(sc_guard.t[1] - sc_guard.t[0])
                else:
                    guard_dt = 1.0 / 10_000.0
                late_start = min(len(f_guard) - 1, max(0, int(round(guard_late_start_s / guard_dt))))
                late_err = f_guard[late_start:] - sc_guard.f_true[late_start:]
                guard_late_rmse = float(np.sqrt(np.mean(late_err ** 2))) if len(late_err) else guard_rmse
                if not np.isfinite(guard_rmse):
                    return _failed_eval_profile("nonfinite_guard_rmse")
                if not np.isfinite(guard_late_rmse):
                    return _failed_eval_profile("nonfinite_guard_late_rmse")
                if guard_rmse > guard_fail_rmse or guard_late_rmse > guard_fail_late_rmse:
                    guard_fail_count += 1
                if guard_late_rmse > guard_fail_late_rmse:
                    guard_late_fail_count += 1
                guard_rmses.append(guard_rmse)
                guard_late_rmses.append(guard_late_rmse)
            if guard_rmses:
                if guard_late_fail_count > 0 and _env_bool_alias("ASTEP_TRACKING_GUARD_HARD_FAIL", "VSTEP_TRACKING_GUARD_HARD_FAIL", True):
                    failed = _failed_eval_profile("tracking_guard_late_fail")
                    failed["guard_fail_count"] = int(guard_fail_count)
                    failed["guard_late_fail_count"] = int(guard_late_fail_count)
                    return failed
                guard_med = float(np.median(guard_rmses))
                guard_p90 = float(np.quantile(guard_rmses, 0.90))
                guard_late_med = float(np.median(guard_late_rmses)) if guard_late_rmses else guard_med
                guard_late_p90 = float(np.quantile(guard_late_rmses, 0.90)) if guard_late_rmses else guard_p90
                guard_score = guard_med + 0.50 * guard_p90 + 0.75 * guard_late_med + 0.25 * guard_late_p90 + 2.0 * float(guard_fail_count)
                score += float(tracking_guard_weight) * guard_score
                stability_score += float(tracking_guard_weight) * guard_score
        if not np.isfinite(score) or not np.isfinite(stability_score):
            return _failed_eval_profile("nonfinite_score")
        return {
            "score": float(score),
            "amplitude_stability_score": float(amplitude_stability_score),
            "stability_score": float(stability_score),
            "rmse_mean": rmse_mean,
            "rmse_median": rmse_med,
            "rmse_p90": rmse_p90,
            "rmse_p95": rmse_p95,
            "rmse_cvar90": rmse_cvar90,
            "peak_median": peak_med,
            "peak_p90": peak_p90,
            "peak_p95": peak_p95,
            "rfe_p90": rfe_p90,
            "bound_hit_rate_mean": bound_hit_rate_mean,
            "bound_hit_rate_p90": bound_hit_rate_p90,
            "fail_count": int(fail_count),
            "fail_rate": fail_rate,
            "guard_score": guard_score,
            "guard_fail_count": int(guard_fail_count),
            "guard_late_fail_count": int(guard_late_fail_count),
            "reason": "ok",
        }
    except Exception:
        return _failed_eval_profile("exception")


def _evaluate_params_rmse(
    est_cls: type,
    params: dict[str, Any],
    scenarios_eval: list[Any],
    eval_start: int,
    *,
    objective_mode: str | None = None,
    tracking_guard_scenarios: list[TuningSignal] | None = None,
    tracking_guard_weight: float = 0.0,
) -> float:
    profile = _evaluate_params_profile(
        est_cls,
        params,
        scenarios_eval,
        eval_start,
        objective_mode=objective_mode,
        tracking_guard_scenarios=tracking_guard_scenarios,
        tracking_guard_weight=tracking_guard_weight,
    )
    score = float(profile.get("score", 1e6))
    return score if np.isfinite(score) else 1e6


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


def _apply_amplitude_step_safety_params(est_name: str, params: dict[str, Any], step_percent: float) -> dict[str, Any]:
    """Apply non-tunable anti-artifact policy to every tuning candidate."""
    out = dict(params)
    if "output_smoothing" in out:
        try:
            out["output_smoothing"] = max(float(out["output_smoothing"]), _min_tracking_output_alpha())
        except Exception:
            out["output_smoothing"] = _min_tracking_output_alpha()

    if est_name in {"RLS", "SOGI-PLL", "SOGI-FLL", "Type-3 SOGI-PLL", "IPDFT"}:
        out.update(_fixed_frequency_bound_params())

    if est_name == "SOGI-FLL" and "gamma" in out:
        try:
            out["gamma"] = max(float(out["gamma"]), _min_sogi_fll_gamma())
        except Exception:
            out["gamma"] = _min_sogi_fll_gamma()

    if est_name == "TKEO" and "input_smoothing" in out:
        try:
            out["input_smoothing"] = min(1.0, max(float(out["input_smoothing"]), 0.08))
        except Exception:
            out["input_smoothing"] = 1.0

    if est_name == "LKF2":
        out["freq_dev_limit_hz"] = _fixed_freq_dev_limit_hz()

    if est_name == "RA-EKF":
        amp_post = _post_step_amp_pu(step_percent)
        out.update(_fixed_frequency_bound_params(min_key="freq_min_hz", max_key="freq_max_hz"))
        out["amp_max"] = max(20.0, 1.5 * amp_post)
        out["rocof_limit_hz_s"] = _fixed_rocof_limit_hz_s()

    return out


def _seeded_tuning_candidates(est_name: str, defaults: dict[str, Any], step_percent: float) -> list[dict[str, Any]]:
    """Known sane anchors so low-budget Optuna runs do not miss valid tracking regimes."""
    defaults = _apply_amplitude_step_safety_params(est_name, defaults, step_percent)
    seeds: list[dict[str, Any]] = [dict(defaults)]

    if est_name == "RLS":
        for lambda_fixed, p0, smooth, r_min in (
            (0.9950, 0.01, 0.03, 0.95),
            (0.9990, 0.01, 0.01, 0.95),
            (0.9995, 0.001, 0.005, 0.95),
            (0.9999, 0.001, 0.003, 0.98),
            (0.9999, 0.0001, 0.001, 0.99),
        ):
            params = dict(defaults)
            params.update(
                {
                    "is_vff": False,
                    "lambda_fixed": lambda_fixed,
                    "alpha_vff": 0.2,
                    "lambda_min": 0.90,
                    "lambda_max": 0.9995,
                    "vff_beta": 0.02,
                    "p0": p0,
                    "normalize_input": False,
                    "amp_lpf_alpha": 0.08,
                    "amp_floor": 0.05,
                    "robust_update": False,
                    "innovation_clip": 3.0,
                    "transient_reject": False,
                    "transient_clip": 3.0,
                    "transient_hold_samples": 0,
                    "pole_radius_min": r_min,
                    "pole_radius_max": 1.0,
                    "output_smoothing": smooth,
                    "f_min_hz": 40.0,
                    "f_max_hz": 80.0,
                }
            )
            seeds.append(params)
    elif est_name == "LKF2":
        for beta, q_scale, r_val, p0 in (
            (75.0, 1e-3, 1.0, 100.0),
            (100.0, 1e-2, 1.0, 1_000.0),
            (200.0, 1e-1, 0.5, 1_000.0),
            (350.0, 1.0, 0.25, 10_000.0),
        ):
            params = dict(defaults)
            params.update(
                {
                    "q_vc": q_scale,
                    "q_vs": q_scale,
                    "r": r_val,
                    "beta": beta,
                    "lpf_mu": 1.0,
                    "omega_leak": 0.995,
                    "freq_dev_limit_hz": 5.0,
                    "p0": p0,
                }
            )
            seeds.append(params)
    elif est_name == "EKF":
        for q_omega, p_omega_hz, alpha in (
            (1e-2, 1.0, 0.01),
            (1e0, 5.0, 0.005),
            (1e1, 10.0, 0.002),
        ):
            params = dict(defaults)
            params.update({"q_omega": q_omega, "p_omega_hz": p_omega_hz, "output_smoothing": alpha})
            seeds.append(params)
    elif est_name == "RA-EKF":
        amp_post = _post_step_amp_pu(step_percent)
        for q_omega, q_rocof, p_omega_hz in (
            (1e-4, 1e-3, 2.0),
            (1e-3, 1e-2, 4.0),
            (1e-2, 5e-2, 5.0),
            (1e-6, 1e-5, 0.5),
        ):
            params = dict(defaults)
            params.update(
                {
                    "q_omega": q_omega,
                    "q_rocof": q_rocof,
                    "p_omega_hz": p_omega_hz,
                    "derivative_noise_scale": max(float(defaults.get("derivative_noise_scale", 200.0)), 50.0),
                    "tau_rocof": 0.10,
                    "amp_max": max(20.0, 1.5 * amp_post),
                    "rocof_limit_hz_s": 20.0,
                }
            )
            seeds.append(params)
    elif est_name == "Type-3 SOGI-PLL":
        for kp, ki, ki2, alpha in (
            (60.0, 0.02, 120.0, 0.005),
            (45.0, 0.2, 0.05, 0.002),
            (90.0, 2700.0, 27000.0, 0.003),
        ):
            params = dict(defaults)
            params.update(
                {
                    "kp": kp,
                    "ki": ki,
                    "ki2": ki2,
                    "k_sogi": 1.414,
                    "err_clip": 1.0,
                    "int1_limit": 5.0,
                    "int2_limit": 5.0,
                    "output_smoothing": alpha,
                    "f_min_hz": 40.0,
                    "f_max_hz": 80.0,
                }
            )
            seeds.append(params)
    elif est_name == "TKEO":
        for smooth, pre_smooth in (
            (0.001, 0.50),
            (0.003, 0.35),
            (0.005, 0.25),
            (0.010, 0.15),
        ):
            params = dict(defaults)
            params.update({"output_smoothing": smooth, "input_smoothing": pre_smooth})
            seeds.append(params)

    # De-duplicate while preserving order.
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for params in seeds:
        params = _apply_amplitude_step_safety_params(est_name, params, step_percent)
        key = json.dumps(benchmark._to_builtin(params), sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(params)
    return unique


def _oracle_enabled(est_name: str) -> bool:
    if not _env_bool("ASTEP_ORACLE_TUNING", True):
        return False
    if est_name == "UKF" and not _env_bool("ASTEP_UKF_ORACLE_TUNING", True):
        return False
    include = _csv_set("ASTEP_ORACLE_ESTIMATORS")
    if not include:
        include = set(AMPLITUDE_ORACLE_ESTIMATORS)
    include_norm = {_normalize_estimator_token(x) for x in include}
    return _normalize_estimator_token(est_name) in include_norm


def _ukf_oracle_enabled() -> bool:
    return _oracle_enabled("UKF")


def _normalize_estimator_token(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _oracle_trial_count(est_name: str, requested_trials: int) -> int:
    key = _sanitize_env_key(est_name)
    base_default = max(300, int(requested_trials))
    if est_name in {"SOGI-PLL", "SOGI-FLL", "IPDFT"}:
        base_default = max(180, int(requested_trials))
    elif est_name == "RLS":
        base_default = max(240, int(requested_trials))
    elif est_name == "Type-3 SOGI-PLL":
        base_default = max(240, int(requested_trials))
    default_floor = _env_int("ASTEP_ORACLE_TRIALS", base_default, minimum=1)
    if est_name == "UKF":
        return _env_int("ASTEP_UKF_ORACLE_TRIALS", max(default_floor, int(requested_trials)), minimum=1)
    return _env_int(f"ASTEP_{key}_ORACLE_TRIALS", max(default_floor, int(requested_trials)), minimum=1)


def _stability_oracle_enabled(est_name: str) -> bool:
    if not _env_bool("ASTEP_STABILITY_ORACLE", True):
        return False
    include = _csv_set("ASTEP_STABILITY_ESTIMATORS")
    if not include:
        include = set(STABILITY_ORACLE_ESTIMATORS)
    include_norm = {_normalize_estimator_token(x) for x in include}
    return _normalize_estimator_token(est_name) in include_norm


def _candidate_key(params: dict[str, Any]) -> str:
    return json.dumps(benchmark._to_builtin(params), sort_keys=True, separators=(",", ":"))


def _finite_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _param_value_distance(key: str, a: Any, b: Any) -> float:
    if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
        return 0.0 if bool(a) == bool(b) else 1.0
    if isinstance(a, str) or isinstance(b, str):
        return 0.0 if str(a) == str(b) else 1.0
    fa = _finite_or_none(a)
    fb = _finite_or_none(b)
    if fa is None or fb is None:
        return 0.0 if a == b else 1.0
    if fa == fb:
        return 0.0

    key_l = key.lower()
    if "lambda" in key_l or key_l in {"rho", "omega_leak"}:
        return min(abs(fa - fb) / 0.02, 10.0)
    if key_l.startswith("f_") or key_l.startswith("freq_") or key_l in {"fmin", "fmax"}:
        return min(abs(fa - fb) / 20.0, 10.0)

    log_like = (
        fa > 0.0
        and fb > 0.0
        and (
            key_l.startswith(("q", "r", "p"))
            or any(token in key_l for token in ("alpha", "beta", "gamma", "sigma", "scale", "gain", "clip", "floor", "limit", "smooth", "epsilon"))
            or max(fa, fb) / max(min(fa, fb), 1e-300) > 10.0
        )
    )
    if log_like:
        return min(abs(math.log10(max(fa, 1e-300)) - math.log10(max(fb, 1e-300))), 10.0)
    return min(abs(fa - fb) / (1.0 + max(abs(fa), abs(fb))), 10.0)


def _param_distance(params: dict[str, Any], anchor: dict[str, Any]) -> float:
    keys = sorted(set(params) | set(anchor))
    if not keys:
        return 0.0
    total = 0.0
    for key in keys:
        if key not in params or key not in anchor:
            total += 1.0
        else:
            total += _param_value_distance(key, params[key], anchor[key])
    return float(total / max(1, len(keys)))


def _compact_profile(profile: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "score",
        "amplitude_stability_score",
        "stability_score",
        "rmse_mean",
        "rmse_median",
        "rmse_p90",
        "rmse_p95",
        "rmse_cvar90",
        "peak_p90",
        "peak_p95",
        "rfe_p90",
        "fail_count",
        "fail_rate",
        "guard_score",
        "guard_fail_count",
        "guard_late_fail_count",
        "reason",
    ]
    return {key: profile.get(key) for key in keep if key in profile}


def _make_candidate_record(
    *,
    source: str,
    step_percent: float,
    params: dict[str, Any],
    profile: dict[str, Any] | None = None,
    training_loss: float | None = None,
    selection_score: float | None = None,
    continuity_penalty: float | None = None,
    trial_number: int | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "source": source,
        "step_percent": float(step_percent),
        "params": benchmark._to_builtin(params),
    }
    if profile is not None:
        record["profile"] = benchmark._to_builtin(_compact_profile(profile))
    if training_loss is not None and math.isfinite(float(training_loss)):
        record["training_loss"] = float(training_loss)
    if selection_score is not None and math.isfinite(float(selection_score)):
        record["selection_score"] = float(selection_score)
    if continuity_penalty is not None and math.isfinite(float(continuity_penalty)):
        record["continuity_penalty"] = float(continuity_penalty)
    if trial_number is not None:
        record["trial_number"] = int(trial_number)
    return record


def _select_stability_neighbor_records(records: list[dict[str, Any]], step_percent: float) -> list[dict[str, Any]]:
    if not records:
        return []
    max_records = _env_int("ASTEP_STABILITY_MAX_NEIGHBOR_CANDIDATES", 24, minimum=0)
    if max_records <= 0:
        return []
    max_ratio = _env_float("ASTEP_STABILITY_NEIGHBOR_STEP_RATIO", 2.5, minimum=1.0)
    current = max(float(step_percent), 1e-9)
    deduped: list[tuple[float, dict[str, Any]]] = []
    seen: set[str] = set()
    for record in records:
        params = record.get("params")
        if not isinstance(params, dict):
            continue
        try:
            source_step = max(float(record.get("step_percent", current)), 1e-9)
        except Exception:
            source_step = current
        ratio = max(current / source_step, source_step / current)
        if ratio > max_ratio:
            continue
        key = _candidate_key(params)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((abs(math.log(current / source_step)), record))
    deduped.sort(key=lambda item: (item[0], float(item[1].get("selection_score", item[1].get("training_loss", 1e9)) or 1e9)))
    return [record for _distance, record in deduped[:max_records]]


def _ukf_oracle_space(trial: optuna.Trial, step_percent: float = 0.0) -> dict[str, Any]:
    return {
        "q_dc": trial.suggest_float("q_dc", 1e-12, 1e-1, log=True),
        "q_alpha": trial.suggest_float("q_alpha", 1e-12, 1e1, log=True),
        "q_beta": trial.suggest_float("q_beta", 1e-12, 1e1, log=True),
        "q_omega": trial.suggest_float("q_omega", 1e-12, 1e2, log=True),
        "r_meas": trial.suggest_float("r_meas", 1e-8, 1e3, log=True),
        "output_smoothing": trial.suggest_float("output_smoothing", _min_tracking_output_alpha(), 0.7, log=True),
        "alpha_ut": trial.suggest_float("alpha_ut", 0.05, 1.0, log=True),
        "beta_ut": trial.suggest_float("beta_ut", 1.0, 4.0),
        "kappa_ut": trial.suggest_float("kappa_ut", -1.0, 3.0),
        "p_dc": trial.suggest_float("p_dc", 1e-4, 1e3, log=True),
        "p_alpha": trial.suggest_float("p_alpha", 1e-4, 1e3, log=True),
        "p_beta": trial.suggest_float("p_beta", 1e-4, 1e3, log=True),
        "p_omega_hz": trial.suggest_float("p_omega_hz", 0.01, 25.0, log=True),
    }


def _ekf_oracle_space(trial: optuna.Trial, step_percent: float = 0.0) -> dict[str, Any]:
    return {
        "q_dc": trial.suggest_float("q_dc", 1e-12, 1e-1, log=True),
        "q_alpha": trial.suggest_float("q_alpha", 1e-12, 1e1, log=True),
        "q_beta": trial.suggest_float("q_beta", 1e-12, 1e1, log=True),
        "q_omega": trial.suggest_float("q_omega", 1e-12, 1e2, log=True),
        "r_meas": trial.suggest_float("r_meas", 1e-8, 1e3, log=True),
        "output_smoothing": trial.suggest_float("output_smoothing", _min_tracking_output_alpha(), 0.7, log=True),
        "p_dc": trial.suggest_float("p_dc", 1e-4, 1e3, log=True),
        "p_alpha": trial.suggest_float("p_alpha", 1e-4, 1e3, log=True),
        "p_beta": trial.suggest_float("p_beta", 1e-4, 1e3, log=True),
        "p_omega_hz": trial.suggest_float("p_omega_hz", 0.01, 25.0, log=True),
    }


def _rls_oracle_space(trial: optuna.Trial, step_percent: float = 0.0) -> dict[str, Any]:
    return {
        "is_vff": False,
        "lambda_fixed": trial.suggest_float("lambda_fixed", 0.97, 0.99999),
        "alpha_vff": 0.2,
        "lambda_min": 0.90,
        "lambda_max": 0.9995,
        "vff_beta": 0.02,
        "output_smoothing": trial.suggest_float("output_smoothing", _min_tracking_output_alpha(), 0.15, log=True),
        "pole_radius_min": trial.suggest_float("pole_radius_min", 0.94, 0.999),
        "pole_radius_max": trial.suggest_float("pole_radius_max", 0.999, 1.0),
        "p0": trial.suggest_float("p0", 1e-5, 1e4, log=True),
        "normalize_input": False,
        "amp_lpf_alpha": 0.08,
        "amp_floor": 0.05,
        "robust_update": False,
        "innovation_clip": 3.0,
        "transient_reject": False,
        "transient_clip": 3.0,
        "transient_hold_samples": 0,
        **_fixed_frequency_bound_params(),
    }


def _lkf_oracle_space(trial: optuna.Trial, step_percent: float = 0.0) -> dict[str, Any]:
    return {
        "q": trial.suggest_float("q", 1e-10, 1e-1, log=True),
        "r": trial.suggest_float("r", 1e-6, 1e1, log=True),
        "rho": trial.suggest_float("rho", 0.995, 1.0),
        "output_smoothing": trial.suggest_float("output_smoothing", _min_tracking_output_alpha(), 0.2, log=True),
        "phase_lag_samples": trial.suggest_int("phase_lag_samples", 8, 90),
        "normalize_input": True,
        "amp_lpf_alpha": trial.suggest_float("amp_lpf_alpha", 0.005, 1.0, log=True),
        "amp_floor": trial.suggest_float("amp_floor", 0.02, 0.2, log=True),
        "p_x1": trial.suggest_float("p_x1", 1e-4, 1e4, log=True),
        "p_x2": trial.suggest_float("p_x2", 1e-4, 1e4, log=True),
    }


def _lkf2_oracle_space(trial: optuna.Trial, step_percent: float = 0.0) -> dict[str, Any]:
    amp_post = _post_step_amp_pu(step_percent)
    q_hi = max(1e-1, min(10.0, 0.02 * amp_post * amp_post))
    p_hi = max(1e3, min(1e5, 1e3 * amp_post))
    return {
        "q_dc": trial.suggest_float("q_dc", 1e-10, q_hi, log=True),
        "q_vc": trial.suggest_float("q_vc", 1e-10, q_hi, log=True),
        "q_vs": trial.suggest_float("q_vs", 1e-10, q_hi, log=True),
        "r": trial.suggest_float("r", 1e-6, 1e2, log=True),
        "beta": trial.suggest_float("beta", 20.0, 500.0, log=True),
        "lpf_mu": trial.suggest_float("lpf_mu", 0.25, 1.0),
        "omega_leak": trial.suggest_float("omega_leak", 0.98, 0.9999),
        "freq_dev_limit_hz": _fixed_freq_dev_limit_hz(),
        "p0": trial.suggest_float("p0", 1e-2, p_hi, log=True),
    }


def _post_step_amp_pu(step_percent: float) -> float:
    return 1.0 + max(0.0, float(step_percent)) / 100.0


def _ra_ekf_oracle_space(trial: optuna.Trial, step_percent: float = 0.0) -> dict[str, Any]:
    amp_post = _post_step_amp_pu(step_percent)
    amp_max_low = max(2.0, 1.05 * amp_post)
    amp_max_high = max(25.0, 2.0 * amp_post)
    p_amp_high = max(25.0, 4.0 * amp_post * amp_post)
    sigma_v_high = max(20.0, 5.0 * amp_post)
    return {
        "q_theta": trial.suggest_float("q_theta", 1e-12, 1e-1, log=True),
        "q_omega": trial.suggest_float("q_omega", 1e-12, 1e1, log=True),
        "q_A": trial.suggest_float("q_A", 1e-12, 1e2, log=True),
        "q_rocof": trial.suggest_float("q_rocof", 1e-10, 1e1, log=True),
        "r_meas": trial.suggest_float("r_meas", 1e-8, 1e3, log=True),
        "sigma_v": trial.suggest_float("sigma_v", 1e-4, sigma_v_high, log=True),
        "derivative_noise_scale": trial.suggest_float("derivative_noise_scale", 1.0, 500.0, log=True),
        "gamma": trial.suggest_float("gamma", 2.0, 100.0, log=True),
        "deriv_lpf_alpha": trial.suggest_float("deriv_lpf_alpha", 0.001, 0.4),
        "tau_rocof": trial.suggest_float("tau_rocof", 0.01, 2.0, log=True),
        **_fixed_frequency_bound_params(min_key="freq_min_hz", max_key="freq_max_hz"),
        "amp_min": trial.suggest_float("amp_min", 1e-4, 0.25, log=True),
        "amp_max": max(20.0, 1.5 * amp_post),
        "rocof_limit_hz_s": _fixed_rocof_limit_hz_s(),
        "derivative_step_reject": True,
        "dv_step_factor": trial.suggest_float("dv_step_factor", 2.0, 12.0),
        "dv_step_scale": trial.suggest_float("dv_step_scale", 20.0, 500.0, log=True),
        "p_theta": trial.suggest_float("p_theta", 1e-4, 10.0, log=True),
        "p_omega_hz": trial.suggest_float("p_omega_hz", 0.01, 10.0, log=True),
        "p_amp": trial.suggest_float("p_amp", 1e-3, p_amp_high, log=True),
        "p_rocof_hz_s": trial.suggest_float("p_rocof_hz_s", 0.01, 50.0, log=True),
    }


def _sogi_pll_oracle_space(trial: optuna.Trial, step_percent: float = 0.0) -> dict[str, Any]:
    return {
        "settle_time": trial.suggest_float("settle_time", 0.002, 0.5, log=True),
        "k_sogi": trial.suggest_float("k_sogi", 0.05, 8.0),
        "output_smoothing": trial.suggest_float("output_smoothing", _min_tracking_output_alpha(), 0.8, log=True),
        "kp_scale": trial.suggest_float("kp_scale", 0.05, 100.0, log=True),
        "ki_scale": trial.suggest_float("ki_scale", 0.005, 100.0, log=True),
        **_fixed_frequency_bound_params(),
    }


def _sogi_fll_oracle_space(trial: optuna.Trial, step_percent: float = 0.0) -> dict[str, Any]:
    return {
        "gamma": trial.suggest_float("gamma", _min_sogi_fll_gamma(), 1e5, log=True),
        "k_sogi": trial.suggest_float("k_sogi", 0.05, 10.0),
        "normalize_amplitude": trial.suggest_categorical("normalize_amplitude", [True, False]),
        "amp_epsilon": trial.suggest_float("amp_epsilon", 1e-4, 10.0, log=True),
        "output_smoothing": trial.suggest_float("output_smoothing", _min_tracking_output_alpha(), 0.8, log=True),
        **_fixed_frequency_bound_params(),
    }


def _type3_sogi_pll_oracle_space(trial: optuna.Trial, step_percent: float = 0.0) -> dict[str, Any]:
    return {
        "kp": trial.suggest_float("kp", 5.0, 200.0, log=True),
        "ki": trial.suggest_float("ki", 1e-2, 1e4, log=True),
        "ki2": trial.suggest_float("ki2", 1e-2, 1e5, log=True),
        "k_sogi": trial.suggest_float("k_sogi", 0.5, 6.0),
        "err_clip": trial.suggest_float("err_clip", 0.05, 1.0, log=True),
        "int1_limit": trial.suggest_float("int1_limit", 0.05, 10.0, log=True),
        "int2_limit": trial.suggest_float("int2_limit", 1e-3, 10.0, log=True),
        "output_smoothing": trial.suggest_float("output_smoothing", _min_tracking_output_alpha(), 0.8, log=True),
        **_fixed_frequency_bound_params(),
    }


def _ipdft_oracle_space(trial: optuna.Trial, step_percent: float = 0.0) -> dict[str, Any]:
    return {
        "cycles": trial.suggest_float("cycles", 0.5, 4.0),
        "decim": 1,
        "window": trial.suggest_categorical("window", ["hann", "blackman"]),
        "delta_limit_bins": trial.suggest_float("delta_limit_bins", 0.25, 3.0, log=True),
        "output_smoothing": trial.suggest_float("output_smoothing", _min_tracking_output_alpha(), 0.2, log=True),
        **_fixed_frequency_bound_params(),
        "verbose": False,
    }


ORACLE_SEARCH_SPACES: dict[str, Any] = {
    "EKF": _ekf_oracle_space,
    "UKF": _ukf_oracle_space,
    "LKF": _lkf_oracle_space,
    "LKF2": _lkf2_oracle_space,
    "RA-EKF": _ra_ekf_oracle_space,
    "SOGI-PLL": _sogi_pll_oracle_space,
    "SOGI-FLL": _sogi_fll_oracle_space,
    "Type-3 SOGI-PLL": _type3_sogi_pll_oracle_space,
    "IPDFT": _ipdft_oracle_space,
    "RLS": _rls_oracle_space,
}


def _tune_estimator_for_scenario(
    est_name: str,
    est_cls: type,
    scenario_cls: type,
    *,
    n_trials: int,
    tune_eval_runs: int,
    base_seed: int,
    neighbor_candidate_records: list[dict[str, Any]] | None = None,
    tuning_scenarios_override: list[Any] | None = None,
    policy_step_percent: float | None = None,
    fixed_policy_training_steps: list[float] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    defaults: dict[str, Any] = est_cls.default_params() if hasattr(est_cls, "default_params") else {}
    step_percent = float(policy_step_percent if policy_step_percent is not None else getattr(scenario_cls, "STEP_PERCENT", 0.0))
    defaults = _apply_amplitude_step_safety_params(est_name, defaults, step_percent)
    fixed_policy = fixed_policy_training_steps is not None
    tuning_meta: dict[str, Any] = {
        "mode": "per_scenario_best_tuning",
        "tuning_policy": "fixed_policy" if fixed_policy else "per_step_oracle",
        "objective": _tuning_objective_mode(),
        "step_percent": step_percent,
        "n_trials_requested": int(n_trials),
        "n_trials_executed": 0,
        "tune_eval_runs": int(max(1, tune_eval_runs)),
        "tuning_base_seed": int(base_seed),
        "sampler_mode_effective": None,
        "best_objective": None,
    }
    is_oracle = _oracle_enabled(est_name) and est_name in ORACLE_SEARCH_SPACES
    stability_enabled = bool(is_oracle and _stability_oracle_enabled(est_name))
    neighbor_candidate_records = list(neighbor_candidate_records or [])
    if is_oracle:
        tuning_meta["mode"] = FIXED_POLICY_LABEL if fixed_policy else (UKF_ORACLE_LABEL if est_name == "UKF" else ORACLE_LABEL)
        tuning_meta["interpretation"] = (
            "single hyperparameter policy trained on representative amplitude steps; deployment-transferable within this sweep"
            if fixed_policy
            else "per-scenario practical lower-bound tuning; not deployment-transferable"
        )
        tuning_meta["search_space"] = f"{est_name}_scenario_oracle_space"
        tuning_meta["n_trials_requested_base"] = int(n_trials)
        n_trials = max(int(n_trials), _oracle_trial_count(est_name, int(n_trials)))
        tuning_meta["n_trials_effective_requested"] = int(n_trials)
    elif fixed_policy:
        tuning_meta["mode"] = FIXED_POLICY_LABEL
        tuning_meta["interpretation"] = "single hyperparameter policy trained on representative amplitude steps; deployment-transferable within this sweep"
    if fixed_policy:
        tuning_meta["fixed_policy_training_steps_percent"] = [float(v) for v in fixed_policy_training_steps or []]
        tuning_meta["fixed_policy_eval_runs_per_step"] = int(max(1, tune_eval_runs))
    tuning_meta["stability_oracle"] = {
        "enabled": stability_enabled,
        "selection_metric": "holdout_amplitude_stability_plus_guard_tiebreak_and_continuity" if stability_enabled else "objective_score",
        "neighbor_candidates_received": int(len(neighbor_candidate_records)),
        "param_penalty_weight": _env_float("ASTEP_STABILITY_PARAM_PENALTY", 0.035, minimum=0.0),
        "guard_tie_weight": _env_float("ASTEP_STABILITY_GUARD_TIE_WEIGHT", 0.02, minimum=0.0),
    }

    if est_name not in benchmark.SEARCH_SPACES:
        tuning_meta["reason"] = "no_search_space"
        return defaults, tuning_meta
    if n_trials <= 0:
        tuning_meta["reason"] = "n_trials<=0"
        return defaults, tuning_meta

    if is_oracle:
        raw_space_fn = ORACLE_SEARCH_SPACES[est_name]

        def space_fn(trial: optuna.Trial) -> dict[str, Any]:
            return _apply_amplitude_step_safety_params(est_name, raw_space_fn(trial, step_percent), step_percent)
    else:
        raw_space_fn = benchmark.SEARCH_SPACES[est_name]

        def space_fn(trial: optuna.Trial) -> dict[str, Any]:
            return _apply_amplitude_step_safety_params(est_name, raw_space_fn(trial), step_percent)
    if not benchmark._grid_space_for_estimator(space_fn, n_trials=2):
        tuning_meta["reason"] = "empty_search_space"
        return defaults, tuning_meta

    scenarios_eval = list(tuning_scenarios_override or [])
    if not scenarios_eval:
        scenarios_eval = _build_tuning_scenarios(
            scenario_cls,
            base_seed=base_seed,
            n_runs=max(1, tune_eval_runs),
        )
    fs_dsp = 1.0 / float(scenarios_eval[0].t[1] - scenarios_eval[0].t[0])
    eval_start = int(0.15 * fs_dsp)
    tuning_meta["phase_stratified"] = _env_bool_alias("ASTEP_PHASE_STRATIFIED", "VSTEP_PHASE_STRATIFIED", True)
    tuning_meta["phase_bins"] = _env_int_alias("ASTEP_PHASE_BINS", "VSTEP_PHASE_BINS", max(8, min(32, int(max(1, tune_eval_runs)))), minimum=1)
    tracking_guard_scenarios: list[TuningSignal] = []
    tracking_guard_weight = 0.0
    if _tracking_guard_enabled(est_name):
        guard_runs = _env_int_alias(
            "ASTEP_TRACKING_GUARD_RUNS",
            "VSTEP_TRACKING_GUARD_RUNS",
            max(4, min(8, int(max(1, tune_eval_runs)))),
            minimum=1,
        )
        tracking_guard_weight = _env_float_alias(
            "ASTEP_TRACKING_GUARD_WEIGHT",
            "VSTEP_TRACKING_GUARD_WEIGHT",
            0.35,
            minimum=0.0,
        )
        tracking_guard_scenarios = _build_tracking_guard_scenarios(n_runs=guard_runs)
        tuning_meta["tracking_guard"] = {
            "enabled": True,
            "type": "c0_frequency_step_pm1hz",
            "runs": int(guard_runs),
            "weight": float(tracking_guard_weight),
            "fail_rmse_hz": _env_float_alias("ASTEP_TRACKING_GUARD_FAIL_RMSE_HZ", "VSTEP_TRACKING_GUARD_FAIL_RMSE_HZ", 0.5, minimum=0.0),
            "fail_late_rmse_hz": _env_float_alias("ASTEP_TRACKING_GUARD_FAIL_LATE_RMSE_HZ", "VSTEP_TRACKING_GUARD_FAIL_LATE_RMSE_HZ", 0.40, minimum=0.0),
            "late_start_s": _env_float_alias("ASTEP_TRACKING_GUARD_LATE_START_S", "VSTEP_TRACKING_GUARD_LATE_START_S", 1.0, minimum=0.0),
            "hard_fail_basis": "late_rmse",
            "purpose": "reject nominal-hold parameterizations during amplitude-only tuning",
        }
    else:
        tuning_meta["tracking_guard"] = {"enabled": False}

    seeded_candidates = _seeded_tuning_candidates(est_name, defaults, step_percent)
    seeded_best_params: dict[str, Any] | None = None
    seeded_best_loss = float("inf")
    for cand in seeded_candidates:
        cand = _apply_amplitude_step_safety_params(est_name, cand, step_percent)
        loss = _evaluate_params_rmse(
            est_cls,
            cand,
            scenarios_eval,
            eval_start,
            objective_mode=tuning_meta["objective"],
            tracking_guard_scenarios=tracking_guard_scenarios,
            tracking_guard_weight=tracking_guard_weight,
        )
        if loss < seeded_best_loss:
            seeded_best_loss = float(loss)
            seeded_best_params = dict(cand)
    tuning_meta["seeded_candidates"] = int(len(seeded_candidates))
    tuning_meta["seeded_best_objective"] = float(seeded_best_loss) if math.isfinite(seeded_best_loss) else None

    grid_candidates = _small_grid_candidates(est_name)
    if grid_candidates:
        best_params = seeded_best_params or defaults
        best_loss = seeded_best_loss if seeded_best_params is not None else 1e6
        for cand in grid_candidates:
            params = _apply_amplitude_step_safety_params(est_name, {**defaults, **cand}, step_percent)
            loss = _evaluate_params_rmse(
                est_cls,
                params,
                scenarios_eval,
                eval_start,
                objective_mode=tuning_meta["objective"],
                tracking_guard_scenarios=tracking_guard_scenarios,
                tracking_guard_weight=tracking_guard_weight,
            )
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
        params = _apply_amplitude_step_safety_params(est_name, {**defaults, **suggested}, step_percent)
        return _evaluate_params_rmse(
            est_cls,
            params,
            scenarios_eval,
            eval_start,
            objective_mode=tuning_meta["objective"],
            tracking_guard_scenarios=tracking_guard_scenarios,
            tracking_guard_weight=tracking_guard_weight,
        )

    study, n_trials_exec, sampler_mode_effective = benchmark._build_optuna_study(space_fn=space_fn, n_trials=int(n_trials))
    tuning_meta["n_trials_executed"] = int(n_trials_exec)
    tuning_meta["sampler_mode_effective"] = sampler_mode_effective
    study.optimize(objective, n_trials=n_trials_exec)
    if study.best_value >= 1e6:
        if seeded_best_params is not None and seeded_best_loss < 1e6:
            tuning_meta["reason"] = "optuna_all_trials_failed_seeded_fallback"
            tuning_meta["best_objective"] = float(seeded_best_loss)
            return seeded_best_params, tuning_meta
        tuning_meta["reason"] = "all_trials_failed"
        return defaults, tuning_meta

    best_suggested = space_fn(study.best_trial)
    tuning_meta["best_objective"] = float(study.best_value)
    top_k_requested = _env_int_alias("ASTEP_TUNE_TOPK_REEVAL", "VSTEP_TUNE_TOPK_REEVAL", 1, minimum=1)
    if stability_enabled:
        top_k = max(top_k_requested, _env_int("ASTEP_STABILITY_TOPK", 20, minimum=1))
    else:
        top_k = top_k_requested

    if not stability_enabled and top_k <= 1:
        if seeded_best_params is not None and seeded_best_loss < float(study.best_value):
            tuning_meta["reason"] = "seeded_candidate_beats_optuna"
            tuning_meta["best_objective"] = float(seeded_best_loss)
            return _apply_amplitude_step_safety_params(est_name, seeded_best_params, step_percent), tuning_meta
        return _apply_amplitude_step_safety_params(est_name, {**defaults, **best_suggested}, step_percent), tuning_meta

    completed_trials = [
        trial for trial in study.trials
        if trial.value is not None and math.isfinite(float(trial.value))
    ]
    completed_trials = sorted(completed_trials, key=lambda trial: float(trial.value))
    top_trials = completed_trials[:min(int(top_k), len(completed_trials))]

    candidate_pool: list[dict[str, Any]] = []
    seen_candidates: set[str] = set()

    def add_candidate(
        *,
        source: str,
        params: dict[str, Any],
        source_step: float | None = None,
        training_loss: float | None = None,
        trial_number: int | None = None,
    ) -> None:
        full_params = _apply_amplitude_step_safety_params(est_name, {**defaults, **dict(params)}, step_percent)
        key = _candidate_key(full_params)
        if key in seen_candidates:
            return
        seen_candidates.add(key)
        candidate_pool.append(
            {
                "source": source,
                "source_step": float(step_percent if source_step is None else source_step),
                "params": full_params,
                "training_loss": training_loss,
                "trial_number": trial_number,
            }
        )

    add_candidate(
        source="optuna_best",
        params=best_suggested,
        training_loss=float(study.best_value),
        trial_number=int(study.best_trial.number),
    )
    if seeded_best_params is not None and math.isfinite(seeded_best_loss):
        add_candidate(source="seeded_best", params=seeded_best_params, training_loss=float(seeded_best_loss))
    for trial in top_trials:
        add_candidate(
            source="optuna_topk",
            params=space_fn(trial),
            training_loss=float(trial.value) if trial.value is not None else None,
            trial_number=int(trial.number),
        )
    if stability_enabled:
        for record in neighbor_candidate_records:
            params = record.get("params")
            if not isinstance(params, dict):
                continue
            add_candidate(
                source=f"neighbor:{record.get('source', 'unknown')}",
                params=params,
                source_step=float(record.get("step_percent", step_percent)),
                training_loss=_finite_or_none(record.get("selection_score", record.get("training_loss"))),
            )

    validation_runs = _env_int_alias(
        "ASTEP_TUNE_TOPK_VALIDATION_RUNS",
        "VSTEP_TUNE_TOPK_VALIDATION_RUNS",
        max(4, int(tune_eval_runs)),
        minimum=1,
    )
    if stability_enabled:
        validation_runs = max(validation_runs, _env_int("ASTEP_STABILITY_VALIDATION_RUNS", validation_runs, minimum=1))
    if fixed_policy:
        validation_runs = _env_int(
            "ASTEP_FIXED_POLICY_VALIDATION_RUNS_PER_STEP",
            max(1, min(6, int(validation_runs))),
            minimum=1,
        )
    validation_seed = int(base_seed) + _env_int_alias(
        "ASTEP_TUNE_VALIDATION_SEED_OFFSET",
        "VSTEP_TUNE_VALIDATION_SEED_OFFSET",
        2_000_000,
        minimum=1,
    )
    if fixed_policy:
        validation_scenarios = _build_fixed_policy_tuning_scenarios(
            base_seed=validation_seed,
            n_runs_per_step=validation_runs,
        )
    else:
        validation_scenarios = _build_tuning_scenarios(
            scenario_cls,
            base_seed=validation_seed,
            n_runs=validation_runs,
        )

    anchor_params = [
        dict(record["params"])
        for record in neighbor_candidate_records
        if isinstance(record.get("params"), dict)
    ]
    penalty_weight = _env_float("ASTEP_STABILITY_PARAM_PENALTY", 0.035, minimum=0.0)
    best_validation_loss = float("inf")
    best_validation_params = {**defaults, **best_suggested}
    evaluated_records: list[dict[str, Any]] = []
    selection_score_key = "amplitude_stability_score" if stability_enabled else "score"
    guard_tie_weight = _env_float("ASTEP_STABILITY_GUARD_TIE_WEIGHT", 0.02, minimum=0.0)

    for candidate in candidate_pool:
        params = dict(candidate["params"])
        profile = _evaluate_params_profile(
            est_cls,
            params,
            validation_scenarios,
            eval_start,
            objective_mode=tuning_meta["objective"],
            tracking_guard_scenarios=tracking_guard_scenarios,
            tracking_guard_weight=tracking_guard_weight,
        )
        base_loss = float(profile.get(selection_score_key, profile.get("score", 1e6)))
        if not math.isfinite(base_loss):
            base_loss = 1e6
        continuity_penalty = 0.0
        if stability_enabled and anchor_params and penalty_weight > 0.0:
            continuity_penalty = penalty_weight * min(_param_distance(params, anchor) for anchor in anchor_params)
        guard_tie_penalty = 0.0
        if stability_enabled and guard_tie_weight > 0.0:
            guard_score = _finite_or_none(profile.get("guard_score"))
            if guard_score is not None:
                guard_tie_penalty = float(guard_tie_weight * guard_score)
        selection_loss = float(base_loss + continuity_penalty + guard_tie_penalty)
        if selection_loss < best_validation_loss:
            best_validation_loss = selection_loss
            best_validation_params = params
        evaluated_records.append(
            _make_candidate_record(
                source=str(candidate["source"]),
                step_percent=float(step_percent),
                params=params,
                profile=profile,
                training_loss=_finite_or_none(candidate.get("training_loss")),
                selection_score=selection_loss,
                continuity_penalty=float(continuity_penalty + guard_tie_penalty),
                trial_number=candidate.get("trial_number"),
            )
        )

    evaluated_records = sorted(evaluated_records, key=lambda record: float(record.get("selection_score", 1e6)))
    record_top_n = _env_int("ASTEP_STABILITY_RECORD_TOP_N", 16 if stability_enabled else 8, minimum=0)
    tuning_meta["topk_reeval_k"] = int(len(top_trials))
    tuning_meta["topk_validation_runs"] = int(validation_runs)
    tuning_meta["topk_validation_seed"] = int(validation_seed)
    tuning_meta["topk_best_objective"] = float(best_validation_loss)
    tuning_meta["candidate_pool_size"] = int(len(candidate_pool))
    tuning_meta["selection_score_key"] = selection_score_key
    if evaluated_records:
        selected = evaluated_records[0]
        tuning_meta["selected_candidate_source"] = selected.get("source")
        tuning_meta["selected_candidate_selection_score"] = selected.get("selection_score")
        tuning_meta["selected_candidate_profile"] = selected.get("profile")
    if record_top_n > 0:
        tuning_meta["stability_candidate_records"] = evaluated_records[:record_top_n]
    return best_validation_params, tuning_meta


def _fixed_policy_spec_path(out_dir: Path, estimator_name: str) -> Path:
    return out_dir / "_fixed_policy_tuning" / estimator_name / "fixed_policy_tuning_spec.json"


def _can_reuse_fixed_policy_tuning(
    spec_path: Path,
    *,
    estimator_name: str,
    requested_tune_trials: int,
    requested_eval_runs_per_step: int,
    requested_tuning_base_seed: int,
) -> bool:
    if not spec_path.exists():
        return False
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if spec.get("pipeline_method_version") != PIPELINE_METHOD_VERSION:
        return False
    if spec.get("estimator") != estimator_name:
        return False
    meta = spec.get("tuning_meta", {}) if isinstance(spec.get("tuning_meta", {}), dict) else {}
    if meta.get("mode") != FIXED_POLICY_LABEL:
        return False
    try:
        if int(meta.get("n_trials_requested", -1)) != int(requested_tune_trials):
            return False
        if int(meta.get("tune_eval_runs", -1)) != int(requested_eval_runs_per_step):
            return False
        if int(meta.get("tuning_base_seed", -1)) != int(requested_tuning_base_seed):
            return False
    except Exception:
        return False
    saved_steps = [float(v) for v in meta.get("fixed_policy_training_steps_percent", [])]
    expected_steps = [float(v) for v in _fixed_policy_training_steps()]
    if saved_steps != expected_steps:
        return False
    return isinstance(spec.get("best_params"), dict)


def _tune_estimator_for_fixed_policy(
    est_name: str,
    est_cls: type,
    *,
    n_trials: int,
    tune_eval_runs: int,
    base_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    eval_runs_per_step = _env_int(
        "ASTEP_FIXED_POLICY_EVAL_RUNS_PER_STEP",
        max(2, min(6, int(max(1, tune_eval_runs)))),
        minimum=1,
    )
    training_steps = _fixed_policy_training_steps()
    tuning_scenarios = _build_fixed_policy_tuning_scenarios(
        base_seed=base_seed,
        n_runs_per_step=eval_runs_per_step,
    )
    policy_step = _fixed_policy_safety_step_percent()
    policy_scenario = _create_step_variant(policy_step).scenario_cls
    params, meta = _tune_estimator_for_scenario(
        est_name=est_name,
        est_cls=est_cls,
        scenario_cls=policy_scenario,
        n_trials=n_trials,
        tune_eval_runs=eval_runs_per_step,
        base_seed=base_seed,
        tuning_scenarios_override=tuning_scenarios,
        policy_step_percent=policy_step,
        fixed_policy_training_steps=training_steps,
    )
    meta["mode"] = FIXED_POLICY_LABEL
    meta["tuning_policy"] = "fixed_policy"
    meta["fixed_policy_training_signal_count"] = int(len(tuning_scenarios))
    meta["fixed_policy_safety_step_percent"] = float(policy_step)
    return params, meta


def _load_or_tune_fixed_policy_bank(
    out_dir: Path,
    estimators: dict[str, type],
    run_configs: dict[str, EstimatorRunConfig],
    *,
    resume_run: bool,
    tuning_base_seed: int,
) -> dict[str, dict[str, Any]]:
    bank: dict[str, dict[str, Any]] = {}
    for est_name, est_cls in estimators.items():
        cfg = run_configs[est_name]
        eval_runs_per_step = _env_int(
            "ASTEP_FIXED_POLICY_EVAL_RUNS_PER_STEP",
            max(2, min(6, int(max(1, cfg.tune_eval_runs)))),
            minimum=1,
        )
        spec_path = _fixed_policy_spec_path(out_dir, est_name)
        if resume_run and _can_reuse_fixed_policy_tuning(
            spec_path,
            estimator_name=est_name,
            requested_tune_trials=cfg.tune_trials,
            requested_eval_runs_per_step=eval_runs_per_step,
            requested_tuning_base_seed=tuning_base_seed,
        ):
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            bank[est_name] = spec
            continue

        print(
            f"  - fixed-policy tune {est_name} "
            f"[{cfg.tier}: tune={cfg.tune_trials} trials, steps={_fixed_policy_training_steps()}, runs/step={eval_runs_per_step}]",
            flush=True,
        )
        t0 = time.perf_counter()
        params, meta = _tune_estimator_for_fixed_policy(
            est_name=est_name,
            est_cls=est_cls,
            n_trials=cfg.tune_trials,
            tune_eval_runs=cfg.tune_eval_runs,
            base_seed=tuning_base_seed,
        )
        elapsed_s = float(time.perf_counter() - t0)
        spec = {
            "pipeline_method_version": PIPELINE_METHOD_VERSION,
            "estimator": est_name,
            "family": ESTIMATOR_FAMILIES.get(est_name, "Unknown"),
            "best_params": benchmark._to_builtin(params),
            "tuning_meta": benchmark._to_builtin(meta),
            "run_tier": cfg.tier,
            "tuning_base_seed": int(tuning_base_seed),
            "timing": {"fixed_policy_tuning_elapsed_s": elapsed_s},
        }
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(json.dumps(benchmark._to_builtin(spec), indent=2, ensure_ascii=False), encoding="utf-8")
        bank[est_name] = spec
    return bank


def _fixed_policy_reuse_meta(spec: dict[str, Any], *, step_percent: float) -> dict[str, Any]:
    meta = dict(spec.get("tuning_meta", {}) if isinstance(spec.get("tuning_meta", {}), dict) else {})
    meta["mode"] = FIXED_POLICY_REUSED_LABEL
    meta["tuning_policy"] = "fixed_policy"
    meta["step_percent"] = float(step_percent)
    meta["fixed_policy_source_mode"] = FIXED_POLICY_LABEL
    meta["fixed_policy_interpretation"] = "same selected hyperparameters reused at every amplitude step; this is the primary physical degradation curve"
    return meta


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

    family_order = ["Loop-based", "Model-based", "Window-based", "Adaptive", "Data-driven", "Exotic"]
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

    ref_default = 50.0 if 50.0 in step_ticks else (step_ticks[len(step_ticks) // 2] if step_ticks else 50.0)
    ref_step = _env_float_alias("ASTEP_REFERENCE_STEP_PCT", "VSTEP_REFERENCE_STEP_PCT", ref_default, minimum=0.0)

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

    protocol_title = "Fixed Policy" if _fixed_policy_enabled() else "Per-Step Oracle"
    fig.suptitle(f"{title_prefix}: by estimator family ({protocol_title})", fontsize=13, y=0.995)
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
    policy_note = (
        " Fixed-policy mode: one parameter set per estimator is reused across all amplitudes."
        if _fixed_policy_enabled()
        else " Oracle mode: each amplitude may select a different parameter set; interpret as a lower-bound envelope, not a physical curve."
    )
    fig.text(
        0.5,
        0.965,
        METHODOLOGY_TEXT + policy_note,
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
    value_col = "m1_rmse_hz_mean"
    value_label = "log10 mean RMSE [Hz]"
    higher_is_better = False
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
    cbar = fig.colorbar(im, ax=ax, fraction=0.024, pad=0.012)
    cbar.set_label(value_label)

    ax2 = axes[1]
    summary_rows = []
    guide = _env_float_alias("ASTEP_LIMIT_RMSE_GUIDE", "VSTEP_LIMIT_RMSE_IEEE", 0.05, minimum=0.0)
    strict = _env_float_alias("ASTEP_LIMIT_RMSE_STRICT", "VSTEP_LIMIT_RMSE_IEC", 0.01, minimum=0.0)
    for est, df_est in df.sort_values("step_percent").groupby("estimator", sort=False):
        rmse = pd.to_numeric(df_est.get("m1_rmse_hz_mean"), errors="coerce")
        steps_est = pd.to_numeric(df_est["step_percent"], errors="coerce")
        fail = df_est.loc[rmse > strict, "step_percent"] if rmse is not None else pd.Series(dtype=float)
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
    ax2.set_xlabel("First amplitude step where mean RMSE exceeds strict guide [%]")
    ax2.set_title("Critical Step Summary", loc="left", fontweight="bold")
    _shade_amplitude_regions(ax2, min(steps), max(steps), include_labels=True)
    ax2.grid(True, which="both", alpha=0.22)
    ax2.legend(loc="best", fontsize=6.4, frameon=True)

    protocol_title = "Fixed Policy" if _fixed_policy_enabled() else "Per-Step Oracle"
    fig.suptitle(f"Amplitude-Step Method Atlas ({protocol_title})", fontsize=13, y=0.995)
    _add_methodology_text(fig)
    fig.tight_layout(rect=[0.06, 0.04, 0.98, 0.93])
    png_path = out_dir / SUMMARY_MAP_PNG_NAME
    pdf_path = out_dir / SUMMARY_MAP_PDF_NAME
    fig.savefig(png_path, dpi=240)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [png_path, pdf_path], {}


def _load_phase_dispersion_data(out_dir: Path) -> pd.DataFrame:
    estimators = set(_env_csv("ASTEP_PHASE_DISPERSION_ESTIMATORS") or PHASE_DISPERSION_ESTIMATORS)
    configured_steps = _env_float_csv("ASTEP_PHASE_DISPERSION_STEPS") or list(PHASE_DISPERSION_STEPS)
    step_targets = [float(step) for step in configured_steps]
    rows: list[pd.DataFrame] = []
    for run_spec_path in out_dir.glob("Sweep_AmplitudeStep_*/*/run_spec.json"):
        try:
            spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        estimator = str(spec.get("estimator", run_spec_path.parent.name))
        if estimator not in estimators:
            continue
        try:
            step = float(spec.get("step_percent", float("nan")))
        except Exception:
            continue
        if not math.isfinite(step):
            continue
        if step_targets and not any(abs(step - target) <= max(1e-9, 1e-6 * target) for target in step_targets):
            continue
        summary_files = list(run_spec_path.parent.glob("*_summary.csv"))
        if not summary_files:
            continue
        try:
            df_summary = pd.read_csv(summary_files[0])
        except Exception:
            continue
        required = {"phase_rad", "m1_rmse_hz"}
        if not required.issubset(df_summary.columns):
            continue
        piece_cols = [
            col for col in [
                "run_idx",
                "phase_rad",
                "m1_rmse_hz",
                "m25_post_1cy_rmse_hz",
                "m27_post_100ms_rmse_hz",
                "m29_late_event_rmse_hz",
                "noise_sigma",
                "t_step_s",
            ]
            if col in df_summary.columns
        ]
        piece = df_summary[piece_cols].copy()
        piece["estimator"] = estimator
        piece["family"] = str(spec.get("family", ESTIMATOR_FAMILIES.get(estimator, "Unknown")))
        piece["step_percent"] = step
        phase = pd.to_numeric(piece["phase_rad"], errors="coerce").to_numpy(dtype=float)
        piece["phase_deg"] = np.mod(np.degrees(phase), 360.0)
        rows.append(piece)
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    df = df[np.isfinite(pd.to_numeric(df["phase_deg"], errors="coerce"))]
    df = df[np.isfinite(pd.to_numeric(df["m1_rmse_hz"], errors="coerce"))]
    return df


def _make_phase_dispersion_figure(out_dir: Path) -> plt.Figure | None:
    df = _load_phase_dispersion_data(out_dir)
    if df.empty:
        return None
    estimators = [est for est in PHASE_DISPERSION_ESTIMATORS if est in set(df["estimator"].astype(str))]
    estimators += sorted(set(df["estimator"].astype(str)) - set(estimators))
    if not estimators:
        return None
    ncols = 2
    nrows = int(math.ceil(len(estimators) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12.0, max(4.0, 3.35 * nrows)), sharex=True)
    axes_arr = np.atleast_1d(axes).ravel()
    steps = sorted(float(v) for v in df["step_percent"].dropna().unique())
    cmap = matplotlib.colormaps["viridis"]
    step_colors = {step: cmap(i / max(1, len(steps) - 1)) for i, step in enumerate(steps)}

    for idx, estimator in enumerate(estimators):
        ax = axes_arr[idx]
        df_est = df[df["estimator"].astype(str) == estimator].copy()
        for step, df_step in df_est.sort_values(["step_percent", "phase_deg"]).groupby("step_percent", sort=True):
            x_vals = pd.to_numeric(df_step["phase_deg"], errors="coerce").to_numpy(dtype=float)
            y_vals = pd.to_numeric(df_step["m1_rmse_hz"], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(x_vals) & np.isfinite(y_vals) & (y_vals > 0.0)
            if not np.any(ok):
                continue
            ax.scatter(
                x_vals[ok],
                np.maximum(y_vals[ok], 1e-9),
                s=14,
                alpha=0.70,
                color=step_colors[float(step)],
                label=f"{float(step):g}%",
                edgecolors="none",
            )
        ax.set_yscale("log")
        ax.set_xlim(0.0, 360.0)
        ax.set_xticks([0, 60, 120, 180, 240, 300, 360])
        ax.grid(True, which="both", alpha=0.24)
        ax.set_title(estimator, loc="left", fontweight="bold")
        ax.set_ylabel("RMSE [Hz]")
        ax.set_xlabel("Step phase [deg]")
        if idx == 0:
            ax.legend(loc="best", fontsize=6.2, frameon=True, ncol=2)

    for j in range(len(estimators), len(axes_arr)):
        axes_arr[j].set_visible(False)
    protocol_title = "Fixed Policy" if _fixed_policy_enabled() else "Per-Step Oracle"
    fig.suptitle(f"Phase Dispersion of Sensitive Estimators ({protocol_title})", fontsize=13, y=0.995)
    fig.text(
        0.5,
        0.965,
        "Each marker is one Monte Carlo run. This page exposes phase-dependent AM-to-FM leakage that is hidden by median-only curves.",
        ha="center",
        va="top",
        fontsize=7.4,
        color="#263238",
        wrap=True,
    )
    fig.tight_layout(rect=[0.04, 0.04, 0.98, 0.92])
    return fig


def _save_phase_dispersion_map(out_dir: Path) -> list[Path]:
    fig = _make_phase_dispersion_figure(out_dir)
    if fig is None:
        return []
    png_path = out_dir / PHASE_DISPERSION_PNG_NAME
    pdf_path = out_dir / PHASE_DISPERSION_PDF_NAME
    fig.savefig(png_path, dpi=240)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [png_path, pdf_path]


def _linear_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    yy = y_true[finite]
    pp = y_pred[finite]
    sst = float(np.sum((yy - np.mean(yy)) ** 2))
    if sst <= 0.0:
        return float("nan")
    sse = float(np.sum((yy - pp) ** 2))
    return float(1.0 - sse / sst)


def _fit_hypothesis_models(x: np.ndarray, y: np.ndarray) -> dict[str, dict[str, float | str]]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    x = x[finite]
    y = y[finite]
    if len(x) < 4:
        return {}

    models: dict[str, dict[str, float | str]] = {}

    def add_model(name: str, pred: np.ndarray, n_params: int, description: str) -> None:
        pred = np.asarray(pred, dtype=float)
        ok = np.isfinite(pred)
        if np.count_nonzero(ok) < 4:
            return
        yy = y[ok]
        pp = pred[ok]
        sse = float(np.sum((yy - pp) ** 2))
        n = int(len(yy))
        aic = float(n * math.log(max(sse / max(n, 1), 1e-30)) + 2 * n_params)
        models[name] = {
            "r2_y": _linear_r2(yy, pp),
            "sse_y": sse,
            "aic_y": aic,
            "description": description,
        }

    lx = np.log10(x)
    ly = np.log(y)
    add_model("constant", np.full_like(y, float(np.mean(y))), 1, "RMSE approximately constant over amplitude step.")

    p_linear = np.polyfit(x, y, 1)
    add_model("linear", np.polyval(p_linear, x), 2, f"RMSE = {p_linear[0]:.6g}*step + {p_linear[1]:.6g}")
    models["linear"]["slope"] = float(p_linear[0])

    p_log = np.polyfit(lx, y, 1)
    add_model("logarithmic", np.polyval(p_log, lx), 2, f"RMSE = {p_log[0]:.6g}*log10(step) + {p_log[1]:.6g}")
    models["logarithmic"]["slope"] = float(p_log[0])

    p_exp = np.polyfit(x, ly, 1)
    pred_exp = np.exp(np.polyval(p_exp, x))
    add_model("exponential", pred_exp, 2, f"RMSE = exp({p_exp[0]:.6g}*step + {p_exp[1]:.6g})")
    models["exponential"]["slope_log"] = float(p_exp[0])
    models["exponential"]["r2_log"] = _linear_r2(ly, np.polyval(p_exp, x))

    p_power = np.polyfit(lx, ly, 1)
    pred_power = np.exp(np.polyval(p_power, lx))
    add_model("power_law", pred_power, 2, f"RMSE = exp({p_power[1]:.6g})*step^{p_power[0]:.6g}")
    models["power_law"]["slope_loglog"] = float(p_power[0])
    models["power_law"]["r2_loglog"] = _linear_r2(ly, np.polyval(p_power, lx))

    return models


def _best_hypothesis_model(models: dict[str, dict[str, float | str]]) -> str:
    finite = [
        (name, float(values.get("aic_y", float("inf"))))
        for name, values in models.items()
        if math.isfinite(float(values.get("aic_y", float("inf"))))
    ]
    if not finite:
        return "none"
    return min(finite, key=lambda item: item[1])[0]


def _load_estimator_run_metric(out_dir: Path, estimator: str, metric_col: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for run_spec_path in out_dir.glob("Sweep_AmplitudeStep_*/*/run_spec.json"):
        try:
            spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(spec.get("estimator")) != str(estimator):
            continue
        summary_files = list(run_spec_path.parent.glob("*_summary.csv"))
        if not summary_files:
            continue
        try:
            df_summary = pd.read_csv(summary_files[0])
        except Exception:
            continue
        if metric_col not in df_summary.columns or "run_idx" not in df_summary.columns:
            continue
        step = float(spec.get("step_percent", float("nan")))
        if not math.isfinite(step):
            continue
        piece = df_summary[["run_idx", metric_col]].copy()
        piece["step_percent"] = step
        rows.append(piece)
    if not rows:
        return pd.DataFrame(columns=["run_idx", metric_col, "step_percent"])
    return pd.concat(rows, ignore_index=True)


def _bootstrap_power_slope_ci(run_df: pd.DataFrame, metric_col: str, *, seed: int = 20260516) -> tuple[float, float, float, int]:
    if run_df.empty or metric_col not in run_df.columns:
        return float("nan"), float("nan"), float("nan"), 0
    df = run_df.copy()
    df = df[np.isfinite(pd.to_numeric(df["step_percent"], errors="coerce"))]
    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
    df = df[np.isfinite(df[metric_col]) & (df[metric_col] > 0.0)]
    steps = sorted(float(v) for v in df["step_percent"].dropna().unique())
    if len(steps) < 4:
        return float("nan"), float("nan"), float("nan"), 0

    grouped = {step: df[np.isclose(df["step_percent"].astype(float), step)].copy() for step in steps}
    run_sets = [set(g["run_idx"].astype(int).tolist()) for g in grouped.values() if not g.empty]
    common_runs = sorted(set.intersection(*run_sets)) if run_sets else []
    n_boot = _env_int("ASTEP_HYPOTHESIS_BOOTSTRAP_RUNS", 300, minimum=0)
    if n_boot <= 0:
        return float("nan"), float("nan"), float("nan"), 0

    rng = np.random.default_rng(seed)
    slopes: list[float] = []
    log_steps = np.log10(np.asarray(steps, dtype=float))
    for _ in range(n_boot):
        medians: list[float] = []
        if len(common_runs) >= 3:
            sample_runs = rng.choice(common_runs, size=len(common_runs), replace=True)
            for step in steps:
                g = grouped[step]
                vals = g[g["run_idx"].astype(int).isin(sample_runs)][metric_col].to_numpy(dtype=float)
                if len(vals) == 0:
                    break
                medians.append(float(np.median(vals)))
        else:
            for step in steps:
                vals_all = grouped[step][metric_col].to_numpy(dtype=float)
                if len(vals_all) == 0:
                    break
                vals = rng.choice(vals_all, size=len(vals_all), replace=True)
                medians.append(float(np.median(vals)))
        if len(medians) != len(steps) or min(medians) <= 0.0:
            continue
        slope = float(np.polyfit(log_steps, np.log(np.asarray(medians, dtype=float)), 1)[0])
        if math.isfinite(slope):
            slopes.append(slope)
    if not slopes:
        return float("nan"), float("nan"), float("nan"), 0
    arr = np.asarray(slopes, dtype=float)
    return float(np.median(arr)), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)), int(len(arr))


def _build_tuning_continuity_audit(out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    by_estimator: dict[str, list[dict[str, Any]]] = {}
    for run_spec_path in out_dir.glob("Sweep_AmplitudeStep_*/*/run_spec.json"):
        try:
            spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        estimator = str(spec.get("estimator", ""))
        if not estimator:
            continue
        by_estimator.setdefault(estimator, []).append(spec)

    for estimator, specs in sorted(by_estimator.items()):
        specs = sorted(specs, key=lambda s: float(s.get("step_percent", float("nan"))))
        keys: list[str] = []
        distances: list[float] = []
        fallback_steps: list[float] = []
        modes: set[str] = set()
        selected_sources: set[str] = set()
        prev_params: dict[str, Any] | None = None
        switches = 0
        for spec in specs:
            params = dict(spec.get("best_params", {}) if isinstance(spec.get("best_params"), dict) else {})
            meta = spec.get("tuning_meta", {}) if isinstance(spec.get("tuning_meta", {}), dict) else {}
            modes.add(str(meta.get("mode", "")))
            if meta.get("selected_candidate_source") is not None:
                selected_sources.add(str(meta.get("selected_candidate_source")))
            reason = str(meta.get("reason", ""))
            if "fallback" in reason or "all_trials_failed" in reason:
                try:
                    fallback_steps.append(float(spec.get("step_percent", float("nan"))))
                except Exception:
                    pass
            key = _candidate_key(params)
            keys.append(key)
            if prev_params is not None:
                distance = _param_distance(params, prev_params)
                if math.isfinite(distance):
                    distances.append(float(distance))
                if key != _candidate_key(prev_params):
                    switches += 1
            prev_params = params
        n_points = len(specs)
        switch_rate = float(switches) / float(max(1, n_points - 1))
        max_distance = float(max(distances)) if distances else 0.0
        median_distance = float(np.median(distances)) if distances else 0.0
        modes_clean = sorted(m for m in modes if m)
        if any(m == FIXED_POLICY_REUSED_LABEL for m in modes_clean):
            warning = "fixed_policy_primary_curve"
            claim_status_override = "claim_ready_fixed_policy"
        elif switches >= max(3, n_points // 4):
            warning = "oracle_parameter_switching"
            claim_status_override = "oracle_envelope_not_physical_curve"
        elif fallback_steps:
            warning = "tuning_fallback_detected"
            claim_status_override = "diagnostic_only_tuning_fallback"
        else:
            warning = ""
            claim_status_override = ""
        rows.append(
            {
                "estimator": estimator,
                "n_points": int(n_points),
                "unique_parameter_sets": int(len(set(keys))),
                "parameter_switches": int(switches),
                "switch_rate": switch_rate,
                "max_neighbor_param_distance": max_distance,
                "median_neighbor_param_distance": median_distance,
                "tuning_modes": ",".join(modes_clean),
                "selected_candidate_sources": ",".join(sorted(selected_sources)),
                "fallback_count": int(len(fallback_steps)),
                "fallback_steps_percent": ",".join(f"{v:g}" for v in fallback_steps if math.isfinite(v)),
                "audit_warning": warning,
                "claim_status_override": claim_status_override,
            }
        )
    return pd.DataFrame(rows)


def _save_tuning_continuity_audit(out_dir: Path) -> Path:
    df_audit = _build_tuning_continuity_audit(out_dir)
    path = out_dir / TUNING_CONTINUITY_CSV_NAME
    df_audit.to_csv(path, index=False)
    return path


def _classify_deterioration_regime(
    df_est: pd.DataFrame,
    *,
    out_dir: Path,
    method_version: str,
    metric_col: str = "m1_rmse_hz",
) -> dict[str, Any]:
    estimator = str(df_est["estimator"].iloc[0])
    family = str(df_est["family"].iloc[0]) if "family" in df_est.columns else ESTIMATOR_FAMILIES.get(estimator, "Unknown")
    max_step = _env_float("ASTEP_HYPOTHESIS_MAX_STEP_PCT", 1000.0, minimum=1.0)
    min_points = _env_int("ASTEP_HYPOTHESIS_MIN_POINTS", 6, minimum=4)
    flat_ratio_limit = _env_float("ASTEP_HYPOTHESIS_FLAT_RATIO", 1.25, minimum=1.0)
    flat_slope_abs = _env_float("ASTEP_HYPOTHESIS_FLAT_SLOPE_ABS", 0.12, minimum=0.0)
    improving_slope = -_env_float("ASTEP_HYPOTHESIS_IMPROVING_SLOPE_ABS", 0.12, minimum=0.0)
    improving_tail_ratio = _env_float("ASTEP_HYPOTHESIS_IMPROVING_TAIL_RATIO", 0.90, minimum=0.0)
    power_r2_min = _env_float("ASTEP_HYPOTHESIS_POWER_R2_MIN", 0.85, minimum=0.0)
    weak_power_r2_min = _env_float("ASTEP_HYPOTHESIS_WEAK_POWER_R2_MIN", 0.75, minimum=0.0)
    power_slope_min = _env_float("ASTEP_HYPOTHESIS_POWER_SLOPE_MIN", 0.15, minimum=0.0)
    plateau_ratio_limit = _env_float("ASTEP_HYPOTHESIS_PLATEAU_RATIO", 1.25, minimum=1.0)
    plateau_growth_min = _env_float("ASTEP_HYPOTHESIS_PLATEAU_GROWTH", 1.50, minimum=1.0)
    bound_hit_threshold = _env_float("ASTEP_HYPOTHESIS_BOUND_HIT_RATE", 0.01, minimum=0.0)
    monotone_fraction_min = _env_float("ASTEP_HYPOTHESIS_MONOTONE_FRACTION", 0.85, minimum=0.5)
    flat_ratio_slack = _env_float("ASTEP_HYPOTHESIS_FLAT_RATIO_SLACK", 1.10, minimum=1.0)

    value_col = f"{metric_col}_median" if f"{metric_col}_median" in df_est.columns else f"{metric_col}_mean"
    if value_col not in df_est.columns:
        value_col = "m1_rmse_hz_mean"
    df = df_est.copy()
    df = df[pd.to_numeric(df["step_percent"], errors="coerce") <= max_step].sort_values("step_percent")
    x = pd.to_numeric(df["step_percent"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    x = x[finite]
    y = y[finite]

    base: dict[str, Any] = {
        "estimator": estimator,
        "family": family,
        "metric": metric_col,
        "value_column": value_col,
        "hypothesis_max_step_percent": float(max_step),
        "n_points": int(len(x)),
        "step_min_percent": float(np.min(x)) if len(x) else float("nan"),
        "step_max_percent": float(np.max(x)) if len(x) else float("nan"),
        "method_version": method_version,
    }
    if len(x) < min_points:
        base.update(
            {
                "primary_regime": "Erratic",
                "claim_status": "insufficient_range",
                "interpretation": "Too few amplitude levels for automatic deterioration classification.",
                "h_insensitive_flat": False,
                "h_snr_improving": False,
                "h_power_law_like": False,
                "h_monotone_growth": False,
                "h_saturation_plateau": False,
                "h_erratic": True,
            }
        )
        return base

    models = _fit_hypothesis_models(x, y)
    best_model = _best_hypothesis_model(models)
    power = models.get("power_law", {})
    exp_model = models.get("exponential", {})
    power_slope = float(power.get("slope_loglog", float("nan")))
    power_r2 = float(power.get("r2_loglog", float("nan")))
    exp_r2 = float(exp_model.get("r2_log", float("nan")))

    diffs = np.diff(np.log(np.maximum(y, 1e-30)))
    significant = np.abs(diffs) > math.log(1.07)
    signs = np.sign(diffs[significant])
    large_reversals = int(np.sum(signs[1:] * signs[:-1] < 0)) if len(signs) > 1 else 0
    monotone_tol = math.log(1.05)
    if len(diffs):
        nondecreasing_fraction = float(np.mean(diffs >= -monotone_tol))
        nonincreasing_fraction = float(np.mean(diffs <= monotone_tol))
    else:
        nondecreasing_fraction = 1.0
        nonincreasing_fraction = 1.0
    monotone_fraction = max(nondecreasing_fraction, nonincreasing_fraction)
    if nondecreasing_fraction >= nonincreasing_fraction:
        monotone_direction = "nondecreasing"
    else:
        monotone_direction = "nonincreasing"
    is_smooth_monotone = bool(monotone_fraction >= monotone_fraction_min and large_reversals <= 1)
    total_ratio = float(np.max(y) / max(np.min(y), 1e-30))
    head_n = max(2, min(len(y) // 4, 5))
    tail_n = max(3, min(len(y), max(4, int(math.ceil(0.25 * len(y))))))
    head_median = float(np.median(y[:head_n]))
    tail_median = float(np.median(y[-tail_n:]))
    tail_ratio = float(np.max(y[-tail_n:]) / max(np.min(y[-tail_n:]), 1e-30))
    tail_to_head_ratio = float(tail_median / max(head_median, 1e-30))
    final_to_initial_ratio = float(tail_median / max(float(np.median(y[:tail_n])), 1e-30))

    bound_cols = [
        col for col in [
            "m31_freq_bound_hit_rate_mean",
            "m31_freq_bound_hit_rate_p90",
            "m33_freq_upper_bound_hit_rate_mean",
            "m33_freq_upper_bound_hit_rate_p90",
        ]
        if col in df.columns
    ]
    bound_hit_max = 0.0
    if bound_cols:
        bound_hit_max = float(np.nanmax(pd.to_numeric(df[bound_cols].stack(), errors="coerce").to_numpy(dtype=float)))
        if not math.isfinite(bound_hit_max):
            bound_hit_max = 0.0

    run_df = _load_estimator_run_metric(out_dir, estimator, metric_col)
    slope_boot, slope_ci_lo, slope_ci_hi, slope_boot_n = _bootstrap_power_slope_ci(run_df, metric_col)

    has_v12_protocol = str(method_version) == PIPELINE_METHOD_VERSION
    has_bound_metrics = bool(bound_cols)
    claim_status = "claim_ready_screening" if has_v12_protocol and has_bound_metrics else "diagnostic_only"
    protocol_warning = ""
    if not has_v12_protocol:
        protocol_warning = "Run predates current v12 anti-artifact protocol."
    elif not has_bound_metrics:
        protocol_warning = "Run lacks bound-hit metrics needed for saturation screening."

    h_saturation = bool(
        (bound_hit_max >= bound_hit_threshold)
        or (tail_ratio <= plateau_ratio_limit and tail_to_head_ratio >= plateau_growth_min and power_slope > 0.0)
    )
    h_flat = bool(total_ratio <= flat_ratio_limit and abs(power_slope) <= flat_slope_abs and not h_saturation)
    h_near_flat = bool(
        total_ratio <= flat_ratio_limit * flat_ratio_slack
        and abs(power_slope) <= flat_slope_abs
        and is_smooth_monotone
        and not h_saturation
    )
    h_improving = bool(power_slope <= improving_slope and final_to_initial_ratio <= improving_tail_ratio and not h_saturation)
    h_erratic = bool(
        (large_reversals >= max(3, len(x) // 4) and not is_smooth_monotone)
        or (power_r2 < 0.65 and total_ratio > flat_ratio_limit and not is_smooth_monotone)
    )
    h_power = bool(
        power_slope >= power_slope_min
        and power_r2 >= power_r2_min
        and not h_saturation
        and not h_erratic
    )
    h_monotone_growth = bool(
        power_slope >= power_slope_min
        and power_r2 >= weak_power_r2_min
        and is_smooth_monotone
        and not h_saturation
        and not h_power
        and not h_erratic
    )

    if h_saturation:
        primary = "Saturation/plateau"
        interpretation = "Error reaches a high-step plateau or hits estimator frequency rails; interpret as robustness/saturation limit, not a pure growth law."
    elif h_flat or h_near_flat:
        primary = "Insensitive/flat"
        interpretation = "RMSE changes only within, or just above, the configured practical-equivalence band over the tested amplitude range."
    elif h_improving:
        primary = "SNR-improving"
        interpretation = "RMSE decreases as amplitude grows; likely improved effective SNR or threshold-crossing geometry."
    elif h_power:
        primary = "Power-law-like"
        interpretation = "Log-log trend is approximately linear over the non-saturated range; report exponent b, not an exponential law."
    elif h_monotone_growth:
        primary = "Monotone growth"
        interpretation = "RMSE grows consistently with amplitude, but the strict power-law fit is below threshold; interpret as smooth model-mismatch deterioration."
    else:
        primary = "Erratic"
        interpretation = "Trend is not stable enough for a clean law; repeat under v12/full MC or inspect estimator-specific bifurcations."
        h_erratic = True

    if not has_v12_protocol:
        claim_status = "diagnostic_only_legacy_protocol"
    elif not has_bound_metrics:
        claim_status = "diagnostic_only_missing_bound_metrics"

    base.update(
        {
            "primary_regime": primary,
            "claim_status": claim_status,
            "protocol_warning": protocol_warning,
            "interpretation": interpretation,
            "h_insensitive_flat": h_flat,
            "h_snr_improving": h_improving,
            "h_power_law_like": h_power,
            "h_monotone_growth": h_monotone_growth,
            "h_saturation_plateau": h_saturation,
            "h_erratic": h_erratic,
            "monotone_direction": monotone_direction,
            "monotone_fraction": monotone_fraction,
            "nondecreasing_fraction": nondecreasing_fraction,
            "nonincreasing_fraction": nonincreasing_fraction,
            "total_ratio_max_min": total_ratio,
            "tail_ratio_max_min": tail_ratio,
            "tail_to_head_ratio": tail_to_head_ratio,
            "final_to_initial_ratio": final_to_initial_ratio,
            "large_reversals": large_reversals,
            "best_model_aic": best_model,
            "power_slope_b": power_slope,
            "power_r2_loglog": power_r2,
            "exponential_r2_log": exp_r2,
            "power_slope_bootstrap_median": slope_boot,
            "power_slope_bootstrap_ci95_lo": slope_ci_lo,
            "power_slope_bootstrap_ci95_hi": slope_ci_hi,
            "power_slope_bootstrap_n": slope_boot_n,
            "bound_hit_rate_max": bound_hit_max,
            "threshold_flat_ratio": flat_ratio_limit,
            "threshold_weak_power_r2": weak_power_r2_min,
            "threshold_monotone_fraction": monotone_fraction_min,
            "threshold_power_r2": power_r2_min,
            "threshold_bound_hit_rate": bound_hit_threshold,
        }
    )
    for model_name, values in models.items():
        base[f"{model_name}_r2_y"] = values.get("r2_y")
        base[f"{model_name}_aic_y"] = values.get("aic_y")
        base[f"{model_name}_description"] = values.get("description")
    return base


def _save_deterioration_hypothesis_tests(df_global: pd.DataFrame, out_dir: Path) -> tuple[Path, Path, Path]:
    manifest_path = out_dir / MANIFEST_NAME
    method_version = ""
    if manifest_path.exists():
        try:
            method_version = str(json.loads(manifest_path.read_text(encoding="utf-8")).get("pipeline_method_version", ""))
        except Exception:
            method_version = ""
    if not method_version:
        method_version = PIPELINE_METHOD_VERSION if "m31_freq_bound_hit_rate_mean" in df_global.columns else "unknown"

    rows = [
        _classify_deterioration_regime(df_est, out_dir=out_dir, method_version=method_version)
        for _est, df_est in df_global.groupby("estimator", sort=True)
    ]
    df_tests = pd.DataFrame(rows)
    df_audit = _build_tuning_continuity_audit(out_dir)
    if not df_tests.empty and not df_audit.empty:
        if "protocol_warning" not in df_tests.columns:
            df_tests["protocol_warning"] = ""
        audit_cols = [
            "estimator",
            "unique_parameter_sets",
            "parameter_switches",
            "switch_rate",
            "fallback_count",
            "audit_warning",
            "claim_status_override",
        ]
        df_tests = df_tests.merge(df_audit[[col for col in audit_cols if col in df_audit.columns]], on="estimator", how="left")
        override = df_tests.get("claim_status_override")
        if override is not None:
            mask = override.fillna("").astype(str) != ""
            df_tests.loc[mask, "claim_status"] = override[mask].astype(str)
            df_tests.loc[mask, "protocol_warning"] = (
                df_tests.loc[mask, "protocol_warning"].fillna("").astype(str)
                + " "
                + df_tests.loc[mask, "audit_warning"].fillna("").astype(str)
            ).str.strip()
    csv_path = out_dir / HYPOTHESIS_CSV_NAME
    json_path = out_dir / HYPOTHESIS_JSON_NAME
    md_path = out_dir / HYPOTHESIS_MD_NAME
    df_tests.to_csv(csv_path, index=False)

    counts = df_tests["primary_regime"].value_counts().to_dict() if "primary_regime" in df_tests.columns else {}
    claim_counts = df_tests["claim_status"].value_counts().to_dict() if "claim_status" in df_tests.columns else {}
    payload = {
        "artifact": str(out_dir),
        "method_version": method_version,
        "current_pipeline_method_version": PIPELINE_METHOD_VERSION,
        "max_step_percent": _env_float("ASTEP_HYPOTHESIS_MAX_STEP_PCT", 1000.0, minimum=1.0),
        "metric": "m1_rmse_hz",
        "regime_counts": counts,
        "claim_status_counts": claim_counts,
        "hypotheses": {
            "Insensitive/flat": "Practical-equivalence test: total RMSE ratio and log-log slope stay below configured thresholds.",
            "SNR-improving": "Negative log-log slope and lower high-step RMSE than low-step RMSE.",
            "Power-law-like": "Positive log-log slope with high log-log R2 and no saturation/erratic flags.",
            "Monotone growth": "Mostly monotone positive deterioration with acceptable, but not strict, log-log support.",
            "Saturation/plateau": "High-step plateau and/or frequency-bound hit rate exceeds threshold.",
            "Erratic": "Multiple large reversals or weak smooth-model support; do not claim a law.",
        },
        "csv": HYPOTHESIS_CSV_NAME,
    }
    json_path.write_text(json.dumps(benchmark._to_builtin(payload), indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Deterioration Hypothesis Tests",
        "",
        f"- Artifact: `{out_dir}`",
        f"- Method version: `{method_version}`",
        f"- Metric: `m1_rmse_hz`",
        f"- Max amplitude step included: `{payload['max_step_percent']:g}%`",
        "",
        "These are automatic screening tests. They classify the observed deterioration regime; they do not force monotonicity or smooth the curves.",
        "",
        "## Regime Counts",
        "",
    ]
    for regime, count in counts.items():
        lines.append(f"- `{regime}`: {count}")
    lines.extend(["", "## Per-Estimator Classification", ""])
    if not df_tests.empty:
        show_cols = [
            "estimator", "primary_regime", "claim_status", "power_slope_b",
            "power_r2_loglog", "bound_hit_rate_max", "large_reversals", "interpretation",
        ]
        for _, row in df_tests.sort_values(["primary_regime", "estimator"]).iterrows():
            lines.append(
                f"- `{row['estimator']}`: `{row['primary_regime']}`; "
                f"b={float(row.get('power_slope_b', float('nan'))):.3g}, "
                f"R2_loglog={float(row.get('power_r2_loglog', float('nan'))):.3g}, "
                f"bound_hit_max={float(row.get('bound_hit_rate_max', 0.0)):.3g}, "
                f"status=`{row.get('claim_status', '')}`, "
                f"switches={int(row.get('parameter_switches', 0)) if pd.notna(row.get('parameter_switches', 0)) else 0}."
            )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


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
        ("m25_post_1cy_rmse_hz_mean", "Post-step RMSE, 1 cycle [Hz]", "Post-step 1-cycle RMSE", "log", _env_float_alias("ASTEP_LIMIT_RMSE_GUIDE", "VSTEP_LIMIT_RMSE_IEEE", 0.05, minimum=0.0), _env_float_alias("ASTEP_LIMIT_RMSE_STRICT", "VSTEP_LIMIT_RMSE_IEC", 0.01, minimum=0.0)),
        ("m27_post_100ms_rmse_hz_mean", "Post-step RMSE, 100 ms [Hz]", "Post-step 100-ms RMSE", "log", _env_float_alias("ASTEP_LIMIT_RMSE_GUIDE", "VSTEP_LIMIT_RMSE_IEEE", 0.05, minimum=0.0), _env_float_alias("ASTEP_LIMIT_RMSE_STRICT", "VSTEP_LIMIT_RMSE_IEC", 0.01, minimum=0.0)),
        ("m28_post_event_peak_hz_mean", "Post-step peak FE [Hz]", "Post-step peak FE", "log", _env_float_alias("ASTEP_LIMIT_FE_MAX_GUIDE", "VSTEP_LIMIT_FE_MAX_IEEE", 0.5, minimum=0.0), _env_float_alias("ASTEP_LIMIT_FE_MAX_STRICT", "VSTEP_LIMIT_FE_MAX_IEC", 0.01, minimum=0.0)),
        ("m30_event_settling_time_s_mean", "Event settling time [s]", "Event settling time", "linear", _env_float_alias("ASTEP_LIMIT_TOB_GUIDE", "VSTEP_LIMIT_TOB_IEEE", 0.1, minimum=0.0), _env_float_alias("ASTEP_LIMIT_TOB_STRICT", "VSTEP_LIMIT_TOB_IEC", 0.02, minimum=0.0)),
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
        fig_phase = _make_phase_dispersion_figure(out_dir)
        if fig_phase is not None:
            pdf.savefig(fig_phase)
            plt.close(fig_phase)
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
            "policy": _tuning_policy(),
            "fixed_policy_enabled": _fixed_policy_enabled(),
            "fixed_policy_training_steps_percent": _fixed_policy_training_steps() if _fixed_policy_enabled() else [],
            "fixed_policy_eval_runs_per_step": _env_int(
                "ASTEP_FIXED_POLICY_EVAL_RUNS_PER_STEP",
                max(2, min(6, int(max(1, fast_tune_eval_runs)))),
                minimum=1,
            ) if _fixed_policy_enabled() else None,
            "oracle_tuning_enabled": _env_bool("ASTEP_ORACLE_TUNING", True),
            "oracle_estimators": sorted(_csv_set("ASTEP_ORACLE_ESTIMATORS") or AMPLITUDE_ORACLE_ESTIMATORS),
            "oracle_trials_floor": _env_int("ASTEP_ORACLE_TRIALS", 300, minimum=1),
            "ukf_oracle_tuning_enabled": _ukf_oracle_enabled(),
            "ukf_oracle_trials": _env_int("ASTEP_UKF_ORACLE_TRIALS", _env_int("ASTEP_ORACLE_TRIALS", 300, minimum=1), minimum=1),
            "oracle_interpretation": "per-scenario practical lower-bound tuning; not deployment-transferable",
            "seed_policy": "tuning seeds are offset from final MC seeds to prevent seed reuse leakage",
            "stability_oracle_enabled": _env_bool("ASTEP_STABILITY_ORACLE", True),
            "stability_estimators": sorted(_csv_set("ASTEP_STABILITY_ESTIMATORS") or STABILITY_ORACLE_ESTIMATORS),
            "stability_topk": _env_int("ASTEP_STABILITY_TOPK", 20, minimum=1),
            "stability_param_penalty": _env_float("ASTEP_STABILITY_PARAM_PENALTY", 0.035, minimum=0.0),
            "stability_guard_tie_weight": _env_float("ASTEP_STABILITY_GUARD_TIE_WEIGHT", 0.02, minimum=0.0),
            "stability_fail_weight": _env_float_alias("ASTEP_STABILITY_FAIL_WEIGHT", "VSTEP_STABILITY_FAIL_WEIGHT", 8.0, minimum=0.0),
            "stability_neighbor_step_ratio": _env_float("ASTEP_STABILITY_NEIGHBOR_STEP_RATIO", 2.5, minimum=1.0),
            "fixed_frequency_bounds_hz": list(_fixed_frequency_bounds()),
            "fixed_freq_dev_limit_hz": _fixed_freq_dev_limit_hz(),
            "fixed_rocof_limit_hz_s": _fixed_rocof_limit_hz_s(),
            "bound_hit_weight": _env_float("ASTEP_BOUND_HIT_WEIGHT", 4.0, minimum=0.0),
            "mc_stratified_covariates": _env_bool_alias("ASTEP_MC_STRATIFIED_COVARIATES", "VSTEP_MC_STRATIFIED_COVARIATES", True),
            "noise_mode": os.getenv("ASTEP_NOISE_MODE", os.getenv("VSTEP_NOISE_MODE", "fixed_absolute")),
            "mc_stratification_note": "phase, noise_sigma, and t_step_s are deterministic Latin-hypercube-like covariates by run_idx; the same run_idx maps to the same covariates for every amplitude step.",
            "stability_interpretation": "Top-K holdout selection with neighbor-candidate reevaluation and continuity penalty; tracking guard is a hard late-failure rejection and light tie-breaker, not a dominant amplitude-step score term.",
        },
        "execution": {
            "force_vectorized_engine": _env_bool("ASTEP_FORCE_VECTORIZED_ENGINE", False),
            "slow_vectorized_engine": _env_bool("ASTEP_SLOW_VECTORIZED_ENGINE", False),
            "vectorized_interpretation": "When enabled, slow estimators are evaluated through step_vectorized for practical runtime; CPU timing remains a full process_time pass.",
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
            "deterioration_hypothesis_tests_csv": HYPOTHESIS_CSV_NAME,
            "deterioration_hypothesis_tests_json": HYPOTHESIS_JSON_NAME,
            "deterioration_hypothesis_tests_md": HYPOTHESIS_MD_NAME,
            "tuning_parameter_continuity_csv": TUNING_CONTINUITY_CSV_NAME,
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
        tuning_audit_path = _save_tuning_continuity_audit(OUTPUT_DIR)
        hypothesis_paths = _save_deterioration_hypothesis_tests(df_global=df_global, out_dir=OUTPUT_DIR)
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
        print(f"  - {tuning_audit_path.relative_to(ROOT)}")
        for path in hypothesis_paths:
            print(f"  - {path.relative_to(ROOT)}")
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
    print(f"  Tuning objective: {_tuning_objective_mode()}; policy: {_tuning_policy()}; oracle tuning: {_env_bool('ASTEP_ORACLE_TUNING', True)}")
    print(f"  Stability oracle: {_env_bool('ASTEP_STABILITY_ORACLE', True)}; topK={_env_int('ASTEP_STABILITY_TOPK', 20, minimum=1)}; neighbor ratio={_env_float('ASTEP_STABILITY_NEIGHBOR_STEP_RATIO', 2.5, minimum=1.0):g}")
    print("  Slow tier configs:")
    for est_name, cfg in run_configs.items():
        if cfg.tier == "slow":
            print(f"    - {est_name}: MC={cfg.n_mc_runs}, tune_trials={cfg.tune_trials}, tune_eval_runs={cfg.tune_eval_runs}, cost_reps={cfg.n_cost_reps}")
    print(f"  Output dir: {OUTPUT_DIR}")

    rows_agg: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    stability_candidate_bank: dict[str, list[dict[str, Any]]] = {}
    fixed_policy_bank: dict[str, dict[str, Any]] = {}
    if _fixed_policy_enabled():
        print("\nFixed-policy tuning phase")
        fixed_policy_bank = _load_or_tune_fixed_policy_bank(
            OUTPUT_DIR,
            estimators,
            run_configs,
            resume_run=resume_run,
            tuning_base_seed=tuning_base_seed,
        )
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
                try:
                    run_spec_current = json.loads(run_spec_path.read_text(encoding="utf-8"))
                except Exception:
                    run_spec_current = {}
            else:
                tune_label = "fixed-policy reuse" if _fixed_policy_enabled() else f"tune={cfg.tune_trials}x{cfg.tune_eval_runs}"
                print(f"  - {est_name} [{cfg.tier}: MC={cfg.n_mc_runs}, {tune_label}, cost={cfg.n_cost_reps}]", flush=True)
                t_estimator_start = time.perf_counter()
                t_tune_start = time.perf_counter()
                if _fixed_policy_enabled():
                    fixed_spec = fixed_policy_bank.get(est_name, {})
                    best_params = dict(fixed_spec.get("best_params", {}) if isinstance(fixed_spec.get("best_params"), dict) else {})
                    if not best_params:
                        raise RuntimeError(f"Missing fixed-policy parameters for {est_name}")
                    tuning_meta = _fixed_policy_reuse_meta(fixed_spec, step_percent=sc.step_percent)
                else:
                    neighbor_records = _select_stability_neighbor_records(
                        stability_candidate_bank.get(est_name, []),
                        sc.step_percent,
                    )
                    best_params, tuning_meta = _tune_estimator_for_scenario(
                        est_name=est_name,
                        est_cls=est_cls,
                        scenario_cls=sc.scenario_cls,
                        n_trials=cfg.tune_trials,
                        tune_eval_runs=cfg.tune_eval_runs,
                        base_seed=tuning_base_seed,
                        neighbor_candidate_records=neighbor_records,
                    )
                tune_elapsed_s = float(time.perf_counter() - t_tune_start)
                engine = MonteCarloEngine(
                    scenario_cls=sc.scenario_cls,
                    estimator_cls=est_cls,
                    estimator_params=best_params,
                    n_runs=cfg.n_mc_runs,
                    base_seed=base_seed,
                    n_cost_reps=cfg.n_cost_reps,
                    enforce_standardized_step=not _use_vectorized_engine(est_name, cfg),
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
                    "vectorized_engine": bool(_use_vectorized_engine(est_name, cfg)),
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
                run_spec_current = run_spec
                candidate_records = tuning_meta.get("stability_candidate_records", [])
                if isinstance(candidate_records, list) and candidate_records:
                    audit_rows: list[dict[str, Any]] = []
                    for rank, record in enumerate(candidate_records, start=1):
                        profile = record.get("profile", {}) if isinstance(record.get("profile"), dict) else {}
                        audit_rows.append(
                            {
                                "rank": rank,
                                "source": record.get("source"),
                                "step_percent": float(sc.step_percent),
                                "selection_score": record.get("selection_score"),
                                "continuity_penalty": record.get("continuity_penalty"),
                                "training_loss": record.get("training_loss"),
                                "rmse_median": profile.get("rmse_median"),
                                "rmse_p90": profile.get("rmse_p90"),
                                "rmse_cvar90": profile.get("rmse_cvar90"),
                                "peak_p90": profile.get("peak_p90"),
                                "bound_hit_rate_p90": profile.get("bound_hit_rate_p90"),
                                "fail_count": profile.get("fail_count"),
                                "fail_rate": profile.get("fail_rate"),
                                "params_json": json.dumps(benchmark._to_builtin(record.get("params", {})), sort_keys=True),
                            }
                        )
                    pd.DataFrame(audit_rows).to_csv(out_dir / "tuning_candidates.csv", index=False)

            if run_spec_current:
                tune_meta_current = run_spec_current.get("tuning_meta", {})
                records_current = []
                if isinstance(tune_meta_current, dict):
                    raw_records = tune_meta_current.get("stability_candidate_records", [])
                    if isinstance(raw_records, list):
                        records_current.extend([record for record in raw_records if isinstance(record, dict)])
                best_params_current = run_spec_current.get("best_params")
                if isinstance(best_params_current, dict):
                    records_current.append(
                        _make_candidate_record(
                            source="selected",
                            step_percent=float(sc.step_percent),
                            params=best_params_current,
                            selection_score=None,
                        )
                    )
                if records_current:
                    bank = stability_candidate_bank.setdefault(est_name, [])
                    bank.extend(records_current)
                    max_bank = _env_int("ASTEP_STABILITY_BANK_MAX_PER_ESTIMATOR", 160, minimum=1)
                    stability_candidate_bank[est_name] = bank[-max_bank:]

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
    phase_dispersion_paths = _save_phase_dispersion_map(OUTPUT_DIR)
    tuning_audit_path = _save_tuning_continuity_audit(OUTPUT_DIR)
    hypothesis_paths = _save_deterioration_hypothesis_tests(df_global=df_global, out_dir=OUTPUT_DIR)
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
    for path in phase_dispersion_paths:
        print(f"  - {path.relative_to(ROOT)}")
    print(f"  - {multipage_pdf_path.relative_to(ROOT)}")
    print(f"  - {tuning_audit_path.relative_to(ROOT)}")
    for path in hypothesis_paths:
        print(f"  - {path.relative_to(ROOT)}")
    print(f"  - {legend_map_path.relative_to(ROOT)}")
    print(f"  - {manifest_path.relative_to(ROOT)}")
    print(f"\n[DONE] Amplitude-step sweep completed in {elapsed:.1f} min.")


if __name__ == "__main__":
    main()
