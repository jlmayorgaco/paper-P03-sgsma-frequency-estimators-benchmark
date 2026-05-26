from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
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
from openfreqbench.artifacts import (
    build_artifact_index,
    write_artifact_index,
    write_environment_report,
    write_evidence_manifest,
    write_paper_traceability,
)
from openfreqbench.reproducibility import git_manifest, sha256_file
from pipelines.benchmark_definition import ESTIMATOR_FAMILIES, load_active_estimators
import pipelines.full_mc_benchmark as benchmark
from scenarios.ibr_harmonics_large import IBRHarmonicsLargeScenario
from scenarios.ibr_harmonics_medium import IBRHarmonicsMediumScenario
from scenarios.ieee_freq_ramp import IEEEFreqRampScenario
from scenarios.ieee_freq_step import IEEEFreqStepScenario
from scenarios.ieee_mag_step import IEEEMagStepScenario
from scenarios.ieee_modulation_am import IEEEModulationAMScenario
from scenarios.ieee_modulation_fm import IEEEModulationFMScenario
from scenarios.ieee_phase_jump_20 import IEEEPhaseJump20Scenario
from scenarios.ieee_single_sinwave import IEEESingleSinWaveScenario


METHOD_VERSION = "atlas_sweep_v2_2026_05_25_phase_modulation_p0"

GLOBAL_CSV_NAME = "global_metrics_report.csv"
RMSE_EST_CSV_NAME = "rmse_by_estimator.csv"
RMSE_FAM_CSV_NAME = "rmse_by_family.csv"
TIMING_CSV_NAME = "timing_profile.csv"
TUNING_CONTINUITY_CSV_NAME = "tuning_parameter_continuity.csv"
HYPOTHESIS_CSV_NAME = "hypothesis_results.csv"
BENCHMARK_REPORT_NAME = "benchmark_report.json"
MANIFEST_NAME = "manifest.json"
MULTIPAGE_PDF_NAME = "metrics_dashboard_multipage.pdf"
RMSE_FAMILY_PDF_NAME = "rmse_deterioration_by_family.pdf"
RMSE_FAMILY_PNG_NAME = "rmse_deterioration_by_family.png"
RMSE_ALL_ESTIMATORS_PDF_NAME = "rmse_all_estimators_small_multiples.pdf"
RMSE_ALL_ESTIMATORS_PNG_NAME = "rmse_all_estimators_small_multiples.png"
METHOD_MAP_PDF_NAME = "atlas_method_map.pdf"
METHOD_MAP_PNG_NAME = "atlas_method_map.png"
ASYMMETRY_PDF_NAME = "atlas_sign_asymmetry.pdf"
ASYMMETRY_PNG_NAME = "atlas_sign_asymmetry.png"
PARETO_PDF_NAME = "atlas_accuracy_latency_cpu_pareto.pdf"
PARETO_PNG_NAME = "atlas_accuracy_latency_cpu_pareto.png"
LEGEND_CSV_NAME = "rmse_plot_method_legend.csv"
READINESS_JSON_NAME = "atlas_readiness_report.json"
READINESS_MD_NAME = "atlas_readiness_report.md"

CANONICAL_ESTIMATORS = (
    "ZCD,IPDFT,TFT,RLS,PLL,SOGI-PLL,SOGI-FLL,Type-3 SOGI-PLL,"
    "LKF,LKF2,EKF,UKF,RA-EKF,TKEO,Prony,ESPRIT,Koopman (RK-DPMU),PI-GRU"
)
REQUIRED_ATLAS_SWEEPS = (
    "magnitude_step",
    "rocof",
    "frequency_step",
    "phase_jump_sweep",
    "modulation_am_sweep",
    "modulation_fm_sweep",
    "harmonics",
    "interharmonics",
    "noise_snr",
)
PAPER_GRADE_MIN_RUNS = 30
JOURNAL_GRADE_MIN_RUNS = 100
MIN_LEVELS_PER_SWEEP = 4
PAPER_READY_POLICIES = {"fixed_policy"}

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

FAMILY_PALETTE = {
    "Model-based": "#1565C0",
    "Loop-based": "#2E7D32",
    "Window-based": "#E65100",
    "Adaptive": "#6A1B9A",
    "Data-driven": "#B71C1C",
    "Exotic": "#455A64",
    "Unknown": "#616161",
}

FAMILY_ORDER = ["Loop-based", "Model-based", "Window-based", "Adaptive", "Data-driven", "Exotic"]

HIGH_CONTRAST_ESTIMATOR_COLORS = {
    "PLL": "#D00000",
    "SOGI-PLL": "#008000",
    "SOGI-FLL": "#FFB000",
    "Type-3 SOGI-PLL": "#0072B2",
    "ZCD": "#6F2DBD",
    "EKF": "#E69F00",
    "LKF": "#D55E00",
    "LKF2": "#009E73",
    "RA-EKF": "#111111",
    "UKF": "#7B2CBF",
    "IPDFT": "#FF7F0E",
    "TFT": "#0057B8",
    "RLS": "#5A189A",
    "TKEO": "#E63946",
    "Prony": "#8C564B",
    "ESPRIT": "#17BECF",
    "Koopman (RK-DPMU)": "#004E98",
    "PI-GRU": "#E7298A",
    "MUSIC": "#7F7F7F",
}

HIGH_CONTRAST_FALLBACK_COLORS = (
    "#D00000",
    "#008000",
    "#FFB000",
    "#0072B2",
    "#6F2DBD",
    "#D55E00",
    "#009E73",
    "#111111",
    "#E63946",
    "#17BECF",
    "#8C564B",
)

HIGH_CONTRAST_ESTIMATOR_MARKERS = {
    "PLL": "o",
    "SOGI-PLL": "s",
    "SOGI-FLL": "^",
    "Type-3 SOGI-PLL": "D",
    "ZCD": "P",
    "EKF": "o",
    "LKF": "s",
    "LKF2": "^",
    "RA-EKF": "X",
    "UKF": "D",
    "IPDFT": "o",
    "TFT": "s",
    "RLS": "o",
    "TKEO": "s",
    "Prony": "^",
    "ESPRIT": "D",
    "Koopman (RK-DPMU)": "P",
    "PI-GRU": "X",
    "MUSIC": "v",
}

HIGH_CONTRAST_FALLBACK_MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<", ">")

SEVERITY_REGIONS: dict[str, tuple[tuple[str, float, float, str], ...]] = {
    "magnitude_step": (
        ("Voltage PMU", 1.0, 10.0, "#66BB6A"),
        ("Grid stress", 10.0, 25.0, "#DCE775"),
        ("IBR normal", 25.0, 100.0, "#FDD835"),
        ("IBR stress", 100.0, 500.0, "#FFB74D"),
        ("Mega stress", 500.0, 1000.0, "#EF5350"),
    ),
    "rocof": (
        ("Low", 0.10, 1.00, "#66BB6A"),
        ("Standard-like", 1.00, 3.00, "#DCE775"),
        ("IBR stress", 3.00, 10.00, "#FDD835"),
        ("Severe", 10.00, 30.00, "#FFB74D"),
        ("Extreme", 30.00, 50.00, "#EF5350"),
    ),
    "frequency_step": (
        ("Small", 0.05, 0.20, "#66BB6A"),
        ("Nominal", 0.20, 1.00, "#FDD835"),
        ("Severe", 1.00, 3.00, "#FFB74D"),
        ("Extreme", 3.00, 5.00, "#EF5350"),
    ),
    "phase_jump_sweep": (
        ("Small", 5.0, 10.0, "#66BB6A"),
        ("IEEE 1547", 10.0, 20.0, "#DCE775"),
        ("Stress", 20.0, 45.0, "#FDD835"),
        ("Severe", 45.0, 60.0, "#EF5350"),
    ),
    "modulation_am_sweep": (
        ("Slow", 0.10, 0.50, "#66BB6A"),
        ("Reference", 0.50, 2.00, "#DCE775"),
        ("Fast", 2.00, 5.00, "#FDD835"),
        ("Severe", 5.00, 10.00, "#EF5350"),
    ),
    "modulation_fm_sweep": (
        ("Slow", 0.10, 0.50, "#66BB6A"),
        ("Reference", 0.50, 2.00, "#DCE775"),
        ("Fast", 2.00, 5.00, "#FDD835"),
        ("Severe", 5.00, 10.00, "#EF5350"),
    ),
    "harmonics": (
        ("Low THD", 1.0, 3.0, "#66BB6A"),
        ("Reference", 3.0, 5.0, "#DCE775"),
        ("High THD", 5.0, 10.0, "#FDD835"),
        ("Severe", 10.0, 20.0, "#FFB74D"),
        ("Extreme", 20.0, 30.0, "#EF5350"),
    ),
    "interharmonics": (
        ("Trace", 0.5, 2.0, "#66BB6A"),
        ("Reference", 2.0, 5.0, "#DCE775"),
        ("High", 5.0, 10.0, "#FDD835"),
        ("Severe", 10.0, 20.0, "#EF5350"),
    ),
    "noise_snr": (
        ("Low noise", 0.0001, 0.001, "#66BB6A"),
        ("Nominal", 0.001, 0.003, "#DCE775"),
        ("High", 0.003, 0.03, "#FDD835"),
        ("Severe", 0.03, 0.10, "#EF5350"),
    ),
}


@dataclass(frozen=True)
class SweepSpec:
    key: str
    label: str
    x_col: str
    x_label: str
    signed_col: str
    default_levels: tuple[float, ...]
    reference_value: float
    methodology: str
    directional: bool = True


@dataclass(frozen=True)
class AtlasScenario:
    sweep_key: str
    sweep_label: str
    scenario_name: str
    scenario_cls: type
    signed_value: float
    abs_value: float
    direction: str
    params: dict[str, Any]


SWEEP_SPECS: dict[str, SweepSpec] = {
    "magnitude_step": SweepSpec(
        key="magnitude_step",
        label="Magnitude Step",
        x_col="abs_step_percent",
        x_label="|Magnitude step| [%]",
        signed_col="step_percent",
        default_levels=(1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0, 35.0, 50.0, 75.0, 90.0),
        reference_value=10.0,
        methodology=(
            "Pure magnitude step: true frequency remains nominal. Frequency errors quantify "
            "AM-to-FM cross-sensitivity and numerical robustness. Upward steps are swells; "
            "downward steps are sags. The default sag grid stops below 100% to avoid a zero-voltage test."
        ),
    ),
    "rocof": SweepSpec(
        key="rocof",
        label="RoCoF Ramp",
        x_col="abs_rocof_hz_s",
        x_label="|RoCoF| [Hz/s]",
        signed_col="rocof_hz_s",
        default_levels=(0.1, 0.5, 1.0, 3.0, 5.0, 10.0, 20.0, 50.0),
        reference_value=3.0,
        methodology=(
            "Frequency-ramp RoCoF atlas: true frequency changes linearly during the ramp and "
            "then holds. Curves expose dynamic tracking error, RoCoF sensitivity, latency and "
            "directional bias."
        ),
    ),
    "frequency_step": SweepSpec(
        key="frequency_step",
        label="Frequency Step",
        x_col="abs_step_hz",
        x_label="|Frequency step| [Hz]",
        signed_col="step_hz",
        default_levels=(0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0),
        reference_value=1.0,
        methodology=(
            "C0-continuous frequency-step atlas: true frequency changes abruptly without an "
            "artificial phase jump. Curves expose transient tracking, overshoot and settling."
        ),
    ),
    "phase_jump_sweep": SweepSpec(
        key="phase_jump_sweep",
        label="Phase Jump",
        x_col="abs_phase_jump_deg",
        x_label="|Phase jump| [deg]",
        signed_col="phase_jump_deg",
        default_levels=(5.0, 10.0, 20.0, 30.0, 45.0, 60.0),
        reference_value=20.0,
        methodology=(
            "Phase-jump atlas: true frequency remains nominal while voltage phase changes "
            "instantaneously. Frequency errors quantify phase-discontinuity rejection, "
            "post-event recovery and sign asymmetry."
        ),
    ),
    "modulation_am_sweep": SweepSpec(
        key="modulation_am_sweep",
        label="AM Modulation",
        x_col="modulation_frequency_hz",
        x_label="AM modulation frequency [Hz]",
        signed_col="modulation_frequency_hz",
        default_levels=(0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0),
        reference_value=2.0,
        methodology=(
            "Pure AM atlas: true frequency is fixed at nominal while the voltage envelope "
            "is sinusoidally modulated at a fixed depth. Frequency error is AM-to-FM "
            "cross-coupling rather than tracking error."
        ),
        directional=False,
    ),
    "modulation_fm_sweep": SweepSpec(
        key="modulation_fm_sweep",
        label="FM Modulation",
        x_col="modulation_frequency_hz",
        x_label="FM modulation frequency [Hz]",
        signed_col="modulation_frequency_hz",
        default_levels=(0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0),
        reference_value=2.0,
        methodology=(
            "Pure FM atlas: amplitude is fixed and true frequency oscillates sinusoidally "
            "with a fixed peak deviation. Curves expose estimator bandwidth, attenuation, "
            "phase lag and latency-driven tracking limits."
        ),
        directional=False,
    ),
    "harmonics": SweepSpec(
        key="harmonics",
        label="Integer Harmonics",
        x_col="thd_percent",
        x_label="Integer-harmonic THD [%]",
        signed_col="thd_percent",
        default_levels=(1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0),
        reference_value=5.0,
        methodology=(
            "Integer-harmonic isolation sweep: the true frequency is fixed at nominal and "
            "integer harmonic coefficients are scaled to a target THD. Interharmonics, "
            "subharmonics, impulses and frequency events are disabled so degradation can be "
            "attributed to harmonic distortion rather than mixed IBR artifacts."
        ),
        directional=False,
    ),
    "interharmonics": SweepSpec(
        key="interharmonics",
        label="Interharmonics",
        x_col="interharmonic_percent",
        x_label="Interharmonic amplitude [%]",
        signed_col="interharmonic_percent",
        default_levels=(0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0),
        reference_value=2.0,
        methodology=(
            "Non-synchronous interharmonic isolation sweep: integer harmonics and RoCoF are "
            "disabled while a 75 Hz component is scaled. This isolates spectral leakage and "
            "off-bin disturbance sensitivity."
        ),
        directional=False,
    ),
    "noise_snr": SweepSpec(
        key="noise_snr",
        label="White Noise / SNR",
        x_col="noise_sigma_pu",
        x_label="White-noise sigma [pu]",
        signed_col="snr_db",
        default_levels=(0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.10),
        reference_value=0.001,
        methodology=(
            "Single-tone white-noise sweep: the signal is a constant-frequency sinusoid and "
            "only additive white noise changes. Reported SNR assumes a 1 pu peak sine wave "
            "with RMS 1/sqrt(2)."
        ),
        directional=False,
    ),
}


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        value = int(raw)
    except ValueError:
        return int(default)
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
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _float_csv(raw: str | None) -> list[float]:
    out: list[float] = []
    for item in _csv(raw):
        try:
            out.append(float(item))
        except ValueError:
            continue
    return out


def _sanitize_token(value: float) -> str:
    return f"{abs(float(value)):g}".replace(".", "p").replace("-", "m")


def _snr_db_from_sigma(noise_sigma: float, amplitude_peak: float = 1.0) -> float:
    if noise_sigma <= 0.0:
        return float("inf")
    return float(20.0 * math.log10((abs(amplitude_peak) / math.sqrt(2.0)) / noise_sigma))


def _integer_harmonic_coefficients_for_thd(thd_pu: float) -> dict[str, float]:
    weights = {
        "h2_pct": 0.25,
        "h3_pct": 0.50,
        "h5_pct": 1.00,
        "h7_pct": 0.75,
        "h11_pct": 0.375,
        "h13_pct": 0.25,
    }
    norm = math.sqrt(sum(v * v for v in weights.values()))
    return {key: float(max(0.0, thd_pu) * value / norm) for key, value in weights.items()}


def _env_key_for_estimator(est_name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in est_name.upper()).strip("_")


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


def _estimator_defaults(est_cls: type) -> dict[str, Any]:
    if hasattr(est_cls, "default_params"):
        return dict(est_cls.default_params())
    return {}


def _apply_frequency_bounds(est_name: str, est_cls: type, params: dict[str, Any], bounds: tuple[float, float] | None) -> dict[str, Any]:
    if bounds is None or not _env_bool("ATLAS_FORCE_FREQUENCY_BOUNDS", True):
        return dict(params)
    out = dict(params)
    f_min, f_max = bounds
    accepted = _accepted_init_params(est_cls)
    accepts_any = "**kwargs" in accepted
    for lo_key, hi_key in (("f_min_hz", "f_max_hz"), ("freq_min_hz", "freq_max_hz")):
        if accepts_any or lo_key in accepted:
            out[lo_key] = float(f_min)
        if accepts_any or hi_key in accepted:
            out[hi_key] = float(f_max)
    if est_name == "LKF2" and (accepts_any or "freq_dev_limit_hz" in accepted):
        out["freq_dev_limit_hz"] = max(abs(60.0 - f_min), abs(f_max - 60.0))
    if est_name == "RA-EKF" and (accepts_any or "rocof_limit_hz_s" in accepted):
        out["rocof_limit_hz_s"] = max(float(out.get("rocof_limit_hz_s", 0.0)), 50.0)
    return out


def _noise_bounds() -> tuple[float, float]:
    lo = _env_float("ATLAS_NOISE_LOW", 0.0005, minimum=0.0)
    hi = _env_float("ATLAS_NOISE_HIGH", 0.0020, minimum=lo)
    return lo, hi


def _apply_atlas_overrides(cls: type, params: dict[str, Any], run_idx: int, n_runs: int, base_seed: int) -> dict[str, Any]:
    del base_seed
    if not _env_bool("ATLAS_MC_STRATIFIED_COVARIATES", True):
        return params
    n = max(1, int(n_runs))
    u = (int(run_idx) + 0.5) / n
    phase_u = (0.61803398875 * (int(run_idx) + 1)) % 1.0
    time_u = (0.41421356237 * (int(run_idx) + 1)) % 1.0
    noise_lo, noise_hi = _noise_bounds()
    out = dict(params)
    sweep_key = getattr(cls, "ATLAS_SWEEP_KEY", "")
    if sweep_key in {
        "magnitude_step",
        "rocof",
        "frequency_step",
        "phase_jump_sweep",
        "modulation_am_sweep",
        "modulation_fm_sweep",
        "harmonics",
        "interharmonics",
        "noise_snr",
    }:
        out["phase_rad"] = float(2.0 * math.pi * phase_u)
    if sweep_key in {
        "magnitude_step",
        "rocof",
        "frequency_step",
        "phase_jump_sweep",
        "modulation_am_sweep",
        "modulation_fm_sweep",
    }:
        noise = float(noise_lo + (noise_hi - noise_lo) * u)
        if sweep_key == "magnitude_step":
            mode = os.getenv("ATLAS_MAG_NOISE_MODE", "fixed_absolute").strip().lower()
            if mode in {"fixed_snr", "amplitude_scaled", "post_fixed_snr"}:
                noise *= max(abs(float(out.get("amp_post_pu", 1.0))), 1e-12)
            elif mode in {"none", "noise_free", "no_noise"}:
                noise = 0.0
        out["noise_sigma"] = noise
    elif sweep_key == "noise_snr":
        out["noise_sigma"] = float(out.get("noise_sigma", 0.0))
    elif sweep_key in {"harmonics", "interharmonics"}:
        out["white_noise_sigma"] = float(out.get("white_noise_sigma", 0.0))
    if "t_step_s" in out:
        out["t_step_s"] = float(float(out["t_step_s"]) - 0.025 + 0.050 * time_u)
    if "t_start_s" in out:
        out["t_start_s"] = float(float(out["t_start_s"]) - 0.025 + 0.050 * time_u)
    if "t_jump_s" in out:
        out["t_jump_s"] = float(float(out["t_jump_s"]) - 0.025 + 0.050 * time_u)
    out["seed"] = int(out.get("seed", 0) or 0)
    return out


def _make_scenario_variant(sweep_key: str, signed_value: float) -> AtlasScenario:
    spec = SWEEP_SPECS[sweep_key]
    value = float(signed_value)
    abs_value = abs(value)
    direction = "pos" if value >= 0.0 else "neg"
    if not spec.directional:
        value = abs_value
        direction = "level"
    token = _sanitize_token(value)
    monte_carlo_space: dict[str, Any]

    if sweep_key == "magnitude_step":
        amp_pre = 1.0
        amp_post = amp_pre + abs_value / 100.0 if direction == "pos" else max(1e-3, amp_pre - abs_value / 100.0)
        scenario_name = f"Atlas_MagnitudeStep_{direction}_{token}pct"
        class_name = f"AtlasMagnitudeStep{direction.title()}{token}"
        default_params = {
            **IEEEMagStepScenario.DEFAULT_PARAMS,
            "duration_s": _env_float("ATLAS_MAG_DURATION_S", 1.8, minimum=0.3),
            "amp_pre_pu": amp_pre,
            "amp_post_pu": amp_post,
            "t_step_s": _env_float("ATLAS_MAG_T_STEP_S", 0.50, minimum=0.0),
            "noise_sigma": 0.001,
        }
        base_cls = IEEEMagStepScenario
        signed_col = {"step_percent": value, "abs_step_percent": abs_value}
        monte_carlo_space = {
            "phase_rad": {"kind": "uniform", "low": 0.0, "high": 2.0 * math.pi},
            "noise_sigma": {"kind": "uniform", "low": _noise_bounds()[0], "high": _noise_bounds()[1]},
        }
    elif sweep_key == "rocof":
        t_start = _env_float("ATLAS_ROCOF_T_START_S", 0.30, minimum=0.0)
        ramp_duration = _env_float("ATLAS_ROCOF_RAMP_DURATION_S", 0.40, minimum=0.02)
        scenario_name = f"Atlas_RoCoF_{direction}_{token}Hzs"
        class_name = f"AtlasRoCoF{direction.title()}{token}"
        default_params = {
            **IEEEFreqRampScenario.DEFAULT_PARAMS,
            "duration_s": _env_float("ATLAS_ROCOF_DURATION_S", t_start + ramp_duration + 0.50, minimum=0.3),
            "rocof_hz_s": value,
            "t_start_s": t_start,
            "freq_cap_hz": 60.0 + value * ramp_duration,
            "noise_sigma": 0.001,
        }
        base_cls = IEEEFreqRampScenario
        signed_col = {"rocof_hz_s": value, "abs_rocof_hz_s": abs_value}
        monte_carlo_space = {
            "phase_rad": {"kind": "uniform", "low": 0.0, "high": 2.0 * math.pi},
            "noise_sigma": {"kind": "uniform", "low": _noise_bounds()[0], "high": _noise_bounds()[1]},
        }
    elif sweep_key == "frequency_step":
        scenario_name = f"Atlas_FrequencyStep_{direction}_{token}Hz"
        class_name = f"AtlasFrequencyStep{direction.title()}{token}"
        default_params = {
            **IEEEFreqStepScenario.DEFAULT_PARAMS,
            "duration_s": _env_float("ATLAS_FREQSTEP_DURATION_S", 1.5, minimum=0.3),
            "freq_pre_hz": 60.0,
            "freq_post_hz": 60.0 + value,
            "t_step_s": _env_float("ATLAS_FREQSTEP_T_STEP_S", 0.50, minimum=0.0),
            "noise_sigma": 0.001,
        }
        base_cls = IEEEFreqStepScenario
        signed_col = {"step_hz": value, "abs_step_hz": abs_value}
        monte_carlo_space = {
            "phase_rad": {"kind": "uniform", "low": 0.0, "high": 2.0 * math.pi},
            "noise_sigma": {"kind": "uniform", "low": _noise_bounds()[0], "high": _noise_bounds()[1]},
        }
    elif sweep_key == "phase_jump_sweep":
        jump_rad = math.radians(value)
        scenario_name = f"Atlas_PhaseJump_{direction}_{token}deg"
        class_name = f"AtlasPhaseJump{direction.title()}{token}"
        default_params = {
            **IEEEPhaseJump20Scenario.DEFAULT_PARAMS,
            "duration_s": _env_float("ATLAS_PHASE_JUMP_DURATION_S", 1.5, minimum=0.3),
            "freq_hz": 60.0,
            "phase_jump_rad": jump_rad,
            "t_jump_s": _env_float("ATLAS_PHASE_JUMP_T_S", 0.70, minimum=0.0),
            "noise_sigma": 0.001,
        }
        base_cls = IEEEPhaseJump20Scenario
        signed_col = {
            "phase_jump_deg": value,
            "abs_phase_jump_deg": abs_value,
            "phase_jump_rad": jump_rad,
        }
        monte_carlo_space = {
            "phase_rad": {"kind": "uniform", "low": 0.0, "high": 2.0 * math.pi},
            "noise_sigma": {"kind": "uniform", "low": _noise_bounds()[0], "high": _noise_bounds()[1]},
        }
    elif sweep_key == "modulation_am_sweep":
        am_depth_pct = _env_float("ATLAS_AM_DEPTH_PCT", 10.0, minimum=0.0)
        scenario_name = f"Atlas_ModulationAM_{token}Hz"
        class_name = f"AtlasModulationAm{token}"
        default_params = {
            **IEEEModulationAMScenario.DEFAULT_PARAMS,
            "duration_s": _env_float("ATLAS_AM_DURATION_S", 2.0, minimum=0.3),
            "freq_nom_hz": 60.0,
            "kx": am_depth_pct / 100.0,
            "fm_hz": abs_value,
            "noise_sigma": 0.001,
        }
        base_cls = IEEEModulationAMScenario
        signed_col = {
            "modulation_frequency_hz": abs_value,
            "am_depth_percent": am_depth_pct,
            "am_depth_pu": am_depth_pct / 100.0,
        }
        monte_carlo_space = {
            "phase_rad": {"kind": "uniform", "low": 0.0, "high": 2.0 * math.pi},
            "noise_sigma": {"kind": "uniform", "low": _noise_bounds()[0], "high": _noise_bounds()[1]},
        }
    elif sweep_key == "modulation_fm_sweep":
        peak_dev_hz = _env_float("ATLAS_FM_PEAK_DEV_HZ", 0.20, minimum=0.0)
        ka_rad = peak_dev_hz / max(abs_value, 1e-12)
        scenario_name = f"Atlas_ModulationFM_{token}Hz"
        class_name = f"AtlasModulationFm{token}"
        default_params = {
            **IEEEModulationFMScenario.DEFAULT_PARAMS,
            "duration_s": _env_float("ATLAS_FM_DURATION_S", 2.0, minimum=0.3),
            "freq_nom_hz": 60.0,
            "ka": ka_rad,
            "fm_hz": abs_value,
            "noise_sigma": 0.001,
        }
        base_cls = IEEEModulationFMScenario
        signed_col = {
            "modulation_frequency_hz": abs_value,
            "peak_freq_dev_hz": peak_dev_hz,
            "ka_rad": ka_rad,
        }
        monte_carlo_space = {
            "phase_rad": {"kind": "uniform", "low": 0.0, "high": 2.0 * math.pi},
            "noise_sigma": {"kind": "uniform", "low": _noise_bounds()[0], "high": _noise_bounds()[1]},
        }
    elif sweep_key == "harmonics":
        thd_pu = abs_value / 100.0
        harmonic_coeffs = _integer_harmonic_coefficients_for_thd(thd_pu)
        scenario_name = f"Atlas_Harmonics_THD_{token}pct"
        class_name = f"AtlasHarmonicsThd{token}"
        default_params = {
            **IBRHarmonicsLargeScenario.DEFAULT_PARAMS,
            "duration_s": _env_float("ATLAS_HARMONICS_DURATION_S", 2.0, minimum=0.3),
            "freq_nom_hz": 60.0,
            "freq_step_hz": 0.0,
            "t_event_s": _env_float("ATLAS_HARMONICS_MARKER_S", 1.0, minimum=0.0),
            "amp_pu": 1.0,
            **harmonic_coeffs,
            "sub_pct": 0.0,
            "ih325_pct": 0.0,
            "ih85_pct": 0.0,
            "phase_rad": 0.0,
            "white_noise_sigma": _env_float("ATLAS_DISTORTION_NOISE_SIGMA", 0.0, minimum=0.0),
            "brown_noise_sigma": 0.0,
            "impulse_prob": 0.0,
            "impulse_mag": 0.0,
        }
        base_cls = IBRHarmonicsLargeScenario
        signed_col = {"thd_percent": abs_value, "thd_pu": thd_pu, **harmonic_coeffs}
        monte_carlo_space = {
            "phase_rad": {"kind": "uniform", "low": 0.0, "high": 2.0 * math.pi},
        }
    elif sweep_key == "interharmonics":
        interharmonic_pu = abs_value / 100.0
        scenario_name = f"Atlas_Interharmonics_75Hz_{token}pct"
        class_name = f"AtlasInterharmonics75Hz{token}"
        default_params = {
            **IBRHarmonicsMediumScenario.DEFAULT_PARAMS,
            "duration_s": _env_float("ATLAS_INTERHARMONICS_DURATION_S", 2.0, minimum=0.3),
            "freq_nom_hz": 60.0,
            "rocof_hz_s": 0.0,
            "rocof_duration_s": 0.0,
            "t_event_s": _env_float("ATLAS_INTERHARMONICS_MARKER_S", 1.0, minimum=0.0),
            "amp_pu": 1.0,
            "h3_pct": 0.0,
            "h5_pct": 0.0,
            "h7_pct": 0.0,
            "h11_pct": 0.0,
            "h13_pct": 0.0,
            "ih75_pct": interharmonic_pu,
            "phase_rad": 0.0,
            "white_noise_sigma": _env_float("ATLAS_DISTORTION_NOISE_SIGMA", 0.0, minimum=0.0),
            "brown_noise_sigma": 0.0,
        }
        base_cls = IBRHarmonicsMediumScenario
        signed_col = {"interharmonic_percent": abs_value, "interharmonic_pu": interharmonic_pu, "interharmonic_hz": 75.0}
        monte_carlo_space = {
            "phase_rad": {"kind": "uniform", "low": 0.0, "high": 2.0 * math.pi},
        }
    elif sweep_key == "noise_snr":
        sigma = abs_value
        snr_db = _snr_db_from_sigma(sigma, amplitude_peak=1.0)
        scenario_name = f"Atlas_NoiseSigma_{token}pu"
        class_name = f"AtlasNoiseSigma{token}"
        default_params = {
            **IEEESingleSinWaveScenario.DEFAULT_PARAMS,
            "duration_s": _env_float("ATLAS_NOISE_DURATION_S", 1.5, minimum=0.3),
            "amplitude": 1.0,
            "freq_hz": 60.0,
            "phase_rad": 0.0,
            "noise_sigma": sigma,
        }
        base_cls = IEEESingleSinWaveScenario
        signed_col = {"noise_sigma_pu": sigma, "snr_db": snr_db}
        monte_carlo_space = {
            "phase_rad": {"kind": "uniform", "low": 0.0, "high": 2.0 * math.pi},
        }
    else:
        raise ValueError(f"Unknown atlas sweep: {sweep_key}")

    attrs = {
        "SCENARIO_NAME": scenario_name,
        "DEFAULT_PARAMS": default_params,
        "MONTE_CARLO_SPACE": monte_carlo_space,
        "DISABLE_EVENT_METRICS": sweep_key in {
            "harmonics",
            "interharmonics",
            "noise_snr",
            "modulation_am_sweep",
            "modulation_fm_sweep",
        },
        "ATLAS_SWEEP_KEY": sweep_key,
        "ATLAS_SWEEP_VALUE": value,
        "ATLAS_ABS_VALUE": abs_value,
        "ATLAS_DIRECTION": direction,
        "get_name": classmethod(lambda cls: cls.SCENARIO_NAME),
        "apply_run_index_overrides": classmethod(_apply_atlas_overrides),
    }
    new_cls = type(class_name, (base_cls,), attrs)
    new_cls.__module__ = __name__
    globals()[class_name] = new_cls
    return AtlasScenario(
        sweep_key=sweep_key,
        sweep_label=spec.label,
        scenario_name=scenario_name,
        scenario_cls=new_cls,
        signed_value=value,
        abs_value=abs_value,
        direction=direction,
        params=signed_col,
    )


def _levels_for_sweep(sweep_key: str) -> list[float]:
    spec = SWEEP_SPECS[sweep_key]
    env_name = {
        "magnitude_step": "ATLAS_MAG_LEVELS_PCT",
        "rocof": "ATLAS_ROCOF_LEVELS_HZ_S",
        "frequency_step": "ATLAS_FREQSTEP_LEVELS_HZ",
        "phase_jump_sweep": "ATLAS_PHASE_JUMP_LEVELS_DEG",
        "modulation_am_sweep": "ATLAS_AM_MOD_FREQ_LEVELS_HZ",
        "modulation_fm_sweep": "ATLAS_FM_MOD_FREQ_LEVELS_HZ",
        "harmonics": "ATLAS_HARMONICS_THD_LEVELS_PCT",
        "interharmonics": "ATLAS_INTERHARMONIC_LEVELS_PCT",
        "noise_snr": "ATLAS_NOISE_SIGMA_LEVELS_PU",
    }[sweep_key]
    configured = _float_csv(os.getenv(env_name))
    values = configured if configured else list(spec.default_levels)
    return sorted({round(abs(float(v)), 9) for v in values if abs(float(v)) > 0.0})


def _directions_for_sweep(sweep_key: str) -> list[str]:
    if not SWEEP_SPECS[sweep_key].directional:
        return ["level"]
    raw = os.getenv(f"ATLAS_{sweep_key.upper()}_DIRECTIONS") or os.getenv("ATLAS_DIRECTIONS", "pos,neg")
    requested = [item.lower() for item in _csv(raw)]
    out: list[str] = []
    if any(item in {"pos", "+", "up", "positive", "swell"} for item in requested):
        out.append("pos")
    if any(item in {"neg", "-", "down", "negative", "sag"} for item in requested):
        out.append("neg")
    return out or ["pos", "neg"]


def _expand_sweep_keys(requested: list[str]) -> list[str]:
    expanded: list[str] = []
    aliases = {
        "phase_jump": "phase_jump_sweep",
        "phase_jump_sweep": "phase_jump_sweep",
        "modulation_am": "modulation_am_sweep",
        "modulation_am_sweep": "modulation_am_sweep",
        "modlation_am_sweep": "modulation_am_sweep",
        "am_modulation": "modulation_am_sweep",
        "am_modulation_sweep": "modulation_am_sweep",
        "modulation_fm": "modulation_fm_sweep",
        "modulation_fm_sweep": "modulation_fm_sweep",
        "modlation_fm_sweep": "modulation_fm_sweep",
        "fm_modulation": "modulation_fm_sweep",
        "fm_modulation_sweep": "modulation_fm_sweep",
    }
    for item in requested:
        key = item.strip().lower().replace("-", "_")
        if key == "all":
            expanded.extend(SWEEP_SPECS.keys())
        elif key == "core":
            expanded.extend(["magnitude_step", "rocof", "frequency_step"])
        elif key == "p0":
            expanded.extend(
                [
                    "phase_jump_sweep",
                    "modulation_am_sweep",
                    "modulation_fm_sweep",
                    "harmonics",
                    "interharmonics",
                    "noise_snr",
                ]
            )
        else:
            expanded.append(aliases.get(key, key))
    return list(dict.fromkeys(expanded))


def build_atlas_scenarios(sweep_keys: list[str]) -> list[AtlasScenario]:
    scenarios: list[AtlasScenario] = []
    for sweep_key in _expand_sweep_keys(sweep_keys):
        if sweep_key not in SWEEP_SPECS:
            raise ValueError(f"Unknown ATLAS sweep: {sweep_key}. Known: {sorted(SWEEP_SPECS)}")
        levels = _levels_for_sweep(sweep_key)
        for direction in _directions_for_sweep(sweep_key):
            for level in levels:
                if sweep_key == "magnitude_step" and direction == "neg" and level >= 100.0:
                    continue
                signed = level if direction in {"pos", "level"} else -level
                scenarios.append(_make_scenario_variant(sweep_key, signed))
    include = set(_csv(os.getenv("ATLAS_INCLUDE_SCENARIOS")))
    if include:
        scenarios = [sc for sc in scenarios if sc.scenario_name in include]
    if not scenarios:
        raise ValueError("ATLAS scenario selection is empty.")
    return scenarios


def select_estimators() -> dict[str, type]:
    estimators = load_active_estimators()
    include_raw = os.getenv("ATLAS_INCLUDE_ESTIMATORS", CANONICAL_ESTIMATORS).strip()
    exclude_raw = os.getenv("ATLAS_EXCLUDE_ESTIMATORS", "").strip()
    by_lower = {label.lower(): label for label in estimators}
    if include_raw:
        selected: set[str] = set()
        missing: list[str] = []
        for item in _csv(include_raw):
            hit = by_lower.get(item.lower())
            if hit:
                selected.add(hit)
            else:
                missing.append(item)
        if missing:
            print(f"[WARN] Unknown ATLAS_INCLUDE_ESTIMATORS entries ignored: {missing}")
        if not selected:
            raise ValueError("ATLAS_INCLUDE_ESTIMATORS did not match any active estimator.")
        estimators = {k: v for k, v in estimators.items() if k in selected}
    if exclude_raw:
        excluded = {by_lower[item.lower()] for item in _csv(exclude_raw) if item.lower() in by_lower}
        estimators = {k: v for k, v in estimators.items() if k not in excluded}
    if not estimators:
        raise ValueError("Estimator filter removed all estimators.")
    return estimators


def _run_engine_local(engine: MonteCarloEngine) -> MonteCarloResult:
    rows: list[dict[str, Any]] = []
    signals: list[pd.DataFrame] = []
    for run_idx in range(engine.n_runs):
        row, signal_df = engine.run_once(run_idx)
        rows.append(row)
        if not signal_df.empty:
            signals.append(signal_df)
    summary_df = pd.DataFrame(rows).sort_values("run_idx").reset_index(drop=True)
    signals_df = (
        pd.concat(signals, ignore_index=True).sort_values(["run_idx", "t_s"]).reset_index(drop=True)
        if signals
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
            "execution_mode": "atlas_local_sequential",
        },
    )


def _score_params(est_name: str, est_cls: type, params: dict[str, Any], scenarios: list[Any], eval_start_s: float) -> float:
    try:
        rmses: list[float] = []
        peaks: list[float] = []
        late_rmses: list[float] = []
        rfes: list[float] = []
        for sc in scenarios:
            scoring_engine = MonteCarloEngine(
                scenario_cls=IEEESingleSinWaveScenario,
                estimator_cls=est_cls,
                estimator_params=params,
                n_runs=1,
                base_seed=0,
                n_cost_reps=1,
                enforce_standardized_step=est_name not in {"PI-GRU", "Koopman (RK-DPMU)"},
                capture_signals=False,
            )
            est_out = scoring_engine._run_estimator(sc.v, t=sc.t)
            f_hat = np.asarray(est_out.get("f_hat", []), dtype=float)
            f_true = np.asarray(sc.f_true, dtype=float)
            if len(f_hat) != len(f_true):
                return 1e9
            dt = float(sc.t[1] - sc.t[0]) if len(sc.t) > 1 else 1e-4
            start = min(len(f_true) - 1, max(0, int(round(eval_start_s / dt))))
            err = np.asarray(f_hat[start:], dtype=float) - f_true[start:]
            if len(err) < 4 or not np.all(np.isfinite(err)):
                return 1e9
            late_start = min(len(f_true) - 1, max(start, int(round(1.0 / dt))))
            late_err = np.asarray(f_hat[late_start:], dtype=float) - f_true[late_start:]
            rfe = np.diff(err) / dt if len(err) > 1 else np.zeros(1)
            rmses.append(float(np.sqrt(np.mean(err**2))))
            peaks.append(float(np.max(np.abs(err))))
            late_rmses.append(float(np.sqrt(np.mean(late_err**2))) if len(late_err) else rmses[-1])
            rfes.append(float(np.sqrt(np.mean(np.clip(rfe, -500.0, 500.0) ** 2))))
        if not rmses:
            return 1e9
        score = (
            float(np.median(rmses))
            + 0.50 * float(np.quantile(rmses, 0.90))
            + 0.05 * float(np.quantile(peaks, 0.90))
            + 0.25 * float(np.median(late_rmses))
            + 0.001 * float(np.median(rfes))
        )
        return score if math.isfinite(score) else 1e9
    except Exception:
        return 1e9


def _build_eval_scenarios(scenarios: list[AtlasScenario], base_seed: int, runs_per_level: int) -> list[Any]:
    out: list[Any] = []
    for idx, sc in enumerate(scenarios):
        for run_idx in range(max(1, int(runs_per_level))):
            sampler = MonteCarloEngine(sc.scenario_cls, n_runs=max(1, int(runs_per_level)), base_seed=base_seed + 1000 * idx)
            params = sampler.sample_params(run_idx)
            out.append(sc.scenario_cls.run(**params))
    return out


def _tune_estimator(
    est_name: str,
    est_cls: type,
    eval_scenarios: list[Any],
    *,
    n_trials: int,
    tune_eval_runs: int,
    mode: str,
    frequency_bounds: tuple[float, float] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    del tune_eval_runs
    defaults = _apply_frequency_bounds(est_name, est_cls, _estimator_defaults(est_cls), frequency_bounds)
    meta: dict[str, Any] = {
        "mode": mode,
        "n_trials_requested": int(n_trials),
        "n_trials_executed": 0,
        "n_eval_scenarios": int(len(eval_scenarios)),
        "best_objective": None,
        "frequency_bounds_hz": list(frequency_bounds) if frequency_bounds else None,
    }
    if est_name not in benchmark.SEARCH_SPACES:
        meta["reason"] = "no_search_space"
        return defaults, meta
    if n_trials <= 0:
        meta["reason"] = "n_trials<=0"
        return defaults, meta
    if not eval_scenarios:
        meta["reason"] = "no_eval_scenarios"
        return defaults, meta

    raw_space_fn = benchmark.SEARCH_SPACES[est_name]
    if not benchmark._grid_space_for_estimator(raw_space_fn, n_trials=2):
        meta["reason"] = "empty_search_space"
        return defaults, meta

    def objective(trial: optuna.Trial) -> float:
        suggested = raw_space_fn(trial)
        params = _apply_frequency_bounds(est_name, est_cls, {**defaults, **suggested}, frequency_bounds)
        return _score_params(est_name, est_cls, params, eval_scenarios, eval_start_s=0.15)

    study, n_exec, sampler_mode = benchmark._build_optuna_study(raw_space_fn, n_trials=int(n_trials))
    meta["n_trials_executed"] = int(n_exec)
    meta["sampler_mode_effective"] = sampler_mode
    study.optimize(objective, n_trials=n_exec)
    if study.best_value >= 1e9:
        meta["reason"] = "all_trials_failed"
        return defaults, meta
    best = _apply_frequency_bounds(est_name, est_cls, {**defaults, **raw_space_fn(study.best_trial)}, frequency_bounds)
    meta["best_objective"] = float(study.best_value)
    return best, meta


def _frequency_bounds_for_sweep(scenarios: list[AtlasScenario]) -> tuple[float, float] | None:
    values: list[float] = []
    for sc in scenarios:
        params = sc.scenario_cls.get_default_params()
        if sc.sweep_key == "magnitude_step":
            values.extend([40.0, 80.0])
        elif sc.sweep_key == "rocof":
            values.extend([float(params.get("freq_nom_hz", 60.0)), float(params.get("freq_cap_hz", 60.0))])
        elif sc.sweep_key == "frequency_step":
            values.extend([float(params.get("freq_pre_hz", 60.0)), float(params.get("freq_post_hz", 60.0))])
        elif sc.sweep_key == "phase_jump_sweep":
            values.append(float(params.get("freq_hz", 60.0)))
        elif sc.sweep_key == "modulation_am_sweep":
            values.append(float(params.get("freq_nom_hz", 60.0)))
        elif sc.sweep_key == "modulation_fm_sweep":
            f_nom = float(params.get("freq_nom_hz", 60.0))
            peak_dev = abs(float(params.get("ka", 0.0)) * float(params.get("fm_hz", 0.0)))
            values.extend([f_nom - peak_dev, f_nom + peak_dev])
        elif sc.sweep_key in {"harmonics", "interharmonics"}:
            values.append(float(params.get("freq_nom_hz", 60.0)))
        elif sc.sweep_key == "noise_snr":
            values.append(float(params.get("freq_hz", 60.0)))
    if not values:
        return None
    margin = _env_float("ATLAS_FREQ_BOUND_MARGIN_HZ", 10.0, minimum=0.0)
    return max(0.0, min(values) - margin), max(values) + margin


def _can_reuse(run_spec_path: Path, summary_csv: Path, expected: dict[str, Any]) -> bool:
    if not run_spec_path.exists() or not summary_csv.exists():
        return False
    try:
        spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    for key, value in expected.items():
        if spec.get(key) != value:
            return False
    return True


def _bootstrap_ci_mean(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return float("nan"), float("nan")
    if len(finite) == 1:
        val = float(finite[0])
        return val, val
    n_boot = _env_int("ATLAS_BOOTSTRAP_RUNS", 500, minimum=50)
    rng = np.random.default_rng(_env_int("ATLAS_BOOTSTRAP_SEED", 20260525, minimum=0))
    idx = rng.integers(0, len(finite), size=(n_boot, len(finite)))
    means = finite[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _aggregate_summary(summary_df: pd.DataFrame) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for metric in METRIC_COLUMNS:
        if metric not in summary_df.columns:
            continue
        series = pd.to_numeric(summary_df[metric], errors="coerce")
        valid = series.dropna().to_numpy(dtype=float)
        if len(valid) == 0:
            continue
        ci_lo, ci_hi = _bootstrap_ci_mean(valid)
        row[f"{metric}_mean"] = float(np.mean(valid))
        row[f"{metric}_median"] = float(np.median(valid))
        row[f"{metric}_std"] = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
        row[f"{metric}_p05"] = float(np.quantile(valid, 0.05))
        row[f"{metric}_p95"] = float(np.quantile(valid, 0.95))
        row[f"{metric}_ci95_low"] = ci_lo
        row[f"{metric}_ci95_high"] = ci_hi
        row[f"{metric}_n"] = int(len(valid))
    return row


def _add_derived_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for base_col, derived_prefix, scale in [
        ("m15_pcb_compliant", "m15_pass_rate_pct", 100.0),
        ("m16_heatmap_pass", "m16_heatmap_pass_rate_pct", 100.0),
        ("m22_invalid_output_rate", "m22_invalid_output_rate_pct", 100.0),
    ]:
        for suffix in ["mean", "median", "p05", "p95", "std", "ci95_low", "ci95_high"]:
            col = f"{base_col}_{suffix}"
            if col in out.columns:
                out[f"{derived_prefix}_{suffix}"] = pd.to_numeric(out[col], errors="coerce") * scale
    return out


def _line_style(direction: str) -> str:
    return "--" if direction == "neg" else "-"


def _direction_label_suffix(direction: str) -> str:
    if direction == "pos":
        return " +"
    if direction == "neg":
        return " -"
    return ""


def _estimator_color_map(estimators: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for idx, est in enumerate(sorted(estimators)):
        color = HIGH_CONTRAST_ESTIMATOR_COLORS.get(
            str(est),
            HIGH_CONTRAST_FALLBACK_COLORS[idx % len(HIGH_CONTRAST_FALLBACK_COLORS)],
        )
        out[str(est)] = matplotlib.colors.to_rgba(color)
    return out


def _estimator_marker(estimator: str) -> str:
    if estimator in HIGH_CONTRAST_ESTIMATOR_MARKERS:
        return HIGH_CONTRAST_ESTIMATOR_MARKERS[estimator]
    idx = sum(ord(ch) for ch in estimator) % len(HIGH_CONTRAST_FALLBACK_MARKERS)
    return HIGH_CONTRAST_FALLBACK_MARKERS[idx]


def _dominant_policy_label(df: pd.DataFrame) -> str:
    if "policy" not in df.columns or df["policy"].dropna().empty:
        return "policy"
    raw = str(df["policy"].dropna().astype(str).mode().iloc[0]).replace("_", " ")
    return raw.strip() or "policy"


def _metric_interval_columns(metric_col: str, df: pd.DataFrame) -> tuple[str | None, str | None]:
    for lo_col, hi_col in [
        (metric_col.replace("_mean", "_p10"), metric_col.replace("_mean", "_p90")),
        (metric_col.replace("_mean", "_p05"), metric_col.replace("_mean", "_p95")),
        (metric_col.replace("_mean", "_ci95_low"), metric_col.replace("_mean", "_ci95_high")),
    ]:
        if lo_col in df.columns and hi_col in df.columns:
            return lo_col, hi_col
    return None, None


def _shade_severity_regions(ax: plt.Axes, spec: SweepSpec, x_lo: float, x_hi: float, *, labels: bool = True) -> None:
    for idx, (label, lo, hi, color) in enumerate(SEVERITY_REGIONS.get(spec.key, ())):
        band_lo = max(float(lo), float(x_lo))
        band_hi = min(float(hi), float(x_hi))
        if band_hi <= band_lo:
            continue
        ax.axvspan(band_lo, band_hi, color=color, alpha=0.075 if idx < 3 else 0.055, zorder=0)
        if labels:
            x_mid = math.sqrt(max(band_lo, 1e-12) * max(band_hi, 1e-12))
            ax.text(
                x_mid,
                0.985 - 0.04 * (idx % 3),
                label,
                transform=ax.get_xaxis_transform(),
                va="top",
                ha="center",
                fontsize=6.4,
                color="#263238",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.65),
            )


def _reference_cases_for_sweep(sweep_key: str) -> list[tuple[AtlasScenario, str, str]]:
    spec = SWEEP_SPECS[sweep_key]
    if spec.directional:
        return [
            (_make_scenario_variant(sweep_key, spec.reference_value), "-", f"+{spec.reference_value:g}"),
            (_make_scenario_variant(sweep_key, -spec.reference_value), "--", f"-{spec.reference_value:g}"),
        ]
    return [(_make_scenario_variant(sweep_key, spec.reference_value), "-", f"{spec.reference_value:g}")]


def _plot_reference_panel(ax: plt.Axes, sweep_key: str) -> None:
    spec = SWEEP_SPECS[sweep_key]
    ylabel = "Signal [pu]"
    for scenario, line_style, label in _reference_cases_for_sweep(sweep_key):
        try:
            data = scenario.scenario_cls.run(seed=0)
        except Exception as exc:
            ax.text(0.5, 0.5, f"Reference unavailable: {exc}", ha="center", va="center", wrap=True)
            continue
        x = np.asarray(data.t, dtype=float)
        if sweep_key in {
            "magnitude_step",
            "phase_jump_sweep",
            "modulation_am_sweep",
            "harmonics",
            "interharmonics",
            "noise_snr",
        }:
            y = np.asarray(data.v, dtype=float)
            if sweep_key == "phase_jump_sweep" and len(x):
                event_t = float(scenario.scenario_cls.get_default_params().get("t_jump_s", 0.70))
                keep = (x >= max(float(x[0]), event_t - 0.055)) & (x <= min(float(x[-1]), event_t + 0.085))
                x = x[keep]
                y = y[keep]
                ax.axvline(event_t, color="#B71C1C", linestyle=":", linewidth=1.1, label="jump")
            if sweep_key in {"harmonics", "interharmonics", "noise_snr", "modulation_am_sweep"} and len(x):
                keep = x <= min(float(x[0]) + 0.12, float(x[-1]))
                x = x[keep]
                y = y[keep]
            ylabel = "Signal [pu]"
        else:
            y = np.asarray(data.f_true, dtype=float)
            ylabel = "Frequency [Hz]"
        ax.plot(x, y, color="#111111", linestyle=line_style, linewidth=1.25, label=label)
    title_suffix = " (zoom)" if sweep_key == "phase_jump_sweep" else ""
    ax.set_title(f"Reference {spec.label}{title_suffix}", loc="left", fontweight="bold")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=7, frameon=True)


def _plot_metric_page(df: pd.DataFrame, sweep_key: str, metric_col: str, metric_label: str, yscale: str = "log") -> tuple[plt.Figure, dict[str, Any]]:
    spec = SWEEP_SPECS[sweep_key]
    df_sweep = df[df["sweep_key"] == sweep_key].copy()
    if df_sweep.empty or metric_col not in df_sweep.columns:
        fig, _ax = plt.subplots(figsize=(8.0, 4.0))
        return fig, {}

    fig, axes = plt.subplots(3, 2, figsize=(12.0, 10.95), sharex=False, sharey=False)
    axes_arr = axes.flatten()
    families = [fam for fam in FAMILY_ORDER if fam in set(df_sweep["family"])]
    color_map = _estimator_color_map(sorted(df_sweep["estimator"].unique()))
    ticks = sorted(pd.to_numeric(df_sweep[spec.x_col], errors="coerce").dropna().unique().tolist())
    lo_col, hi_col = _metric_interval_columns(metric_col, df_sweep)

    _plot_reference_panel(axes_arr[0], sweep_key)
    for ax, family in zip(axes_arr[1:], families):
        part = df_sweep[df_sweep["family"] == family]
        for (estimator, direction), df_est in part.sort_values([spec.x_col, "estimator"]).groupby(["estimator", "direction"], sort=True):
            x = pd.to_numeric(df_est[spec.x_col], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(df_est[metric_col], errors="coerce").to_numpy(dtype=float)
            valid_xy = np.isfinite(x) & np.isfinite(y)
            if not np.any(valid_xy):
                continue
            x = x[valid_xy]
            y = y[valid_xy]
            if yscale == "log":
                y = np.maximum(y, 1e-12)
            if lo_col and hi_col and len(df_est) > 1:
                lo = pd.to_numeric(df_est[lo_col], errors="coerce").to_numpy(dtype=float)[valid_xy]
                hi = pd.to_numeric(df_est[hi_col], errors="coerce").to_numpy(dtype=float)[valid_xy]
                if np.any(np.isfinite(lo)) and np.any(np.isfinite(hi)):
                    if yscale == "log":
                        lo = np.maximum(lo, 1e-12)
                        hi = np.maximum(hi, 1e-12)
                    ax.fill_between(x, lo, hi, color=color_map[str(estimator)], alpha=0.08, linewidth=0)
            ax.plot(
                x,
                y,
                marker=_estimator_marker(str(estimator)),
                markersize=3.2,
                markeredgecolor="#111111",
                markeredgewidth=0.25,
                linewidth=1.05,
                color=color_map[str(estimator)],
                linestyle=_line_style(str(direction)),
                label=f"{estimator}{_direction_label_suffix(str(direction))}",
            )
        if ticks:
            _shade_severity_regions(ax, spec, min(ticks), max(ticks), labels=True)
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x:g}" if x in ticks else ""))
            for tick in ax.get_xticklabels():
                tick.set_rotation(70)
                tick.set_ha("right")
                tick.set_fontsize(6)
        if yscale == "log":
            ax.set_yscale("log")
        ax.axvline(spec.reference_value, color="#7B1FA2", linestyle=":", linewidth=0.9, label=f"{spec.reference_value:g} ref")
        ax.grid(True, which="both", alpha=0.25)
        ax.set_title(family, loc="left", fontweight="bold")
        ax.set_ylabel(metric_label)
        ax.legend(loc="best", fontsize=5.8, frameon=True)
    for idx in range(1 + len(families), len(axes_arr)):
        axes_arr[idx].set_visible(False)
    for ax in axes_arr[1 : 1 + len(families)]:
        ax.set_xlabel(spec.x_label)
    fig.suptitle(f"{spec.label}: {metric_label} by estimator family ({_dominant_policy_label(df_sweep)})", fontsize=13, y=0.995)
    fig.text(
        0.5,
        0.965,
        spec.methodology,
        ha="center",
        va="top",
        fontsize=7.2,
        color="#263238",
        wrap=True,
    )
    fig.text(
        0.5,
        0.006,
        "Solid/dashed lines mark perturbation sign when applicable. Bands show available run intervals; thresholds and regions are interpretation guides.",
        ha="center",
        va="bottom",
        fontsize=7,
        color="#37474F",
    )
    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.92])
    return fig, color_map


def save_rmse_family_plot(df_global: pd.DataFrame, out_dir: Path) -> tuple[list[Path], dict[str, Any]]:
    pdf_path = out_dir / RMSE_FAMILY_PDF_NAME
    png_path = out_dir / RMSE_FAMILY_PNG_NAME
    color_map: dict[str, Any] = {}
    first_fig: plt.Figure | None = None
    with PdfPages(pdf_path) as pdf:
        for sweep_key in sorted(df_global["sweep_key"].unique()):
            fig, cmap = _plot_metric_page(df_global, str(sweep_key), "m1_rmse_hz_mean", "RMSE [Hz]", "log")
            color_map.update(cmap)
            if first_fig is None:
                first_fig = fig
            pdf.savefig(fig)
            if fig is not first_fig:
                plt.close(fig)
    if first_fig is not None:
        first_fig.savefig(png_path, dpi=240)
        plt.close(first_fig)
    return [png_path, pdf_path], color_map


def _plot_all_estimators_page(
    df_global: pd.DataFrame,
    sweep_key: str,
    metric_col: str,
    metric_label: str,
    yscale: str = "log",
) -> plt.Figure:
    spec = SWEEP_SPECS[sweep_key]
    part = df_global[df_global["sweep_key"] == sweep_key].copy()
    estimators = sorted(
        part["estimator"].dropna().astype(str).unique().tolist(),
        key=lambda est: (
            FAMILY_ORDER.index(ESTIMATOR_FAMILIES.get(est, "Exotic"))
            if ESTIMATOR_FAMILIES.get(est, "Exotic") in FAMILY_ORDER
            else 99,
            est,
        ),
    )
    n_estimators = max(1, len(estimators))
    n_cols = 3
    n_rows = int(math.ceil(n_estimators / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.25 * n_cols, 2.45 * n_rows + 0.55), squeeze=False)
    axes_flat = axes.flatten()
    color_map = _estimator_color_map(estimators)
    ticks = sorted(pd.to_numeric(part[spec.x_col], errors="coerce").dropna().unique().tolist())
    lo_col, hi_col = _metric_interval_columns(metric_col, part)
    signed = spec.directional and "direction" in part.columns

    for idx, (ax, estimator) in enumerate(zip(axes_flat, estimators)):
        df_est_all = part[part["estimator"].astype(str) == estimator].copy()
        fallback_family = "Unknown"
        if "family" in df_est_all and not df_est_all["family"].dropna().empty:
            fallback_family = str(df_est_all["family"].dropna().iloc[0])
        family = ESTIMATOR_FAMILIES.get(estimator, fallback_family)
        group_cols = ["direction"] if signed else ["estimator"]
        for group_key, df_est in df_est_all.sort_values(spec.x_col).groupby(group_cols, sort=True):
            direction = str(group_key[0] if isinstance(group_key, tuple) else group_key)
            x = pd.to_numeric(df_est[spec.x_col], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(df_est[metric_col], errors="coerce").to_numpy(dtype=float)
            valid_xy = np.isfinite(x) & np.isfinite(y)
            if not np.any(valid_xy):
                continue
            x = x[valid_xy]
            y = y[valid_xy]
            if yscale == "log":
                y = np.maximum(y, 1e-12)
            if lo_col and hi_col and len(df_est) > 1:
                lo = pd.to_numeric(df_est[lo_col], errors="coerce").to_numpy(dtype=float)[valid_xy]
                hi = pd.to_numeric(df_est[hi_col], errors="coerce").to_numpy(dtype=float)[valid_xy]
                if np.any(np.isfinite(lo)) and np.any(np.isfinite(hi)):
                    if yscale == "log":
                        lo = np.maximum(lo, 1e-12)
                        hi = np.maximum(hi, 1e-12)
                    ax.fill_between(x, lo, hi, color=color_map[estimator], alpha=0.10, linewidth=0)
            ax.plot(
                x,
                y,
                marker=_estimator_marker(estimator),
                markersize=3.0,
                markeredgecolor="#111111",
                markeredgewidth=0.25,
                linewidth=1.25,
                color=color_map[estimator],
                linestyle=_line_style(direction) if signed else "-",
                label=_direction_label_suffix(direction).strip() if signed else estimator,
            )
        if ticks:
            _shade_severity_regions(ax, spec, min(ticks), max(ticks), labels=False)
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x:g}" if x in ticks else ""))
            for tick in ax.get_xticklabels():
                tick.set_rotation(55)
                tick.set_ha("right")
                tick.set_fontsize(6.2)
        if yscale == "log":
            ax.set_yscale("log")
        ax.axvline(spec.reference_value, color="#7B1FA2", linestyle=":", linewidth=0.85)
        ax.set_title(f"{estimator}\n{family}", loc="left", fontsize=8.2, fontweight="bold", pad=2.5)
        ax.set_ylabel(metric_label if idx % n_cols == 0 else "", fontsize=7.2)
        ax.set_xlabel(spec.x_label if idx >= (n_rows - 1) * n_cols else "", fontsize=7.2)
        ax.tick_params(axis="both", labelsize=6.4)
        ax.grid(True, which="both", alpha=0.24)
        if signed:
            ax.legend(loc="best", fontsize=5.8, frameon=True, title="sign", title_fontsize=5.8)
    for ax in axes_flat[n_estimators:]:
        ax.set_visible(False)
    fig.suptitle(f"{spec.label}: all canonical estimators ({_dominant_policy_label(part)})", fontsize=13.0, y=0.997)
    fig.text(
        0.5,
        0.975,
        f"Small multiples use one panel per estimator to avoid hiding methods in crowded family legends. Metric: {metric_label}.",
        ha="center",
        va="top",
        fontsize=7.2,
        color="#263238",
    )
    fig.text(
        0.5,
        0.006,
        "Solid/dashed lines mark positive/negative events when applicable. Violet dotted line marks the canonical reference level.",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color="#37474F",
    )
    fig.tight_layout(rect=[0.02, 0.03, 0.99, 0.95])
    return fig


def save_rmse_all_estimators_plot(df_global: pd.DataFrame, out_dir: Path) -> list[Path]:
    sweeps = sorted(df_global["sweep_key"].unique())
    pdf_path = out_dir / RMSE_ALL_ESTIMATORS_PDF_NAME
    png_path = out_dir / RMSE_ALL_ESTIMATORS_PNG_NAME
    first_fig: plt.Figure | None = None
    with PdfPages(pdf_path) as pdf:
        for sweep_key in sweeps:
            fig = _plot_all_estimators_page(df_global, str(sweep_key), "m1_rmse_hz_mean", "RMSE [Hz]", "log")
            if first_fig is None:
                first_fig = fig
            pdf.savefig(fig)
            if fig is not first_fig:
                plt.close(fig)
    paths = [png_path, pdf_path]
    if first_fig is not None:
        first_fig.savefig(png_path, dpi=240)
        if len(sweeps) == 1:
            sweep_key = str(sweeps[0])
            alias_png = out_dir / f"{sweep_key}_all_estimators_rmse.png"
            alias_pdf = out_dir / f"{sweep_key}_all_estimators_rmse.pdf"
            first_fig.savefig(alias_png, dpi=240)
            first_fig.savefig(alias_pdf)
            paths.extend([alias_png, alias_pdf])
        plt.close(first_fig)
    return paths


def save_multipage_dashboard(df_global: pd.DataFrame, out_dir: Path) -> Path:
    pdf_path = out_dir / MULTIPAGE_PDF_NAME
    df_plot = _add_derived_metric_columns(df_global)
    pages = [
        ("m1_rmse_hz_mean", "RMSE [Hz]", "log"),
        ("m3_max_peak_hz_mean", "Peak FE [Hz]", "log"),
        ("m27_post_100ms_rmse_hz_mean", "Post-event 100 ms RMSE [Hz]", "log"),
        ("m29_late_event_rmse_hz_mean", "Late-window RMSE [Hz]", "log"),
        ("m30_event_settling_time_s_mean", "Settling time [s]", "linear"),
        ("m5_trip_risk_s_mean", "Trip-risk time [s]", "linear"),
        ("m15_pass_rate_pct_mean", "Pass rate [%]", "linear"),
        ("m13_cpu_time_us_mean", "CPU time [us/pass]", "log"),
    ]
    with PdfPages(pdf_path) as pdf:
        for sweep_key in sorted(df_plot["sweep_key"].unique()):
            for metric_col, label, yscale in pages:
                if metric_col not in df_plot.columns:
                    continue
                fig, _ = _plot_metric_page(df_plot, str(sweep_key), metric_col, label, yscale)
                pdf.savefig(fig)
                plt.close(fig)
        fig, ax = plt.subplots(figsize=(11.0, 8.0))
        ax.axis("off")
        lines = [
            "OpenFreqBench ATLAS",
            "",
            f"Method version: {METHOD_VERSION}",
            f"Sweeps: {', '.join(sorted(df_global['sweep_key'].unique()))}",
            f"Estimators: {df_global['estimator'].nunique()}",
            f"Scenarios: {df_global['scenario'].nunique()}",
            f"Median MC runs per pair: {int(df_global['n_mc_runs'].median())}",
            "",
            "Interpretation:",
            "- Every sweep uses the same runner, metric aggregation, manifests and plotting code.",
            "- Signed sweeps plot positive and negative events together; level-only sweeps isolate one disturbance amplitude.",
            "- CI bands are bootstrap intervals over Monte Carlo runs; n=1 runs are diagnostic only.",
            "- Guide lines in older plots are not formal compliance claims unless tied to a preregistered test.",
        ]
        ax.text(0.04, 0.95, "\n".join(lines), va="top", ha="left", fontsize=11, color="#263238", wrap=True)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    return pdf_path


def _plot_method_map_page(df_global: pd.DataFrame, sweep_key: str) -> plt.Figure:
    spec = SWEEP_SPECS[sweep_key]
    part = df_global[df_global["sweep_key"] == sweep_key].copy()
    pivot = (
        part.groupby(["estimator", spec.x_col], as_index=False)["m1_rmse_hz_mean"]
        .mean()
        .pivot(index="estimator", columns=spec.x_col, values="m1_rmse_hz_mean")
    )
    families = part[["estimator", "family"]].drop_duplicates().set_index("estimator")["family"].to_dict()
    family_order = {name: idx for idx, name in enumerate(FAMILY_ORDER)}
    ordered = sorted(pivot.index, key=lambda est: (family_order.get(families.get(est, ""), 99), str(est)))
    pivot = pivot.loc[ordered]
    x_vals = [float(x) for x in pivot.columns]
    values = np.log10(np.maximum(pivot.to_numpy(dtype=float), 1e-12))

    fig, axes = plt.subplots(2, 1, figsize=(13.0, 8.2), gridspec_kw={"height_ratios": [3.2, 1.15]})
    ax = axes[0]
    finite = values[np.isfinite(values)]
    vmax = float(np.percentile(finite, 95)) if finite.size else 0.0
    im = ax.imshow(values, aspect="auto", cmap="magma_r", vmin=-4.0, vmax=max(vmax, -4.0 + 1e-6))
    ax.set_title(f"{spec.label} Method Stress Map", loc="left", fontweight="bold")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_xticks(np.arange(len(x_vals)))
    ax.set_xticklabels([f"{v:g}" for v in x_vals], rotation=45, ha="right", fontsize=7)
    ax.set_xlabel(spec.x_label)
    ax.set_ylabel("Estimator")
    cbar = fig.colorbar(im, ax=ax, fraction=0.024, pad=0.012)
    cbar.set_label("log10 mean RMSE [Hz]")

    ax2 = axes[1]
    guide = _env_float("ATLAS_LIMIT_RMSE_GUIDE", 0.05, minimum=0.0)
    rows = []
    for est, df_est in part.sort_values(spec.x_col).groupby("estimator", sort=False):
        reduced = df_est.groupby(spec.x_col, as_index=False)["m1_rmse_hz_mean"].mean().sort_values(spec.x_col)
        fail = reduced[reduced["m1_rmse_hz_mean"] > guide]
        critical = float(fail.iloc[0][spec.x_col]) if not fail.empty else float("nan")
        rows.append((est, families.get(est, ""), critical, float(reduced[spec.x_col].max())))
    summary = pd.DataFrame(rows, columns=["estimator", "family", "critical_level", "max_level"])
    summary = summary.sort_values(["family", "critical_level"], na_position="last")
    y = np.arange(len(summary))
    x = summary["critical_level"].fillna(summary["max_level"] * 1.05).to_numpy(dtype=float)
    ax2.scatter(x, y, s=24, color="#263238")
    for i, row in enumerate(summary.itertuples(index=False)):
        ax2.text(float(x[i]) * 1.03, i, str(row.estimator), va="center", fontsize=6.4)
    ax2.axvline(guide, color="#303F9F", linestyle="--", linewidth=0.9, label=f"RMSE guide {guide:g} Hz")
    if x_vals:
        ax2.set_xscale("log")
        ax2.set_xlim(min(x_vals), max(x_vals) * 1.4)
        _shade_severity_regions(ax2, spec, min(x_vals), max(x_vals), labels=True)
    ax2.set_yticks([])
    ax2.set_xlabel(f"First {spec.x_label} where mean RMSE exceeds guide")
    ax2.set_title(f"Critical {spec.label} Summary", loc="left", fontweight="bold")
    ax2.grid(True, which="both", alpha=0.22)
    ax2.legend(loc="best", fontsize=6.4, frameon=True)

    fig.suptitle(f"{spec.label} Method Atlas ({_dominant_policy_label(part)})", fontsize=13, y=0.995)
    fig.text(0.5, 0.965, spec.methodology, ha="center", va="top", fontsize=7.2, color="#263238", wrap=True)
    fig.tight_layout(rect=[0.06, 0.04, 0.98, 0.93])
    return fig


def save_method_map(df_global: pd.DataFrame, out_dir: Path) -> list[Path]:
    sweeps = sorted(df_global["sweep_key"].unique())
    png = out_dir / METHOD_MAP_PNG_NAME
    pdf = out_dir / METHOD_MAP_PDF_NAME
    first_fig: plt.Figure | None = None
    with PdfPages(pdf) as pages:
        for sweep_key in sweeps:
            fig = _plot_method_map_page(df_global, str(sweep_key))
            if first_fig is None:
                first_fig = fig
            pages.savefig(fig)
            if fig is not first_fig:
                plt.close(fig)
    if first_fig is not None:
        first_fig.savefig(png, dpi=240)
        if len(sweeps) == 1:
            sweep_key = str(sweeps[0])
            alias_png = out_dir / f"{sweep_key}_method_map.png"
            alias_pdf = out_dir / f"{sweep_key}_method_map.pdf"
            first_fig.savefig(alias_png, dpi=240)
            first_fig.savefig(alias_pdf)
            plt.close(first_fig)
            return [png, pdf, alias_png, alias_pdf]
        plt.close(first_fig)
    return [png, pdf]


def save_sign_asymmetry(df_global: pd.DataFrame, out_dir: Path) -> list[Path]:
    rows: list[dict[str, Any]] = []
    for (sweep_key, estimator), part in df_global.groupby(["sweep_key", "estimator"], sort=True):
        spec = SWEEP_SPECS[str(sweep_key)]
        ratios: list[float] = []
        for _x, by_x in part.groupby(spec.x_col):
            vals = by_x.set_index("direction")["m1_rmse_hz_mean"].to_dict()
            if "pos" in vals and "neg" in vals:
                pos = max(float(vals["pos"]), 1e-12)
                neg = max(float(vals["neg"]), 1e-12)
                ratios.append(max(pos, neg) / min(pos, neg))
        if ratios:
            rows.append({"sweep_key": sweep_key, "estimator": estimator, "max_asymmetry_ratio": max(ratios)})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(12.0, 6.0))
    if df.empty:
        level_rows: list[dict[str, Any]] = []
        for (sweep_key, estimator), part in df_global.groupby(["sweep_key", "estimator"], sort=True):
            spec = SWEEP_SPECS[str(sweep_key)]
            if spec.directional:
                continue
            reduced = (
                part.groupby(spec.x_col, as_index=False)["m1_rmse_hz_mean"]
                .mean()
                .sort_values(spec.x_col)
            )
            x = pd.to_numeric(reduced[spec.x_col], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(reduced["m1_rmse_hz_mean"], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
            if np.count_nonzero(ok) < 2:
                continue
            lx = np.log10(x[ok])
            ly = np.log10(np.maximum(y[ok], 1e-12))
            slope, _intercept = np.polyfit(lx, ly, 1)
            ratio = float(np.max(y[ok]) / max(float(np.min(y[ok])), 1e-12))
            level_rows.append(
                {
                    "sweep_key": sweep_key,
                    "estimator": estimator,
                    "family": str(part["family"].iloc[0]),
                    "slope": float(slope),
                    "ratio": ratio,
                }
            )
        df_level = pd.DataFrame(level_rows)
        if df_level.empty:
            ax.text(0.5, 0.5, "No paired sign or level-sensitivity diagnostic available.", ha="center", va="center")
        else:
            df_level = df_level.sort_values(["family", "ratio"], ascending=[True, False])
            color_map = _estimator_color_map(df_level["estimator"].astype(str).tolist())
            colors = [color_map.get(str(est), matplotlib.colors.to_rgba("#616161")) for est in df_level["estimator"]]
            x = np.arange(len(df_level))
            ax.bar(x, df_level["ratio"].to_numpy(dtype=float), color=colors, alpha=0.90, edgecolor="#111111", linewidth=0.25)
            ax.axhline(2.0, color="#303F9F", linestyle="--", linewidth=0.95, label="2x ratio guide")
            ax.set_yscale("log")
            ax.set_xticks(x)
            ax.set_xticklabels(df_level["estimator"], rotation=70, ha="right", fontsize=7)
            ax.set_ylabel("max RMSE / min RMSE across level sweep")
            ax.set_title("Level-only sensitivity by estimator", loc="left", fontweight="bold")
            ax.grid(True, which="both", axis="y", alpha=0.25)
            ax.legend(fontsize=7)
    else:
        pivot = df.pivot(index="estimator", columns="sweep_key", values="max_asymmetry_ratio").fillna(1.0)
        pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index]
        x = np.arange(len(pivot.index))
        width = 0.8 / max(1, len(pivot.columns))
        for idx, col in enumerate(pivot.columns):
            ax.bar(x + idx * width, pivot[col].to_numpy(dtype=float), width=width, label=str(col))
        ax.axhline(3.0, color="#B71C1C", linestyle="--", linewidth=1.0, label="diagnostic threshold")
        ax.set_xticks(x + width * (len(pivot.columns) - 1) / 2)
        ax.set_xticklabels(pivot.index, rotation=70, ha="right", fontsize=7)
        ax.set_ylabel("max(pos, neg) / min(pos, neg)")
        ax.set_title("Sign asymmetry by estimator and sweep", loc="left", fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=7)
    fig.tight_layout()
    png = out_dir / ASYMMETRY_PNG_NAME
    pdf = out_dir / ASYMMETRY_PDF_NAME
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    sweep_keys = sorted(df_global["sweep_key"].unique())
    extra_paths: list[Path] = []
    if len(sweep_keys) == 1:
        suffix = "sign_asymmetry" if SWEEP_SPECS[str(sweep_keys[0])].directional else "level_sensitivity"
        alias_png = out_dir / f"{sweep_keys[0]}_{suffix}.png"
        alias_pdf = out_dir / f"{sweep_keys[0]}_{suffix}.pdf"
        fig.savefig(alias_png, dpi=240)
        fig.savefig(alias_pdf)
        extra_paths.extend([alias_png, alias_pdf])
    plt.close(fig)
    return [png, pdf, *extra_paths]


def save_pareto_plot(df_global: pd.DataFrame, out_dir: Path) -> list[Path]:
    sweeps = sorted(df_global["sweep_key"].unique())
    n_sweeps = max(1, len(sweeps))
    n_cols = min(3, n_sweeps)
    n_rows = int(math.ceil(n_sweeps / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 4.2 * n_rows), sharey=False, squeeze=False)
    axes_flat = axes.flatten()
    color_map = _estimator_color_map(sorted(df_global["estimator"].unique()))
    for ax, sweep_key in zip(axes_flat, sweeps):
        part = (
            df_global[df_global["sweep_key"] == sweep_key]
            .groupby(["estimator", "family"], as_index=False)
            .agg(
                rmse=("m1_rmse_hz_mean", "median"),
                cpu=("m13_cpu_time_us_mean", "median"),
                latency=("m14_struct_latency_ms_mean", "median"),
            )
        )
        for _, row in part.iterrows():
            ax.scatter(
                max(float(row["cpu"]), 1e-12),
                max(float(row["rmse"]), 1e-12),
                s=25 + 4 * max(float(row.get("latency", 0.0)), 0.0),
                color=color_map.get(str(row["estimator"]), "#616161"),
                edgecolors="#111111",
                linewidths=0.25,
                alpha=0.85,
            )
        top = part.sort_values("rmse").head(5)
        for _, row in top.iterrows():
            ax.text(max(float(row["cpu"]), 1e-12), max(float(row["rmse"]), 1e-12), str(row["estimator"]), fontsize=6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("CPU time [us/pass]")
        ax.set_ylabel("median RMSE [Hz]")
        ax.set_title(SWEEP_SPECS[str(sweep_key)].label, loc="left", fontweight="bold")
        ax.grid(True, which="both", alpha=0.25)
    for ax in axes_flat[len(sweeps):]:
        ax.set_visible(False)
    fig.suptitle("Accuracy vs CPU vs structural latency", fontsize=13)
    fig.tight_layout()
    png = out_dir / PARETO_PNG_NAME
    pdf = out_dir / PARETO_PDF_NAME
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def save_summary_tables(df_global: pd.DataFrame, timing_rows: list[dict[str, Any]], out_dir: Path) -> tuple[Path, Path, Path, Path]:
    global_csv = out_dir / GLOBAL_CSV_NAME
    df_global.to_csv(global_csv, index=False)
    rmse_cols = [
        "sweep_key",
        "scenario",
        "direction",
        "estimator",
        "family",
        "n_mc_runs",
        "m1_rmse_hz_mean",
        "m1_rmse_hz_median",
        "m1_rmse_hz_ci95_low",
        "m1_rmse_hz_ci95_high",
        "m1_rmse_hz_std",
    ]
    for spec in SWEEP_SPECS.values():
        if spec.x_col in df_global.columns:
            rmse_cols.append(spec.x_col)
        if spec.signed_col in df_global.columns:
            rmse_cols.append(spec.signed_col)
    rmse_cols = [c for c in dict.fromkeys(rmse_cols) if c in df_global.columns]
    rmse_est = out_dir / RMSE_EST_CSV_NAME
    df_global[rmse_cols].to_csv(rmse_est, index=False)
    family = (
        df_global.groupby(["sweep_key", "family"], as_index=False)
        .agg(
            family_rmse_median=("m1_rmse_hz_mean", "median"),
            family_rmse_min=("m1_rmse_hz_mean", "min"),
            family_rmse_max=("m1_rmse_hz_mean", "max"),
            estimator_count=("estimator", "nunique"),
        )
        .sort_values(["sweep_key", "family"])
    )
    rmse_family = out_dir / RMSE_FAM_CSV_NAME
    family.to_csv(rmse_family, index=False)
    timing_csv = out_dir / TIMING_CSV_NAME
    pd.DataFrame(timing_rows).to_csv(timing_csv, index=False)
    return global_csv, rmse_est, rmse_family, timing_csv


def save_hypothesis_results(df_global: pd.DataFrame, out_dir: Path) -> Path:
    rows: list[dict[str, Any]] = []
    for (sweep_key, estimator), part in df_global.groupby(["sweep_key", "estimator"], sort=True):
        spec = SWEEP_SPECS[str(sweep_key)]
        reduced = (
            part.groupby(spec.x_col, as_index=False)["m1_rmse_hz_mean"]
            .mean()
            .sort_values(spec.x_col)
        )
        x = reduced[spec.x_col].to_numpy(dtype=float)
        y = np.maximum(reduced["m1_rmse_hz_mean"].to_numpy(dtype=float), 1e-12)
        if len(x) < 4:
            regime = "too_few_points"
            slope = float("nan")
            ratio = float("nan")
        else:
            slope, _intercept = np.polyfit(np.log10(x), np.log10(y), 1)
            ratio = float(y[-1] / max(y[0], 1e-12))
            diffs = np.diff(np.log10(y))
            if ratio <= 1.35 and abs(float(slope)) <= 0.12:
                regime = "flat"
            elif float(slope) > 0.15 and float(np.mean(diffs >= -0.08)) >= 0.75:
                regime = "monotone_deterioration"
            elif float(slope) < -0.12:
                regime = "improves_with_severity"
            else:
                regime = "nonmonotone_or_noise_limited"
        rows.append(
            {
                "hypothesis_id": f"{sweep_key}_{estimator}_severity_trend",
                "sweep_key": sweep_key,
                "estimator": estimator,
                "family": ESTIMATOR_FAMILIES.get(estimator, "Unknown"),
                "metric": "m1_rmse_hz",
                "x_axis": spec.x_col,
                "trend_slope_loglog": float(slope) if math.isfinite(float(slope)) else "",
                "high_low_ratio": float(ratio) if math.isfinite(float(ratio)) else "",
                "classification": regime,
                "status": "diagnostic" if int(part["n_mc_runs"].median()) < 30 else "claimable_with_mc_support",
            }
        )
    path = out_dir / HYPOTHESIS_CSV_NAME
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _canonical_estimator_set() -> set[str]:
    return set(_csv(CANONICAL_ESTIMATORS))


def _issue(severity: str, code: str, message: str) -> dict[str, str]:
    return {"severity": severity, "code": code, "message": message}


def build_atlas_readiness_report(df_global: pd.DataFrame, settings: dict[str, Any]) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    if df_global.empty:
        issues.append(_issue("blocker", "empty_results", "No aggregate rows were produced."))
        return {
            "schema_version": "openfreqbench-atlas-readiness-v1",
            "method_version": METHOD_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "diagnostic",
            "scope": "empty",
            "paper_claims_allowed": False,
            "journal_claims_allowed": False,
            "settings": settings,
            "summary": {},
            "issues": issues,
            "required_next_action": "Rerun ATLAS and inspect estimator/scenario failures before using any result.",
        }

    sweeps = sorted(str(item) for item in df_global["sweep_key"].dropna().astype(str).unique()) if "sweep_key" in df_global else []
    estimators = sorted(str(item) for item in df_global["estimator"].dropna().astype(str).unique()) if "estimator" in df_global else []
    policies = sorted(str(item) for item in df_global["policy"].dropna().astype(str).unique()) if "policy" in df_global else []
    n_runs = pd.to_numeric(df_global.get("n_mc_runs", pd.Series(dtype=float)), errors="coerce").dropna()
    min_runs = int(n_runs.min()) if not n_runs.empty else 0
    max_runs = int(n_runs.max()) if not n_runs.empty else 0
    canonical = _canonical_estimator_set()
    missing_sweeps = [item for item in REQUIRED_ATLAS_SWEEPS if item not in set(sweeps)]
    missing_estimators = sorted(canonical.difference(estimators))
    extra_estimators = sorted(set(estimators).difference(canonical))

    if missing_sweeps:
        issues.append(
            _issue(
                "blocker",
                "missing_required_sweeps",
                "Full ATLAS paper claims require all sweeps: " + ", ".join(missing_sweeps) + ".",
            )
        )
    if missing_estimators:
        issues.append(
            _issue(
                "blocker",
                "missing_canonical_estimators",
                "Canonical estimator set is incomplete: " + ", ".join(missing_estimators) + ".",
            )
        )
    if extra_estimators:
        issues.append(
            _issue(
                "warning",
                "noncanonical_estimators_present",
                "Noncanonical estimators are present and must be reported separately: " + ", ".join(extra_estimators) + ".",
            )
        )
    if min_runs < PAPER_GRADE_MIN_RUNS:
        issues.append(
            _issue(
                "blocker",
                "insufficient_monte_carlo_runs",
                f"Minimum Monte Carlo count is {min_runs}; paper-grade ATLAS requires at least {PAPER_GRADE_MIN_RUNS}.",
            )
        )
    if len(policies) != 1:
        issues.append(
            _issue(
                "blocker",
                "mixed_parameter_policies",
                "A paper-grade ATLAS run must use one parameter policy; found: " + ", ".join(policies or ["<missing>"]) + ".",
            )
        )
    elif policies[0] not in PAPER_READY_POLICIES:
        policy = policies[0]
        if policy == "per_scenario_oracle":
            msg = "Oracle tuning is a lower-bound diagnostic, not a deployable estimator policy."
        elif policy == "default":
            msg = "Default parameters are useful for smoke/exploratory analysis; use fixed_policy for paper-grade ATLAS claims."
        else:
            msg = f"Policy {policy!r} is not approved for paper-grade ATLAS claims."
        issues.append(_issue("blocker", "policy_not_paper_ready", msg))

    level_counts: dict[str, int] = {}
    direction_coverage: dict[str, list[str]] = {}
    for sweep_key in sweeps:
        spec = SWEEP_SPECS.get(sweep_key)
        if spec is None or spec.x_col not in df_global.columns:
            continue
        part = df_global[df_global["sweep_key"].astype(str) == sweep_key]
        levels = pd.to_numeric(part[spec.x_col], errors="coerce").dropna().unique()
        level_counts[sweep_key] = int(len(levels))
        if len(levels) < MIN_LEVELS_PER_SWEEP:
            issues.append(
                _issue(
                    "blocker",
                    "insufficient_sweep_levels",
                    f"{spec.label} has {len(levels)} level(s); paper-grade ATLAS requires at least {MIN_LEVELS_PER_SWEEP}.",
                )
            )
        if spec.directional:
            directions = sorted(str(item) for item in part["direction"].dropna().astype(str).unique()) if "direction" in part else []
            direction_coverage[sweep_key] = directions
            missing_directions = [item for item in ("pos", "neg") if item not in directions]
            if missing_directions:
                issues.append(
                    _issue(
                        "blocker",
                        "missing_directional_signs",
                        f"{spec.label} is missing direction(s): " + ", ".join(missing_directions) + ".",
                    )
                )
        else:
            direction_coverage[sweep_key] = sorted(str(item) for item in part["direction"].dropna().astype(str).unique()) if "direction" in part else []

    n_cost_reps = int(settings.get("n_cost_reps", 0) or 0)
    if n_cost_reps < 3:
        issues.append(
            _issue(
                "warning",
                "low_cpu_repetitions",
                f"CPU timing uses n_cost_reps={n_cost_reps}; use at least 3 for publication-quality timing comparisons.",
            )
        )

    has_blockers = any(item["severity"] == "blocker" for item in issues)
    if has_blockers:
        status = "diagnostic"
    elif min_runs >= JOURNAL_GRADE_MIN_RUNS:
        status = "journal_grade"
    else:
        status = "paper_grade"

    scope = "full_atlas" if not missing_sweeps else "subset"
    paper_claims_allowed = status in {"paper_grade", "journal_grade"}
    journal_claims_allowed = status == "journal_grade"
    if status == "journal_grade":
        next_action = "Archive this run with its manifest and use artifact hashes for paper claims."
    elif status == "paper_grade":
        next_action = f"Use for confirmatory paper analysis, or rerun with n_runs>={JOURNAL_GRADE_MIN_RUNS} for journal-grade evidence."
    else:
        next_action = (
            "Treat this output as diagnostic. Rerun with --sweeps all --policy fixed_policy "
            f"--n-runs {JOURNAL_GRADE_MIN_RUNS} and the full canonical estimator set before updating paper numbers."
        )

    return {
        "schema_version": "openfreqbench-atlas-readiness-v1",
        "method_version": METHOD_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "scope": scope,
        "paper_claims_allowed": paper_claims_allowed,
        "journal_claims_allowed": journal_claims_allowed,
        "settings": settings,
        "summary": {
            "sweeps_present": sweeps,
            "required_sweeps": list(REQUIRED_ATLAS_SWEEPS),
            "missing_sweeps": missing_sweeps,
            "estimators_present": estimators,
            "canonical_estimators": sorted(canonical),
            "missing_canonical_estimators": missing_estimators,
            "extra_estimators": extra_estimators,
            "policies": policies,
            "min_n_mc_runs": min_runs,
            "max_n_mc_runs": max_runs,
            "n_rows": int(len(df_global)),
            "level_counts": level_counts,
            "direction_coverage": direction_coverage,
            "minimum_required_runs": PAPER_GRADE_MIN_RUNS,
            "journal_required_runs": JOURNAL_GRADE_MIN_RUNS,
            "minimum_levels_per_sweep": MIN_LEVELS_PER_SWEEP,
        },
        "issues": issues,
        "required_next_action": next_action,
    }


def write_atlas_readiness_report(df_global: pd.DataFrame, settings: dict[str, Any], out_dir: Path) -> tuple[Path, Path, dict[str, Any]]:
    report = build_atlas_readiness_report(df_global, settings)
    json_path = out_dir / READINESS_JSON_NAME
    md_path = out_dir / READINESS_MD_NAME
    json_path.write_text(json.dumps(benchmark._to_builtin(report), indent=2, ensure_ascii=False), encoding="utf-8")

    summary = report.get("summary", {})
    lines = [
        "# ATLAS Readiness Report",
        "",
        f"Status: `{report['status']}`",
        f"Scope: `{report['scope']}`",
        f"Paper claims allowed: `{str(report['paper_claims_allowed']).lower()}`",
        f"Journal claims allowed: `{str(report['journal_claims_allowed']).lower()}`",
        "",
        "## Summary",
        "",
        f"- Sweeps: {len(summary.get('sweeps_present', []))}/{len(REQUIRED_ATLAS_SWEEPS)}",
        f"- Estimators: {len(summary.get('estimators_present', []))}/{len(_canonical_estimator_set())} canonical",
        f"- Monte Carlo runs: {summary.get('min_n_mc_runs', 0)} min, {summary.get('max_n_mc_runs', 0)} max",
        f"- Parameter policies: {', '.join(summary.get('policies', [])) or '<missing>'}",
        "",
        "## Issues",
        "",
    ]
    if report["issues"]:
        for item in report["issues"]:
            lines.append(f"- `{item['severity']}` `{item['code']}`: {item['message']}")
    else:
        lines.append("- None.")
    lines.extend(["", "## Next Action", "", str(report["required_next_action"]), ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path, report


def build_benchmark_report(df_global: pd.DataFrame, artifacts: dict[str, str], settings: dict[str, Any], out_dir: Path) -> Path:
    report_path = out_dir / BENCHMARK_REPORT_NAME
    payload = {
        "schema_version": "openfreqbench-atlas-report-v1",
        "method_version": METHOD_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": settings.get("run_id"),
        "mode": "atlas",
        "metric_profile": "canonical-single-phase-v1",
        "settings": settings,
        "artifacts": artifacts,
        "aggregated_metrics": benchmark._to_builtin(df_global.to_dict(orient="records")),
        "reproducibility": {
            "git": git_manifest(ROOT),
            "command": settings.get("command"),
        },
    }
    report_path.write_text(json.dumps(benchmark._to_builtin(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    return report_path


def write_manifest(
    out_dir: Path,
    scenarios: list[AtlasScenario],
    estimators: dict[str, type],
    settings: dict[str, Any],
    artifacts: dict[str, str],
) -> Path:
    path = out_dir / MANIFEST_NAME
    payload = {
        "schema_version": "openfreqbench-atlas-manifest-v1",
        "method_version": METHOD_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(out_dir.resolve()),
        "settings": settings,
        "sweeps": {
            key: {
                "label": spec.label,
                "x_col": spec.x_col,
                "signed_col": spec.signed_col,
                "methodology": spec.methodology,
            }
            for key, spec in SWEEP_SPECS.items()
            if key in {sc.sweep_key for sc in scenarios}
        },
        "scenarios": [
            {
                "sweep_key": sc.sweep_key,
                "scenario": sc.scenario_name,
                "signed_value": sc.signed_value,
                "abs_value": sc.abs_value,
                "direction": sc.direction,
                "default_params": benchmark._to_builtin(sc.scenario_cls.get_default_params()),
                "monte_carlo_space": benchmark._to_builtin(sc.scenario_cls.get_monte_carlo_space()),
                "event_metrics_enabled": not bool(getattr(sc.scenario_cls, "DISABLE_EVENT_METRICS", False)),
            }
            for sc in scenarios
        ],
        "estimators": list(estimators.keys()),
        "families": {label: ESTIMATOR_FAMILIES.get(label, "Unknown") for label in estimators},
        "artifacts": artifacts,
        "artifact_inventory": [
            row
            for row in build_artifact_index(out_dir)
            if Path(str(row["artifact_path"])).resolve() != path.resolve()
        ],
        "git": git_manifest(ROOT),
    }
    path.write_text(json.dumps(benchmark._to_builtin(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def reproduce_command(settings: dict[str, Any]) -> str:
    parts = [
        "python -m pipelines.atlas_sweep",
        f"--sweeps {settings['sweeps_arg']}",
        f"--policy {settings['policy']}",
        f"--n-runs {settings['n_mc_runs']}",
        f"--base-seed {settings['base_seed']}",
        f"--n-cost-reps {settings['n_cost_reps']}",
        f"--tune-trials {settings['tune_trials']}",
        f"--tune-eval-runs {settings['tune_eval_runs']}",
        f"--output-subdir {settings['output_subdir']}",
    ]
    if settings.get("run_id") and settings["run_id"] != settings["output_subdir"]:
        parts.append(f"--run-id {settings['run_id']}")
    return " ".join(parts)


def write_readme(out_dir: Path, settings: dict[str, Any], readiness: dict[str, Any] | None = None) -> Path:
    path = out_dir / "README.md"
    readiness = readiness or {}
    readiness_status = readiness.get("status", "unknown")
    paper_allowed = str(readiness.get("paper_claims_allowed", False)).lower()
    lines = [
        "# OpenFreqBench ATLAS Run",
        "",
        "This artifact directory was produced by the unified ATLAS pipeline.",
        "",
        "## Readiness",
        "",
        f"- Status: `{readiness_status}`",
        f"- Paper claims allowed: `{paper_allowed}`",
        f"- Full details: `{READINESS_MD_NAME}` and `{READINESS_JSON_NAME}`",
        "",
        "## Reproduce",
        "",
        "```powershell",
        reproduce_command(settings),
        "```",
        "",
        "## Primary Files",
        "",
        f"- `{GLOBAL_CSV_NAME}`",
        f"- `{MULTIPAGE_PDF_NAME}`",
        f"- `{RMSE_FAMILY_PDF_NAME}`",
        f"- `{RMSE_ALL_ESTIMATORS_PDF_NAME}`",
        f"- `{METHOD_MAP_PDF_NAME}`",
        f"- `{ASYMMETRY_PDF_NAME}`",
        f"- `{PARETO_PDF_NAME}`",
        f"- `{READINESS_MD_NAME}`",
        f"- `{READINESS_JSON_NAME}`",
        f"- `{MANIFEST_NAME}`",
        "- `artifact_index.csv`",
        "- `paper_traceability.csv`",
        "- `evidence_manifest.json`",
        "",
        "Runs with `n_runs < 30` are diagnostic. Use the readiness report before moving any number into the paper.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_atlas(args: argparse.Namespace) -> Path:
    t0 = time.time()
    policy = str(args.policy).strip().lower().replace("-", "_")
    if policy == "oracle":
        policy = "per_scenario_oracle"
    if policy not in {"default", "fixed_policy", "per_scenario_oracle"}:
        raise ValueError("ATLAS policy must be one of: default, fixed_policy, oracle/per_scenario_oracle.")

    sweep_keys = _expand_sweep_keys(_csv(args.sweeps))
    unknown = [item for item in sweep_keys if item not in SWEEP_SPECS]
    if unknown:
        raise ValueError(f"Unknown ATLAS sweeps: {unknown}. Known: {sorted(SWEEP_SPECS)}")

    output_subdir = args.output_subdir or os.getenv("ATLAS_OUTPUT_SUBDIR", "atlas_mvp2_representative")
    out_dir = ROOT / "artifacts" / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    n_mc_runs = int(args.n_runs if args.n_runs is not None else _env_int("ATLAS_N_MC_RUNS", 1, minimum=1))
    base_seed = int(args.base_seed if args.base_seed is not None else _env_int("ATLAS_BASE_SEED", 12345, minimum=0))
    n_cost_reps = int(args.n_cost_reps if args.n_cost_reps is not None else _env_int("ATLAS_N_COST_REPS", 1, minimum=1))
    tune_trials = int(args.tune_trials if args.tune_trials is not None else _env_int("ATLAS_TUNE_TRIALS", 0, minimum=0))
    tune_eval_runs = int(args.tune_eval_runs if args.tune_eval_runs is not None else _env_int("ATLAS_TUNE_EVAL_RUNS", 2, minimum=1))
    fixed_eval_runs = _env_int("ATLAS_FIXED_POLICY_EVAL_RUNS_PER_LEVEL", max(1, min(4, tune_eval_runs)), minimum=1)
    capture_signals = bool(args.capture_signals or _env_bool("ATLAS_CAPTURE_SIGNALS", False))
    resume = bool(args.resume or _env_bool("ATLAS_RESUME", True))
    run_id = args.run_id or os.getenv("ATLAS_RUN_ID", output_subdir)

    settings: dict[str, Any] = {
        "run_id": run_id,
        "method_version": METHOD_VERSION,
        "output_subdir": output_subdir,
        "sweeps_arg": args.sweeps,
        "sweeps": sweep_keys,
        "policy": policy,
        "n_mc_runs": n_mc_runs,
        "base_seed": base_seed,
        "n_cost_reps": n_cost_reps,
        "tune_trials": tune_trials,
        "tune_eval_runs": tune_eval_runs,
        "fixed_policy_eval_runs_per_level": fixed_eval_runs,
        "resume": resume,
        "capture_signals": capture_signals,
    }
    settings["command"] = reproduce_command(settings)

    scenarios = build_atlas_scenarios(sweep_keys)
    estimators = select_estimators()
    print(f"Running OpenFreqBench ATLAS: {len(scenarios)} scenarios x {len(estimators)} estimators")
    print(f"  Sweeps: {', '.join(sweep_keys)}")
    print(f"  Policy: {policy}; MC runs: {n_mc_runs}; tuning trials: {tune_trials}")
    print(f"  Output dir: {out_dir}")

    scenarios_by_sweep: dict[str, list[AtlasScenario]] = {}
    for sc in scenarios:
        scenarios_by_sweep.setdefault(sc.sweep_key, []).append(sc)

    fixed_params: dict[tuple[str, str], dict[str, Any]] = {}
    fixed_meta: dict[tuple[str, str], dict[str, Any]] = {}
    if policy == "fixed_policy":
        print("\nFixed-policy tuning phase")
        for sweep_key, sweep_scenarios in scenarios_by_sweep.items():
            frequency_bounds = _frequency_bounds_for_sweep(sweep_scenarios)
            eval_scenarios = _build_eval_scenarios(sweep_scenarios, base_seed + 200000, fixed_eval_runs)
            for est_name, est_cls in estimators.items():
                print(f"  [tune fixed] {sweep_key} / {est_name}", flush=True)
                params, meta = _tune_estimator(
                    est_name,
                    est_cls,
                    eval_scenarios,
                    n_trials=tune_trials,
                    tune_eval_runs=tune_eval_runs,
                    mode="fixed_policy",
                    frequency_bounds=frequency_bounds,
                )
                fixed_params[(sweep_key, est_name)] = params
                fixed_meta[(sweep_key, est_name)] = meta

    rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    for sc in scenarios:
        sc_dir = out_dir / sc.scenario_name
        sc_dir.mkdir(parents=True, exist_ok=True)
        frequency_bounds = _frequency_bounds_for_sweep(scenarios_by_sweep[sc.sweep_key])
        print(f"\nScenario {sc.scenario_name} ({sc.signed_value:+g})")
        for est_name, est_cls in estimators.items():
            est_n_mc = _env_int(f"ATLAS_{_env_key_for_estimator(est_name)}_N_MC_RUNS", n_mc_runs, minimum=1)
            est_n_cost = _env_int(f"ATLAS_{_env_key_for_estimator(est_name)}_N_COST_REPS", n_cost_reps, minimum=1)
            out_est = sc_dir / est_name
            out_est.mkdir(parents=True, exist_ok=True)
            summary_csv = out_est / f"{sc.scenario_name}__{est_name}_summary.csv"
            run_spec_path = out_est / "run_spec.json"
            expected = {
                "method_version": METHOD_VERSION,
                "scenario": sc.scenario_name,
                "estimator": est_name,
                "sweep_key": sc.sweep_key,
                "signed_value": sc.signed_value,
                "policy": policy,
                "n_mc_runs": int(est_n_mc),
                "n_cost_reps": int(est_n_cost),
                "base_seed": int(base_seed),
                "capture_signals": bool(capture_signals),
                "tune_trials": int(tune_trials),
                "tune_eval_runs": int(tune_eval_runs),
            }

            if resume and _can_reuse(run_spec_path, summary_csv, expected):
                summary_df = pd.read_csv(summary_csv)
                timing = {}
                spec_current = json.loads(run_spec_path.read_text(encoding="utf-8"))
                best_params = spec_current.get("best_params", {})
                tuning_meta = spec_current.get("tuning_meta", {})
            else:
                if policy == "fixed_policy":
                    best_params = dict(fixed_params.get((sc.sweep_key, est_name), {}))
                    tuning_meta = dict(fixed_meta.get((sc.sweep_key, est_name), {}))
                elif policy == "per_scenario_oracle":
                    eval_scenarios = _build_eval_scenarios([sc], base_seed + 200000, max(1, tune_eval_runs))
                    best_params, tuning_meta = _tune_estimator(
                        est_name,
                        est_cls,
                        eval_scenarios,
                        n_trials=tune_trials,
                        tune_eval_runs=tune_eval_runs,
                        mode="per_scenario_oracle",
                        frequency_bounds=frequency_bounds,
                    )
                else:
                    best_params = _apply_frequency_bounds(est_name, est_cls, _estimator_defaults(est_cls), frequency_bounds)
                    tuning_meta = {"mode": "default", "n_trials_requested": 0, "n_trials_executed": 0}

                print(f"  - {est_name}", flush=True)
                run_start = time.perf_counter()
                engine = MonteCarloEngine(
                    scenario_cls=sc.scenario_cls,
                    estimator_cls=est_cls,
                    estimator_params=best_params,
                    n_runs=est_n_mc,
                    base_seed=base_seed,
                    n_cost_reps=est_n_cost,
                    enforce_standardized_step=est_name not in {"PI-GRU", "Koopman (RK-DPMU)"},
                    capture_signals=capture_signals,
                )
                result = _run_engine_local(engine)
                summary_df = result.summary_df
                summary_df.to_csv(summary_csv, index=False)
                if capture_signals and not result.signals_df.empty:
                    result.signals_df.to_csv(out_est / f"{sc.scenario_name}__{est_name}_signals.csv", index=False)
                timing = {"total_elapsed_s": float(time.perf_counter() - run_start)}
                spec_current = {
                    **expected,
                    "family": ESTIMATOR_FAMILIES.get(est_name, "Unknown"),
                    "best_params": benchmark._to_builtin(best_params),
                    "tuning_meta": benchmark._to_builtin(tuning_meta),
                    "timing": timing,
                    "frequency_bounds_hz": list(frequency_bounds) if frequency_bounds else None,
                }
                run_spec_path.write_text(json.dumps(benchmark._to_builtin(spec_current), indent=2, ensure_ascii=False), encoding="utf-8")

            agg = _aggregate_summary(summary_df)
            row = {
                "sweep_key": sc.sweep_key,
                "sweep_label": sc.sweep_label,
                "scenario": sc.scenario_name,
                "signed_value": sc.signed_value,
                "abs_value": sc.abs_value,
                "direction": sc.direction,
                **sc.params,
                "estimator": est_name,
                "family": ESTIMATOR_FAMILIES.get(est_name, "Unknown"),
                "n_mc_runs": int(len(summary_df)),
                "policy": policy,
                "scenario_default_params_json": json.dumps(
                    benchmark._to_builtin(sc.scenario_cls.get_default_params()),
                    sort_keys=True,
                    ensure_ascii=False,
                ),
                "monte_carlo_space_json": json.dumps(
                    benchmark._to_builtin(sc.scenario_cls.get_monte_carlo_space()),
                    sort_keys=True,
                    ensure_ascii=False,
                ),
                "event_metrics_enabled": not bool(getattr(sc.scenario_cls, "DISABLE_EVENT_METRICS", False)),
                "best_params_json": json.dumps(benchmark._to_builtin(best_params), sort_keys=True, ensure_ascii=False),
                **agg,
            }
            rows.append(row)
            timing_rows.append(
                {
                    "sweep_key": sc.sweep_key,
                    "scenario": sc.scenario_name,
                    "estimator": est_name,
                    "family": ESTIMATOR_FAMILIES.get(est_name, "Unknown"),
                    "n_mc_runs": int(len(summary_df)),
                    "n_cost_reps": int(est_n_cost),
                    "tune_trials": int(tune_trials),
                    "tune_eval_runs": int(tune_eval_runs),
                    "policy": policy,
                    "total_elapsed_s": timing.get("total_elapsed_s") if isinstance(timing, dict) else None,
                }
            )

    df_global = pd.DataFrame(rows).sort_values(["sweep_key", "abs_value", "direction", "family", "estimator"])
    global_csv, rmse_est, rmse_family, timing_csv = save_summary_tables(df_global, timing_rows, out_dir)
    plot_paths, color_map = save_rmse_family_plot(df_global, out_dir)
    plot_paths.extend(save_rmse_all_estimators_plot(df_global, out_dir))
    plot_paths.extend(save_method_map(df_global, out_dir))
    plot_paths.extend(save_sign_asymmetry(df_global, out_dir))
    plot_paths.extend(save_pareto_plot(df_global, out_dir))
    dashboard_pdf = save_multipage_dashboard(df_global, out_dir)
    hypothesis_csv = save_hypothesis_results(df_global, out_dir)
    readiness_json_path, readiness_md_path, readiness_report = write_atlas_readiness_report(df_global, settings, out_dir)

    legend_path = out_dir / LEGEND_CSV_NAME
    pd.DataFrame(
        [
            {"estimator": est, "hex_color": matplotlib.colors.to_hex(rgba), "family": ESTIMATOR_FAMILIES.get(est, "Unknown")}
            for est, rgba in sorted(color_map.items())
        ]
    ).to_csv(legend_path, index=False)

    report_path = out_dir / BENCHMARK_REPORT_NAME
    manifest_path = out_dir / MANIFEST_NAME
    readme_path = out_dir / "README.md"
    env_path = out_dir / "environment_report.json"
    trace_path = out_dir / "paper_traceability.csv"
    artifact_index_path = out_dir / "artifact_index.csv"
    evidence_path = out_dir / "evidence_manifest.json"

    artifacts = {
        "aggregated_metrics_csv": str(global_csv),
        "rmse_by_estimator_csv": str(rmse_est),
        "rmse_by_family_csv": str(rmse_family),
        "timing_profile_csv": str(timing_csv),
        "hypothesis_results_csv": str(hypothesis_csv),
        "metrics_dashboard_pdf": str(dashboard_pdf),
        "rmse_family_pdf": str(out_dir / RMSE_FAMILY_PDF_NAME),
        "rmse_all_estimators_pdf": str(out_dir / RMSE_ALL_ESTIMATORS_PDF_NAME),
        "method_map_pdf": str(out_dir / METHOD_MAP_PDF_NAME),
        "sign_asymmetry_pdf": str(out_dir / ASYMMETRY_PDF_NAME),
        "pareto_pdf": str(out_dir / PARETO_PDF_NAME),
        "atlas_readiness_json": str(readiness_json_path),
        "atlas_readiness_md": str(readiness_md_path),
        "benchmark_report_json": str(report_path),
        "manifest_json": str(manifest_path),
        "readme": str(readme_path),
        "environment_report": str(env_path),
        "paper_traceability_csv": str(trace_path),
        "artifact_index_csv": str(artifact_index_path),
        "evidence_manifest_json": str(evidence_path),
    }
    report_path = build_benchmark_report(df_global, artifacts, settings, out_dir)
    readme_path = write_readme(out_dir, settings, readiness_report)
    env_path = write_environment_report(ROOT, env_path, source_root=SRC)
    trace_path = write_paper_traceability(report_path, trace_path)
    manifest_path = write_manifest(out_dir, scenarios, estimators, settings, artifacts)
    evidence_path = write_evidence_manifest(out_dir, evidence_path, source_report=report_path)
    manifest_path = write_manifest(out_dir, scenarios, estimators, settings, artifacts)
    artifact_index_path = write_artifact_index(out_dir, artifact_index_path)

    elapsed = (time.time() - t0) / 60.0
    print("\nArtifacts:")
    for path in [
        global_csv,
        rmse_est,
        rmse_family,
        timing_csv,
        hypothesis_csv,
        *plot_paths,
        dashboard_pdf,
        readiness_json_path,
        readiness_md_path,
        report_path,
        manifest_path,
        trace_path,
        artifact_index_path,
        evidence_path,
    ]:
        print(f"  - {path.relative_to(ROOT)}")
    print(
        f"\nReadiness: {readiness_report['status']} "
        f"(paper_claims_allowed={readiness_report['paper_claims_allowed']})"
    )
    print(f"\n[DONE] ATLAS completed in {elapsed:.1f} min.")
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified OpenFreqBench ATLAS sweep runner.")
    parser.add_argument(
        "--sweeps",
        default=os.getenv("ATLAS_SWEEPS", "all"),
        help=(
            "Comma list: all, core, p0, magnitude_step, rocof, frequency_step, "
            "phase_jump_sweep, modulation_am_sweep, modulation_fm_sweep, "
            "harmonics, interharmonics, noise_snr."
        ),
    )
    parser.add_argument("--policy", default=os.getenv("ATLAS_POLICY", "default"), help="default, fixed_policy, or oracle/per_scenario_oracle.")
    parser.add_argument("--n-runs", type=int, default=None, help="Monte Carlo runs per scenario/estimator.")
    parser.add_argument("--base-seed", type=int, default=None)
    parser.add_argument("--n-cost-reps", type=int, default=None)
    parser.add_argument("--tune-trials", type=int, default=None)
    parser.add_argument("--tune-eval-runs", type=int, default=None)
    parser.add_argument("--output-subdir", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--capture-signals", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_atlas(args)


if __name__ == "__main__":
    main()
