"""
Full active benchmark pipeline: scenario-wise Optuna tuning, Monte Carlo,
artifact generation, and paper-facing dashboard export.

Phases
------
1  Tune each estimator per scenario, run N_MC_RUNS, save CSV artifacts.
2  Aggregate summary CSVs into per-scenario and global statistics.
3  Generate performance dashboards from the aggregated report.
4  Export a large unified JSON report with full run specs, Monte Carlo statistics,
   trends, rankings, Pareto front, PCA, KMeans, and exploratory hypothesis tests.

Usage
-----
As a script (full benchmark):
    python -m pipelines.full_mc_benchmark

Legacy wrapper entry point:
    python tests/montecarlo/test_dedicated_smoke_test.py

As pytest (infrastructure sanity):
    pytest src/pipelines/full_mc_benchmark.py -q
"""

from __future__ import annotations

import gc
import inspect
import json
import os
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
except Exception:
    KMeans = None
    PCA = None
    StandardScaler = None
    silhouette_score = None

# â”€â”€ Thread safety: must be set before NumPy/MKL load in subprocesses â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# â”€â”€ Path bootstrap â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipelines.paths import (
    BENCHMARK_OUTPUT_DIR,
    FIGURE1_BASENAME,
    FIGURE2_BASENAME,
    JSON_REPORT_NAME,
)
from pipelines.benchmark_definition import (
    BENCHMARK_AUTHORITY_STATEMENT,
    BENCHMARK_IDENTITY,
    BENCHMARK_SCOPE,
    ESTIMATOR_FAMILIES,
    PAPER_ALIGNMENT_POLICY,
    build_estimator_registry_manifest,
    excluded_estimator_specs,
    load_active_estimators,
)

# â”€â”€ Project imports â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from analysis.monte_carlo_engine import MonteCarloEngine
from scenarios.ibr_multi_event import IBRMultiEventScenario
from scenarios.ieee_freq_ramp import IEEEFreqRampScenario
from scenarios.ieee_freq_step import IEEEFreqStepScenario
from scenarios.ieee_mag_step import IEEEMagStepScenario
from scenarios.ieee_single_sinwave import IEEESingleSinWaveScenario
from scenarios.ibr_power_imbalance_ringdown import IBRPowerImbalanceRingdownScenario
from scenarios.ieee_modulation import IEEEModulationScenario
from scenarios.ieee_modulation_am import IEEEModulationAMScenario
from scenarios.ieee_modulation_fm import IEEEModulationFMScenario
from scenarios.ieee_oob_interference import IEEEOOBInterferenceScenario
from scenarios.ieee_phase_jump_20 import IEEEPhaseJump20Scenario
from scenarios.ieee_phase_jump_60 import IEEEPhaseJump60Scenario
from scenarios.nerc_phase_jump_60 import NERCPhaseJump60Scenario
from scenarios.ibr_harmonics_small import IBRHarmonicsSmallScenario
from scenarios.ibr_harmonics_medium import IBRHarmonicsMediumScenario
from scenarios.ibr_harmonics_large import IBRHarmonicsLargeScenario

optuna.logging.set_verbosity(optuna.logging.WARNING)

# â”€â”€ Run configuration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        print(f"[WARN] Invalid integer for {name}={raw!r}; using default={default}.")
        return default
    if value < minimum:
        print(f"[WARN] {name}={value} < {minimum}; clamping to {minimum}.")
        return minimum
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_choice(name: str, default: str, choices: set[str]) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value not in choices:
        print(
            f"[WARN] Invalid value for {name}={raw!r}; "
            f"expected one of {sorted(choices)}. Using {default!r}."
        )
        return default
    return value


def _env_csv_list(name: str) -> list[str]:
    raw = os.getenv(name)
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


N_TRIALS_TUNING = _env_int("BENCHMARK_N_TRIALS_TUNING", 500, minimum=0)
N_MC_RUNS = _env_int("BENCHMARK_N_MC_RUNS", 100, minimum=1)
OPTUNA_SAMPLER_MODE = _env_choice(
    "BENCHMARK_OPTUNA_SAMPLER",
    "tpe",
    {"tpe", "random", "grid"},
)
OPTUNA_SEED = _env_int("BENCHMARK_OPTUNA_SEED", 42, minimum=0)
APPLY_TRIAL_OVERRIDES = _env_bool("BENCHMARK_APPLY_TRIAL_OVERRIDES", True)
EXCLUDED_ESTIMATOR_LABELS = _env_csv_list("BENCHMARK_EXCLUDE_ESTIMATORS")

BASE_RESULTS_DIR = BENCHMARK_OUTPUT_DIR
# --- Generador de Escenarios DinÃ¡micos ---
def create_variant(base_cls: type, name_suffix: str, param_overrides: dict[str, Any]) -> type:
    """Dynamically creates a subclass with overridden default parameters."""
    # El nombre visible (para carpetas y grÃ¡ficos) puede tener puntos
    new_name = f"{base_cls.SCENARIO_NAME}_{name_suffix}"
    new_params = {**base_cls.DEFAULT_PARAMS, **param_overrides}
    
    # Â¡NUEVO FIX!: El nombre INTERNO de la clase Python no puede tener puntos '.'
    safe_class_suffix = name_suffix.replace(".", "p").replace("-", "m")
    class_name = f"{base_cls.__name__}_{safe_class_suffix}"
    
    new_cls = type(class_name, (base_cls,), {
        "SCENARIO_NAME": new_name,
        "DEFAULT_PARAMS": new_params,
        "get_name": classmethod(lambda cls: cls.SCENARIO_NAME)
    })
    
    # Forzamos a que Python registre esta clase dinÃ¡mica en el espacio 
    # de nombres global para que 'pickle' pueda enviarla a los otros procesos.
    new_cls.__module__ = __name__
    globals()[class_name] = new_cls
    
    return new_cls

BASE_SCENARIOS = [
    IBRMultiEventScenario,
    IBRPowerImbalanceRingdownScenario,
    IBRHarmonicsSmallScenario,
    IBRHarmonicsMediumScenario,
    IBRHarmonicsLargeScenario,
    IEEEFreqStepScenario,
    IEEEModulationScenario,
    IEEEModulationAMScenario,
    IEEEModulationFMScenario,
    IEEEOOBInterferenceScenario,
    IEEEPhaseJump20Scenario,
    IEEEPhaseJump60Scenario,
    NERCPhaseJump60Scenario,
    IEEESingleSinWaveScenario,
]

# 1. Variaciones de Magnitude Step (Sag/Swell)
mag_steps = [
    (1.01, "1pct"),
    (1.05, "5pct"),
    (1.10, "10pct"),
    (1.15, "15pct"),
    (1.25, "25pct"),
    (1.50, "50pct"),
]
mag_variants = [
    create_variant(IEEEMagStepScenario, suffix, {"amp_post_pu": val}) 
    for val, suffix in mag_steps
]

# 2. Variaciones de Frequency Ramp (RoCoF)
ramp_steps = [
    (0.25, "0.25Hzs"),
    (0.5, "0.5Hzs"),
    (1.0, "1Hzs"),
    (2.0, "2Hzs"),
    (5.0, "5Hzs"),
    (10.0, "10Hzs"),
    (15.0, "15Hzs"),
    (20.0, "20Hzs"),
]
ramp_variants = [
    create_variant(IEEEFreqRampScenario, suffix, {"rocof_hz_s": val})
    for val, suffix in ramp_steps
]

# 3. Variaciones de Ringdown (Testing de SNR e Inter-armÃ³nicos)
ringdown_stress_tests = [
    (0.002, 0.01, "Low_Noise"),       # ~50 dB SNR, 1% interharmonic
    (0.007, 0.02, "Normal_Noise"),    # ~40 dB SNR, 2% interharmonic (Base)
    (0.022, 0.05, "Medium_Noise"),    # ~30 dB SNR, 5% interharmonic (PLL Killer)
    (0.03, 0.1, "Severe_Noise"),    # ~30 dB SNR, 10% interharmonic (PLL Killer)
]
ringdown_variants = [
    create_variant(IBRPowerImbalanceRingdownScenario, suffix, {
        "white_noise_sigma": noise_lvl,
        "interharmonic_pu": ih_lvl
    })
    for noise_lvl, ih_lvl, suffix in ringdown_stress_tests
]

# Lista final unificada
SCENARIOS = BASE_SCENARIOS + mag_variants + ramp_variants + ringdown_variants

# Metric columns written by MonteCarloEngine / metrics.py
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
    "m17_hw_class"
]

METRIC_LABELS: dict[str, str] = {
    "m1_rmse_hz": "RMSE_Hz",
    "m2_mae_hz": "MAE_Hz",
    "m3_max_peak_hz": "MAX_PEAK_Hz",
    "m4_std_error_hz": "STD_ERR_Hz",
    "m5_trip_risk_s": "TRIP_RISK_s",
    "m5_trip_risk_resolution_s": "TRIP_RISK_RES_s",
    "m6_max_contig_trip_s": "MAX_CONTIG_TRIP_s",
    "m7_pcb_hz": "PCB_Hz",
    "m8_settling_time_s": "SETTLING_TIME_s",
    "m9_rfe_max_hz_s": "RFE_MAX_Hz/s",
    "m10_rfe_rms_hz_s": "RFE_RMS_Hz/s",
    "m11_rnaf_db": "RNAF_dB",
    "m12_isi_pu": "ISI_pu",
    "m13_cpu_time_us": "CPU_TIME_us",
    "m14_struct_latency_ms": "STRUCT_LATENCY_ms",
    "m15_pcb_compliant": "PCB_COMPLIANT",
    "m16_heatmap_pass": "HEATMAP_PASS",
    "m17_hw_class": "HW_CLASS"
}

# â”€â”€ IEEE publication-quality style â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_IEEE_RC: dict[str, Any] = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.4,
    "axes.linewidth": 0.6,
    "lines.linewidth": 0.8,
    "patch.linewidth": 0.5,
    "mathtext.fontset": "stix",
}
_IEEE_FULL_W = 7.16
_IEEE_PAGE_H = 11.0

_ESTIMATOR_FAMILIES: dict[str, str] = ESTIMATOR_FAMILIES

_FAMILY_PALETTE: dict[str, str] = {
    "Model-based": "#1565C0",
    "Loop-based": "#2E7D32",
    "Window-based": "#E65100",
    "Adaptive": "#6A1B9A",
    "Data-driven": "#B71C1C",
}

# â”€â”€ Search spaces (PRO LEVEL - EXPANDED FOR 300 TRIALS) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SEARCH_SPACES: dict[str, Any] = {
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # Loop-based
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    "ZCD": lambda trial: {
        # ZCD estÃ¡ndar no tiene parÃ¡metros sintonizables en su __init__
    },
    "PLL": lambda trial: {
        # Ganancias pueden ser minÃºsculas (lazos muy amortiguados) o gigantes.
        "kp_scale": trial.suggest_float("kp_scale", 1e-4, 50.0, log=True),
        "ki_scale": trial.suggest_float("ki_scale", 1e-4, 100.0, log=True),
    },
    "SOGI-PLL": lambda trial: {
        # Tiempos de establecimiento desde 1/4 de ciclo (extremadamente agresivo) hasta 1 segundo
        "settle_time": trial.suggest_float("settle_time", 0.004, 1.0, log=True),
        "k_sogi": trial.suggest_float("k_sogi", 0.1, 5.0),
    },
    "SOGI-FLL": lambda trial: {
        "gamma": trial.suggest_float("gamma", 1.0, 1000.0, log=True),
        "k_sogi": trial.suggest_float("k_sogi", 0.1, 5.0),
    },
    "Type-3 SOGI-PLL": lambda trial: {
        # Type-3 requiere sintonÃ­a muy fina; ampliamos a rangos masivos logarÃ­tmicos
        "kp": trial.suggest_float("kp", 1.0, 1000.0, log=True),
        "ki": trial.suggest_float("ki", 10.0, 50000.0, log=True),
        "ki2": trial.suggest_float("ki2", 100.0, 500000.0, log=True),
    },

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # Model-based (Kalman Filters)
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    "LKF": lambda trial: {
        # Alineado con lkf.py (Narrowband 2-state)
        "q": trial.suggest_float("q", 1e-12, 1e2, log=True),
        "r": trial.suggest_float("r", 1e-8, 1e4, log=True),
        "rho": trial.suggest_float("rho", 0.90, 1.0),
        "output_smoothing": trial.suggest_float("output_smoothing", 1e-4, 0.5, log=True),
    },
    "LKF2": lambda trial: {
        # Alineado con lkf2.py (Ahmed-style 3-state)
        "q_dc": trial.suggest_float("q_dc", 1e-12, 1e0, log=True),
        "q_vc": trial.suggest_float("q_vc", 1e-12, 1e2, log=True),
        "q_vs": trial.suggest_float("q_vs", 1e-12, 1e2, log=True),
        "r": trial.suggest_float("r", 1e-8, 1e4, log=True),
        "beta": trial.suggest_float("beta", 1.0, 1000.0, log=True),
        "lpf_mu": trial.suggest_float("lpf_mu", 0.01, 1.0),
    },
    "EKF": lambda trial: {
        # Asumiendo parÃ¡metros estÃ¡ndar de ekf.py
        "q_omega": trial.suggest_float("q_omega", 1e-12, 1e2, log=True),
        "r_meas": trial.suggest_float("r_meas", 1e-8, 1e4, log=True),
    },
    "UKF": lambda trial: {
        # Alineado con ukf.py (4-state nonlinear)
        "q_dc": trial.suggest_float("q_dc", 1e-12, 1e0, log=True),
        "q_alpha": trial.suggest_float("q_alpha", 1e-12, 1e2, log=True),
        "q_beta": trial.suggest_float("q_beta", 1e-12, 1e2, log=True),
        "q_omega": trial.suggest_float("q_omega", 1e-12, 1e3, log=True),
        "r_meas": trial.suggest_float("r_meas", 1e-8, 1e4, log=True),
        "output_smoothing": trial.suggest_float("output_smoothing", 1e-4, 0.5, log=True),
    },
    "RA-EKF": lambda trial: {
        # Con 300 trials, podemos exprimir la matriz de covarianza completa
        "q_theta": trial.suggest_float("q_theta", 1e-12, 1e-1, log=True),
        "q_omega": trial.suggest_float("q_omega", 1e-12, 1e1, log=True),
        "q_A": trial.suggest_float("q_A", 1e-12, 1e-1, log=True),
        "q_rocof": trial.suggest_float("q_rocof", 1e-10, 1e2, log=True),
        "r_meas": trial.suggest_float("r_meas", 1e-8, 1e3, log=True),
        "sigma_v": trial.suggest_float("sigma_v", 1e-4, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.5, 100.0, log=True),
        "deriv_lpf_alpha": trial.suggest_float("deriv_lpf_alpha", 0.001, 0.9),
        "tau_rocof": trial.suggest_float("tau_rocof", 0.005, 2.0, log=True),
    },

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # Window-based
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # Permitimos ventanas sub-ciclo (0.5) para latencia extrema, hasta 10 ciclos para robustez masiva.
    "IPDFT": lambda trial: {
        "cycles": trial.suggest_float("cycles", 0.5, 10.0),
    },
    "TFT": lambda trial: {
        "n_cycles": trial.suggest_float("n_cycles", 0.5, 10.0),
    },
    "Prony": lambda trial: {
        # Ã“rdenes altos permiten modelar ruido/armÃ³nicos como polos matemÃ¡ticos
        "order": trial.suggest_int("order", 2, 14),
        "n_cycles": trial.suggest_float("n_cycles", 0.5, 8.0),
    },
    "ESPRIT": lambda trial: {
        "n_cycles": trial.suggest_float("n_cycles", 0.5, 8.0),
    },

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # Adaptive & Data-driven
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    "RLS": lambda trial: {
        # Factores de olvido agresivos (rÃ¡pido) vs casi 1.0 (lento pero estable)
        "alpha_vff": trial.suggest_float("alpha_vff", 1e-4, 0.99, log=True),
        "lambda_min": trial.suggest_float("lambda_min", 0.50, 0.9999),
    },
    "TKEO": lambda trial: {
        "output_smoothing": trial.suggest_float("output_smoothing", 1e-6, 0.5, log=True),
    },
    "Koopman (RK-DPMU)": lambda trial: {
        "n_cycles": trial.suggest_float("n_cycles", 0.5, 10.0),
    },
    "PI-GRU": lambda trial: {
        # Fixed pretrained model. No Optuna hyperparameters in the benchmark.
    },
}


# Override trials for computationally heavy estimators
N_TRIALS_OVERRIDES: dict[str, int] = {
    "Prony": _env_int("BENCHMARK_N_TRIALS_OVERRIDE_PRONY", 3, minimum=0),
    "ESPRIT": _env_int("BENCHMARK_N_TRIALS_OVERRIDE_ESPRIT", 3, minimum=0),
    "Koopman (RK-DPMU)": _env_int("BENCHMARK_N_TRIALS_OVERRIDE_KOOPMAN", 10, minimum=0),
}


def _effective_n_trials(est_name: str) -> int:
    if APPLY_TRIAL_OVERRIDES and est_name in N_TRIALS_OVERRIDES:
        return int(N_TRIALS_OVERRIDES[est_name])
    return int(N_TRIALS_TUNING)


class _SpaceRecorder:
    def __init__(self) -> None:
        self.specs: list[dict[str, Any]] = []

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: float | None = None,
        log: bool = False,
    ) -> float:
        self.specs.append(
            {
                "kind": "float",
                "name": name,
                "low": float(low),
                "high": float(high),
                "step": None if step is None else float(step),
                "log": bool(log),
            }
        )
        return float(low)

    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        *,
        step: int = 1,
        log: bool = False,
    ) -> int:
        self.specs.append(
            {
                "kind": "int",
                "name": name,
                "low": int(low),
                "high": int(high),
                "step": int(step),
                "log": bool(log),
            }
        )
        return int(low)

    def suggest_categorical(self, name: str, choices: list[Any] | tuple[Any, ...]) -> Any:
        vals = list(choices)
        self.specs.append({"kind": "categorical", "name": name, "choices": vals})
        return vals[0] if vals else None


def _levels_per_dimension(n_trials: int, n_dimensions: int) -> int:
    if n_dimensions <= 0:
        return 1
    return max(2, int(round(max(1, n_trials) ** (1.0 / n_dimensions))))


def _build_float_grid(low: float, high: float, n: int, log: bool) -> list[float]:
    if n <= 1:
        return [float(low)]
    if log and low > 0.0 and high > 0.0:
        vals = np.geomspace(low, high, num=n)
    else:
        vals = np.linspace(low, high, num=n)
    return [float(v) for v in vals]


def _build_int_grid(low: int, high: int, n: int, step: int = 1, log: bool = False) -> list[int]:
    if n <= 1:
        return [int(low)]
    if log and low > 0 and high > 0:
        raw = np.geomspace(low, high, num=n)
    else:
        raw = np.linspace(low, high, num=n)
    vals = []
    for v in raw:
        q = int(round(float(v) / step) * step)
        q = min(high, max(low, q))
        vals.append(q)
    vals.extend([int(low), int(high)])
    return sorted(set(vals))


def _grid_space_for_estimator(space_fn: Any, n_trials: int) -> dict[str, list[Any]]:
    recorder = _SpaceRecorder()
    space_fn(recorder)
    specs = recorder.specs
    if not specs:
        return {}

    n_levels = _levels_per_dimension(n_trials=n_trials, n_dimensions=len(specs))
    grid_space: dict[str, list[Any]] = {}
    for spec in specs:
        name = str(spec["name"])
        kind = str(spec["kind"])
        if kind == "float":
            grid_space[name] = _build_float_grid(
                low=float(spec["low"]),
                high=float(spec["high"]),
                n=n_levels,
                log=bool(spec["log"]),
            )
        elif kind == "int":
            grid_space[name] = _build_int_grid(
                low=int(spec["low"]),
                high=int(spec["high"]),
                n=n_levels,
                step=max(1, int(spec.get("step", 1))),
                log=bool(spec.get("log", False)),
            )
        elif kind == "categorical":
            grid_space[name] = list(spec["choices"])
    return grid_space


def _grid_cardinality(grid_space: dict[str, list[Any]]) -> int:
    total = 1
    for values in grid_space.values():
        total *= max(1, len(values))
    return int(total)


def _build_optuna_study(space_fn: Any, n_trials: int) -> tuple[optuna.Study, int, str]:
    mode = OPTUNA_SAMPLER_MODE

    if mode == "grid":
        grid_space = _grid_space_for_estimator(space_fn, n_trials)
        if grid_space:
            sampler = optuna.samplers.GridSampler(search_space=grid_space, seed=OPTUNA_SEED)
            budget = min(max(1, n_trials), _grid_cardinality(grid_space))
            return optuna.create_study(direction="minimize", sampler=sampler), budget, "grid"
        print("[WARN] Empty grid search space; falling back to TPE sampler.")
        mode = "tpe"

    if mode == "random":
        sampler = optuna.samplers.RandomSampler(seed=OPTUNA_SEED)
        return (
            optuna.create_study(direction="minimize", sampler=sampler),
            max(1, n_trials),
            "random",
        )

    sampler = optuna.samplers.TPESampler(seed=OPTUNA_SEED)
    return (
        optuna.create_study(direction="minimize", sampler=sampler),
        max(1, n_trials),
        "tpe",
    )

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Estimator registry
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def load_estimators() -> dict[str, type]:
    """Load the canonical active estimator registry for the modular benchmark."""
    estimators = load_active_estimators()
    if not EXCLUDED_ESTIMATOR_LABELS:
        return estimators

    by_lower = {label.lower(): label for label in estimators}
    excluded_labels: set[str] = set()
    unknown: list[str] = []

    for raw in EXCLUDED_ESTIMATOR_LABELS:
        hit = by_lower.get(raw.lower())
        if hit is None:
            unknown.append(raw)
            continue
        excluded_labels.add(hit)

    if unknown:
        print(
            "[WARN] BENCHMARK_EXCLUDE_ESTIMATORS contains unknown labels: "
            f"{unknown}. Known labels: {sorted(estimators.keys())}"
        )

    filtered = {label: cls for label, cls in estimators.items() if label not in excluded_labels}
    if not filtered:
        raise ValueError("Estimator filter removed all estimators. Check BENCHMARK_EXCLUDE_ESTIMATORS.")

    if excluded_labels:
        print(f"  Excluding estimators via BENCHMARK_EXCLUDE_ESTIMATORS: {sorted(excluded_labels)}")
    return filtered


def _print_registry_summary(estimators: dict[str, type]) -> None:
    excluded = excluded_estimator_specs()
    print(f"  Active: {list(estimators)}")
    if not excluded:
        return
    print("  Explicitly excluded:")
    for spec in excluded:
        print(f"    - {spec.label}: {spec.reason}")


def validate_search_spaces(estimators: dict[str, type]) -> None:
    """
    Raise ValueError if any SEARCH_SPACES key suggests a param that does not
    exist in the estimator's __init__ signature.
    """

    class _FakeTrial:
        def __init__(self) -> None:
            self.suggested: set[str] = set()

        def suggest_float(self, name: str, *args: Any, **kwargs: Any) -> float:
            self.suggested.add(name)
            return 0.5

        def suggest_int(self, name: str, *args: Any, **kwargs: Any) -> int:
            self.suggested.add(name)
            return 1

    errors: list[str] = []
    for est_name, space_fn in SEARCH_SPACES.items():
        cls = estimators.get(est_name)
        if cls is None:
            continue
        fake = _FakeTrial()
        suggested_params = set(space_fn(fake).keys())
        valid_params = set(inspect.signature(cls.__init__).parameters) - {"self"}
        unknown = suggested_params - valid_params
        if unknown:
            errors.append(
                f"SEARCH_SPACES['{est_name}'] suggests unknown params: {unknown}. "
                f"Valid params: {valid_params}"
            )

    if errors:
        raise ValueError(
            "Search space validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Noise-profile compatibility helper
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _noise_kwargs(sc_cls: type, level: float) -> dict[str, Any]:
    """
    Return noise override kwargs compatible with both old-style scenarios
    and new multi-spectral scenarios.
    """
    params = sc_cls.DEFAULT_PARAMS
    if "noise_sigma" in params:
        return {"noise_sigma": level}
    overrides: dict[str, Any] = {}
    if "white_noise_sigma" in params:
        overrides["white_noise_sigma"] = level
    if "brown_noise_sigma" in params:
        overrides["brown_noise_sigma"] = 0.0
    if "impulse_prob" in params:
        overrides["impulse_prob"] = 0.0
    return overrides


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Tuning
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Tuning
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def tune_estimator(
    est_name: str,
    est_cls: type,
    scenario_cls: type,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Run Optuna search for one estimator on one scenario.
    Returns (best_params, tuning_metadata).
    """
    defaults: dict[str, Any] = (
        est_cls.default_params() if hasattr(est_cls, "default_params") else {}
    )

    tuning_meta: dict[str, Any] = {
        "sampler_mode_requested": OPTUNA_SAMPLER_MODE,
        "sampler_mode_effective": None,
        "n_trials_requested": None,
        "n_trials_executed": 0,
        "n_trials_override_applied": bool(APPLY_TRIAL_OVERRIDES and est_name in N_TRIALS_OVERRIDES),
    }

    if est_name not in SEARCH_SPACES:
        tuning_meta["reason"] = "no_search_space"
        return defaults, tuning_meta

    n_trials_requested = _effective_n_trials(est_name)
    tuning_meta["n_trials_requested"] = int(n_trials_requested)

    if n_trials_requested <= 0:
        print(f"      [!] Skipping tuning for {est_name} - using defaults.")
        tuning_meta["reason"] = "n_trials_requested<=0"
        return defaults, tuning_meta

    space_fn = SEARCH_SPACES[est_name]
    if not _grid_space_for_estimator(space_fn, n_trials=2):
        tuning_meta["reason"] = "empty_search_space"
        tuning_meta["sampler_mode_effective"] = "none"
        tuning_meta["n_trials_executed"] = 0
        return defaults, tuning_meta

    sc = scenario_cls.run(duration_s=2.0, seed=42, **_noise_kwargs(scenario_cls, 0.001))
    fs_dsp = 1.0 / (sc.t[1] - sc.t[0])
    eval_start = int(0.100 * fs_dsp)

    def objective(trial: optuna.Trial) -> float:
        suggested = space_fn(trial)
        params = {**defaults, **suggested}
        try:
            est = est_cls(**params)
            f_hat = _run_estimator(est, sc.v)

            error = f_hat[eval_start:] - sc.f_true[eval_start:]
            rmse = float(np.sqrt(np.mean(error ** 2)))
            max_peak = float(np.max(np.abs(error)))
            tail_idx = int(0.9 * len(error))
            steady_tail_error = float(np.mean(np.abs(error[tail_idx:])))

            tol_rmse = 0.05
            tol_peak = 0.50
            tol_tail = 0.02

            norm_rmse = rmse / tol_rmse
            norm_peak = max_peak / tol_peak
            norm_tail = steady_tail_error / tol_tail

            w_rmse, w_peak, w_tail = 0.50, 0.30, 0.20
            composite_loss = (w_rmse * norm_rmse) + (w_peak * norm_peak) + (w_tail * norm_tail)
            return composite_loss if np.isfinite(composite_loss) else 1e6
        except Exception:
            return 1e6

    study, n_trials_exec, sampler_mode_effective = _build_optuna_study(
        space_fn=space_fn,
        n_trials=n_trials_requested,
    )
    tuning_meta["sampler_mode_effective"] = sampler_mode_effective
    tuning_meta["n_trials_executed"] = int(n_trials_exec)

    if defaults:
        try:
            dummy_trial = optuna.trial.FixedTrial({k: 0.5 for k in defaults.keys()})
            dummy_suggested = space_fn(dummy_trial)
            seed_params = {k: v for k, v in defaults.items() if k in dummy_suggested}
            if seed_params:
                study.enqueue_trial(seed_params, skip_if_exists=True)
        except Exception:
            pass

    study.optimize(objective, n_trials=n_trials_exec)

    if study.best_value >= 1e6:
        print(f"      [!] All trials failed for {est_name} - using defaults.")
        tuning_meta["reason"] = "all_trials_failed"
        return defaults, tuning_meta

    best_suggested = space_fn(study.best_trial)
    return {**defaults, **best_suggested}, tuning_meta


def _safe_float(value: Any) -> float | None:
    try:
        v = float(value)
        if np.isfinite(v):
            return v
        return None
    except Exception:
        return None


def _to_builtin(obj: Any) -> Any:
    """Convert numpy/pandas/path objects to JSON-safe builtin Python types."""
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, np.ndarray):
        return [_to_builtin(v) for v in obj.tolist()]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if pd.isna(obj):
        return None
    return obj


def _run_estimator(est: Any, v: np.ndarray) -> np.ndarray:
    """Call step_vectorized if available, otherwise fall back to per-sample step."""
    if hasattr(est, "step_vectorized"):
        return np.asarray(est.step_vectorized(v), dtype=float)
    return np.array([est.step(float(sample)) for sample in v], dtype=float)


def _save_scenario_artifacts(sc: Any, sc_dir: Path, sc_name: str) -> None:
    """Write scenario CSV and a voltage + frequency overview plot."""
    df = pd.DataFrame({
        "t_s": sc.t,
        "v_pu": sc.v,
        "f_true_hz": sc.f_true,
    })
    df.to_csv(sc_dir / f"{sc_name}_scenario.csv", index=False)

    fig, (ax_v, ax_f) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax_v.plot(sc.t, sc.v, color="navy", linewidth=0.7)
    ax_v.set_ylabel("Voltage [pu]")
    ax_v.grid(True, alpha=0.3)
    ax_f.plot(sc.t, sc.f_true, color="darkred", linewidth=1.5)
    ax_f.set_ylabel("Frequency [Hz]")
    ax_f.set_xlabel("Time [s]")
    ax_f.grid(True, alpha=0.3)
    fig.suptitle(sc_name, fontweight="bold")
    fig.tight_layout()
    fig.savefig(sc_dir / f"{sc_name}_plot.png", dpi=150)
    plt.close(fig)


def _save_tracking_plot(
    est_cls: type,
    params: dict[str, Any],
    sc: Any,
    out_dir: Path,
    est_name: str,
    sc_name: str,
) -> None:
    """Plot estimator frequency tracking against ground truth."""
    try:
        est = est_cls(**params)
        f_hat = _run_estimator(est, sc.v)

        margin = max(5.0, (sc.f_true.max() - sc.f_true.min()) * 0.15)
        with plt.rc_context(_IEEE_RC):
            fig, ax = plt.subplots(figsize=(3.5, 2.2))
            ax.plot(sc.t, sc.f_true, "k--", alpha=0.6, linewidth=0.8, label="True")
            ax.plot(sc.t, f_hat, color="#C62828", linewidth=0.7, label=f"{est_name}")
            ax.set_ylim(sc.f_true.min() - margin, sc.f_true.max() + margin)
            ax.set_title(f"{est_name} | {sc_name}", pad=2)
            ax.set_xlabel("Time [s]", labelpad=1)
            ax.set_ylabel("Frequency [Hz]", labelpad=1)
            ax.legend(loc="upper right", handlelength=1.0)
            fig.tight_layout(pad=0.4)
            fig.savefig(out_dir / f"tracking_{est_name}.png", dpi=150)
            plt.close(fig)
    except Exception as exc:
        print(f"      [!] Tracking plot failed for {est_name}: {exc}")


def _save_scenario_zoom_plot(sc: Any, sc_dir: Path, sc_name: str) -> None:
    """
    Save a publication-quality zoom plot focused on the key event region.
    """
    p = sc.meta.get("parameters", {})
    t_total = float(sc.t[-1])

    event_t: float | None = None
    for key in ("t_step_s", "t_event_s", "t_ramp_s"):
        if key in p and p[key] is not None:
            event_t = float(p[key])
            break
    if event_t is None and "t_start_s" in p:
        event_t = float(p["t_start_s"])

    if event_t is not None:
        z_lo = max(0.0, event_t - 0.08)
        z_hi = min(t_total, event_t + 0.35)
    else:
        t_mid = t_total / 2.0
        z_lo = max(0.0, t_mid - 0.20)
        z_hi = min(t_total, t_mid + 0.20)

    mask = (sc.t >= z_lo) & (sc.t <= z_hi)
    t_z, v_z, f_z = sc.t[mask], sc.v[mask], sc.f_true[mask]
    if len(t_z) < 10:
        return

    f_nom = float(p.get("freq_nom_hz", p.get("freq_hz", 60.0)))

    with plt.rc_context(_IEEE_RC):
        fig, (ax_v, ax_f) = plt.subplots(2, 1, figsize=(3.5, 3.8), sharex=True)
        fig.subplots_adjust(hspace=0.08, left=0.15, right=0.97, top=0.89, bottom=0.12)

        ax_v.plot(t_z, v_z, lw=0.65, color="#1A237E", rasterized=True)
        if event_t is not None:
            ax_v.axvline(
                event_t,
                color="black",
                lw=0.8,
                ls="--",
                alpha=0.65,
                label="Event onset",
            )
            ax_v.legend(loc="upper right", fontsize=5.5, handlelength=0.9)
        ax_v.set_ylabel(r"$v(t)$ [pu]", labelpad=2)
        v_rng = max(float(np.ptp(v_z)), 0.05)
        ax_v.set_ylim(
            float(v_z.min()) - 0.10 * v_rng,
            float(v_z.max()) + 0.22 * v_rng,
        )
        ax_v.set_title(f"{sc_name} â€” Zoom Detail", fontweight="bold", pad=3)

        ax_f.plot(
            t_z,
            f_z,
            lw=1.0,
            color="#B71C1C",
            rasterized=True,
            label=r"$f_\mathrm{true}(t)$",
        )
        ax_f.axhline(
            f_nom,
            color="#444",
            lw=0.55,
            ls=":",
            alpha=0.7,
            label=f"$f_0={f_nom:.0f}$ Hz",
        )
        if event_t is not None:
            ax_f.axvline(event_t, color="black", lw=0.8, ls="--", alpha=0.65)
        ax_f.set_xlabel("Time [s]", labelpad=1)
        ax_f.set_ylabel(r"$f(t)$ [Hz]", labelpad=2)
        f_rng = max(float(np.ptp(f_z)), 0.02)
        ax_f.set_ylim(
            float(f_z.min()) - 0.18 * f_rng,
            float(f_z.max()) + 0.28 * f_rng,
        )
        ax_f.set_xlim(z_lo, z_hi)
        ax_f.legend(
            loc="upper right",
            fontsize=5.5,
            ncol=2,
            handlelength=1.0,
            borderpad=0.3,
        )

        out = sc_dir / f"{sc_name}_zoom.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
    print(f"      Zoom plot â†’ {out.relative_to(ROOT)}")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Phase 1 â€” Tuning + Monte Carlo + per-estimator artifacts
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Phase 1 â€” Tuning + Monte Carlo + per-estimator artifacts
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def run_phase_1(estimators: dict[str, type]) -> None:
    """
    For every (scenario, estimator) pair:
      1. Tune hyperparameters with Optuna.
      2. Save a tracking plot.
      3. Run N_MC_RUNS Monte Carlo iterations and save summary + signals CSVs.
      4. Save a per-run spec JSON for later large-report export.
    """
    print("\n>>> PHASE 1: TUNING + MC + ARTIFACTS <<<")

    for sc_cls in SCENARIOS:
        sc_name = sc_cls.get_name()
        sc_dir = BASE_RESULTS_DIR / sc_name
        sc_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  Scenario: {sc_name}")

        # ---> CORRECCIÃ“N: Alineado con la optimizaciÃ³n de Optuna a 2.0s <---
        sc_base = sc_cls.run(duration_s=2.0, seed=42, **_noise_kwargs(sc_cls, 0.0))
        _save_scenario_artifacts(sc_base, sc_dir, sc_name)
        _save_scenario_zoom_plot(sc_base, sc_dir, sc_name)

        for est_name, est_cls in estimators.items():
            print(f"    [{est_name}] tuning ...", end=" ", flush=True)
            out_dir = sc_dir / est_name
            out_dir.mkdir(parents=True, exist_ok=True)

            best_params, tuning_meta = tune_estimator(est_name, est_cls, sc_cls)
            print("MC ...", end=" ", flush=True)

            _save_tracking_plot(est_cls, best_params, sc_base, out_dir, est_name, sc_name)

            # Si tu MonteCarloEngine no tiene kwargs para pasar duration_s,
            # asegÃºrate de que internamente construya el escenario respetando
            # la lÃ³gica o que los defaults de tus clases no fuercen los 5s.
            engine = MonteCarloEngine(
                scenario_cls=sc_cls,
                estimator_cls=est_cls,
                estimator_params=best_params,
                n_runs=N_MC_RUNS,
            )
            result = engine.run()
            engine.save_csv(result, out_dir)

            summary_files = [str(p.relative_to(ROOT)) for p in sorted(out_dir.glob("*_summary.csv"))]
            signal_files = [str(p.relative_to(ROOT)) for p in sorted(out_dir.glob("*_signals.csv"))]
            tracking_files = [str(p.relative_to(ROOT)) for p in sorted(out_dir.glob("tracking_*.png"))]

            run_spec = {
                "scenario": sc_name,
                "scenario_class": sc_cls.__name__,
                "scenario_meta": _to_builtin(sc_base.meta),
                "estimator": est_name,
                "estimator_class": est_cls.__name__,
                "family": _ESTIMATOR_FAMILIES.get(est_name, "Unknown"),
                "best_params": _to_builtin(best_params),
                "defaults": _to_builtin(
                    est_cls.default_params() if hasattr(est_cls, "default_params") else {}
                ),
                "tuning_meta": _to_builtin(tuning_meta),
                "n_trials_tuning": tuning_meta.get("n_trials_executed", _effective_n_trials(est_name)),
                "n_trials_tuning_requested": tuning_meta.get("n_trials_requested", _effective_n_trials(est_name)),
                "tuning_sampler_requested": tuning_meta.get("sampler_mode_requested", OPTUNA_SAMPLER_MODE),
                "tuning_sampler_effective": tuning_meta.get("sampler_mode_effective", OPTUNA_SAMPLER_MODE),
                "tuning_trials_override_applied": tuning_meta.get(
                    "n_trials_override_applied",
                    bool(APPLY_TRIAL_OVERRIDES and est_name in N_TRIALS_OVERRIDES),
                ),
                "n_mc_runs": N_MC_RUNS,
                "artifacts": {
                    "summary_csv": summary_files,
                    "signals_csv": signal_files,
                    "tracking_plots": tracking_files,
                    "scenario_csv": str((sc_dir / f"{sc_name}_scenario.csv").relative_to(ROOT)),
                    "scenario_zoom_plot": str((sc_dir / f"{sc_name}_zoom.png").relative_to(ROOT)),
                },
            }

            (out_dir / "run_spec.json").write_text(
                json.dumps(_to_builtin(run_spec), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            print("done")
            gc.collect()

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Phase 2 â€” Metric aggregation
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def run_phase_2(allowed_estimators: set[str] | None = None) -> None:
    """
    Read every *_summary.csv written by Phase 1, compute mean Â± std across
    MC runs, and write per-scenario and global summary CSVs.
    """
    print("\n>>> PHASE 2: METRIC AGGREGATION <<<")
    all_stats: list[dict[str, Any]] = []

    for sc_dir in sorted(BASE_RESULTS_DIR.iterdir()):
        if not sc_dir.is_dir():
            continue
        sc_name = sc_dir.name
        sc_stats: list[dict[str, Any]] = []

        for est_dir in sorted(sc_dir.iterdir()):
            if not est_dir.is_dir():
                continue
            est_name = est_dir.name
            if allowed_estimators is not None and est_name not in allowed_estimators:
                continue
            summary_file = next(est_dir.glob("*_summary.csv"), None)
            if summary_file is None:
                print(f"    [?] No summary CSV in {est_dir.relative_to(ROOT)}")
                continue

            df = pd.read_csv(summary_file)
            available = [c for c in METRIC_COLUMNS if c in df.columns]
            row: dict[str, Any] = {
                "scenario": sc_name,
                "estimator": est_name,
                "family": _ESTIMATOR_FAMILIES.get(est_name, "Unknown"),
            }
            for col in available:
                row[f"{col}_mean"] = float(df[col].mean())
                row[f"{col}_std"] = float(df[col].std())
            sc_stats.append(row)
            all_stats.append(row)

        if sc_stats:
            pd.DataFrame(sc_stats).to_csv(sc_dir / "summary_stats.csv", index=False)
            print(f"    {sc_name}: {len(sc_stats)} estimators aggregated")

    if all_stats:
        global_path = BASE_RESULTS_DIR / "global_metrics_report.csv"
        pd.DataFrame(all_stats).to_csv(global_path, index=False)
        print(f"  Global report â†’ {global_path.relative_to(ROOT)}")
    else:
        print("  [!] No statistics collected â€” run Phase 1 first.")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Phase 3 â€” Dashboards
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def run_phase_3() -> None:
    """
    Generate all dashboards:
      â€¢ Per-scenario 6-metric bar dashboards
      â€¢ Global accuracyâ€“latency scatter
      â€¢ Mega Dashboard 1 â€” Scenario signal overview
      â€¢ Mega Dashboard 2 â€” Estimator performance summary
    """
    print("\n>>> PHASE 3: DASHBOARDS <<<")
    from plotting.benchmark.generate_mega_dashboards import generate_benchmark_figures

    report_path = BASE_RESULTS_DIR / "global_metrics_report.csv"
    if not report_path.exists():
        print("  [!] global_metrics_report.csv missing â€” skipping MC-based dashboards.")
        return

    df_global = pd.read_csv(report_path)
    _plot_per_scenario_dashboards(df_global)
    _plot_global_tradeoff(df_global)
    generate_benchmark_figures(BASE_RESULTS_DIR)


def _plot_per_scenario_dashboards(df_global: pd.DataFrame) -> None:
    """
    Per-scenario performance dashboard with 6 metrics.
    """
    panel_metrics = [
        ("m1_rmse_hz_mean", r"RMSE [Hz]", "log"),
        ("m2_mae_hz_mean", r"MAE [Hz]", "log"),
        ("m3_max_peak_hz_mean", r"Max Peak [Hz]", "log"),
        ("m5_trip_risk_s_mean", r"Trip Risk $T$ [s]", "log"),
        ("m8_settling_time_s_mean", r"Settling Time [s]", "linear"),
        ("m13_cpu_time_us_mean", r"CPU [$\mu$s/sample]", "linear"),
    ]

    for sc_name in df_global["scenario"].unique():
        df_sc = (
            df_global[df_global["scenario"] == sc_name]
            .sort_values("m1_rmse_hz_mean")
            .reset_index(drop=True)
        )
        sc_dir = BASE_RESULTS_DIR / sc_name
        available = [
            (col, title, scale)
            for col, title, scale in panel_metrics
            if col in df_sc.columns
        ]
        if not available:
            continue

        n_panels = len(available)
        ncols = 2
        nrows = (n_panels + 1) // ncols
        x = np.arange(len(df_sc))
        clrs = [
            _FAMILY_PALETTE.get(_ESTIMATOR_FAMILIES.get(e, "Adaptive"), "#757575")
            for e in df_sc["estimator"]
        ]

        with plt.rc_context(_IEEE_RC):
            fig, axes = plt.subplots(nrows, ncols, figsize=(_IEEE_FULL_W, 2.8 * nrows))
            axes_flat = np.array(axes).flatten()

            for idx, (col, title, scale) in enumerate(available):
                ax = axes_flat[idx]
                std_col = col.replace("_mean", "_std")
                yerr = df_sc[std_col].values if std_col in df_sc.columns else None
                ax.bar(x, df_sc[col], color=clrs, edgecolor="none", alpha=0.87, zorder=3)
                if yerr is not None:
                    ax.errorbar(
                        x,
                        df_sc[col],
                        yerr=yerr,
                        fmt="none",
                        ecolor="#333",
                        elinewidth=0.6,
                        capsize=2,
                        zorder=4,
                    )
                ax.set_title(title, fontsize=7, pad=2)
                ax.set_yscale(scale)
                ax.set_xticks(x)
                ax.set_xticklabels(df_sc["estimator"], rotation=40, ha="right", fontsize=5.5)

                best_idx = int(df_sc[col].idxmin())
                if best_idx < len(x):
                    axes_flat[idx].patches[best_idx].set_edgecolor("#000")
                    axes_flat[idx].patches[best_idx].set_linewidth(0.9)

            for idx in range(len(available), len(axes_flat)):
                axes_flat[idx].set_visible(False)

            legend_patches = [
                plt.Rectangle((0, 0), 1, 1, fc=clr, label=fam)
                for fam, clr in _FAMILY_PALETTE.items()
            ]
            fig.legend(
                handles=legend_patches,
                loc="lower center",
                ncol=len(_FAMILY_PALETTE),
                fontsize=5.5,
                frameon=True,
                handlelength=0.9,
                bbox_to_anchor=(0.5, 0.0),
            )
            fig.suptitle(
                f"Performance Dashboard â€” {sc_name}\n"
                r"(sorted by RMSE; error bars = $\sigma$ across MC runs)",
                fontsize=7.5,
                fontweight="bold",
                y=1.01,
            )
            fig.tight_layout(rect=[0, 0.04, 1, 1.0])
            out = sc_dir / "dashboard.png"
            fig.savefig(out, dpi=200)
            plt.close(fig)
        print(f"    Dashboard â†’ {out.relative_to(ROOT)}")


def _plot_global_tradeoff(df_global: pd.DataFrame) -> None:
    if "m13_cpu_time_us_mean" not in df_global or "m1_rmse_hz_mean" not in df_global:
        return

    df_avg = (
        df_global.groupby("estimator")[["m13_cpu_time_us_mean", "m1_rmse_hz_mean"]]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    for _, row in df_avg.iterrows():
        ax.scatter(
            row["m13_cpu_time_us_mean"],
            row["m1_rmse_hz_mean"],
            s=280,
            edgecolors="black",
            alpha=0.75,
            zorder=3,
        )
        ax.text(
            row["m13_cpu_time_us_mean"] * 1.08,
            row["m1_rmse_hz_mean"],
            row["estimator"],
            fontsize=8,
            fontweight="bold",
        )

    ax.axhline(0.1, color="red", linestyle="--", alpha=0.6, label="100 mHz threshold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Avg CPU Time / sample [Âµs]")
    ax.set_ylabel("Avg RMSE [Hz]")
    ax.set_title("Global Trade-off: Latency vs Accuracy")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    out_path = BASE_RESULTS_DIR / "dashboard_global_tradeoff.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    Global trade-off plot â†’ {out_path.relative_to(ROOT)}")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Mega Dashboard 1 â€” Scenario Signal Overview
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _plot_megadashboard1() -> None:
    import matplotlib.gridspec as gridspec

    out_dir = BASE_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n  [MEGA1] Building scenario signal overview mega-dashboard ...")

    ROWS = [
        dict(
            cls=IEEESingleSinWaveScenario,
            kw=dict(seed=42),
            label="(a) Clean Sinusoid",
            sub=r"$f_0=60\,\mathrm{Hz}$, $A=1\,\mathrm{pu}$ â€” noise-free baseline",
            vc="#1B5E20",
            fc="#1B5E20",
            ev=None,
            regs=[],
        ),
        dict(
            cls=IEEEMagStepScenario,
            kw=dict(noise_sigma=0.001, seed=42),
            label="(b) Amplitude Step (IEC 60255-118-1 Scen. A)",
            sub=r"$\Delta A=+10\,\%$ step at $t=0.5\,\mathrm{s}$; $\sigma_n=10^{-3}\,\mathrm{pu}$",
            vc="#0D47A1",
            fc="#0D47A1",
            ev=0.5,
            regs=[(0.0, 0.5, "Pre-step", "#43A047", 0.06), (0.5, 1.5, "Post-step", "#FB8C00", 0.06)],
        ),
        dict(
            cls=IEEEFreqRampScenario,
            kw=dict(rocof_hz_s=3.0, t_start_s=0.3, freq_cap_hz=61.5, noise_sigma=0.001, seed=42),
            label="(c) Frequency Ramp (IEEE 1547-2018 Cat. III)",
            sub=r"RoCoF $=+3\,\mathrm{Hz/s}$, onset $t=0.3\,\mathrm{s}$, cap at $61.5\,\mathrm{Hz}$",
            vc="#E65100",
            fc="#BF360C",
            ev=0.3,
            regs=[(0.0, 0.3, "Nominal", "#43A047", 0.06), (0.3, 0.8, "RoCoF ramp", "#FB8C00", 0.06), (0.8, 1.5, "Cap hold", "#C62828", 0.06)],
        ),
        dict(
            cls=IEEEModulationScenario,
            kw=dict(kx=0.1, ka=0.1, fm_hz=2.0, noise_sigma=0.001, seed=42),
            label="(d) AM/PM Modulation (IEC 60255-118-1 Bandwidth Test)",
            sub=r"$k_x=10\,\%$, $k_a=0.1\,\mathrm{rad}$, $f_m=2\,\mathrm{Hz}$; $f_\mathrm{true}(t)=f_0 - k_a f_m \sin(2\pi f_m t)$",
            vc="#6A1B9A",
            fc="#6A1B9A",
            ev=None,
            regs=[],
        ),
        dict(
            cls=IBRHarmonicsMediumScenario,
            kw=dict(seed=42),
            label="(e) IBR Harmonic Distortion (THD â‰ˆ 8 %)",
            sub=r"3rdâ€“13th harmonics + 75\,Hz inter-harmonic; RoCoF $=-1\,\mathrm{Hz/s}$ at $t=0.8\,\mathrm{s}$",
            vc="#37474F",
            fc="#B71C1C",
            ev=0.8,
            regs=[(0.0, 0.8, "Nominal + harmonics", "#43A047", 0.05), (0.8, 2.0, "RoCoF + harmonics", "#FB8C00", 0.05)],
        ),
        dict(
            cls=IBRMultiEventScenario,
            kw=dict(seed=42),
            label="(f) IBR Multi-Event Composite Fault (worst-case)",
            sub=r"Sag $50\,\%$ + $\Delta\phi=45Â°$ + RoCoF $-2\,\mathrm{Hz/s}$ + DC offset + 5th/7th harmonics + multi-spectral noise",
            vc="#263238",
            fc="#B71C1C",
            ev=0.5,
            regs=[(0.0, 0.5, "Pre-fault", "#43A047", 0.05), (0.5, 1.5, "RoCoF ramp", "#FB8C00", 0.05), (1.5, 5.0, "Nadir hold", "#C62828", 0.04)],
        ),
    ]

    nrows = len(ROWS)
    fig_h = _IEEE_PAGE_H * 0.70

    with plt.rc_context(_IEEE_RC):
        fig = plt.figure(figsize=(_IEEE_FULL_W, fig_h))
        gs = gridspec.GridSpec(
            nrows,
            2,
            figure=fig,
            hspace=0.52,
            wspace=0.13,
            left=0.075,
            right=0.980,
            top=0.940,
            bottom=0.055,
        )

        for ridx, spec in enumerate(ROWS):
            try:
                sc = spec["cls"].run(**spec["kw"])
            except Exception as exc:
                print(f"      [!] {spec['cls'].__name__} failed: {exc}")
                continue

            t, v, f = sc.t, sc.v, sc.f_true
            pars = sc.meta.get("parameters", {})
            f_nom = float(pars.get("freq_nom_hz", pars.get("freq_hz", 60.0)))

            ax_v = fig.add_subplot(gs[ridx, 0])
            ax_f = fig.add_subplot(gs[ridx, 1])

            ax_v.plot(t, v, lw=0.45, color=spec["vc"], rasterized=True)
            for r_lo, r_hi, _, rc, ra in spec["regs"]:
                ax_v.axvspan(r_lo, min(r_hi, float(t[-1])), alpha=ra, color=rc, lw=0)
            if spec["ev"] is not None:
                ax_v.axvline(spec["ev"], color="#000", lw=0.75, ls="--", alpha=0.55)

            ax_v.set_xlim(float(t[0]), float(t[-1]))
            v_pk = max(abs(float(v.min())), abs(float(v.max())))
            ax_v.set_ylim(-v_pk * 1.22, v_pk * 1.32)
            ax_v.set_ylabel(r"$v(t)$ [pu]", labelpad=1, fontsize=6.5)
            ax_v.set_title(spec["label"], fontsize=6.8, fontweight="bold", pad=2)
            ax_v.text(
                0.015,
                0.96,
                spec["sub"],
                transform=ax_v.transAxes,
                fontsize=5.0,
                va="top",
                style="italic",
                color="#444444",
            )

            ax_f.plot(t, f, lw=0.85, color=spec["fc"], rasterized=True, label=r"$f_\mathrm{true}(t)$")
            ax_f.axhline(f_nom, color="#555", lw=0.5, ls=":", alpha=0.7, label=f"$f_0={f_nom:.0f}$ Hz")

            for r_lo, r_hi, _, rc, ra in spec["regs"]:
                ax_f.axvspan(r_lo, min(r_hi, float(t[-1])), alpha=ra, color=rc, lw=0)
            if spec["ev"] is not None:
                ax_f.axvline(spec["ev"], color="#000", lw=0.75, ls="--", alpha=0.55)

            ax_f.set_xlim(float(t[0]), float(t[-1]))
            f_rng = max(float(np.ptp(f)), 0.04)
            ax_f.set_ylim(float(f.min()) - 0.22 * f_rng, float(f.max()) + 0.38 * f_rng)
            ax_f.set_ylabel(r"$f(t)$ [Hz]", labelpad=1, fontsize=6.5)
            ax_f.legend(loc="upper right", fontsize=5.5, ncol=2, handlelength=1.0, borderpad=0.3, labelspacing=0.2)

            if ridx == nrows - 1:
                ax_v.set_xlabel("Time [s]", labelpad=1)
                ax_f.set_xlabel("Time [s]", labelpad=1)
            else:
                ax_v.tick_params(labelbottom=False)
                ax_f.tick_params(labelbottom=False)

        fig.text(0.300, 0.974, "Voltage Waveform", ha="center", va="top", fontsize=7.5, fontweight="bold", color="#222")
        fig.text(0.730, 0.974, "Instantaneous Frequency", ha="center", va="top", fontsize=7.5, fontweight="bold", color="#222")
        fig.text(0.530, 0.993, "IBR Frequency Estimator Benchmark â€” Scenario Signal Overview", ha="center", va="top", fontsize=8, fontweight="bold", color="#111")

        out = out_dir / "megadashboard1.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
    print(f"    megadashboard1 saved â†’ {out.relative_to(ROOT)}")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Mega Dashboard 2 â€” Estimator Performance Summary
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _plot_megadashboard2(df_global: pd.DataFrame) -> None:
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LogNorm

    out_dir = BASE_RESULTS_DIR
    print("\n  [MEGA2] Building estimator performance mega-dashboard ...")

    def safe_pivot(col: str) -> pd.DataFrame | None:
        if col not in df_global.columns:
            return None
        piv = df_global.pivot_table(index="estimator", columns="scenario", values=col, aggfunc="mean")
        return piv if not piv.empty else None

    def fam_clr(est: str) -> str:
        return _FAMILY_PALETTE.get(_ESTIMATOR_FAMILIES.get(est, "Adaptive"), "#757575")

    def sc_short(name: str) -> str:
        return (
            name.replace("IEEE_Single_SinWave", "SinWave")
            .replace("IEEE_Mag_Step", "MagStep")
            .replace("IEEE_Freq_Ramp", "FreqRamp")
            .replace("IEEE_Freq_Step", "FreqStep")
            .replace("IEEE_Modulation", "Modul.")
            .replace("IEEE_OOB_Interference", "OOB")
            .replace("IEEE_Phase_Jump_20", "PJ20Â°")
            .replace("IEEE_Phase_Jump_60", "PJ60Â°")
            .replace("NERC_Phase_Jump_60", "NERC\nPJ60Â°")
            .replace("IBR_Multi_Event", "IBR\nMulti")
            .replace("IBR_Power_Imbalance_Ringdown", "IBR\nRing.")
            .replace("IBR_Harmonics_Small", "Harm\nSm")
            .replace("IBR_Harmonics_Medium", "Harm\nMed")
            .replace("IBR_Harmonics_Large", "Harm\nLg")
        )

    def avg_col(col: str) -> pd.Series:
        if col not in df_global.columns:
            return pd.Series(dtype=float)
        return df_global.groupby("estimator")[col].mean()

    piv_rmse = safe_pivot("m1_rmse_hz_mean")
    piv_trip = safe_pivot("m5_trip_risk_s_mean")
    piv_peak = safe_pivot("m3_max_peak_hz_mean")

    avg_rmse = avg_col("m1_rmse_hz_mean").sort_values()
    avg_cpu = avg_col("m13_cpu_time_us_mean")
    avg_settle = avg_col("m8_settling_time_s_mean").sort_values()

    def _draw_heatmap(ax: Any, piv: pd.DataFrame | None, cmap: str, label: str, annotate: bool = True) -> None:
        if piv is None or piv.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", fontsize=7, color="#888")
            return
        data = piv.values.astype(float)
        finite = data[np.isfinite(data) & (data > 0)]
        if len(finite) == 0:
            return
        vmin = max(finite.min() / 2.0, 1e-5)
        vmax = min(finite.max() * 2.0, 50.0)
        clipped = np.clip(data, vmin, vmax)
        im = ax.imshow(clipped, aspect="auto", cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
        xl = [sc_short(c) for c in piv.columns]
        ax.set_xticks(range(len(xl)))
        ax.set_xticklabels(xl, rotation=38, ha="right", fontsize=4.8)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index, fontsize=5.0)
        cb = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.88, aspect=22)
        cb.set_label(label, fontsize=5.5)
        cb.ax.tick_params(labelsize=4.5)
        if annotate:
            thresh = np.nanpercentile(finite, 72)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    val = data[i, j]
                    if np.isfinite(val):
                        tc = "w" if val >= thresh else "k"
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=3.8, color=tc, fontweight="bold")

    fig_h = _IEEE_PAGE_H * 0.70

    with plt.rc_context(_IEEE_RC):
        fig = plt.figure(figsize=(_IEEE_FULL_W, fig_h))
        gs = gridspec.GridSpec(
            4,
            2,
            figure=fig,
            hspace=0.78,
            wspace=0.24,
            left=0.08,
            right=0.98,
            top=0.930,
            bottom=0.055,
        )

        ax00 = fig.add_subplot(gs[0, 0])
        _draw_heatmap(ax00, piv_rmse, "RdYlGn_r", "RMSE [Hz]")
        ax00.set_title("RMSE [Hz] â€” Scenario Ã— Estimator", fontsize=7, fontweight="bold", pad=3)

        ax01 = fig.add_subplot(gs[0, 1])
        _draw_heatmap(ax01, piv_trip, "YlOrRd", r"$T_\mathrm{trip}$ [s]")
        ax01.set_title(r"Trip Risk $T_\mathrm{trip}$ [s]", fontsize=7, fontweight="bold", pad=3)

        ax10 = fig.add_subplot(gs[1, 0])
        if not avg_rmse.empty and not avg_cpu.empty:
            common = avg_rmse.index.intersection(avg_cpu.index)
            seen_fam: set[str] = set()
            for est in common:
                fam = _ESTIMATOR_FAMILIES.get(est, "Adaptive")
                clr = _FAMILY_PALETTE.get(fam, "#757575")
                lbl = fam if fam not in seen_fam else "_"
                seen_fam.add(fam)
                ax10.scatter(avg_cpu[est], avg_rmse[est], s=22, color=clr, edgecolors="white", lw=0.4, zorder=4, label=lbl)
                ax10.annotate(est, (avg_cpu[est], avg_rmse[est]), xytext=(3, 2), textcoords="offset points", fontsize=4.0, color=clr)
            ax10.axhline(0.1, color="#B71C1C", ls="--", lw=0.7, alpha=0.75, label="IEC 100 mHz")
            ax10.set_xscale("log")
            ax10.set_yscale("log")
            ax10.set_xlabel(r"Avg CPU [$\mu$s / sample]", labelpad=1)
            ax10.set_ylabel("Avg RMSE [Hz]", labelpad=1)
            ax10.legend(loc="upper left", fontsize=5.5, ncol=1, handlelength=0.9, borderpad=0.3)
        ax10.set_title("Accuracyâ€“Latency Trade-off (mean across all scenarios)", fontsize=7, fontweight="bold", pad=3)

        ax11 = fig.add_subplot(gs[1, 1])
        _draw_heatmap(ax11, piv_peak, "RdPu", "Max Peak [Hz]", annotate=False)
        ax11.set_title("Max Peak Error [Hz]", fontsize=7, fontweight="bold", pad=3)

        ax20 = fig.add_subplot(gs[2, 0])
        rmse_col = "m1_rmse_hz_mean"
        if rmse_col in df_global.columns:
            sc_list = sorted(df_global["scenario"].unique())
            est_list = df_global.groupby("estimator")[rmse_col].mean().sort_values().index.tolist()
            x20 = np.arange(len(sc_list))
            w = max(0.05, 0.72 / max(len(est_list), 1))
            cmap20 = plt.get_cmap("tab20", len(est_list))
            for i, est in enumerate(est_list):
                vals = []
                for sc in sc_list:
                    sub = df_global[(df_global["estimator"] == est) & (df_global["scenario"] == sc)]
                    vals.append(float(sub[rmse_col].mean()) if len(sub) else np.nan)
                offset = (i - len(est_list) / 2.0 + 0.5) * w
                ax20.bar(x20 + offset, vals, width=w * 0.88, color=cmap20(i), label=est, alpha=0.87, edgecolor="none")
            ax20.set_xticks(x20)
            ax20.set_xticklabels([sc_short(s) for s in sc_list], rotation=30, ha="right", fontsize=4.8)
            ax20.set_yscale("log")
            ax20.set_ylabel("RMSE [Hz]", labelpad=1)
            ax20.legend(loc="upper right", fontsize=3.8, ncol=2, handlelength=0.6, borderpad=0.2, labelspacing=0.12)
        ax20.set_title("RMSE by Estimator per Scenario", fontsize=7, fontweight="bold", pad=3)

        ax21 = fig.add_subplot(gs[2, 1])
        if not avg_settle.empty:
            clrs21 = [fam_clr(e) for e in avg_settle.index]
            ax21.barh(range(len(avg_settle)), avg_settle.values, color=clrs21, edgecolor="none", height=0.65)
            ax21.set_yticks(range(len(avg_settle)))
            ax21.set_yticklabels(avg_settle.index, fontsize=5.2)
            ax21.set_xlabel("Avg Settling Time [s]", labelpad=1)
            ax21.invert_yaxis()
        else:
            ax21.text(0.5, 0.5, "No data", transform=ax21.transAxes, ha="center", fontsize=7, color="#888")
        for fam, clr in _FAMILY_PALETTE.items():
            ax21.barh(0, 0, color=clr, label=fam, height=0)
        ax21.legend(loc="lower right", fontsize=5, ncol=1, handlelength=0.7, borderpad=0.3)
        ax21.set_title("Average Settling Time by Estimator", fontsize=7, fontweight="bold", pad=3)

        ax30 = fig.add_subplot(gs[3, 0])
        if not avg_rmse.empty:
            clrs30 = [fam_clr(e) for e in avg_rmse.index]
            ax30.barh(range(len(avg_rmse)), avg_rmse.values, color=clrs30, edgecolor="none", height=0.65)
            ax30.set_yticks(range(len(avg_rmse)))
            ax30.set_yticklabels(avg_rmse.index, fontsize=5.2)
            ax30.set_xscale("log")
            ax30.set_xlabel("Mean RMSE [Hz] â€” log scale", labelpad=1)
            ax30.axvline(0.1, color="#B71C1C", ls="--", lw=0.7, alpha=0.8)
            ax30.text(0.107, len(avg_rmse) - 0.5, "IEC\n100 mHz", fontsize=4.0, color="#B71C1C", va="top")
            ax30.invert_yaxis()
            for i, (_, val) in enumerate(avg_rmse.items()):
                if np.isfinite(val):
                    ax30.text(val * 1.07, i, f"{val:.3f}", va="center", fontsize=4.0, color="#333")
        for fam, clr in _FAMILY_PALETTE.items():
            ax30.barh(0, 0, color=clr, label=fam, height=0)
        ax30.legend(loc="lower right", fontsize=5, ncol=1, handlelength=0.7, borderpad=0.3)
        ax30.set_title("Estimator Ranking â€” Mean RMSE (all scenarios)", fontsize=7, fontweight="bold", pad=3)

        ax31 = fig.add_subplot(gs[3, 1])
        score_cols = [c for c in ("m1_rmse_hz_mean", "m3_max_peak_hz_mean", "m5_trip_risk_s_mean", "m13_cpu_time_us_mean") if c in df_global.columns]
        if score_cols:
            agg = df_global.groupby("estimator")[score_cols].mean()
            normed = pd.DataFrame(index=agg.index)
            for c in score_cols:
                mn, mx = agg[c].min(), agg[c].max()
                normed[c] = (agg[c] - mn) / (mx - mn + 1e-12)
            score = normed.mean(axis=1).sort_values()
            clrs31 = [fam_clr(e) for e in score.index]
            ax31.barh(range(len(score)), score.values, color=clrs31, edgecolor="none", height=0.65)
            ax31.set_yticks(range(len(score)))
            ax31.set_yticklabels(score.index, fontsize=5.2)
            ax31.set_xlabel("Composite Score (lower = better)", labelpad=1)
            ax31.invert_yaxis()
            for i, val in enumerate(score.values):
                ax31.text(val + 0.005, i, f"#{i+1}", va="center", fontsize=4.0, color="#333")
            ax31.text(
                0.98,
                0.02,
                "Normalised: RMSE + Max Peak + Trip Risk + CPU\n(each metric â†’ [0, 1], mean = composite score)",
                transform=ax31.transAxes,
                fontsize=4.5,
                va="bottom",
                ha="right",
                style="italic",
                color="#555",
            )
        else:
            ax31.text(0.5, 0.5, "No data", transform=ax31.transAxes, ha="center", fontsize=7, color="#888")
        for fam, clr in _FAMILY_PALETTE.items():
            ax31.barh(0, 0, color=clr, label=fam, height=0)
        ax31.legend(loc="lower right", fontsize=5, ncol=1, handlelength=0.7, borderpad=0.3)
        ax31.set_title("Composite Performance Score", fontsize=7, fontweight="bold", pad=3)

        fig.text(0.5, 0.962, "IBR Frequency Estimator Benchmark â€” Performance Analysis", ha="center", va="top", fontsize=8, fontweight="bold")
        fig.text(
            0.5,
            0.946,
            rf"$N_\mathrm{{MC}}={N_MC_RUNS}$ runs per estimator/scenario  |  "
            rf"$f_s=10\,\mathrm{{kHz}}$  |  "
            rf"dual-rate physics simulation (1 MHz $\rightarrow$ 10 kHz)",
            ha="center",
            va="top",
            fontsize=5.5,
            color="#555",
        )

        out = out_dir / "megadashboard2.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
    print(f"    megadashboard2 saved â†’ {out.relative_to(ROOT)}")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Phase 4 â€” Full JSON export + advanced analysis
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _metric_series_stats(values: list[float | None]) -> dict[str, Any]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"n": 0}

    q25, q75 = np.percentile(arr, [25, 75])
    iqr = float(q75 - q25)
    median = float(np.median(arr))
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0

    out_low = q25 - 1.5 * iqr
    out_high = q75 + 1.5 * iqr
    outliers = arr[(arr < out_low) | (arr > out_high)]

    stats_out: dict[str, Any] = {
        "n": int(arr.size),
        "mean": mean,
        "std": std,
        "median": median,
        "IQR": iqr,
        "CV": float(std / mean) if abs(mean) > 1e-12 else None,
        "p5": float(np.percentile(arr, 5)),
        "p25": float(q25),
        "p75": float(q75),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n_outliers": int(outliers.size),
        "outlier_values": [float(x) for x in outliers[:20]],
    }

    if scipy_stats is not None and 3 <= arr.size <= 5000:
        try:
            sw_stat, sw_p = scipy_stats.shapiro(arr)
            stats_out["SW_stat"] = float(sw_stat)
            stats_out["SW_p"] = float(sw_p)
            stats_out["is_normal_SW"] = bool(sw_p >= 0.05)
        except Exception:
            stats_out["SW_stat"] = None
            stats_out["SW_p"] = None
            stats_out["is_normal_SW"] = None
    else:
        stats_out["SW_stat"] = None
        stats_out["SW_p"] = None
        stats_out["is_normal_SW"] = None

    if scipy_stats is not None and arr.size >= 3:
        try:
            stats_out["skew"] = float(scipy_stats.skew(arr, bias=False))
            stats_out["kurtosis_excess"] = float(scipy_stats.kurtosis(arr, bias=False))
        except Exception:
            stats_out["skew"] = None
            stats_out["kurtosis_excess"] = None
    else:
        stats_out["skew"] = None
        stats_out["kurtosis_excess"] = None

    return stats_out


def _load_long_run_dataframe(allowed_estimators: set[str] | None = None) -> pd.DataFrame:
    """
    Load all per-run summary CSVs into one long dataframe:
    one row = one MC run for one (scenario, estimator).
    """
    rows: list[pd.DataFrame] = []

    for sc_dir in sorted(BASE_RESULTS_DIR.iterdir()):
        if not sc_dir.is_dir():
            continue
        sc_name = sc_dir.name

        for est_dir in sorted(sc_dir.iterdir()):
            if not est_dir.is_dir():
                continue
            est_name = est_dir.name
            if allowed_estimators is not None and est_name not in allowed_estimators:
                continue
            summary_file = next(est_dir.glob("*_summary.csv"), None)
            if summary_file is None:
                continue

            df = pd.read_csv(summary_file).copy()
            if df.empty:
                continue

            df["scenario"] = sc_name
            df["estimator"] = est_name
            df["family"] = _ESTIMATOR_FAMILIES.get(est_name, "Unknown")

            if "run_idx" not in df.columns:
                df["run_idx"] = np.arange(len(df), dtype=int)

            rows.append(df)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def _build_aggregated_dataframe(df_long: pd.DataFrame) -> pd.DataFrame:
    if df_long.empty:
        return pd.DataFrame()

    group_cols = ["scenario", "estimator", "family"]
    agg_map: dict[str, list[str]] = {}

    for metric in METRIC_COLUMNS:
        if metric in df_long.columns:
            agg_map[metric] = ["mean", "std", "median", "min", "max"]

    if not agg_map:
        return pd.DataFrame()

    df_agg = df_long.groupby(group_cols, dropna=False).agg(agg_map)
    df_agg.columns = [f"{col}_{stat}" for col, stat in df_agg.columns]
    df_agg = df_agg.reset_index()
    return df_agg


def _build_run_specs_manifest(allowed_estimators: set[str] | None = None) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []

    for sc_dir in sorted(BASE_RESULTS_DIR.iterdir()):
        if not sc_dir.is_dir():
            continue
        for est_dir in sorted(sc_dir.iterdir()):
            if not est_dir.is_dir():
                continue
            if allowed_estimators is not None and est_dir.name not in allowed_estimators:
                continue
            spec_file = est_dir / "run_spec.json"
            if spec_file.exists():
                try:
                    specs.append(json.loads(spec_file.read_text(encoding="utf-8")))
                except Exception as exc:
                    specs.append({
                        "scenario": sc_dir.name,
                        "estimator": est_dir.name,
                        "error": f"failed_to_read_run_spec: {exc}",
                    })
    return specs


def _build_monte_carlo_stats(df_long: pd.DataFrame) -> dict[str, Any]:
    if df_long.empty:
        return {
            "description": "No Monte Carlo data found.",
            "n_runs": 0,
            "methods": {},
        }

    out: dict[str, Any] = {
        "description": (
            "Monte Carlo robustness analysis exported from per-run summary CSVs. "
            "Each estimator/scenario pair includes descriptive statistics for all tracked metrics."
        ),
        "n_runs": int(df_long.groupby(["scenario", "estimator"]).size().max()),
        "methods": {},
    }

    for est_name, df_est in df_long.groupby("estimator"):
        est_block: dict[str, Any] = {}
        for sc_name, df_sc in df_est.groupby("scenario"):
            sc_block: dict[str, Any] = {}
            for metric, metric_label in METRIC_LABELS.items():
                if metric in df_sc.columns:
                    sc_block[metric_label] = _metric_series_stats(
                        [_safe_float(v) for v in df_sc[metric].tolist()]
                    )
            est_block[sc_name] = sc_block
        out["methods"][est_name] = est_block

    return out


def _build_trends(df_agg: pd.DataFrame) -> dict[str, Any]:
    if df_agg.empty:
        return {}

    trends: dict[str, Any] = {}

    if "m1_rmse_hz_mean" in df_agg.columns:
        scenario_difficulty = (
            df_agg.groupby("scenario")["m1_rmse_hz_mean"]
            .median()
            .sort_values(ascending=False)
        )
        trends["scenario_difficulty_by_median_rmse"] = [
            {"scenario": sc, "median_rmse_hz": float(v)}
            for sc, v in scenario_difficulty.items()
        ]

        winners = []
        for sc_name, df_sc in df_agg.groupby("scenario"):
            row = df_sc.loc[df_sc["m1_rmse_hz_mean"].idxmin()]
            winners.append({
                "scenario": sc_name,
                "winner_estimator": row["estimator"],
                "family": row["family"],
                "rmse_hz": float(row["m1_rmse_hz_mean"]),
            })
        trends["scenario_winners_rmse"] = winners

    if "m13_cpu_time_us_mean" in df_agg.columns:
        fastest = (
            df_agg.groupby("estimator")["m13_cpu_time_us_mean"]
            .mean()
            .sort_values()
        )
        trends["global_fastest_estimators_cpu"] = [
            {"estimator": est, "cpu_time_us_mean": float(v)}
            for est, v in fastest.items()
        ]

    family_win_counter: dict[str, int] = defaultdict(int)
    if "m1_rmse_hz_mean" in df_agg.columns:
        for _, df_sc in df_agg.groupby("scenario"):
            best_row = df_sc.loc[df_sc["m1_rmse_hz_mean"].idxmin()]
            family_win_counter[str(best_row["family"])] += 1
    trends["family_wins_by_rmse"] = dict(sorted(family_win_counter.items()))

    return trends


def _build_rankings(df_agg: pd.DataFrame) -> dict[str, Any]:
    if df_agg.empty:
        return {}

    out: dict[str, Any] = {}
    metrics_for_rank = [m for m in METRIC_COLUMNS if f"{m}_mean" in df_agg.columns]

    for metric in metrics_for_rank:
        pivot = df_agg.pivot_table(
            index="scenario",
            columns="estimator",
            values=f"{metric}_mean",
            aggfunc="mean",
        )
        if pivot.empty:
            continue

        ranks = pivot.rank(axis=1, method="average", ascending=True)
        mean_rank = ranks.mean(axis=0).sort_values()

        out[METRIC_LABELS.get(metric, metric)] = [
            {"estimator": est, "mean_rank": float(rank)}
            for est, rank in mean_rank.items()
        ]

    return out


def _build_pareto_front(df_agg: pd.DataFrame) -> dict[str, Any]:
    needed = {"m1_rmse_hz_mean", "m13_cpu_time_us_mean"}
    if df_agg.empty or not needed.issubset(df_agg.columns):
        return {}

    df_est = (
        df_agg.groupby(["estimator", "family"])[["m1_rmse_hz_mean", "m13_cpu_time_us_mean"]]
        .mean()
        .reset_index()
    )

    points = df_est[["m1_rmse_hz_mean", "m13_cpu_time_us_mean"]].to_numpy(dtype=float)
    is_pareto = np.ones(len(points), dtype=bool)

    for i in range(len(points)):
        for j in range(len(points)):
            if i == j:
                continue
            better_or_equal = np.all(points[j] <= points[i])
            strictly_better = np.any(points[j] < points[i])
            if better_or_equal and strictly_better:
                is_pareto[i] = False
                break

    return {
        "objective": "minimize RMSE and CPU time simultaneously",
        "points": [
            {
                "estimator": row["estimator"],
                "family": row["family"],
                "rmse_hz_mean": float(row["m1_rmse_hz_mean"]),
                "cpu_time_us_mean": float(row["m13_cpu_time_us_mean"]),
                "is_pareto": bool(flag),
            }
            for (_, row), flag in zip(df_est.iterrows(), is_pareto)
        ],
        "pareto_estimators": [
            str(df_est.iloc[i]["estimator"]) for i in range(len(df_est)) if is_pareto[i]
        ],
    }


def _build_correlations(df_agg: pd.DataFrame) -> dict[str, Any]:
    corr_cols = [f"{m}_mean" for m in METRIC_COLUMNS if f"{m}_mean" in df_agg.columns]
    if len(corr_cols) < 2:
        return {}

    corr = df_agg[corr_cols].corr(method="spearman")
    corr = corr.rename(index=lambda c: c.replace("_mean", ""), columns=lambda c: c.replace("_mean", ""))
    return {"spearman": _to_builtin(corr.round(6).to_dict())}


def _build_pca_kmeans(df_agg: pd.DataFrame) -> dict[str, Any]:
    needed = [f"{m}_mean" for m in METRIC_COLUMNS if f"{m}_mean" in df_agg.columns]
    if len(needed) < 2 or StandardScaler is None or PCA is None:
        return {
            "pca": {"available": False, "reason": "sklearn not installed or insufficient metrics"},
            "kmeans": {"available": False, "reason": "sklearn not installed or insufficient metrics"},
        }

    df_est = (
        df_agg.groupby(["estimator", "family"])[needed]
        .mean()
        .reset_index()
    )

    if len(df_est) < 3:
        return {
            "pca": {"available": False, "reason": "too few estimators for PCA"},
            "kmeans": {"available": False, "reason": "too few estimators for clustering"},
        }

    X = df_est[needed].to_numpy(dtype=float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca_model = PCA(n_components=min(2, Xs.shape[1]))
    Xp = pca_model.fit_transform(Xs)

    pca_block = {
        "available": True,
        "features": needed,
        "explained_variance_ratio": [float(v) for v in pca_model.explained_variance_ratio_],
        "components": _to_builtin(pca_model.components_.tolist()),
        "scores": [
            {
                "estimator": row["estimator"],
                "family": row["family"],
                "PC1": float(Xp[i, 0]),
                "PC2": float(Xp[i, 1]) if Xp.shape[1] > 1 else 0.0,
            }
            for i, (_, row) in enumerate(df_est.iterrows())
        ],
    }

    if KMeans is None:
        kmeans_block: dict[str, Any] = {"available": False, "reason": "sklearn not installed"}
    else:
        best = None
        best_score = -np.inf
        best_labels = None

        for k in range(2, min(4, len(df_est) - 1) + 1):
            try:
                model = KMeans(n_clusters=k, n_init=20, random_state=42)
                labels = model.fit_predict(Xs)
                score = silhouette_score(Xs, labels) if silhouette_score is not None else -1.0
                if score > best_score:
                    best_score = score
                    best = model
                    best_labels = labels
            except Exception:
                continue

        if best is None or best_labels is None:
            kmeans_block = {"available": False, "reason": "kmeans fitting failed"}
        else:
            kmeans_block = {
                "available": True,
                "n_clusters": int(best.n_clusters),
                "silhouette_score": float(best_score),
                "clusters": [
                    {
                        "estimator": row["estimator"],
                        "family": row["family"],
                        "cluster": int(best_labels[i]),
                    }
                    for i, (_, row) in enumerate(df_est.iterrows())
                ],
                "cluster_centers_standardized": _to_builtin(best.cluster_centers_.tolist()),
            }

    return {
        "pca": pca_block,
        "kmeans": kmeans_block,
    }


def _build_family_hypothesis_tests(df_long: pd.DataFrame) -> dict[str, Any]:
    if df_long.empty:
        return {}

    out: dict[str, Any] = {
        "note": (
            "Exploratory family-level tests across Monte Carlo run rows. "
            "Use as secondary evidence, not as the sole paper claim source."
        ),
        "tests": {},
    }

    for metric, label in METRIC_LABELS.items():
        if metric not in df_long.columns:
            continue

        valid = df_long[["family", metric]].dropna().copy()
        if valid.empty:
            continue

        groups = []
        group_names = []
        for fam, df_fam in valid.groupby("family"):
            vals = df_fam[metric].to_numpy(dtype=float)
            if len(vals) >= 2:
                groups.append(vals)
                group_names.append(str(fam))

        if len(groups) < 2:
            continue

        test_block: dict[str, Any] = {"groups": group_names}

        if scipy_stats is not None:
            try:
                f_stat, p_val = scipy_stats.f_oneway(*groups)
                test_block["anova"] = {
                    "F": float(f_stat),
                    "p_value": float(p_val),
                    "reject_H0_alpha_0p05": bool(p_val < 0.05),
                }
            except Exception as exc:
                test_block["anova"] = {"error": str(exc)}

            try:
                h_stat, p_val = scipy_stats.kruskal(*groups)
                test_block["kruskal"] = {
                    "H": float(h_stat),
                    "p_value": float(p_val),
                    "reject_H0_alpha_0p05": bool(p_val < 0.05),
                }
            except Exception as exc:
                test_block["kruskal"] = {"error": str(exc)}
        else:
            test_block["anova"] = {"available": False, "reason": "scipy not installed"}
            test_block["kruskal"] = {"available": False, "reason": "scipy not installed"}

        out["tests"][label] = test_block

    return out


def _export_full_benchmark_json(estimators: dict[str, type]) -> Path:
    print("\n>>> PHASE 4: FULL JSON EXPORT + ADVANCED ANALYSIS <<<")
    allowed_estimators = set(estimators.keys())

    df_long = _load_long_run_dataframe(allowed_estimators=allowed_estimators)
    df_agg = _build_aggregated_dataframe(df_long)
    run_specs = _build_run_specs_manifest(allowed_estimators=allowed_estimators)

    report: dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "description": "Unified benchmark export with full run specs, Monte Carlo summaries, and advanced analysis.",
            "pc_hostname": platform.node(),
            "machine_arch": platform.machine(),
            "cpu_processor": platform.processor(),
            "os_platform": platform.platform(),
            "python_version": sys.version,
        },
        "run_configuration": {
            "benchmark_identity": BENCHMARK_IDENTITY,
            "benchmark_scope": BENCHMARK_SCOPE,
            "authority_statement": BENCHMARK_AUTHORITY_STATEMENT,
            "paper_alignment_policy": PAPER_ALIGNMENT_POLICY,
            "n_trials_tuning": N_TRIALS_TUNING,
            "n_mc_runs": N_MC_RUNS,
            "optuna_sampler_mode": OPTUNA_SAMPLER_MODE,
            "optuna_seed": OPTUNA_SEED,
            "apply_trial_overrides": APPLY_TRIAL_OVERRIDES,
            "n_trials_overrides": _to_builtin(N_TRIALS_OVERRIDES),
            "excluded_estimators": _to_builtin(EXCLUDED_ESTIMATOR_LABELS),
            "base_results_dir": str(BASE_RESULTS_DIR),
            "scenarios": [sc.get_name() for sc in SCENARIOS],
            "metrics": METRIC_COLUMNS,
            "metric_labels": METRIC_LABELS,
            "estimator_families": _ESTIMATOR_FAMILIES,
        },
        "estimators_loaded": sorted(list(estimators.keys())),
        "estimator_registry": build_estimator_registry_manifest(),
        "estimators_excluded": [spec.label for spec in excluded_estimator_specs()],
        "run_specs": run_specs,
        "raw_run_records": _to_builtin(df_long.to_dict(orient="records")) if not df_long.empty else [],
        "aggregated_metrics": _to_builtin(df_agg.to_dict(orient="records")) if not df_agg.empty else [],
        "monte_carlo": _build_monte_carlo_stats(df_long),
        "advanced_analysis": {
            "trends": _build_trends(df_agg),
            "rankings": _build_rankings(df_agg),
            "pareto_front": _build_pareto_front(df_agg),
            "correlations": _build_correlations(df_agg),
            **_build_pca_kmeans(df_agg),
            "family_hypothesis_tests": _build_family_hypothesis_tests(df_long),
        },
        "artifacts_manifest": {
            "global_metrics_report_csv": str(BASE_RESULTS_DIR / "global_metrics_report.csv"),
            "figure_1_png": str(BASE_RESULTS_DIR / f"{FIGURE1_BASENAME}.png"),
            "figure_1_pdf": str(BASE_RESULTS_DIR / f"{FIGURE1_BASENAME}.pdf"),
            "figure_2_png": str(BASE_RESULTS_DIR / f"{FIGURE2_BASENAME}.png"),
            "figure_2_pdf": str(BASE_RESULTS_DIR / f"{FIGURE2_BASENAME}.pdf"),
        },
    }

    out_path = BASE_RESULTS_DIR / JSON_REPORT_NAME
    out_path.write_text(
        json.dumps(_to_builtin(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Full JSON report â†’ {out_path.relative_to(ROOT)}")
    return out_path


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Pytest entry points
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def test_estimators_load() -> None:
    """All canonical estimators must import and expose a 'name' attribute."""
    estimators = load_estimators()
    assert len(estimators) > 0, "No estimators loaded"
    for _, cls in estimators.items():
        assert hasattr(cls, "name"), f"{cls.__name__} missing class attribute 'name'"


def test_estimator_registry_contains_lkf_and_pi_gru() -> None:
    """The active canonical registry must include both LKF variants and PI-GRU."""
    estimators = load_estimators()
    assert "LKF" in estimators, "LKF must load through the active pipeline"
    assert "LKF2" in estimators, "LKF2 must load through the active pipeline"
    assert "PI-GRU" in estimators, "PI-GRU must load through the active pipeline"


def test_search_spaces_valid() -> None:
    """Every SEARCH_SPACES key must use only real __init__ parameter names."""
    estimators = load_estimators()
    validate_search_spaces(estimators)


def test_single_scenario_single_estimator() -> None:
    """Smoke: one MC run of EKF on the sinwave scenario must produce finite RMSE."""
    from estimators.ekf import EKF_Estimator

    engine = MonteCarloEngine(
        scenario_cls=IEEESingleSinWaveScenario,
        estimator_cls=EKF_Estimator,
        estimator_params=EKF_Estimator.default_params(),
        n_runs=1,
    )
    result = engine.run()
    rmse_col = "m1_rmse_hz"
    assert rmse_col in result.summary_df.columns, "RMSE column missing from MC output"
    rmse = float(result.summary_df[rmse_col].iloc[0])
    assert np.isfinite(rmse), f"RMSE is not finite: {rmse}"
    assert rmse < 1.0, f"RMSE {rmse:.4f} Hz is unexpectedly large on a clean sinwave"


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Script entry point
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def main() -> None:
    t_start = time.time()
    BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading estimators ...")
    estimators = load_estimators()
    _print_registry_summary(estimators)

    print("Validating search spaces ...")
    validate_search_spaces(estimators)
    print("  OK")
    print(
        "Run config: "
        f"N_MC_RUNS={N_MC_RUNS}, "
        f"N_TRIALS_TUNING={N_TRIALS_TUNING}, "
        f"OPTUNA_SAMPLER={OPTUNA_SAMPLER_MODE}, "
        f"APPLY_OVERRIDES={APPLY_TRIAL_OVERRIDES}, "
        f"EXCLUDED={EXCLUDED_ESTIMATOR_LABELS}"
    )

    run_phase_1(estimators)
    run_phase_2(allowed_estimators=set(estimators.keys()))
    run_phase_3()
    _export_full_benchmark_json(estimators)

    elapsed = (time.time() - t_start) / 60.0
    print(f"\n[DONE] Benchmark completed in {elapsed:.1f} min.")


if __name__ == "__main__":
    main()
