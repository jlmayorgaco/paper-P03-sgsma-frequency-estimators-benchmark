"""
Benchmark smoke test: scenario-wise Optuna tuning, Monte Carlo, artifact generation.

Phases
------
1  Tune each estimator per scenario, run N_MC_RUNS, save CSV artifacts.
2  Aggregate summary CSVs into per-scenario and global statistics.
3  Generate performance dashboards from the aggregated report.

Usage
-----
As a script (full benchmark):
    python tests/montecarlo/test_dedicated_smoke_test.py

As pytest (infrastructure sanity, fast):
    pytest tests/montecarlo/test_dedicated_smoke_test.py -v
"""

from __future__ import annotations

import gc
import inspect
import os
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

# ── Thread safety: must be set before NumPy/MKL load in subprocesses ──────────
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# ── Path bootstrap ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ── Project imports ────────────────────────────────────────────────────────────
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

# ── Run configuration ──────────────────────────────────────────────────────────
N_TRIALS_TUNING = 60
N_MC_RUNS = 10
BASE_RESULTS_DIR = ROOT / "tests" / "montecarlo" / "outputs"

SCENARIOS = [
    IBRMultiEventScenario,
    IBRPowerImbalanceRingdownScenario,
    IBRHarmonicsSmallScenario,
    IBRHarmonicsMediumScenario,
    IBRHarmonicsLargeScenario,
    IEEEFreqRampScenario,
    IEEEFreqStepScenario,
    IEEEMagStepScenario,
    IEEEModulationScenario,
    IEEEModulationAMScenario,
    IEEEModulationFMScenario,
    IEEEOOBInterferenceScenario,
    IEEEPhaseJump20Scenario,
    IEEEPhaseJump60Scenario,
    NERCPhaseJump60Scenario,
    IEEESingleSinWaveScenario
]

# Metric columns written by MonteCarloEngine / metrics.py
METRIC_COLUMNS = [
    "m1_rmse_hz",
    "m2_mae_hz",
    "m3_max_peak_hz",
    "m5_trip_risk_s",
    "m8_settling_time_s",
    "m13_cpu_time_us",
]

# ── IEEE publication-quality style ────────────────────────────────────────────
_IEEE_RC: dict = {
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":        7,
    "axes.labelsize":   7,
    "axes.titlesize":   7.5,
    "xtick.labelsize":  6.5,
    "ytick.labelsize":  6.5,
    "legend.fontsize":  6,
    "figure.dpi":       300,
    "savefig.bbox":     "tight",
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.linewidth":   0.4,
    "axes.linewidth":   0.6,
    "lines.linewidth":  0.8,
    "patch.linewidth":  0.5,
    "mathtext.fontset": "stix",
}
_IEEE_FULL_W = 7.16    # two-column text width, inches (IEEE conference)
_IEEE_PAGE_H = 11.0    # letter-size page height, inches

_ESTIMATOR_FAMILIES: dict[str, str] = {
    "ZCD":             "Loop-based",
    "PLL":             "Loop-based",
    "SOGI-PLL":        "Loop-based",
    "SOGI-FLL":        "Loop-based",
    "Type-3 SOGI-PLL": "Loop-based",
    "EKF":             "Model-based",
    "RA-EKF":          "Model-based",
    "UKF":             "Model-based",
    "IPDFT":           "Window-based",
    "TFT":             "Window-based",
    "Prony":           "Window-based",
    "ESPRIT":          "Window-based",
    "TKEO":            "Adaptive",
    "RLS":             "Adaptive",
    "Koopman":         "Data-driven",
}
_FAMILY_PALETTE: dict[str, str] = {
    "Model-based":  "#1565C0",
    "Loop-based":   "#2E7D32",
    "Window-based": "#E65100",
    "Adaptive":     "#6A1B9A",
    "Data-driven":  "#B71C1C",
}

# ── Search spaces ──────────────────────────────────────────────────────────────
# Each lambda receives an optuna.Trial and returns a dict whose keys MUST match
# the corresponding estimator's __init__ parameter names exactly.
# Wrong names are caught at startup by _validate_search_spaces().
SEARCH_SPACES: dict[str, Any] = {
    "PLL": lambda trial: {
        "kp_scale": trial.suggest_float("kp_scale", 0.001, 1.5, log=True),
        "ki_scale": trial.suggest_float("ki_scale", 0.001, 1.5, log=True),
    },
    "SOGI-PLL": lambda trial: { # SRF-PLL monofásico
        "settle_time": trial.suggest_float("settle_time", 0.01, 0.20),
        "k_sogi": trial.suggest_float("k_sogi", 0.7, 1.414),
    },
    "Type-3 SOGI-PLL": lambda trial: { # El arma contra RoCoF
        "kp": trial.suggest_float("kp", 50.0, 150.0),
        "ki": trial.suggest_float("ki", 1000.0, 5000.0),
        "ki2": trial.suggest_float("ki2", 10000.0, 50000.0),
    },
    "SOGI-FLL": lambda trial: {
        "gamma": trial.suggest_float("gamma", 20.0, 100.0),
        "k_sogi": trial.suggest_float("k_sogi", 1.0, 2.0),
    },
    "RLS": lambda trial: {
        # Podemos tunearlo como VFF o Fixed. Aquí lo configuramos como VFF:
        "is_vff": True, 
        "alpha_vff": trial.suggest_float("alpha_vff", 0.05, 0.4),
        "lambda_min": trial.suggest_float("lambda_min", 0.85, 0.98),
    },
    "EKF": lambda trial: {
        "q_omega": trial.suggest_float("q_omega", 1e-6, 1e-1, log=True),
        "r_meas": trial.suggest_float("r_meas", 1e-5, 1.0, log=True),
    },
    "RA-EKF": lambda trial: {
        "q_rocof": trial.suggest_float("q_rocof", 1e-12, 100, log=True),
        "r_meas": trial.suggest_float("r_meas", 1e-12, 100.0, log=True),
    },
    "IPDFT": lambda trial: {
        "cycles": trial.suggest_float("cycles", 2.0, 6.0),
    },
    "TFT": lambda trial: {
        "n_cycles": trial.suggest_float("n_cycles", 2.0, 6.0),
    },
    "Prony": lambda trial: {
        "order": trial.suggest_int("order", 2, 4), # Bajamos a 4 para evitar explosión numérica
        "n_cycles": trial.suggest_float("n_cycles", 2.0, 4.0),
    },
    "ESPRIT": lambda trial: {
        "n_cycles": trial.suggest_float("n_cycles", 2.0, 4.0),
    },
    "Koopman": lambda trial: {
        "n_cycles": trial.suggest_float("n_cycles", 2.0, 5.0),
    },
    "TKEO": lambda trial: {
        "output_smoothing": trial.suggest_float("output_smoothing", 0.001, 0.05),
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# Estimator registry
# ═══════════════════════════════════════════════════════════════════════════════

def load_estimators() -> dict[str, type]:
    """
    Import all benchmark estimators, skipping any that fail to load.
    Returns a dict mapping estimator label → class.
    """

def load_estimators() -> dict[str, type]:

    """Carga todos los estimadores disponibles para el benchmark."""
    candidates = [
        ("zcd",       "ZCDEstimator"),
        ("ipdft",     "IPDFT_Estimator"),
        ("tft",       "TFT_Estimator"),
        
        ("rls",       "RLS_Estimator"),
        ("pll",       "PLL_Estimator"),
        ("sogi_pll",  "SOGIPLLEstimator"),
        ("sogi_fll",  "SOGI_FLL_Estimator"),
        ("type3_sogi_pll", "Type3_SOGI_PLL_Estimator"),

        # Familia Kalman
        ("lkf",       "EKF_Estimator"),
        ("lkf2",      "LKF2_Estimator"),
        ("ekf",       "EKF_Estimator"),
        ("ukf",       "UKF_Estimator"), 
        ("ra_ekf",    "RAEKF_Estimator"),

        ("tkeo",      "TKEO_Estimator"),
        #("prony",     "Prony_Estimator"),
        ("esprit",    "ESPRIT_Estimator"),
        ("koopman",   "Koopman_Estimator"),
    ]

    registry: dict[str, type] = {}
    for module_name, class_name in candidates:
        try:
            module = __import__(f"estimators.{module_name}", fromlist=[class_name])
            cls = getattr(module, class_name)
            # Usamos el atributo 'name' de la clase como etiqueta (ej: "Type-3 SOGI-PLL")
            label = getattr(cls, "name", class_name)
            registry[label] = cls
        except Exception as exc:
            print(f"  [skip] estimators.{module_name}: {exc}")
    return registry


def validate_search_spaces(estimators: dict[str, type]) -> None:
    """
    Raise ValueError if any SEARCH_SPACES key suggests a param that does not
    exist in the estimator's __init__ signature.  Run once at startup so bugs
    are caught before any time-consuming benchmark work begins.
    """
    import optuna.trial as ot

    class _FakeTrial:
        """Minimal stand-in that records suggested param names."""
        def __init__(self):
            self.suggested: set[str] = set()

        def suggest_float(self, name: str, *args, **kwargs) -> float:
            self.suggested.add(name)
            return 0.5

        def suggest_int(self, name: str, *args, **kwargs) -> int:
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


# ═══════════════════════════════════════════════════════════════════════════════
# Noise-profile compatibility helper
# ═══════════════════════════════════════════════════════════════════════════════

def _noise_kwargs(sc_cls: type, level: float) -> dict[str, Any]:
    """
    Return noise override kwargs compatible with both old-style scenarios
    (single `noise_sigma`) and new multi-spectral scenarios
    (`white_noise_sigma` / `brown_noise_sigma` / `impulse_prob`).
    `level` is applied only to white/AWGN noise; brown and impulse are zeroed
    for tuning/baseline use so the signal is deterministic in structure.
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


# ═══════════════════════════════════════════════════════════════════════════════
# Tuning
# ═══════════════════════════════════════════════════════════════════════════════

def tune_estimator(
    est_name: str,
    est_cls: type,
    scenario_cls: type,
) -> dict[str, Any]:
    """
    Run Optuna TPE search for one estimator on one scenario.
    Returns a parameter dict (best found, or defaults if all trials fail).
    """
    defaults: dict[str, Any] = (
        est_cls.default_params() if hasattr(est_cls, "default_params") else {}
    )

    if est_name not in SEARCH_SPACES:
        return defaults

    sc = scenario_cls.run(duration_s=5, seed=42, **_noise_kwargs(scenario_cls, 0.001))
    eval_start = int(0.2 * len(sc.f_true))
    space_fn = SEARCH_SPACES[est_name]

    def objective(trial: optuna.Trial) -> float:
        suggested = space_fn(trial)
        params = {**defaults, **suggested}
        try:
            est = est_cls(**params)
            f_hat = _run_estimator(est, sc.v)
            error = f_hat[eval_start:] - sc.f_true[eval_start:]
            rmse = float(np.sqrt(np.mean(error ** 2)))
            return rmse if np.isfinite(rmse) else 1e6
        except Exception:
            return 1e6

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS_TUNING)

    if study.best_value >= 1e6:
        print(f"      [!] All trials failed for {est_name} — using defaults.")
        return defaults

    best_suggested = space_fn(study.best_trial)
    return {**defaults, **best_suggested}


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

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
            ax.plot(sc.t, f_hat, color="#C62828", linewidth=0.7,
                    label=f"{est_name}")
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

    The zoom window is derived from scenario metadata (event / ramp start time)
    or defaults to the centre 400 ms if no distinct event is detected.
    The output file is ``<sc_name>_zoom.png`` inside *sc_dir*.
    """
    p = sc.meta.get("parameters", {})
    t_total = float(sc.t[-1])

    # Find event time from common metadata keys
    event_t: float | None = None
    for key in ("t_step_s", "t_event_s", "t_ramp_s"):
        if key in p and p[key] is not None:
            event_t = float(p[key])
            break
    if event_t is None and "t_start_s" in p:
        event_t = float(p["t_start_s"])

    if event_t is not None:
        z_lo = max(0.0, event_t - 0.08)          # ~5 cycles pre-event
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

        # ── Voltage zoom ──────────────────────────────────────────────────────
        ax_v.plot(t_z, v_z, lw=0.65, color="#1A237E", rasterized=True)
        if event_t is not None:
            ax_v.axvline(event_t, color="black", lw=0.8, ls="--", alpha=0.65,
                         label="Event onset")
            ax_v.legend(loc="upper right", fontsize=5.5, handlelength=0.9)
        ax_v.set_ylabel(r"$v(t)$ [pu]", labelpad=2)
        v_rng = max(float(np.ptp(v_z)), 0.05)
        ax_v.set_ylim(float(v_z.min()) - 0.10 * v_rng,
                      float(v_z.max()) + 0.22 * v_rng)
        ax_v.set_title(f"{sc_name} — Zoom Detail", fontweight="bold", pad=3)

        # ── Frequency zoom ────────────────────────────────────────────────────
        ax_f.plot(t_z, f_z, lw=1.0, color="#B71C1C", rasterized=True,
                  label=r"$f_\mathrm{true}(t)$")
        ax_f.axhline(f_nom, color="#444", lw=0.55, ls=":", alpha=0.7,
                     label=f"$f_0={f_nom:.0f}$ Hz")
        if event_t is not None:
            ax_f.axvline(event_t, color="black", lw=0.8, ls="--", alpha=0.65)
        ax_f.set_xlabel("Time [s]", labelpad=1)
        ax_f.set_ylabel(r"$f(t)$ [Hz]", labelpad=2)
        f_rng = max(float(np.ptp(f_z)), 0.02)
        ax_f.set_ylim(float(f_z.min()) - 0.18 * f_rng,
                      float(f_z.max()) + 0.28 * f_rng)
        ax_f.set_xlim(z_lo, z_hi)
        ax_f.legend(loc="upper right", fontsize=5.5, ncol=2,
                    handlelength=1.0, borderpad=0.3)

        out = sc_dir / f"{sc_name}_zoom.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
    print(f"      Zoom plot → {out.relative_to(ROOT)}")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Tuning + Monte Carlo + per-estimator artifacts
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase_1(estimators: dict[str, type]) -> None:
    """
    For every (scenario, estimator) pair:
      1. Tune hyperparameters with Optuna.
      2. Save a tracking plot (noise-free baseline).
      3. Run N_MC_RUNS Monte Carlo iterations and save summary + signals CSVs.
    """
    print("\n>>> PHASE 1: TUNING + MC + ARTIFACTS <<<")

    for sc_cls in SCENARIOS:
        sc_name = sc_cls.get_name()
        sc_dir = BASE_RESULTS_DIR / sc_name
        sc_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  Scenario: {sc_name}")

        sc_base = sc_cls.run(duration_s=5, seed=42, **_noise_kwargs(sc_cls, 0.0))
        _save_scenario_artifacts(sc_base, sc_dir, sc_name)
        _save_scenario_zoom_plot(sc_base, sc_dir, sc_name)

        for est_name, est_cls in estimators.items():
            print(f"    [{est_name}] tuning ...", end=" ", flush=True)
            out_dir = sc_dir / est_name
            out_dir.mkdir(parents=True, exist_ok=True)

            best_params = tune_estimator(est_name, est_cls, sc_cls)
            print("MC ...", end=" ", flush=True)

            _save_tracking_plot(est_cls, best_params, sc_base, out_dir, est_name, sc_name)

            engine = MonteCarloEngine(
                scenario_cls=sc_cls,
                estimator_cls=est_cls,
                estimator_params=best_params,
                n_runs=N_MC_RUNS,
            )
            result = engine.run()
            engine.save_csv(result, out_dir)
            print("done")
            gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Metric aggregation
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase_2() -> None:
    """
    Read every *_summary.csv written by Phase 1, compute mean ± std across
    MC runs, and write per-scenario and global summary CSVs.
    """
    print("\n>>> PHASE 2: METRIC AGGREGATION <<<")
    all_stats: list[dict] = []

    for sc_dir in sorted(BASE_RESULTS_DIR.iterdir()):
        if not sc_dir.is_dir():
            continue
        sc_name = sc_dir.name
        sc_stats: list[dict] = []

        for est_dir in sorted(sc_dir.iterdir()):
            if not est_dir.is_dir():
                continue
            est_name = est_dir.name
            summary_file = next(est_dir.glob("*_summary.csv"), None)
            if summary_file is None:
                print(f"    [?] No summary CSV in {est_dir.relative_to(ROOT)}")
                continue

            df = pd.read_csv(summary_file)
            available = [c for c in METRIC_COLUMNS if c in df.columns]
            row: dict[str, Any] = {"scenario": sc_name, "estimator": est_name}
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
        print(f"  Global report → {global_path.relative_to(ROOT)}")
    else:
        print("  [!] No statistics collected — run Phase 1 first.")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3 — Dashboards
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase_3() -> None:
    """
    Generate all dashboards:
      • Per-scenario 6-metric bar dashboards (dashboard.png per scenario)
      • Global accuracy–latency scatter (dashboard_global_tradeoff.png)
      • Mega Dashboard 1 — Scenario signal overview (megadashboard1.png)
      • Mega Dashboard 2 — Estimator performance summary (megadashboard2.png)
    """
    print("\n>>> PHASE 3: DASHBOARDS <<<")

    # Mega Dashboard 1 is independent of MC results — runs regardless.
    _plot_megadashboard1()

    report_path = BASE_RESULTS_DIR / "global_metrics_report.csv"
    if not report_path.exists():
        print("  [!] global_metrics_report.csv missing — skipping MC-based dashboards.")
        return

    df_global = pd.read_csv(report_path)
    _plot_per_scenario_dashboards(df_global)
    _plot_global_tradeoff(df_global)
    _plot_megadashboard2(df_global)


def _plot_per_scenario_dashboards(df_global: pd.DataFrame) -> None:
    """
    Per-scenario performance dashboard with 6 metrics, IEEE styling.
    Saved as ``dashboard.png`` inside each scenario's output directory.
    """
    panel_metrics = [
        ("m1_rmse_hz_mean",        r"RMSE [Hz]",           "log"),
        ("m2_mae_hz_mean",         r"MAE [Hz]",            "log"),
        ("m3_max_peak_hz_mean",    r"Max Peak [Hz]",       "log"),
        ("m5_trip_risk_s_mean",    r"Trip Risk $T$ [s]",   "log"),
        ("m8_settling_time_s_mean", r"Settling Time [s]",  "linear"),
        ("m13_cpu_time_us_mean",   r"CPU [$\mu$s/sample]", "linear"),
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
            _FAMILY_PALETTE.get(
                _ESTIMATOR_FAMILIES.get(e, "Adaptive"), "#757575"
            )
            for e in df_sc["estimator"]
        ]

        with plt.rc_context(_IEEE_RC):
            fig, axes = plt.subplots(
                nrows, ncols,
                figsize=(_IEEE_FULL_W, 2.8 * nrows),
            )
            axes_flat = np.array(axes).flatten()

            for idx, (col, title, scale) in enumerate(available):
                ax = axes_flat[idx]
                std_col = col.replace("_mean", "_std")
                yerr = df_sc[std_col].values if std_col in df_sc.columns else None
                ax.bar(x, df_sc[col], color=clrs, edgecolor="none",
                       alpha=0.87, zorder=3)
                if yerr is not None:
                    ax.errorbar(x, df_sc[col], yerr=yerr,
                                fmt="none", ecolor="#333", elinewidth=0.6,
                                capsize=2, zorder=4)
                ax.set_title(title, fontsize=7, pad=2)
                ax.set_yscale(scale)
                ax.set_xticks(x)
                ax.set_xticklabels(df_sc["estimator"],
                                   rotation=40, ha="right", fontsize=5.5)
                # Highlight best
                best_idx = int(df_sc[col].idxmin()) if scale == "log" \
                    else int(df_sc[col].idxmin())
                if best_idx < len(x):
                    axes_flat[idx].patches[best_idx].set_edgecolor("#000")
                    axes_flat[idx].patches[best_idx].set_linewidth(0.9)

            for idx in range(len(available), len(axes_flat)):
                axes_flat[idx].set_visible(False)

            # Family legend (shared)
            legend_patches = [
                plt.Rectangle((0, 0), 1, 1, fc=clr, label=fam)
                for fam, clr in _FAMILY_PALETTE.items()
            ]
            fig.legend(
                handles=legend_patches, loc="lower center",
                ncol=len(_FAMILY_PALETTE), fontsize=5.5,
                frameon=True, handlelength=0.9,
                bbox_to_anchor=(0.5, 0.0),
            )
            fig.suptitle(
                f"Performance Dashboard — {sc_name}\n"
                r"(sorted by RMSE; error bars = $\sigma$ across MC runs)",
                fontsize=7.5, fontweight="bold", y=1.01,
            )
            fig.tight_layout(rect=[0, 0.04, 1, 1.0])
            out = sc_dir / "dashboard.png"
            fig.savefig(out, dpi=200)
            plt.close(fig)
        print(f"    Dashboard → {out.relative_to(ROOT)}")


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
    ax.set_xlabel("Avg CPU Time / sample [µs]")
    ax.set_ylabel("Avg RMSE [Hz]")
    ax.set_title("Global Trade-off: Latency vs Accuracy")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    out_path = BASE_RESULTS_DIR / "dashboard_global_tradeoff.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    Global trade-off plot → {out_path.relative_to(ROOT)}")


# ═══════════════════════════════════════════════════════════════════════════════
# Mega Dashboard 1 — Scenario Signal Overview (IEEE 2-column full-width)
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_megadashboard1() -> None:
    """
    Publication-ready scenario signal overview.

    IEEE 2-column full-width (7.16 in), 70 % page height. 6 rows × 2 columns.
    Left column: voltage waveform v(t). Right column: instantaneous frequency f(t).

    Scenario sequence — thematic progression from clean baseline to composite IBR fault:
      Row (a) Clean baseline        — IEEE Single SinWave
      Row (b) Amplitude disturbance — IEEE Mag Step  (+10 %, IEC 60255-118-1 Scen. A)
      Row (c) Frequency dynamics    — IEEE Freq Ramp (+3 Hz/s, IEEE 1547 Cat III)
      Row (d) Modulation bandwidth  — IEEE AM/PM Modulation (fm = 2 Hz)
      Row (e) Harmonic distortion   — IBR Harmonics Medium (THD ≈ 8 %)
      Row (f) Composite IBR fault   — IBR Multi-Event (worst-case)

    Output: ``outputs/megadashboard1.png``
    """
    import matplotlib.gridspec as gridspec

    out_dir = BASE_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n  [MEGA1] Building scenario signal overview mega-dashboard ...")

    # ── Scenario row descriptors ──────────────────────────────────────────────
    ROWS = [
        dict(
            cls=IEEESingleSinWaveScenario,
            kw=dict(seed=42),
            label="(a) Clean Sinusoid",
            sub=r"$f_0=60\,\mathrm{Hz}$, $A=1\,\mathrm{pu}$ — noise-free baseline",
            vc="#1B5E20", fc="#1B5E20",
            ev=None,
            regs=[],
        ),
        dict(
            cls=IEEEMagStepScenario,
            kw=dict(noise_sigma=0.001, seed=42),
            label="(b) Amplitude Step (IEC 60255-118-1 Scen. A)",
            sub=r"$\Delta A=+10\,\%$ step at $t=0.5\,\mathrm{s}$; "
                r"$\sigma_n=10^{-3}\,\mathrm{pu}$",
            vc="#0D47A1", fc="#0D47A1",
            ev=0.5,
            regs=[(0.0, 0.5, "Pre-step",  "#43A047", 0.06),
                  (0.5, 1.5, "Post-step", "#FB8C00", 0.06)],
        ),
        dict(
            cls=IEEEFreqRampScenario,
            kw=dict(rocof_hz_s=3.0, t_start_s=0.3, freq_cap_hz=61.5,
                    noise_sigma=0.001, seed=42),
            label="(c) Frequency Ramp (IEEE 1547-2018 Cat. III)",
            sub=r"RoCoF $=+3\,\mathrm{Hz/s}$, onset $t=0.3\,\mathrm{s}$, "
                r"cap at $61.5\,\mathrm{Hz}$",
            vc="#E65100", fc="#BF360C",
            ev=0.3,
            regs=[(0.0, 0.3,  "Nominal",    "#43A047", 0.06),
                  (0.3, 0.8,  "RoCoF ramp", "#FB8C00", 0.06),
                  (0.8, 1.5,  "Cap hold",   "#C62828", 0.06)],
        ),
        dict(
            cls=IEEEModulationScenario,
            kw=dict(kx=0.1, ka=0.1, fm_hz=2.0, noise_sigma=0.001, seed=42),
            label="(d) AM/PM Modulation (IEC 60255-118-1 Bandwidth Test)",
            sub=r"$k_x=10\,\%$, $k_a=0.1\,\mathrm{rad}$, "
                r"$f_m=2\,\mathrm{Hz}$; "
                r"$f_\mathrm{true}(t)=f_0 - k_a f_m \sin(2\pi f_m t)$",
            vc="#6A1B9A", fc="#6A1B9A",
            ev=None,
            regs=[],
        ),
        dict(
            cls=IBRHarmonicsMediumScenario,
            kw=dict(seed=42),
            label="(e) IBR Harmonic Distortion (THD ≈ 8 %)",
            sub=r"3rd–13th harmonics + 75\,Hz inter-harmonic; "
                r"RoCoF $=-1\,\mathrm{Hz/s}$ at $t=0.8\,\mathrm{s}$",
            vc="#37474F", fc="#B71C1C",
            ev=0.8,
            regs=[(0.0, 0.8, "Nominal + harmonics", "#43A047", 0.05),
                  (0.8, 2.0, "RoCoF + harmonics",   "#FB8C00", 0.05)],
        ),
        dict(
            cls=IBRMultiEventScenario,
            kw=dict(seed=42),
            label="(f) IBR Multi-Event Composite Fault (worst-case)",
            sub=r"Sag $50\,\%$ + $\Delta\phi=45°$ + RoCoF $-2\,\mathrm{Hz/s}$ "
                r"+ DC offset + 5th/7th harmonics + multi-spectral noise",
            vc="#263238", fc="#B71C1C",
            ev=0.5,
            regs=[(0.0, 0.5, "Pre-fault",  "#43A047", 0.05),
                  (0.5, 1.5, "RoCoF ramp", "#FB8C00", 0.05),
                  (1.5, 5.0, "Nadir hold", "#C62828", 0.04)],
        ),
    ]

    nrows = len(ROWS)
    fig_h = _IEEE_PAGE_H * 0.70

    with plt.rc_context(_IEEE_RC):
        fig = plt.figure(figsize=(_IEEE_FULL_W, fig_h))
        gs = gridspec.GridSpec(
            nrows, 2,
            figure=fig,
            hspace=0.52,
            wspace=0.13,
            left=0.075, right=0.980,
            top=0.940, bottom=0.055,
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

            # ── Voltage panel ─────────────────────────────────────────────────
            ax_v.plot(t, v, lw=0.45, color=spec["vc"], rasterized=True)

            for r_lo, r_hi, _, rc, ra in spec["regs"]:
                ax_v.axvspan(r_lo, min(r_hi, float(t[-1])),
                             alpha=ra, color=rc, lw=0)
            if spec["ev"] is not None:
                ax_v.axvline(spec["ev"], color="#000", lw=0.75, ls="--",
                             alpha=0.55)

            ax_v.set_xlim(float(t[0]), float(t[-1]))
            v_pk = max(abs(float(v.min())), abs(float(v.max())))
            ax_v.set_ylim(-v_pk * 1.22, v_pk * 1.32)
            ax_v.set_ylabel(r"$v(t)$ [pu]", labelpad=1, fontsize=6.5)
            ax_v.set_title(spec["label"], fontsize=6.8, fontweight="bold", pad=2)
            ax_v.text(0.015, 0.96, spec["sub"],
                      transform=ax_v.transAxes, fontsize=5.0, va="top",
                      style="italic", color="#444444")

            # ── Frequency panel ───────────────────────────────────────────────
            ax_f.plot(t, f, lw=0.85, color=spec["fc"], rasterized=True,
                      label=r"$f_\mathrm{true}(t)$")
            ax_f.axhline(f_nom, color="#555", lw=0.5, ls=":",
                         alpha=0.7, label=f"$f_0={f_nom:.0f}$ Hz")

            for r_lo, r_hi, _, rc, ra in spec["regs"]:
                ax_f.axvspan(r_lo, min(r_hi, float(t[-1])),
                             alpha=ra, color=rc, lw=0)
            if spec["ev"] is not None:
                ax_f.axvline(spec["ev"], color="#000", lw=0.75, ls="--",
                             alpha=0.55)

            ax_f.set_xlim(float(t[0]), float(t[-1]))
            f_rng = max(float(np.ptp(f)), 0.04)
            ax_f.set_ylim(float(f.min()) - 0.22 * f_rng,
                          float(f.max()) + 0.38 * f_rng)
            ax_f.set_ylabel(r"$f(t)$ [Hz]", labelpad=1, fontsize=6.5)
            ax_f.legend(loc="upper right", fontsize=5.5, ncol=2,
                        handlelength=1.0, borderpad=0.3, labelspacing=0.2)

            # x-label only on last row
            if ridx == nrows - 1:
                ax_v.set_xlabel("Time [s]", labelpad=1)
                ax_f.set_xlabel("Time [s]", labelpad=1)
            else:
                ax_v.tick_params(labelbottom=False)
                ax_f.tick_params(labelbottom=False)

        # ── Column headers ────────────────────────────────────────────────────
        fig.text(0.300, 0.974, "Voltage Waveform",
                 ha="center", va="top", fontsize=7.5, fontweight="bold",
                 color="#222")
        fig.text(0.730, 0.974, "Instantaneous Frequency",
                 ha="center", va="top", fontsize=7.5, fontweight="bold",
                 color="#222")
        fig.text(0.530, 0.993,
                 "IBR Frequency Estimator Benchmark — Scenario Signal Overview",
                 ha="center", va="top", fontsize=8, fontweight="bold",
                 color="#111")

        out = out_dir / "megadashboard1.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
    print(f"    megadashboard1 saved → {out.relative_to(ROOT)}")


# ═══════════════════════════════════════════════════════════════════════════════
# Mega Dashboard 2 — Estimator Performance Summary (IEEE 2-column full-width)
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_megadashboard2(df_global: pd.DataFrame) -> None:
    """
    Estimator benchmark performance summary for an IEEE SGSMA-PES conference paper.

    IEEE 2-column full-width, 70 % page height. 4 rows × 2 columns.

    Layout:
      (0,0) RMSE heatmap (estimator × scenario, log colour scale)
      (0,1) Trip-risk heatmap  T_trip [s]
      (1,0) Accuracy–latency scatter  CPU vs RMSE (log–log, coloured by family)
      (1,1) Max-peak-error heatmap
      (2,0) RMSE per scenario — grouped bar chart (estimator series, log y)
      (2,1) Average settling time — horizontal bar (sorted, family colour)
      (3,0) Global RMSE ranking — sorted horizontal bar
      (3,1) Composite performance score (RMSE + Peak + Trip + CPU normalised)

    Output: ``outputs/megadashboard2.png``
    """
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LogNorm

    out_dir = BASE_RESULTS_DIR
    print("\n  [MEGA2] Building estimator performance mega-dashboard ...")

    # ── Helpers ───────────────────────────────────────────────────────────────
    def safe_pivot(col: str) -> "pd.DataFrame | None":
        if col not in df_global.columns:
            return None
        piv = df_global.pivot_table(
            index="estimator", columns="scenario", values=col, aggfunc="mean"
        )
        return piv if not piv.empty else None

    def fam_clr(est: str) -> str:
        return _FAMILY_PALETTE.get(
            _ESTIMATOR_FAMILIES.get(est, "Adaptive"), "#757575"
        )

    def sc_short(name: str) -> str:
        return (name
                .replace("IEEE_Single_SinWave", "SinWave")
                .replace("IEEE_Mag_Step",       "MagStep")
                .replace("IEEE_Freq_Ramp",       "FreqRamp")
                .replace("IEEE_Freq_Step",       "FreqStep")
                .replace("IEEE_Modulation",      "Modul.")
                .replace("IEEE_OOB_Interference","OOB")
                .replace("IEEE_Phase_Jump_20",   "PJ20°")
                .replace("IEEE_Phase_Jump_60",   "PJ60°")
                .replace("NERC_Phase_Jump_60",   "NERC\nPJ60°")
                .replace("IBR_Multi_Event",      "IBR\nMulti")
                .replace("IBR_Power_Imbalance_Ringdown", "IBR\nRing.")
                .replace("IBR_Harmonics_Small",  "Harm\nSm")
                .replace("IBR_Harmonics_Medium", "Harm\nMed")
                .replace("IBR_Harmonics_Large",  "Harm\nLg"))

    def avg_col(col: str) -> "pd.Series":
        if col not in df_global.columns:
            return pd.Series(dtype=float)
        return df_global.groupby("estimator")[col].mean()

    piv_rmse  = safe_pivot("m1_rmse_hz_mean")
    piv_trip  = safe_pivot("m5_trip_risk_s_mean")
    piv_peak  = safe_pivot("m3_max_peak_hz_mean")

    avg_rmse   = avg_col("m1_rmse_hz_mean").sort_values()
    avg_cpu    = avg_col("m13_cpu_time_us_mean")
    avg_settle = avg_col("m8_settling_time_s_mean").sort_values()

    # ── Helpers for heatmap rendering ─────────────────────────────────────────
    def _draw_heatmap(ax, piv, cmap, label, annotate=True):
        if piv is None or piv.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", fontsize=7, color="#888")
            return
        data = piv.values.astype(float)
        finite = data[np.isfinite(data) & (data > 0)]
        if len(finite) == 0:
            return
        vmin = max(finite.min() / 2.0, 1e-5)
        vmax = min(finite.max() * 2.0, 50.0)
        clipped = np.clip(data, vmin, vmax)
        im = ax.imshow(clipped, aspect="auto", cmap=cmap,
                       norm=LogNorm(vmin=vmin, vmax=vmax))
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
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                                fontsize=3.8, color=tc, fontweight="bold")

    fig_h = _IEEE_PAGE_H * 0.70

    with plt.rc_context(_IEEE_RC):
        fig = plt.figure(figsize=(_IEEE_FULL_W, fig_h))
        gs = gridspec.GridSpec(
            4, 2,
            figure=fig,
            hspace=0.78,
            wspace=0.24,
            left=0.08, right=0.98,
            top=0.930, bottom=0.055,
        )

        # ── (0,0) RMSE heatmap ───────────────────────────────────────────────
        ax00 = fig.add_subplot(gs[0, 0])
        _draw_heatmap(ax00, piv_rmse, "RdYlGn_r", "RMSE [Hz]")
        ax00.set_title("RMSE [Hz] — Scenario × Estimator",
                       fontsize=7, fontweight="bold", pad=3)

        # ── (0,1) Trip-risk heatmap ──────────────────────────────────────────
        ax01 = fig.add_subplot(gs[0, 1])
        _draw_heatmap(ax01, piv_trip, "YlOrRd",
                      r"$T_\mathrm{trip}$ [s]")
        ax01.set_title(r"Trip Risk $T_\mathrm{trip}$ [s]",
                       fontsize=7, fontweight="bold", pad=3)

        # ── (1,0) CPU vs RMSE scatter (log–log) ─────────────────────────────
        ax10 = fig.add_subplot(gs[1, 0])
        if not avg_rmse.empty and not avg_cpu.empty:
            common = avg_rmse.index.intersection(avg_cpu.index)
            seen_fam: set[str] = set()
            for est in common:
                fam = _ESTIMATOR_FAMILIES.get(est, "Adaptive")
                clr = _FAMILY_PALETTE.get(fam, "#757575")
                lbl = fam if fam not in seen_fam else "_"
                seen_fam.add(fam)
                ax10.scatter(avg_cpu[est], avg_rmse[est],
                             s=22, color=clr, edgecolors="white", lw=0.4,
                             zorder=4, label=lbl)
                ax10.annotate(est, (avg_cpu[est], avg_rmse[est]),
                              xytext=(3, 2), textcoords="offset points",
                              fontsize=4.0, color=clr)
            ax10.axhline(0.1, color="#B71C1C", ls="--", lw=0.7, alpha=0.75,
                         label="IEC 100 mHz")
            ax10.set_xscale("log")
            ax10.set_yscale("log")
            ax10.set_xlabel(r"Avg CPU [$\mu$s / sample]", labelpad=1)
            ax10.set_ylabel("Avg RMSE [Hz]", labelpad=1)
            ax10.legend(loc="upper left", fontsize=5.5, ncol=1,
                        handlelength=0.9, borderpad=0.3)
        ax10.set_title("Accuracy–Latency Trade-off (mean across all scenarios)",
                       fontsize=7, fontweight="bold", pad=3)

        # ── (1,1) Max-peak heatmap ───────────────────────────────────────────
        ax11 = fig.add_subplot(gs[1, 1])
        _draw_heatmap(ax11, piv_peak, "RdPu",
                      "Max Peak [Hz]", annotate=False)
        ax11.set_title("Max Peak Error [Hz]",
                       fontsize=7, fontweight="bold", pad=3)

        # ── (2,0) RMSE grouped bar per scenario ──────────────────────────────
        ax20 = fig.add_subplot(gs[2, 0])
        rmse_col = "m1_rmse_hz_mean"
        if rmse_col in df_global.columns:
            sc_list = sorted(df_global["scenario"].unique())
            est_list = (
                df_global.groupby("estimator")[rmse_col]
                .mean().sort_values().index.tolist()
            )
            x20 = np.arange(len(sc_list))
            w = max(0.05, 0.72 / max(len(est_list), 1))
            cmap20 = plt.get_cmap("tab20", len(est_list))
            for i, est in enumerate(est_list):
                vals = []
                for sc in sc_list:
                    sub = df_global[
                        (df_global["estimator"] == est) &
                        (df_global["scenario"] == sc)
                    ]
                    vals.append(float(sub[rmse_col].mean())
                                if len(sub) else np.nan)
                offset = (i - len(est_list) / 2.0 + 0.5) * w
                ax20.bar(x20 + offset, vals, width=w * 0.88,
                         color=cmap20(i), label=est, alpha=0.87,
                         edgecolor="none")
            ax20.set_xticks(x20)
            ax20.set_xticklabels([sc_short(s) for s in sc_list],
                                   rotation=30, ha="right", fontsize=4.8)
            ax20.set_yscale("log")
            ax20.set_ylabel("RMSE [Hz]", labelpad=1)
            ax20.legend(loc="upper right", fontsize=3.8, ncol=2,
                        handlelength=0.6, borderpad=0.2, labelspacing=0.12)
        ax20.set_title("RMSE by Estimator per Scenario",
                       fontsize=7, fontweight="bold", pad=3)

        # ── (2,1) Settling time horizontal bar ───────────────────────────────
        ax21 = fig.add_subplot(gs[2, 1])
        if not avg_settle.empty:
            clrs21 = [fam_clr(e) for e in avg_settle.index]
            ax21.barh(range(len(avg_settle)), avg_settle.values,
                      color=clrs21, edgecolor="none", height=0.65)
            ax21.set_yticks(range(len(avg_settle)))
            ax21.set_yticklabels(avg_settle.index, fontsize=5.2)
            ax21.set_xlabel("Avg Settling Time [s]", labelpad=1)
            ax21.invert_yaxis()
        else:
            ax21.text(0.5, 0.5, "No data", transform=ax21.transAxes,
                      ha="center", fontsize=7, color="#888")
        for fam, clr in _FAMILY_PALETTE.items():
            ax21.barh(0, 0, color=clr, label=fam, height=0)
        ax21.legend(loc="lower right", fontsize=5, ncol=1,
                    handlelength=0.7, borderpad=0.3)
        ax21.set_title("Average Settling Time by Estimator",
                       fontsize=7, fontweight="bold", pad=3)

        # ── (3,0) Global RMSE ranking ────────────────────────────────────────
        ax30 = fig.add_subplot(gs[3, 0])
        if not avg_rmse.empty:
            clrs30 = [fam_clr(e) for e in avg_rmse.index]
            ax30.barh(range(len(avg_rmse)), avg_rmse.values,
                      color=clrs30, edgecolor="none", height=0.65)
            ax30.set_yticks(range(len(avg_rmse)))
            ax30.set_yticklabels(avg_rmse.index, fontsize=5.2)
            ax30.set_xscale("log")
            ax30.set_xlabel("Mean RMSE [Hz] — log scale", labelpad=1)
            ax30.axvline(0.1, color="#B71C1C", ls="--", lw=0.7, alpha=0.8)
            ax30.text(0.107, len(avg_rmse) - 0.5, "IEC\n100 mHz",
                      fontsize=4.0, color="#B71C1C", va="top")
            ax30.invert_yaxis()
            for i, (est, val) in enumerate(avg_rmse.items()):
                if np.isfinite(val):
                    ax30.text(val * 1.07, i, f"{val:.3f}", va="center",
                              fontsize=4.0, color="#333")
        for fam, clr in _FAMILY_PALETTE.items():
            ax30.barh(0, 0, color=clr, label=fam, height=0)
        ax30.legend(loc="lower right", fontsize=5, ncol=1,
                    handlelength=0.7, borderpad=0.3)
        ax30.set_title("Estimator Ranking — Mean RMSE (all scenarios)",
                       fontsize=7, fontweight="bold", pad=3)

        # ── (3,1) Composite performance score ────────────────────────────────
        ax31 = fig.add_subplot(gs[3, 1])
        score_cols = [c for c in (
            "m1_rmse_hz_mean", "m3_max_peak_hz_mean",
            "m5_trip_risk_s_mean", "m13_cpu_time_us_mean",
        ) if c in df_global.columns]
        if score_cols:
            agg = df_global.groupby("estimator")[score_cols].mean()
            normed = pd.DataFrame(index=agg.index)
            for c in score_cols:
                mn, mx = agg[c].min(), agg[c].max()
                normed[c] = (agg[c] - mn) / (mx - mn + 1e-12)
            score = normed.mean(axis=1).sort_values()
            clrs31 = [fam_clr(e) for e in score.index]
            ax31.barh(range(len(score)), score.values,
                      color=clrs31, edgecolor="none", height=0.65)
            ax31.set_yticks(range(len(score)))
            ax31.set_yticklabels(score.index, fontsize=5.2)
            ax31.set_xlabel("Composite Score (lower = better)", labelpad=1)
            ax31.invert_yaxis()
            for i, val in enumerate(score.values):
                ax31.text(val + 0.005, i, f"#{i+1}", va="center",
                          fontsize=4.0, color="#333")
            ax31.text(0.98, 0.02,
                      "Normalised: RMSE + Max Peak + Trip Risk + CPU\n"
                      "(each metric → [0, 1], mean = composite score)",
                      transform=ax31.transAxes, fontsize=4.5,
                      va="bottom", ha="right", style="italic", color="#555")
        else:
            ax31.text(0.5, 0.5, "No data", transform=ax31.transAxes,
                      ha="center", fontsize=7, color="#888")
        for fam, clr in _FAMILY_PALETTE.items():
            ax31.barh(0, 0, color=clr, label=fam, height=0)
        ax31.legend(loc="lower right", fontsize=5, ncol=1,
                    handlelength=0.7, borderpad=0.3)
        ax31.set_title("Composite Performance Score",
                       fontsize=7, fontweight="bold", pad=3)

        # ── Figure-level metadata text ────────────────────────────────────────
        fig.text(0.5, 0.962,
                 "IBR Frequency Estimator Benchmark — Performance Analysis",
                 ha="center", va="top", fontsize=8, fontweight="bold")
        fig.text(0.5, 0.946,
                 rf"$N_\mathrm{{MC}}={N_MC_RUNS}$ runs per estimator/scenario  |  "
                 rf"$f_s=10\,\mathrm{{kHz}}$  |  "
                 rf"dual-rate physics simulation (1 MHz $\rightarrow$ 10 kHz)",
                 ha="center", va="top", fontsize=5.5, color="#555")

        out = out_dir / "megadashboard2.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
    print(f"    megadashboard2 saved → {out.relative_to(ROOT)}")


# ═══════════════════════════════════════════════════════════════════════════════
# Pytest entry points (fast infrastructure checks, not full benchmark)
# ═══════════════════════════════════════════════════════════════════════════════

def test_estimators_load() -> None:
    """All canonical estimators must import and expose a 'name' attribute."""
    estimators = load_estimators()
    assert len(estimators) > 0, "No estimators loaded"
    for label, cls in estimators.items():
        assert hasattr(cls, "name"), f"{cls.__name__} missing class attribute 'name'"


def test_search_spaces_valid() -> None:
    """Every SEARCH_SPACES key must use only real __init__ parameter names."""
    estimators = load_estimators()
    validate_search_spaces(estimators)  # raises ValueError on any mismatch


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


# ═══════════════════════════════════════════════════════════════════════════════
# Script entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t_start = time.time()
    BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading estimators ...")
    estimators = load_estimators()
    print(f"  Loaded: {list(estimators)}")

    print("Validating search spaces ...")
    validate_search_spaces(estimators)
    print("  OK")

    run_phase_1(estimators)
    run_phase_2()
    run_phase_3()

    elapsed = (time.time() - t_start) / 60
    print(f"\n[DONE] Benchmark completed in {elapsed:.1f} min.")
