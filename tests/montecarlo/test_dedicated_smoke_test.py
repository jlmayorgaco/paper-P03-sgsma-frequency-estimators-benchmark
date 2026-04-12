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

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Run configuration ──────────────────────────────────────────────────────────
N_TRIALS_TUNING = 10
N_MC_RUNS = 3
BASE_RESULTS_DIR = ROOT / "tests" / "montecarlo" / "outputs"

SCENARIOS = [
    IEEESingleSinWaveScenario,
    IEEEFreqStepScenario,
    IEEEMagStepScenario,
    IEEEFreqRampScenario,
    IBRMultiEventScenario,
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

# ── Search spaces ──────────────────────────────────────────────────────────────
# Each lambda receives an optuna.Trial and returns a dict whose keys MUST match
# the corresponding estimator's __init__ parameter names exactly.
# Wrong names are caught at startup by _validate_search_spaces().
SEARCH_SPACES: dict[str, Any] = {
    "PLL": lambda trial: {
        "kp_scale": trial.suggest_float("kp_scale", 0.001, 1.5, log=True),
        "ki_scale": trial.suggest_float("ki_scale", 0.001, 1.5, log=True),
    },
    "SOGI-PLL": lambda trial: {
        "settle_time": trial.suggest_float("settle_time", 0.01, 0.25),
        "k_sogi": trial.suggest_float("k_sogi", 0.5, 2.0),
    },
    "EKF": lambda trial: {
        "q_omega": trial.suggest_float("q_omega", 1e-6, 1e-1, log=True),
        "r_meas": trial.suggest_float("r_meas", 1e-5, 1.0, log=True),
    },
    "RA-EKF": lambda trial: {
        "q_rocof": trial.suggest_float("q_rocof", 1e-7, 1e-2, log=True),
        "r_meas": trial.suggest_float("r_meas", 1e-5, 1.0, log=True),
    },
    "RLS": lambda trial: {
        "lambda_fixed": trial.suggest_float("lambda_fixed", 0.940, 0.999),
    },
    "IPDFT": lambda trial: {
        "cycles": trial.suggest_float("cycles", 2.0, 8.0),
    },
    "TFT": lambda trial: {
        "n_cycles": trial.suggest_float("n_cycles", 2.0, 6.0),
    },
    "Prony": lambda trial: {
        "order": trial.suggest_int("order", 2, 6),
        "n_cycles": trial.suggest_float("n_cycles", 1.0, 4.0),
    },
    "ESPRIT": lambda trial: {
        "n_cycles": trial.suggest_float("n_cycles", 1.0, 4.0),
    },
    "Koopman": lambda trial: {
        "n_cycles": trial.suggest_float("n_cycles", 1.0, 4.0),
    },
    "TKEO": lambda trial: {
        "output_smoothing": trial.suggest_float("output_smoothing", 0.001, 0.1),
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
    candidates = [
        ("zcd",       "ZCDEstimator"),
        ("ipdft",     "IPDFT_Estimator"),
        ("tft",       "TFT_Estimator"),
        ("prony",     "Prony_Estimator"),
        ("esprit",    "ESPRIT_Estimator"),
        ("koopman",   "Koopman_Estimator"),
        ("pll",       "PLL_Estimator"),
        ("rls",       "RLS_Estimator"),
        ("sogi_pll",  "SOGIPLLEstimator"),
        ("sogi_fll",  "SOGI_FLL_Estimator"),
        ("ekf",       "EKF_Estimator"),
        ("ra_ekf",    "RAEKF_Estimator"),
        ("tkeo",      "TKEO_Estimator"),
    ]
    registry: dict[str, type] = {}
    for module_name, class_name in candidates:
        try:
            module = __import__(f"estimators.{module_name}", fromlist=[class_name])
            cls = getattr(module, class_name)
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

    sc = scenario_cls.run(duration_s=1.5, noise_sigma=0.001, seed=42)
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
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(sc.t, sc.f_true, "k--", alpha=0.6, linewidth=1.2, label="True")
        ax.plot(sc.t, f_hat, "r-", linewidth=1.0, label=f"Estimated ({est_name})")
        ax.set_ylim(sc.f_true.min() - margin, sc.f_true.max() + margin)
        ax.set_title(f"Tracking: {est_name}  |  {sc_name}")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"tracking_{est_name}.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"      [!] Tracking plot failed for {est_name}: {exc}")


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

        sc_base = sc_cls.run(duration_s=1.5, noise_sigma=0.0, seed=42)
        _save_scenario_artifacts(sc_base, sc_dir, sc_name)

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
    """Generate per-scenario bar charts and a global CPU-vs-RMSE scatter plot."""
    print("\n>>> PHASE 3: DASHBOARDS <<<")

    report_path = BASE_RESULTS_DIR / "global_metrics_report.csv"
    if not report_path.exists():
        print("  [!] global_metrics_report.csv missing — run Phase 2 first.")
        return

    df_global = pd.read_csv(report_path)
    _plot_per_scenario_dashboards(df_global)
    _plot_global_tradeoff(df_global)


def _plot_per_scenario_dashboards(df_global: pd.DataFrame) -> None:
    panel_metrics = [
        ("m1_rmse_hz_mean",         "RMSE [Hz]",        "log"),
        ("m3_max_peak_hz_mean",      "Max Peak [Hz]",    "log"),
        ("m8_settling_time_s_mean",  "Settling Time [s]","linear"),
        ("m13_cpu_time_us_mean",     "CPU Time [µs]",    "linear"),
    ]

    for sc_name in df_global["scenario"].unique():
        df_sc = (
            df_global[df_global["scenario"] == sc_name]
            .sort_values("m1_rmse_hz_mean")
            .reset_index(drop=True)
        )
        sc_dir = BASE_RESULTS_DIR / sc_name
        available = [(col, title, scale) for col, title, scale in panel_metrics if col in df_sc.columns]
        if not available:
            continue

        n_panels = len(available)
        ncols = 2
        nrows = (n_panels + 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
        axes_flat = np.array(axes).flatten()
        x = np.arange(len(df_sc))
        colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(df_sc)))

        for idx, (col, title, scale) in enumerate(available):
            ax = axes_flat[idx]
            ax.bar(x, df_sc[col], color=colors, edgecolor="black", alpha=0.85)
            ax.set_title(title)
            ax.set_yscale(scale)
            ax.set_xticks(x)
            ax.set_xticklabels(df_sc["estimator"], rotation=45, ha="right")
            ax.grid(True, alpha=0.2)

        for idx in range(len(available), len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig.suptitle(f"Performance Dashboard — {sc_name}", fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(sc_dir / "dashboard.png", dpi=150)
        plt.close(fig)
        print(f"    Dashboard saved → {(sc_dir / 'dashboard.png').relative_to(ROOT)}")


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
