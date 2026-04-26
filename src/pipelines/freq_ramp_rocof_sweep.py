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
import numpy as np
import optuna
import pandas as pd

# Keep runtime stable for multiprocessing + BLAS backends.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BENCHMARK_INCLUDE_EXPERIMENTAL", "0")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

# Force this repository to win module resolution over sibling repos
# that also expose top-level packages like `estimators`.
for p in (str(SRC), str(ROOT)):
    while p in sys.path:
        sys.path.remove(p)
sys.path.insert(0, str(SRC))
sys.path.insert(1, str(ROOT))

from analysis.monte_carlo_engine import MonteCarloEngine, MonteCarloResult
from pipelines.benchmark_definition import ESTIMATOR_FAMILIES, load_active_estimators
import pipelines.full_mc_benchmark as benchmark
from scenarios.ieee_freq_ramp import IEEEFreqRampScenario


OUTPUT_DIR = ROOT / "artifacts" / "freq_ramp_rocof_sweep"
MANIFEST_NAME = "scenario_manifest.json"
GLOBAL_CSV_NAME = "global_metrics_report.csv"
RMSE_EST_CSV_NAME = "rmse_by_estimator.csv"
RMSE_FAM_CSV_NAME = "rmse_by_family.csv"
PLOT_NAME = "rmse_deterioration_by_family"
LEGEND_MAP_CSV_NAME = "rmse_plot_method_legend.csv"

ROCOF_MIN_HZ_S = 0.10
ROCOF_MAX_HZ_S = 50.0
ROCOF_ANCHORS_HZ_S: tuple[float, ...] = (
    0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.60, 0.75, 0.90, 1.00,
    1.15, 1.25, 1.40, 1.50, 1.65, 1.75, 2.00, 2.25, 2.50, 2.75,
    3.00, 4.00, 5.00, 6.00, 7.50, 10.00, 12.00, 15.00, 25.00, 50.00,
)
IEEE_STANDARD_NORMAL_ROCOF_HZ_S = 3.0
IEEE_STANDARD_IBR_ROCOF_HZ_S = 5.0
PROPOSED_IBR_ROCOF_HZ_S = 10.0

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
    rocof_hz_s: float
    scenario_cls: type
    scenario_name: str


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


def _sanitize_token(value: float) -> str:
    token = f"{value:g}".replace(".", "p")
    return token.replace("-", "m")


def _create_rocof_variant(rocof_hz_s: float) -> SweepScenario:
    token = _sanitize_token(rocof_hz_s)
    scenario_name = f"Sweep_FreqRamp_RoCoF_{token}Hzs"
    class_name = f"SweepFreqRampRoCoF{token}"

    attrs = {
        "SCENARIO_NAME": scenario_name,
        "DEFAULT_PARAMS": {
            **IEEEFreqRampScenario.DEFAULT_PARAMS,
            "duration_s": 1.8,
            "rocof_hz_s": float(rocof_hz_s),
            "t_start_s": 0.30,
            "freq_cap_hz": 60.0 + (1.2 * float(rocof_hz_s)),
            "noise_sigma": 0.001,
        },
        "MONTE_CARLO_SPACE": {
            "phase_rad": {"kind": "uniform", "low": 0.0, "high": 2.0 * math.pi},
            "noise_sigma": {"kind": "uniform", "low": 0.0005, "high": 0.0020},
        },
        "ROCOF_SWEEP_VALUE": float(rocof_hz_s),
        "get_name": classmethod(lambda cls: cls.SCENARIO_NAME),
    }

    new_cls = type(class_name, (IEEEFreqRampScenario,), attrs)
    new_cls.__module__ = __name__
    globals()[class_name] = new_cls
    return SweepScenario(rocof_hz_s=float(rocof_hz_s), scenario_cls=new_cls, scenario_name=scenario_name)


def _build_rocof_levels() -> list[float]:
    levels = sorted({
        round(float(v), 6)
        for v in ROCOF_ANCHORS_HZ_S
        if ROCOF_MIN_HZ_S <= float(v) <= ROCOF_MAX_HZ_S
    })
    return levels


def _build_scenarios() -> list[SweepScenario]:
    return [_create_rocof_variant(v) for v in _build_rocof_levels()]


def _select_estimators() -> dict[str, type]:
    estimators = load_active_estimators()

    include_raw = os.getenv("FREQRAMP_SWEEP_INCLUDE_ESTIMATORS", "").strip()
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
        valid = series.dropna()
        if valid.empty:
            continue
        row[f"{metric}_mean"] = float(valid.mean())
        row[f"{metric}_std"] = float(valid.std(ddof=1)) if len(valid) > 1 else 0.0
    return row


def _run_engine_local(engine: MonteCarloEngine) -> MonteCarloResult:
    summary_rows: list[dict[str, Any]] = []
    signal_dfs: list[pd.DataFrame] = []
    for run_idx in range(engine.n_runs):
        row, signal_df = engine.run_once(run_idx)
        summary_rows.append(row)
        signal_dfs.append(signal_df)

    summary_df = pd.DataFrame(summary_rows).sort_values(by="run_idx").reset_index(drop=True)
    signals_df = pd.concat(signal_dfs, ignore_index=True).sort_values(by=["run_idx", "t_s"]).reset_index(drop=True)
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


def _tune_estimator_for_scenario(
    est_name: str,
    est_cls: type,
    scenario_cls: type,
    *,
    n_trials: int,
    tune_eval_runs: int,
    base_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    defaults: dict[str, Any] = est_cls.default_params() if hasattr(est_cls, "default_params") else {}
    tuning_meta: dict[str, Any] = {
        "mode": "per_scenario_best_tuning",
        "objective": "minimize_m1_rmse_hz",
        "n_trials_requested": int(n_trials),
        "n_trials_executed": 0,
        "tune_eval_runs": int(max(1, tune_eval_runs)),
        "sampler_mode_effective": None,
        "best_objective": None,
    }

    if est_name not in benchmark.SEARCH_SPACES:
        tuning_meta["reason"] = "no_search_space"
        return defaults, tuning_meta
    if n_trials <= 0:
        tuning_meta["reason"] = "n_trials<=0"
        return defaults, tuning_meta

    space_fn = benchmark.SEARCH_SPACES[est_name]
    if not benchmark._grid_space_for_estimator(space_fn, n_trials=2):
        tuning_meta["reason"] = "empty_search_space"
        return defaults, tuning_meta

    scenarios_eval = [scenario_cls.run(seed=base_seed + i) for i in range(max(1, tune_eval_runs))]
    if not scenarios_eval:
        tuning_meta["reason"] = "no_eval_scenarios"
        return defaults, tuning_meta

    fs_dsp = 1.0 / float(scenarios_eval[0].t[1] - scenarios_eval[0].t[0])
    eval_start = int(0.15 * fs_dsp)

    def objective(trial: optuna.Trial) -> float:
        suggested = space_fn(trial)
        params = {**defaults, **suggested}
        try:
            rmses: list[float] = []
            for sc in scenarios_eval:
                est = est_cls(**params)
                f_hat = benchmark._run_estimator(est, sc.v)
                error = f_hat[eval_start:] - sc.f_true[eval_start:]
                rmse = float(np.sqrt(np.mean(error ** 2)))
                if not np.isfinite(rmse):
                    return 1e6
                rmses.append(rmse)
            return float(np.mean(rmses)) if rmses else 1e6
        except Exception:
            return 1e6

    study, n_trials_exec, sampler_mode_effective = benchmark._build_optuna_study(
        space_fn=space_fn,
        n_trials=int(n_trials),
    )
    tuning_meta["n_trials_executed"] = int(n_trials_exec)
    tuning_meta["sampler_mode_effective"] = sampler_mode_effective

    study.optimize(objective, n_trials=n_trials_exec)
    if study.best_value >= 1e6:
        tuning_meta["reason"] = "all_trials_failed"
        return defaults, tuning_meta

    best_suggested = space_fn(study.best_trial)
    tuning_meta["best_objective"] = float(study.best_value)
    return {**defaults, **best_suggested}, tuning_meta


def _plot_rmse_by_family(df_rmse: pd.DataFrame, out_dir: Path) -> tuple[list[Path], dict[str, tuple[float, float, float, float]]]:
    if df_rmse.empty:
        return [], {}
    families = ["Loop-based", "Model-based", "Window-based", "Adaptive", "Data-driven"]
    panels = ["Reference Ramp"] + families
    rocof_ticks = sorted(df_rmse["rocof_hz_s"].dropna().astype(float).unique().tolist())

    ieee_normal = _env_float("FREQRAMP_LINE_IEEE_NORMAL", IEEE_STANDARD_NORMAL_ROCOF_HZ_S, minimum=0.0)
    ieee_ibr = _env_float("FREQRAMP_LINE_IEEE_IBR", IEEE_STANDARD_IBR_ROCOF_HZ_S, minimum=0.0)
    proposed_ibr = _env_float("FREQRAMP_LINE_PROPOSED_IBR", PROPOSED_IBR_ROCOF_HZ_S, minimum=0.0)
    reference_ramp = _env_float("FREQRAMP_REFERENCE_ROCOF", ieee_normal, minimum=0.0)

    ncols = 2
    nrows = int(math.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12.0, 3.6 * nrows), sharex=False)
    axes_arr = np.atleast_1d(axes).ravel()

    cmap = matplotlib.colormaps["tab20"]
    est_labels = sorted(df_rmse["estimator"].unique().tolist())
    color_map = {label: cmap(i % cmap.N) for i, label in enumerate(est_labels)}

    for idx, panel in enumerate(panels):
        ax = axes_arr[idx]
        if panel == "Reference Ramp":
            sc = IEEEFreqRampScenario.run(
                duration_s=1.8,
                rocof_hz_s=reference_ramp,
                t_start_s=0.30,
                freq_cap_hz=60.0 + (1.2 * reference_ramp),
                noise_sigma=0.0,
                seed=0,
            )
            ax.plot(sc.t, sc.f_true, color="#111111", linewidth=1.4, label=f"Reference RoCoF={reference_ramp:g} Hz/s")
            ax.set_title("Reference Ramp", loc="left", fontweight="bold")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Frequency [Hz]")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(loc="best", fontsize=7, frameon=True)
            continue

        family = panel
        df_family = df_rmse[df_rmse["family"] == family].sort_values(["estimator", "rocof_hz_s"])
        if df_family.empty:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.25)
            ax.set_title(f"{family} (sin datos)", loc="left", fontweight="bold")
            ax.set_ylabel("RMSE [Hz]")
            ax.set_xlabel("RoCoF [Hz/s]")
            if rocof_ticks:
                ax.set_xticks(rocof_ticks)
                ax.set_xticklabels([f"{v:g}" for v in rocof_ticks], rotation=70, ha="right", fontsize=6)
            continue

        for estimator, df_est in df_family.groupby("estimator", sort=True):
            ax.plot(
                df_est["rocof_hz_s"].to_numpy(dtype=float),
                df_est["m1_rmse_hz_mean"].to_numpy(dtype=float),
                marker="o",
                markersize=2.8,
                linewidth=1.0,
                alpha=0.9,
                color=color_map[str(estimator)],
                label=str(estimator),
            )

        ax.axvline(ieee_normal, color="#303F9F", linestyle=":", linewidth=1.0, label=f"IEEE normal ({ieee_normal:g})")
        ax.axvline(ieee_ibr, color="#00897B", linestyle=":", linewidth=1.0, label=f"IEEE IBR ({ieee_ibr:g})")
        ax.axvline(proposed_ibr, color="#E65100", linestyle=":", linewidth=1.0, label=f"Propuesta IBR ({proposed_ibr:g})")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.set_title(family, loc="left", fontweight="bold")
        ax.set_ylabel("RMSE [Hz]")
        if rocof_ticks:
            ax.set_xticks(rocof_ticks)
            ax.set_xticklabels([f"{v:g}" for v in rocof_ticks], rotation=70, ha="right", fontsize=6)
        ax.legend(loc="best", fontsize=6.7, frameon=True, ncol=1)

    for j in range(len(panels), len(axes_arr)):
        axes_arr[j].set_visible(False)

    for ax in axes_arr[1:len(panels)]:
        ax.set_xlabel("RoCoF [Hz/s]")

    fig.suptitle("FreqRamp RoCoF sweep: RMSE deterioration by estimator family", fontsize=13, y=0.995)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.97])

    png_path = out_dir / f"{PLOT_NAME}.png"
    pdf_path = out_dir / f"{PLOT_NAME}.pdf"
    fig.savefig(png_path, dpi=240)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [png_path, pdf_path], color_map


def _write_manifest(out_dir: Path, scenarios: list[SweepScenario], estimators: dict[str, type], n_mc_runs: int) -> Path:
    payload = {
        "benchmark_identity": "freq_ramp_rocof_sweep_active_pipeline",
        "description": "Dedicated RoCoF sweep over IEEE_Freq_Ramp with dense schedule between 0.11 and 50 Hz/s.",
        "output_dir": str(out_dir),
        "n_scenarios": len(scenarios),
        "n_estimators": len(estimators),
        "n_mc_runs": int(n_mc_runs),
        "rocof_levels_hz_s": [float(s.rocof_hz_s) for s in scenarios],
        "estimators": list(estimators.keys()),
        "families": {label: ESTIMATOR_FAMILIES.get(label, "Unknown") for label in estimators},
        "artifacts": {
            "global_metrics": GLOBAL_CSV_NAME,
            "rmse_by_estimator": RMSE_EST_CSV_NAME,
            "rmse_by_family": RMSE_FAM_CSV_NAME,
            "plot_png": f"{PLOT_NAME}.png",
            "plot_pdf": f"{PLOT_NAME}.pdf",
        },
    }
    path = out_dir / MANIFEST_NAME
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> None:
    n_mc_runs = _env_int("FREQRAMP_SWEEP_N_MC_RUNS", _env_int("BENCHMARK_N_MC_RUNS", 20, minimum=1), minimum=1)
    base_seed = _env_int("FREQRAMP_SWEEP_BASE_SEED", 12345, minimum=0)
    resume_run = _env_bool("FREQRAMP_SWEEP_RESUME", True)
    n_cost_reps = _env_int("FREQRAMP_SWEEP_N_COST_REPS", 1, minimum=1)
    tune_trials = _env_int("FREQRAMP_SWEEP_TUNE_TRIALS", 60, minimum=0)
    tune_eval_runs = _env_int("FREQRAMP_SWEEP_TUNE_EVAL_RUNS", 2, minimum=1)

    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scenarios = _build_scenarios()
    estimators = _select_estimators()

    print(f"Running FreqRamp RoCoF sweep: {len(scenarios)} scenarios x {len(estimators)} estimators")
    print(f"  MC runs per pair: {n_mc_runs}")
    print(f"  CPU timing reps per pair: {n_cost_reps}")
    print(f"  Tuning trials per pair: {tune_trials}")
    print(f"  Tuning eval runs: {tune_eval_runs}")
    print(f"  Output dir: {OUTPUT_DIR}")

    rows_agg: list[dict[str, Any]] = []

    for sc in scenarios:
        print(f"\nScenario {sc.scenario_name} (RoCoF={sc.rocof_hz_s:g} Hz/s)")
        sc_dir = OUTPUT_DIR / sc.scenario_name
        sc_dir.mkdir(parents=True, exist_ok=True)

        for est_name, est_cls in estimators.items():
            out_dir = sc_dir / est_name
            out_dir.mkdir(parents=True, exist_ok=True)

            summary_csv = out_dir / f"{sc.scenario_name}__{est_name}_summary.csv"
            run_spec_path = out_dir / "run_spec.json"
            if resume_run and summary_csv.exists() and run_spec_path.exists():
                summary_df = pd.read_csv(summary_csv)
            else:
                print(f"  - {est_name}", flush=True)
                best_params, tuning_meta = _tune_estimator_for_scenario(
                    est_name=est_name,
                    est_cls=est_cls,
                    scenario_cls=sc.scenario_cls,
                    n_trials=tune_trials,
                    tune_eval_runs=tune_eval_runs,
                    base_seed=base_seed,
                )
                engine = MonteCarloEngine(
                    scenario_cls=sc.scenario_cls,
                    estimator_cls=est_cls,
                    estimator_params=best_params,
                    n_runs=n_mc_runs,
                    base_seed=base_seed,
                    n_cost_reps=n_cost_reps,
                )
                result = _run_engine_local(engine)
                result.summary_df.to_csv(summary_csv, index=False)
                summary_df = result.summary_df
                run_spec = {
                    "scenario": sc.scenario_name,
                    "rocof_hz_s": float(sc.rocof_hz_s),
                    "estimator": est_name,
                    "family": ESTIMATOR_FAMILIES.get(est_name, "Unknown"),
                    "best_params": benchmark._to_builtin(best_params),
                    "tuning_meta": benchmark._to_builtin(tuning_meta),
                    "n_mc_runs": int(n_mc_runs),
                }
                run_spec_path.write_text(
                    json.dumps(benchmark._to_builtin(run_spec), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

            agg = _aggregate_summary(summary_df)
            rows_agg.append(
                {
                    "scenario": sc.scenario_name,
                    "rocof_hz_s": float(sc.rocof_hz_s),
                    "estimator": est_name,
                    "family": ESTIMATOR_FAMILIES.get(est_name, "Unknown"),
                    "n_mc_runs": int(len(summary_df)),
                    **agg,
                }
            )

    df_global = pd.DataFrame(rows_agg).sort_values(["rocof_hz_s", "family", "estimator"])
    global_csv = OUTPUT_DIR / GLOBAL_CSV_NAME
    df_global.to_csv(global_csv, index=False)

    keep_cols = ["scenario", "rocof_hz_s", "estimator", "family", "n_mc_runs", "m1_rmse_hz_mean", "m1_rmse_hz_std"]
    rmse_cols = [col for col in keep_cols if col in df_global.columns]
    df_rmse = df_global[rmse_cols].copy()
    rmse_est_csv = OUTPUT_DIR / RMSE_EST_CSV_NAME
    df_rmse.to_csv(rmse_est_csv, index=False)

    df_rmse_family = (
        df_rmse.groupby(["rocof_hz_s", "family"], as_index=False).agg(
            family_rmse_mean=("m1_rmse_hz_mean", "mean"),
            family_rmse_std=("m1_rmse_hz_mean", "std"),
            family_rmse_min=("m1_rmse_hz_mean", "min"),
            family_rmse_max=("m1_rmse_hz_mean", "max"),
        )
        .sort_values(["rocof_hz_s", "family"])
    )
    rmse_family_csv = OUTPUT_DIR / RMSE_FAM_CSV_NAME
    df_rmse_family.to_csv(rmse_family_csv, index=False)

    generated_plots, color_map = _plot_rmse_by_family(df_rmse=df_rmse, out_dir=OUTPUT_DIR)
    legend_rows = []
    for estimator in sorted(color_map):
        rgba = color_map[estimator]
        legend_rows.append(
            {
                "estimator": estimator,
                "hex_color": matplotlib.colors.to_hex(rgba),
                "family": ESTIMATOR_FAMILIES.get(estimator, "Unknown"),
            }
        )
    legend_map_path = OUTPUT_DIR / LEGEND_MAP_CSV_NAME
    pd.DataFrame(legend_rows).to_csv(legend_map_path, index=False)
    manifest_path = _write_manifest(OUTPUT_DIR, scenarios, estimators, n_mc_runs)

    elapsed = (time.time() - t0) / 60.0
    print("\nArtifacts:")
    print(f"  - {global_csv.relative_to(ROOT)}")
    print(f"  - {rmse_est_csv.relative_to(ROOT)}")
    print(f"  - {rmse_family_csv.relative_to(ROOT)}")
    for path in generated_plots:
        print(f"  - {path.relative_to(ROOT)}")
    print(f"  - {legend_map_path.relative_to(ROOT)}")
    print(f"  - {manifest_path.relative_to(ROOT)}")
    print(f"\n[DONE] FreqRamp RoCoF sweep completed in {elapsed:.1f} min.")


if __name__ == "__main__":
    main()
