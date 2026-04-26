"""
Parametric family sweeps for next-run stress testing.

This runner is separate from the canonical full benchmark. It keeps the active
estimator registry, but replaces the scenario catalog with dense 30-point
parametric sweeps so each estimator's degradation can be inspected against one
dominant disturbance variable at a time.

Default optimizations versus the full benchmark:
1. Canonical active estimators only (`BENCHMARK_INCLUDE_EXPERIMENTAL=0`)
2. Family-level tuning reuse: one tuned anchor scenario per family
3. Summary CSVs always; large signals CSVs disabled by default

Usage
-----
Plan only:
    python -m pipelines.parametric_family_sweeps --plan-only

Run:
    python -m pipelines.parametric_family_sweeps
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

# Fast defaults for this runner. Users can override them from the shell.
os.environ.setdefault("BENCHMARK_INCLUDE_EXPERIMENTAL", "0")
os.environ.setdefault("BENCHMARK_N_TRIALS_TUNING", "60")
os.environ.setdefault("BENCHMARK_N_MC_RUNS", "20")
os.environ.setdefault("BENCHMARK_ADV_BOOTSTRAP_ITERS", "400")
os.environ.setdefault("BENCHMARK_APPLY_TRIAL_OVERRIDES", "0")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis.monte_carlo_engine import MonteCarloEngine
from analysis.parametric_sweep_analysis import build_parametric_regression_reports
import pipelines.full_mc_benchmark as benchmark
from pipelines.report_builder import build_benchmark_report
from plotting.benchmark.parametric_sweep_plots import generate_parametric_sweep_figures
from scenarios.ibr_harmonics_large import IBRHarmonicsLargeScenario
from scenarios.ibr_harmonics_medium import IBRHarmonicsMediumScenario
from scenarios.ibr_power_imbalance_ringdown import IBRPowerImbalanceRingdownScenario
from scenarios.ieee_freq_ramp import IEEEFreqRampScenario
from scenarios.ieee_mag_step import IEEEMagStepScenario
from scenarios.ieee_oob_interference import IEEEOOBInterferenceScenario
from scenarios.ieee_phase_jump_60 import IEEEPhaseJump60Scenario
from scenarios.ieee_single_sinwave import IEEESingleSinWaveScenario


SWEEP_BENCHMARK_IDENTITY = "parametric_family_sweeps_active_pipeline"
SWEEP_BENCHMARK_SCOPE = "Dense 30-point family sweeps built on the modular Monte Carlo benchmark."
SWEEP_AUTHORITY_STATEMENT = (
    "This run reuses the active modular estimator registry but replaces the canonical "
    "scenario catalog with dense parametric sweeps to study estimator degradation "
    "against one dominant variable per family."
)
SWEEP_ALIGNMENT_POLICY = "paper_follows_parametric_family_sweeps"
SWEEP_OUTPUT_DIR = ROOT / "artifacts" / "parametric_family_sweeps"
SWEEP_JSON_NAME = "parametric_family_sweeps_report.json"
SWEEP_MANIFEST_NAME = "scenario_sweep_manifest.json"
SWEEP_TUNING_CACHE_NAME = "family_tuning_cache.json"


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, value)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_csv(name: str) -> list[str]:
    raw = os.getenv(name)
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


SWEEP_POINTS = _env_int("PARAM_SWEEP_POINTS", 30, minimum=2)
SAVE_SIGNAL_CSV = _env_bool("PARAM_SWEEP_SAVE_SIGNAL_CSV", False)
SAVE_TRACKING_ALL = _env_bool("PARAM_SWEEP_SAVE_TRACKING_ALL", False)
SAVE_SCENARIO_PLOTS_ALL = _env_bool("PARAM_SWEEP_SAVE_SCENARIO_PLOTS_ALL", False)
RESUME_RUN = _env_bool("PARAM_SWEEP_RESUME", True)
INCLUDED_SWEEP_FAMILIES = set(item.lower() for item in _env_csv("PARAM_SWEEP_INCLUDE_FAMILIES"))


@dataclass(frozen=True)
class SweepScenario:
    scenario_name: str
    scenario_cls: type
    sweep_value: float
    sweep_value_label: str
    overrides: dict[str, Any]
    is_anchor: bool


@dataclass(frozen=True)
class SweepFamily:
    key: str
    display_name: str
    base_scenario_name: str
    sweep_key: str
    sweep_label: str
    unit: str
    xscale: str
    description: str
    scenarios: tuple[SweepScenario, ...]


def _safe_token(value: float, suffix: str) -> str:
    token = f"{value:.4f}".rstrip("0").rstrip(".")
    return token.replace(".", "p").replace("-", "m") + suffix


def _scenario_name(prefix: str, idx: int, token: str) -> str:
    return f"{prefix}_{idx:02d}_{token}"


def _register_variant(
    base_cls: type,
    class_name: str,
    scenario_name: str,
    overrides: dict[str, Any],
    family_key: str,
    sweep_key: str,
    sweep_label: str,
    sweep_value: float,
    unit: str,
) -> type:
    attrs = {
        "SCENARIO_NAME": scenario_name,
        "DEFAULT_PARAMS": {**base_cls.DEFAULT_PARAMS, **overrides},
        "SWEEP_FAMILY_KEY": family_key,
        "SWEEP_KEY": sweep_key,
        "SWEEP_LABEL": sweep_label,
        "SWEEP_VALUE": float(sweep_value),
        "SWEEP_UNIT": unit,
        "get_name": classmethod(lambda cls: cls.SCENARIO_NAME),
    }
    new_cls = type(class_name, (base_cls,), attrs)
    new_cls.__module__ = __name__
    globals()[class_name] = new_cls
    return new_cls


def _percent_levels() -> np.ndarray:
    return np.linspace(0.01, 1.00, SWEEP_POINTS, dtype=float)


def _phase_jump_levels_deg() -> np.ndarray:
    return np.linspace(2.0, 180.0, SWEEP_POINTS, dtype=float)


def _ramp_levels_hz_s() -> np.ndarray:
    return np.geomspace(0.25, 30.0, SWEEP_POINTS, dtype=float)


def _ringdown_time_scales() -> np.ndarray:
    return np.geomspace(0.40, 3.00, SWEEP_POINTS, dtype=float)


def _anchor_index(values: np.ndarray) -> int:
    return int(len(values) // 2)


def _build_sweep_family(
    *,
    key: str,
    display_name: str,
    base_cls: type,
    sweep_key: str,
    sweep_label: str,
    unit: str,
    xscale: str,
    description: str,
    values: np.ndarray,
    naming_prefix: str,
    token_suffix: str,
    overrides_fn: Callable[[float], dict[str, Any]],
) -> SweepFamily:
    scenarios: list[SweepScenario] = []
    anchor_idx = _anchor_index(values)
    for idx, value in enumerate(values, start=1):
        overrides = overrides_fn(float(value))
        token = _safe_token(float(value), token_suffix)
        scenario_name = _scenario_name(naming_prefix, idx, token)
        class_name = f"{naming_prefix}Scenario{idx:02d}"
        scenario_cls = _register_variant(
            base_cls=base_cls,
            class_name=class_name,
            scenario_name=scenario_name,
            overrides=overrides,
            family_key=key,
            sweep_key=sweep_key,
            sweep_label=sweep_label,
            sweep_value=float(value),
            unit=unit,
        )
        scenarios.append(
            SweepScenario(
                scenario_name=scenario_name,
                scenario_cls=scenario_cls,
                sweep_value=float(value),
                sweep_value_label=f"{value:.4g}",
                overrides=overrides,
                is_anchor=(idx - 1) == anchor_idx,
            )
        )
    return SweepFamily(
        key=key,
        display_name=display_name,
        base_scenario_name=base_cls.SCENARIO_NAME,
        sweep_key=sweep_key,
        sweep_label=sweep_label,
        unit=unit,
        xscale=xscale,
        description=description,
        scenarios=tuple(scenarios),
    )


def _build_all_sweep_families() -> tuple[SweepFamily, ...]:
    percent_levels = _percent_levels()
    phase_levels = _phase_jump_levels_deg()
    ramp_levels = _ramp_levels_hz_s()
    time_scales = _ringdown_time_scales()

    families = [
        _build_sweep_family(
            key="ibr_harmonics",
            display_name="IBR Harmonics Severity",
            base_cls=IBRHarmonicsLargeScenario,
            sweep_key="harmonic_level_pu",
            sweep_label="Harmonic level",
            unit="pu",
            xscale="linear",
            description="Integer harmonic-only stress sweep from 1% to 100% of the fundamental.",
            values=percent_levels,
            naming_prefix="Sweep_IBR_Harmonics",
            token_suffix="pu",
            overrides_fn=lambda level: {
                "duration_s": 2.0,
                "freq_step_hz": 0.50,
                "h2_pct": 0.25 * level,
                "h3_pct": 0.50 * level,
                "h5_pct": 1.00 * level,
                "h7_pct": 0.75 * level,
                "h11_pct": 0.375 * level,
                "h13_pct": 0.25 * level,
                "sub_pct": 0.0,
                "ih325_pct": 0.0,
                "ih85_pct": 0.0,
                "white_noise_sigma": 0.003,
                "brown_noise_sigma": 0.0,
                "impulse_prob": 0.0,
                "impulse_mag": 0.0,
            },
        ),
        _build_sweep_family(
            key="phase_jump",
            display_name="Phase Jump Magnitude",
            base_cls=IEEEPhaseJump60Scenario,
            sweep_key="phase_jump_deg",
            sweep_label="Phase jump",
            unit="deg",
            xscale="linear",
            description="Clean phase-step sweep from 2° to 180°.",
            values=phase_levels,
            naming_prefix="Sweep_Phase_Jump",
            token_suffix="deg",
            overrides_fn=lambda deg: {
                "phase_jump_rad": math.radians(deg),
                "noise_sigma": 0.001,
                "t_jump_s": 0.70,
            },
        ),
        _build_sweep_family(
            key="oob_interference",
            display_name="OOB Interference Amplitude",
            base_cls=IEEEOOBInterferenceScenario,
            sweep_key="interf_amp_pu",
            sweep_label="OOB interference amplitude",
            unit="pu",
            xscale="linear",
            description="Out-of-band interference amplitude sweep from 1% to 100% of nominal voltage.",
            values=percent_levels,
            naming_prefix="Sweep_OOB",
            token_suffix="pu",
            overrides_fn=lambda level: {
                "interf_amp_pu": level,
                "interf_freq_hz": 30.0,
                "noise_sigma": 0.001,
            },
        ),
        _build_sweep_family(
            key="single_tone_noise",
            display_name="Single-Tone Noise",
            base_cls=IEEESingleSinWaveScenario,
            sweep_key="noise_sigma",
            sweep_label="Noise sigma",
            unit="pu",
            xscale="linear",
            description="Single-tone baseline with additive white noise from 1% to 100% of the fundamental amplitude.",
            values=percent_levels,
            naming_prefix="Sweep_Single_Tone_Noise",
            token_suffix="pu",
            overrides_fn=lambda level: {
                "duration_s": 2.0,
                "noise_sigma": level,
                "amplitude": 1.0,
            },
        ),
        _build_sweep_family(
            key="interharmonic",
            display_name="Interharmonic Amplitude",
            base_cls=IBRHarmonicsMediumScenario,
            sweep_key="ih75_pct",
            sweep_label="Interharmonic amplitude",
            unit="pu",
            xscale="linear",
            description="Pure interharmonic sweep with integer harmonics disabled.",
            values=percent_levels,
            naming_prefix="Sweep_Interharmonic",
            token_suffix="pu",
            overrides_fn=lambda level: {
                "duration_s": 2.0,
                "rocof_hz_s": 0.0,
                "rocof_duration_s": 0.0,
                "h3_pct": 0.0,
                "h5_pct": 0.0,
                "h7_pct": 0.0,
                "h11_pct": 0.0,
                "h13_pct": 0.0,
                "ih75_pct": level,
                "white_noise_sigma": 0.002,
                "brown_noise_sigma": 0.0,
            },
        ),
        _build_sweep_family(
            key="ibr_ringdown_timescale",
            display_name="IBR Ringdown Time Scale",
            base_cls=IBRPowerImbalanceRingdownScenario,
            sweep_key="time_scale_x",
            sweep_label="Time scale",
            unit="x",
            xscale="log",
            description="Fast-to-slow ringdown sweep that stretches event timing and decay constants together.",
            values=time_scales,
            naming_prefix="Sweep_IBR_Ringdown",
            token_suffix="x",
            overrides_fn=lambda scale: {
                "duration_s": 1.8 + 1.10 * scale,
                "t_ramp_s": 0.25 + 0.20 * scale,
                "t_event_s": 0.55 + 0.45 * scale,
                "ring_tau_s": 0.25 * scale,
                "ring_freq_hz": 4.0 / scale,
                "dc_tau_s": 0.03 * scale,
                "white_noise_sigma": 0.005,
                "interharmonic_pu": 0.02,
            },
        ),
        _build_sweep_family(
            key="ieee_freq_ramp",
            display_name="IEEE Frequency Ramp",
            base_cls=IEEEFreqRampScenario,
            sweep_key="rocof_hz_s",
            sweep_label="RoCoF",
            unit="Hz/s",
            xscale="log",
            description="RoCoF sweep from very slow ramps to aggressive low-inertia events.",
            values=ramp_levels,
            naming_prefix="Sweep_Freq_Ramp",
            token_suffix="hzs",
            overrides_fn=lambda rocof: {
                "duration_s": 1.8,
                "rocof_hz_s": rocof,
                "t_start_s": 0.30,
                "freq_cap_hz": 60.0 + (1.2 * rocof),
                "noise_sigma": 0.001,
            },
        ),
        _build_sweep_family(
            key="ieee_mag_step",
            display_name="IEEE Magnitude Step",
            base_cls=IEEEMagStepScenario,
            sweep_key="amp_post_pu",
            sweep_label="Post-step amplitude",
            unit="pu",
            xscale="linear",
            description="Magnitude-step sweep from +1% to +100% relative to nominal.",
            values=percent_levels,
            naming_prefix="Sweep_Mag_Step",
            token_suffix="pu",
            overrides_fn=lambda level: {
                "amp_pre_pu": 1.0,
                "amp_post_pu": 1.0 + level,
                "t_step_s": 0.50,
                "noise_sigma": 0.001,
            },
        ),
    ]
    return tuple(families)


ALL_SWEEP_FAMILIES = _build_all_sweep_families()


def selected_sweep_families() -> tuple[SweepFamily, ...]:
    if not INCLUDED_SWEEP_FAMILIES:
        return ALL_SWEEP_FAMILIES
    return tuple(
        family
        for family in ALL_SWEEP_FAMILIES
        if family.key.lower() in INCLUDED_SWEEP_FAMILIES
    )


def _flatten_scenarios(families: tuple[SweepFamily, ...]) -> list[type]:
    return [scenario.scenario_cls for family in families for scenario in family.scenarios]


def _family_anchor(family: SweepFamily) -> SweepScenario:
    for scenario in family.scenarios:
        if scenario.is_anchor:
            return scenario
    raise ValueError(f"Family {family.key} has no anchor scenario.")


def _manifest_payload(families: tuple[SweepFamily, ...]) -> dict[str, Any]:
    return {
        "benchmark_identity": SWEEP_BENCHMARK_IDENTITY,
        "benchmark_scope": SWEEP_BENCHMARK_SCOPE,
        "authority_statement": SWEEP_AUTHORITY_STATEMENT,
        "paper_alignment_policy": SWEEP_ALIGNMENT_POLICY,
        "output_dir": str(SWEEP_OUTPUT_DIR),
        "family_count": len(families),
        "scenario_count": sum(len(family.scenarios) for family in families),
        "families": [
            {
                "key": family.key,
                "display_name": family.display_name,
                "base_scenario_name": family.base_scenario_name,
                "sweep_key": family.sweep_key,
                "sweep_label": family.sweep_label,
                "unit": family.unit,
                "xscale": family.xscale,
                "description": family.description,
                "anchor_scenario": _family_anchor(family).scenario_name,
                "scenario_count": len(family.scenarios),
                "scenarios": [
                    {
                        "scenario_name": scenario.scenario_name,
                        "sweep_value": scenario.sweep_value,
                        "sweep_value_label": scenario.sweep_value_label,
                        "overrides": benchmark._to_builtin(scenario.overrides),
                        "is_anchor": scenario.is_anchor,
                    }
                    for scenario in family.scenarios
                ],
            }
            for family in families
        ],
    }


def _configure_benchmark_context(families: tuple[SweepFamily, ...]) -> None:
    benchmark.BASE_RESULTS_DIR = SWEEP_OUTPUT_DIR
    benchmark.SCENARIOS = _flatten_scenarios(families)
    benchmark.BENCHMARK_IDENTITY = SWEEP_BENCHMARK_IDENTITY
    benchmark.BENCHMARK_SCOPE = SWEEP_BENCHMARK_SCOPE
    benchmark.BENCHMARK_AUTHORITY_STATEMENT = SWEEP_AUTHORITY_STATEMENT
    benchmark.PAPER_ALIGNMENT_POLICY = SWEEP_ALIGNMENT_POLICY
    benchmark.JSON_REPORT_NAME = SWEEP_JSON_NAME
    benchmark.FIGURE1_BASENAME = "ParametricSweep_Fig1"
    benchmark.FIGURE2_BASENAME = "ParametricSweep_Fig2"


def _load_tuning_cache(cache_path: Path) -> dict[str, Any]:
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_tuning_cache(cache_path: Path, cache: dict[str, Any]) -> None:
    cache_path.write_text(json.dumps(benchmark._to_builtin(cache), indent=2), encoding="utf-8")


def _summary_path(out_dir: Path, scenario_name: str, estimator_name: str) -> Path:
    return out_dir / f"{scenario_name}__{estimator_name}_summary.csv"


def _signals_path(out_dir: Path, scenario_name: str, estimator_name: str) -> Path:
    return out_dir / f"{scenario_name}__{estimator_name}_signals.csv"


def _save_summary_only(result: Any, out_dir: Path, save_signals: bool) -> tuple[Path, list[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = _summary_path(out_dir, result.scenario_name, result.estimator_name)
    result.summary_df.to_csv(summary_path, index=False)
    signal_files: list[str] = []
    if save_signals:
        signals_path = _signals_path(out_dir, result.scenario_name, result.estimator_name)
        result.signals_df.to_csv(signals_path, index=False)
        signal_files.append(str(signals_path.relative_to(ROOT)))
    return summary_path, signal_files


def _save_scenario_csv(sc: Any, sc_dir: Path, sc_name: str) -> Path:
    sc_dir.mkdir(parents=True, exist_ok=True)
    csv_path = sc_dir / f"{sc_name}_scenario.csv"
    df = pd.DataFrame({"t_s": sc.t, "v_pu": sc.v, "f_true_hz": sc.f_true})
    df.to_csv(csv_path, index=False)
    return csv_path


def _tune_estimator_for_anchor(est_name: str, est_cls: type, scenario_cls: type) -> tuple[dict[str, Any], dict[str, Any]]:
    defaults = est_cls.default_params() if hasattr(est_cls, "default_params") else {}
    tuning_meta: dict[str, Any] = {
        "strategy": "family_anchor_reuse",
        "anchor_scenario": scenario_cls.get_name(),
        "sampler_mode_requested": benchmark.OPTUNA_SAMPLER_MODE,
        "sampler_mode_effective": None,
        "n_trials_requested": None,
        "n_trials_executed": 0,
    }

    if est_name not in benchmark.SEARCH_SPACES:
        tuning_meta["reason"] = "no_search_space"
        return defaults, tuning_meta

    n_trials_requested = benchmark._effective_n_trials(est_name)
    tuning_meta["n_trials_requested"] = int(n_trials_requested)
    if n_trials_requested <= 0:
        tuning_meta["reason"] = "n_trials_requested<=0"
        return defaults, tuning_meta

    space_fn = benchmark.SEARCH_SPACES[est_name]
    if not benchmark._grid_space_for_estimator(space_fn, n_trials=2):
        tuning_meta["reason"] = "empty_search_space"
        tuning_meta["sampler_mode_effective"] = "none"
        return defaults, tuning_meta

    sc = scenario_cls.run(seed=42)
    fs_dsp = 1.0 / float(sc.t[1] - sc.t[0])
    eval_start = int(0.100 * fs_dsp)

    def objective(trial: Any) -> float:
        suggested = space_fn(trial)
        params = {**defaults, **suggested}
        try:
            est = est_cls(**params)
            f_hat = benchmark._run_estimator(est, sc.v)
            error = f_hat[eval_start:] - sc.f_true[eval_start:]
            rmse = float(np.sqrt(np.mean(error ** 2)))
            max_peak = float(np.max(np.abs(error)))
            tail_idx = int(0.9 * len(error))
            steady_tail_error = float(np.mean(np.abs(error[tail_idx:])))

            norm_rmse = rmse / 0.05
            norm_peak = max_peak / 0.50
            norm_tail = steady_tail_error / 0.02
            score = (0.50 * norm_rmse) + (0.30 * norm_peak) + (0.20 * norm_tail)
            return score if np.isfinite(score) else 1e6
        except Exception:
            return 1e6

    study, n_trials_exec, sampler_mode_effective = benchmark._build_optuna_study(
        space_fn=space_fn,
        n_trials=n_trials_requested,
    )
    tuning_meta["sampler_mode_effective"] = sampler_mode_effective
    tuning_meta["n_trials_executed"] = int(n_trials_exec)

    study.optimize(objective, n_trials=n_trials_exec)
    if study.best_value >= 1e6:
        tuning_meta["reason"] = "all_trials_failed"
        return defaults, tuning_meta

    best_params = {**defaults, **space_fn(study.best_trial)}
    return best_params, tuning_meta


def _maybe_write_scenario_artifacts(scenario: SweepScenario, sc_dir: Path) -> dict[str, str | None]:
    sc_name = scenario.scenario_name
    scenario_csv = sc_dir / f"{sc_name}_scenario.csv"
    scenario_plot = sc_dir / f"{sc_name}_plot.png"
    scenario_zoom = sc_dir / f"{sc_name}_zoom.png"

    if RESUME_RUN and scenario_csv.exists() and (scenario.is_anchor or not SAVE_SCENARIO_PLOTS_ALL or scenario_plot.exists()):
        return {
            "scenario_csv": str(scenario_csv.relative_to(ROOT)),
            "scenario_plot": str(scenario_plot.relative_to(ROOT)) if scenario_plot.exists() else None,
            "scenario_zoom_plot": str(scenario_zoom.relative_to(ROOT)) if scenario_zoom.exists() else None,
        }

    sc = scenario.scenario_cls.run(seed=42)
    csv_path = _save_scenario_csv(sc, sc_dir, sc_name)
    if scenario.is_anchor or SAVE_SCENARIO_PLOTS_ALL:
        benchmark._save_scenario_artifacts(sc, sc_dir, sc_name)
        benchmark._save_scenario_zoom_plot(sc, sc_dir, sc_name)

    return {
        "scenario_csv": str(csv_path.relative_to(ROOT)),
        "scenario_plot": str(scenario_plot.relative_to(ROOT)) if scenario_plot.exists() else None,
        "scenario_zoom_plot": str(scenario_zoom.relative_to(ROOT)) if scenario_zoom.exists() else None,
    }


def run_parametric_phase_1(estimators: dict[str, type], families: tuple[SweepFamily, ...]) -> None:
    print("\n>>> PHASE 1: FAMILY-ANCHOR TUNING + PARAMETRIC MC <<<")
    cache_path = SWEEP_OUTPUT_DIR / SWEEP_TUNING_CACHE_NAME
    tuning_cache = _load_tuning_cache(cache_path)

    for family in families:
        anchor = _family_anchor(family)
        print(f"\n  Sweep family: {family.display_name} ({len(family.scenarios)} scenarios)")
        print(f"    Anchor scenario: {anchor.scenario_name}")
        family_cache = tuning_cache.setdefault(family.key, {})

        for est_name, est_cls in estimators.items():
            if RESUME_RUN and est_name in family_cache:
                continue
            print(f"    [{est_name}] tuning on anchor ...", flush=True)
            best_params, tuning_meta = _tune_estimator_for_anchor(est_name, est_cls, anchor.scenario_cls)
            family_cache[est_name] = {
                "best_params": benchmark._to_builtin(best_params),
                "tuning_meta": benchmark._to_builtin(tuning_meta),
            }
            _save_tuning_cache(cache_path, tuning_cache)

        for scenario in family.scenarios:
            sc_name = scenario.scenario_name
            sc_dir = SWEEP_OUTPUT_DIR / sc_name
            print(f"    Scenario: {sc_name}")
            scenario_artifacts = _maybe_write_scenario_artifacts(scenario, sc_dir)

            for est_name, est_cls in estimators.items():
                out_dir = sc_dir / est_name
                out_dir.mkdir(parents=True, exist_ok=True)
                summary_path = _summary_path(out_dir, sc_name, est_name)
                run_spec_path = out_dir / "run_spec.json"
                if RESUME_RUN and summary_path.exists() and run_spec_path.exists():
                    continue

                best_params = dict(family_cache[est_name]["best_params"])
                save_tracking = scenario.is_anchor or SAVE_TRACKING_ALL
                if save_tracking:
                    sc_base = scenario.scenario_cls.run(seed=42)
                    benchmark._save_tracking_plot(est_cls, best_params, sc_base, out_dir, est_name, sc_name)

                engine = MonteCarloEngine(
                    scenario_cls=scenario.scenario_cls,
                    estimator_cls=est_cls,
                    estimator_params=best_params,
                    n_runs=benchmark.N_MC_RUNS,
                )
                result = engine.run()
                summary_csv, signal_files = _save_summary_only(
                    result=result,
                    out_dir=out_dir,
                    save_signals=SAVE_SIGNAL_CSV and (scenario.is_anchor or SAVE_TRACKING_ALL),
                )

                tracking_files = [
                    str(path.relative_to(ROOT))
                    for path in sorted(out_dir.glob("tracking_*.png"))
                ]
                run_spec = {
                    "scenario": sc_name,
                    "scenario_class": scenario.scenario_cls.__name__,
                    "sweep_family_key": family.key,
                    "sweep_family_display": family.display_name,
                    "sweep_key": family.sweep_key,
                    "sweep_label": family.sweep_label,
                    "sweep_unit": family.unit,
                    "sweep_value": scenario.sweep_value,
                    "is_anchor_scenario": scenario.is_anchor,
                    "estimator": est_name,
                    "estimator_class": est_cls.__name__,
                    "family": benchmark._ESTIMATOR_FAMILIES.get(est_name, "Unknown"),
                    "best_params": benchmark._to_builtin(best_params),
                    "tuning_meta": benchmark._to_builtin(family_cache[est_name]["tuning_meta"]),
                    "n_trials_tuning": family_cache[est_name]["tuning_meta"].get("n_trials_executed", benchmark.N_TRIALS_TUNING),
                    "n_trials_tuning_requested": family_cache[est_name]["tuning_meta"].get("n_trials_requested", benchmark.N_TRIALS_TUNING),
                    "n_mc_runs": benchmark.N_MC_RUNS,
                    "artifacts": {
                        "summary_csv": str(summary_csv.relative_to(ROOT)),
                        "signals_csv": signal_files,
                        "tracking_plots": tracking_files,
                        **scenario_artifacts,
                    },
                }
                run_spec_path.write_text(
                    json.dumps(benchmark._to_builtin(run_spec), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                gc.collect()


def _write_manifest(families: tuple[SweepFamily, ...]) -> Path:
    SWEEP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = SWEEP_OUTPUT_DIR / SWEEP_MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(_manifest_payload(families), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest_path


def _print_plan(families: tuple[SweepFamily, ...]) -> None:
    total_scenarios = sum(len(family.scenarios) for family in families)
    print("Parametric sweep plan")
    print(f"  Families: {len(families)}")
    print(f"  Scenarios: {total_scenarios}")
    print(f"  Output dir: {SWEEP_OUTPUT_DIR}")
    print(
        "  Run config: "
        f"N_MC_RUNS={benchmark.N_MC_RUNS}, "
        f"N_TRIALS_TUNING={benchmark.N_TRIALS_TUNING}, "
        f"anchor_reuse=1, "
        f"save_signals={SAVE_SIGNAL_CSV}"
    )
    for family in families:
        anchor = _family_anchor(family)
        print(
            f"    - {family.display_name}: {len(family.scenarios)} scenarios, "
            f"anchor={anchor.scenario_name}, "
            f"sweep={family.sweep_label} [{family.unit}]"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run dense parametric family sweeps.")
    parser.add_argument("--plan-only", action="store_true", help="Write the sweep manifest and stop.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip degradation plot generation.")
    args = parser.parse_args(argv)

    families = selected_sweep_families()
    if not families:
        raise ValueError("No sweep families selected. Check PARAM_SWEEP_INCLUDE_FAMILIES.")

    _configure_benchmark_context(families)
    manifest_path = _write_manifest(families)
    _print_plan(families)
    if args.plan_only:
        print(f"  Manifest -> {manifest_path}")
        return

    t_start = time.time()
    print("Loading estimators ...")
    estimators = benchmark.load_estimators()
    benchmark._print_registry_summary(estimators)
    print("Validating search spaces ...")
    benchmark.validate_search_spaces(estimators)
    print("  OK")

    run_parametric_phase_1(estimators, families)
    benchmark.run_phase_2(allowed_estimators=set(estimators.keys()))

    generated_plots: list[Path] = []
    if not args.skip_plots:
        generated_plots = generate_parametric_sweep_figures(SWEEP_OUTPUT_DIR, manifest_path)
        if generated_plots:
            print(f"  Parametric plots -> {len(generated_plots)} files under {SWEEP_OUTPUT_DIR / 'parametric_plots'}")

    regression_outputs = build_parametric_regression_reports(SWEEP_OUTPUT_DIR, manifest_path)
    print(f"  Regression summary -> {regression_outputs['csv']}")

    json_path = benchmark._export_full_benchmark_json(estimators)
    report_paths = build_benchmark_report(
        input_json=json_path,
        output_dir=SWEEP_OUTPUT_DIR / "report_bundle",
    )
    elapsed_min = (time.time() - t_start) / 60.0

    print(f"  Manifest -> {manifest_path.relative_to(ROOT)}")
    print(f"  JSON report -> {json_path.relative_to(ROOT)}")
    print(f"  Report bundle -> {report_paths['markdown']}")
    print(f"\n[DONE] Parametric family sweeps completed in {elapsed_min:.1f} min.")


if __name__ == "__main__":
    main()
