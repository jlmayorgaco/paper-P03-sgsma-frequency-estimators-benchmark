from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

from analysis.monte_carlo_engine import MonteCarloEngine
from openfreqbench.paths import PROJECT_ROOT
from openfreqbench.registry import CANONICAL_METRIC_PROFILE, METRIC_LABELS, load_estimators, scenario_registry
from openfreqbench.reproducibility import build_reproducibility_manifest
from pipelines.full_mc_benchmark import (
    METRIC_LOWER_IS_BETTER,
    SEARCH_SPACES,
    _to_builtin,
    validate_search_spaces,
)


DEFAULT_OBJECTIVES = ("m1_rmse_hz", "m5_trip_risk_s")


def _safe_token(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def _finite_mean(summary: pd.DataFrame, metric: str) -> float:
    if metric not in summary.columns:
        return math.inf
    values = pd.to_numeric(summary[metric], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.inf
    return float(np.mean(values))


def _objective_score(summary: pd.DataFrame, metric: str) -> tuple[float, float]:
    value = _finite_mean(summary, metric)
    if not math.isfinite(value):
        return math.inf, value
    lower_is_better = METRIC_LOWER_IS_BETTER.get(metric, True)
    return (value if lower_is_better else -value), value


def _default_params(estimator_cls: type) -> dict[str, Any]:
    return dict(estimator_cls.default_params()) if hasattr(estimator_cls, "default_params") else {}


def _run_mc(
    scenario_cls: type,
    estimator_cls: type,
    params: dict[str, Any],
    *,
    n_runs: int,
    base_seed: int,
    n_cost_reps: int,
    capture_signals: bool,
) -> Any:
    engine = MonteCarloEngine(
        scenario_cls=scenario_cls,
        estimator_cls=estimator_cls,
        estimator_params=params,
        n_runs=n_runs,
        base_seed=base_seed,
        n_cost_reps=n_cost_reps,
        capture_signals=capture_signals,
    )
    return engine.run()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_builtin(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _copy_replay_artifacts(
    objective_dir: Path,
    replay_dir: Path,
    run_spec: dict[str, Any],
    eval_summary: pd.DataFrame,
    scenario_name: str,
    estimator_name: str,
) -> None:
    replay_dir.mkdir(parents=True, exist_ok=True)
    _write_json(replay_dir / "run_spec.json", run_spec)
    safe_estimator = estimator_name.replace("/", "_")
    eval_summary.to_csv(
        replay_dir / f"{scenario_name}__{safe_estimator}_summary.csv",
        index=False,
    )
    _write_json(
        replay_dir / "source_objective_artifact.json",
        {
            "source_objective_dir": str(objective_dir),
            "objective": run_spec["objective_metric"],
            "scenario": scenario_name,
            "estimator": estimator_name,
        },
    )


def tune_pair_for_objective(
    scenario_name: str,
    scenario_cls: type,
    estimator_name: str,
    estimator_cls: type,
    objective_metric: str,
    *,
    output_root: Path,
    replay_root: Path,
    run_id: str,
    mc_version: str,
    n_trials: int,
    tune_runs: int,
    eval_runs: int,
    base_seed: int,
    n_cost_reps: int,
    optuna_seed: int,
    capture_signals: bool,
) -> dict[str, Any]:
    safe_estimator = estimator_name.replace("/", "_")
    objective_token = _safe_token(objective_metric)
    objective_dir = output_root / run_id / scenario_name / safe_estimator / objective_token
    objective_dir.mkdir(parents=True, exist_ok=True)
    defaults = _default_params(estimator_cls)
    space_fn = SEARCH_SPACES.get(estimator_name)
    trials: list[dict[str, Any]] = []

    def evaluate(params: dict[str, Any], seed_offset: int) -> tuple[float, float, pd.DataFrame]:
        result = _run_mc(
            scenario_cls,
            estimator_cls,
            params,
            n_runs=tune_runs,
            base_seed=base_seed + seed_offset,
            n_cost_reps=n_cost_reps,
            capture_signals=False,
        )
        score, metric_value = _objective_score(result.summary_df, objective_metric)
        return score, metric_value, result.summary_df

    best_params = dict(defaults)
    best_score = math.inf
    best_metric_value = math.inf
    sampler = optuna.samplers.TPESampler(seed=optuna_seed)
    study: optuna.Study | None = None
    tuning_status = "defaults"

    if space_fn is not None and n_trials > 0:
        study = optuna.create_study(direction="minimize", sampler=sampler)

        def objective(trial: optuna.Trial) -> float:
            suggested = dict(space_fn(trial))
            params = {**defaults, **suggested}
            try:
                score, metric_value, _ = evaluate(params, seed_offset=10_000 + trial.number)
            except Exception as exc:
                score = math.inf
                metric_value = math.inf
                trial.set_user_attr("error", str(exc))
            trial.set_user_attr("objective_metric_value", metric_value)
            trial.set_user_attr("params_json", json.dumps(_to_builtin(params), sort_keys=True))
            trials.append(
                {
                    "trial": int(trial.number),
                    "status": "ok" if math.isfinite(score) else "fail",
                    "objective_metric": objective_metric,
                    "objective_score": None if not math.isfinite(score) else float(score),
                    "objective_metric_value": None
                    if not math.isfinite(metric_value)
                    else float(metric_value),
                    "params_json": json.dumps(_to_builtin(params), sort_keys=True),
                    "error": trial.user_attrs.get("error", ""),
                }
            )
            return score if math.isfinite(score) else 1e12

        study.optimize(objective, n_trials=n_trials)
        if study.best_trial is not None and study.best_value < 1e12:
            best_suggested = dict(space_fn(study.best_trial))
            best_params = {**defaults, **best_suggested}
            best_score = float(study.best_value)
            best_metric_value = float(study.best_trial.user_attrs.get("objective_metric_value", best_score))
            tuning_status = "optuna"

    if not math.isfinite(best_score):
        try:
            best_score, best_metric_value, _ = evaluate(defaults, seed_offset=20_000)
            tuning_status = "default_evaluated"
        except Exception as exc:
            trials.append(
                {
                    "trial": -1,
                    "status": "fail",
                    "objective_metric": objective_metric,
                    "objective_score": None,
                    "objective_metric_value": None,
                    "params_json": json.dumps(_to_builtin(defaults), sort_keys=True),
                    "error": str(exc),
                }
            )

    eval_result = _run_mc(
        scenario_cls,
        estimator_cls,
        best_params,
        n_runs=eval_runs,
        base_seed=base_seed + 30_000,
        n_cost_reps=n_cost_reps,
        capture_signals=capture_signals,
    )
    eval_summary = eval_result.summary_df.copy()
    eval_summary["scenario"] = scenario_name
    eval_summary["estimator"] = estimator_name
    eval_summary["objective_metric"] = objective_metric
    eval_summary_path = objective_dir / "eval_summary.csv"
    eval_summary.to_csv(eval_summary_path, index=False)
    trial_path = objective_dir / "tuning_trials.csv"
    pd.DataFrame(trials).to_csv(trial_path, index=False)

    eval_objective_value = _finite_mean(eval_summary, objective_metric)
    run_spec = {
        "schema_version": "openfreqbench-full-mc-objective-tuning-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "pipeline": "full_mc_tuning_matrix",
        "mc_version": mc_version,
        "metric_profile": CANONICAL_METRIC_PROFILE,
        "scenario": scenario_name,
        "scenario_class": scenario_cls.__name__,
        "estimator": estimator_name,
        "estimator_class": estimator_cls.__name__,
        "objective_metric": objective_metric,
        "objective_label": METRIC_LABELS.get(objective_metric, objective_metric),
        "objective_direction": "minimize" if METRIC_LOWER_IS_BETTER.get(objective_metric, True) else "maximize",
        "params": best_params,
        "estimator_params": best_params,
        "best_params": best_params,
        "defaults": defaults,
        "tuning": {
            "status": tuning_status,
            "n_trials_requested": int(n_trials),
            "n_trials_recorded": int(len(trials)),
            "tune_runs": int(tune_runs),
            "eval_runs": int(eval_runs),
            "base_seed": int(base_seed),
            "optuna_seed": int(optuna_seed),
            "best_tuning_score": None if not math.isfinite(best_score) else float(best_score),
            "best_tuning_metric_value": None
            if not math.isfinite(best_metric_value)
            else float(best_metric_value),
            "eval_objective_metric_value": None
            if not math.isfinite(eval_objective_value)
            else float(eval_objective_value),
        },
        "artifacts": {
            "objective_dir": str(objective_dir),
            "tuning_trials_csv": str(trial_path),
            "eval_summary_csv": str(eval_summary_path),
        },
    }
    run_spec_path = objective_dir / "run_spec.json"
    _write_json(run_spec_path, run_spec)
    replay_dir = replay_root / objective_token / scenario_name / safe_estimator
    _copy_replay_artifacts(objective_dir, replay_dir, run_spec, eval_summary, scenario_name, estimator_name)

    return {
        "run_id": run_id,
        "mc_version": mc_version,
        "scenario": scenario_name,
        "estimator": estimator_name,
        "objective_metric": objective_metric,
        "objective_direction": run_spec["objective_direction"],
        "tuning_status": tuning_status,
        "best_tuning_score": run_spec["tuning"]["best_tuning_score"],
        "best_tuning_metric_value": run_spec["tuning"]["best_tuning_metric_value"],
        "eval_objective_metric_value": run_spec["tuning"]["eval_objective_metric_value"],
        "best_params_json": json.dumps(_to_builtin(best_params), sort_keys=True),
        "run_spec": str(run_spec_path),
        "replay_run_spec": str(replay_dir / "run_spec.json"),
        "tuning_trials_csv": str(trial_path),
        "eval_summary_csv": str(eval_summary_path),
    }


def _selected_scenarios(args: argparse.Namespace) -> list[str]:
    registry = scenario_registry()
    if args.all_scenarios:
        return sorted(registry)
    return list(args.scenario or ["IEEE_Single_SinWave"])


def _selected_estimators(args: argparse.Namespace) -> list[str]:
    if args.all_estimators:
        return sorted(load_estimators().keys())
    return list(args.estimator or ["ZCD"])


def run_matrix(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_dir).expanduser().resolve()
    run_root = output_root / args.run_id
    replay_root = run_root / "selected_replay"
    scenarios = _selected_scenarios(args)
    estimators = _selected_estimators(args)
    objectives = list(args.objective or DEFAULT_OBJECTIVES)
    scenario_map = scenario_registry()
    estimator_map = load_estimators(estimators)
    validate_search_spaces(estimator_map)

    unknown_scenarios = [name for name in scenarios if name not in scenario_map]
    if unknown_scenarios:
        raise ValueError(f"Unknown scenario(s): {unknown_scenarios}. Known: {sorted(scenario_map)}")

    plan = {
        "run_id": args.run_id,
        "pipeline": "full_mc_tuning_matrix",
        "mc_version": args.mc_version,
        "metric_profile": CANONICAL_METRIC_PROFILE,
        "metrics_locked": True,
        "output_root": str(output_root),
        "scenarios": scenarios,
        "estimators": estimators,
        "objectives": objectives,
        "n_trials": int(args.n_trials),
        "tune_runs": int(args.tune_runs),
        "eval_runs": int(args.eval_runs),
        "base_seed": int(args.base_seed),
        "n_cost_reps": int(args.n_cost_reps),
        "capture_signals": bool(args.capture_signals),
    }
    if args.dry_run:
        run_root.mkdir(parents=True, exist_ok=True)
        _write_json(run_root / "plan.json", plan)
        return run_root / "plan.json"

    rows: list[dict[str, Any]] = []
    for scenario_name in scenarios:
        for estimator_name in estimators:
            for objective_metric in objectives:
                print(f"[tune] {scenario_name} / {estimator_name} / {objective_metric}", flush=True)
                rows.append(
                    tune_pair_for_objective(
                        scenario_name,
                        scenario_map[scenario_name],
                        estimator_name,
                        estimator_map[estimator_name],
                        objective_metric,
                        output_root=output_root,
                        replay_root=replay_root,
                        run_id=args.run_id,
                        mc_version=args.mc_version,
                        n_trials=int(args.n_trials),
                        tune_runs=int(args.tune_runs),
                        eval_runs=int(args.eval_runs),
                        base_seed=int(args.base_seed),
                        n_cost_reps=int(args.n_cost_reps),
                        optuna_seed=int(args.optuna_seed),
                        capture_signals=bool(args.capture_signals),
                    )
                )

    run_root.mkdir(parents=True, exist_ok=True)
    matrix_csv = run_root / "tuning_matrix.csv"
    pd.DataFrame(rows).to_csv(matrix_csv, index=False)
    report_path = run_root / "benchmark_report.json"
    _write_json(
        report_path,
        {
            "metadata": {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "description": "Objective-specific full Monte Carlo tuning matrix.",
            },
            "run_configuration": plan,
            "reproducibility": build_reproducibility_manifest(PROJECT_ROOT, None),
            "raw_run_records": [],
            "aggregated_metrics": rows,
            "objective_tuning": rows,
            "artifacts": {
                "run_root": str(run_root),
                "tuning_matrix_csv": str(matrix_csv),
                "selected_replay_root": str(replay_root),
            },
        },
    )
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune estimator parameters per scenario, estimator, and objective metric."
    )
    parser.add_argument("--run-id", default="full-mc-objective-matrix")
    parser.add_argument("--mc-version", default="mc-v1")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "artifacts" / "full_mc_tuning_matrix"))
    parser.add_argument("--scenario", action="append", default=None)
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--estimator", action="append", default=None)
    parser.add_argument("--all-estimators", action="store_true")
    parser.add_argument("--objective", action="append", default=None, help="Metric id to optimize; repeatable.")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--tune-runs", type=int, default=2)
    parser.add_argument("--eval-runs", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--optuna-seed", type=int, default=42)
    parser.add_argument("--n-cost-reps", type=int, default=1)
    parser.add_argument("--capture-signals", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    path = run_matrix(args)
    print(path)


if __name__ == "__main__":
    main()
