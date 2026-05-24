from __future__ import annotations

import importlib
import json
import math
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from analysis.monte_carlo_engine import MonteCarloEngine

from .config import BenchmarkRunConfig, CustomEstimatorSelection, EstimatorSelection
from .registry import (
    CANONICAL_METRIC_IDS,
    CANONICAL_METRIC_PROFILE,
    METRIC_LABELS,
    ESTIMATOR_FAMILIES,
    load_estimators,
    platform_manifest,
    scenario_registry,
)
from .reproducibility import build_reproducibility_manifest, sha256_file

ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def _temporary_env(overrides: dict[str, str]) -> Iterator[None]:
    old = {key: os.environ.get(key) for key in overrides}
    try:
        os.environ.update(overrides)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _load_custom_estimator(spec: CustomEstimatorSelection) -> type:
    if not spec.path.exists():
        raise FileNotFoundError(f"Custom estimator file not found: {spec.path}")
    module_dir = str(spec.path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    module_name = spec.path.stem
    importlib.invalidate_caches()
    module = importlib.import_module(module_name)
    cls = getattr(module, spec.class_name)
    if not hasattr(cls, "name"):
        cls.name = spec.name
    if not (hasattr(cls, "step") or hasattr(cls, "step_vectorized")):
        raise TypeError(f"Custom estimator {spec.class_name} must define step(...) or step_vectorized(...).")
    return cls


def _load_tuned_params(config: BenchmarkRunConfig, scenario_name: str, estimator_name: str) -> dict[str, Any]:
    if config.tuned_artifacts_dir is None:
        raise ValueError("tuned_artifacts_dir is required for artifact_tuned parameter policy.")
    safe_estimator = estimator_name.replace("/", "_")
    candidates = [
        config.tuned_artifacts_dir / scenario_name / estimator_name / "run_spec.json",
        config.tuned_artifacts_dir / scenario_name / safe_estimator / "run_spec.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        spec = json.loads(path.read_text(encoding="utf-8"))
        params = spec.get("params", spec.get("estimator_params", {})) or {}
        if not isinstance(params, dict):
            raise ValueError(f"Tuned params in {path} must be a JSON object.")
        params = dict(params)
        params["_openfreqbench_tuned_source"] = str(path)
        params["_openfreqbench_tuned_source_sha256"] = sha256_file(path)
        return params
    raise FileNotFoundError(
        "No tuned run_spec.json found for "
        f"scenario={scenario_name!r}, estimator={estimator_name!r} under {config.tuned_artifacts_dir}"
    )


def _params_for_pair(
    config: BenchmarkRunConfig,
    scenario_name: str,
    estimator_name: str,
    cls: type,
    explicit_params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    params: dict[str, Any] = {}
    source: dict[str, Any] = {"policy": config.parameter_policy}
    if config.parameter_policy in {"default", "artifact_tuned"} and hasattr(cls, "default_params"):
        params.update(cls.default_params())
        source["default_params"] = True
    if config.parameter_policy == "artifact_tuned":
        tuned = _load_tuned_params(config, scenario_name, estimator_name)
        tuned_source = {
            "path": tuned.pop("_openfreqbench_tuned_source"),
            "sha256": tuned.pop("_openfreqbench_tuned_source_sha256"),
        }
        params.update(tuned)
        source["tuned_artifact"] = tuned_source
    params.update(explicit_params)
    source["explicit_override_keys"] = sorted(explicit_params)
    return params, source


def _resolve_estimator_classes(config: BenchmarkRunConfig) -> dict[str, tuple[type, dict[str, Any], str]]:
    canonical_labels = [item.name for item in config.estimators]
    canonical = load_estimators(canonical_labels) if canonical_labels else {}
    out: dict[str, tuple[type, dict[str, Any], str]] = {}
    for item in config.estimators:
        cls = canonical[item.name]
        out[item.name] = (cls, dict(item.params), ESTIMATOR_FAMILIES.get(item.name, "Unknown"))
    for item in config.custom_estimators:
        cls = _load_custom_estimator(item)
        out[item.name] = (cls, dict(item.params), "Custom")
    return out


def _validate_selection(config: BenchmarkRunConfig) -> None:
    known_scenarios = scenario_registry()
    unknown_scenarios = [name for name in config.scenarios if name not in known_scenarios]
    if unknown_scenarios:
        raise ValueError(f"Unknown scenario(s): {unknown_scenarios}. Known: {sorted(known_scenarios)}")


def dry_run_manifest(config: BenchmarkRunConfig) -> dict[str, Any]:
    _validate_selection(config)
    estimator_names = [item.name for item in config.estimators] + [
        item.name for item in config.custom_estimators
    ]
    return {
        "run_id": config.run_id,
        "mode": config.mode,
        "output_dir": str(config.output_dir / config.run_id),
        "scenarios": config.scenarios,
        "estimators": estimator_names,
        "n_runs": config.n_runs,
        "base_seed": config.base_seed,
        "capture_signals": config.capture_signals,
        "metric_profile": config.metric_profile,
        "metric_include": config.metric_include or list(CANONICAL_METRIC_IDS),
        "metrics_locked": True,
        "parameter_policy": config.parameter_policy,
        "tuned_artifacts_dir": str(config.tuned_artifacts_dir) if config.tuned_artifacts_dir else None,
    }


def _aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    work = raw.copy()
    metric_cols: list[str] = []
    for metric in CANONICAL_METRIC_IDS:
        if metric not in work.columns:
            continue
        converted = pd.to_numeric(work[metric], errors="coerce")
        if converted.notna().any():
            work[metric] = converted
            metric_cols.append(metric)
    if not metric_cols:
        return pd.DataFrame()
    grouped = work.groupby(["scenario", "estimator", "family"], dropna=False)[metric_cols]
    agg = grouped.agg(["mean", "std", "median", "min", "max"]).reset_index()
    agg.columns = [
        "_".join(str(part) for part in col if part)
        if isinstance(col, tuple)
        else str(col)
        for col in agg.columns
    ]
    return agg


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def run_benchmark_config(config: BenchmarkRunConfig, *, dry_run: bool = False) -> dict[str, Any]:
    if dry_run:
        return dry_run_manifest(config)

    _validate_selection(config)
    scenarios = scenario_registry()
    estimators = _resolve_estimator_classes(config)

    run_root = config.output_dir / config.run_id
    run_root.mkdir(parents=True, exist_ok=True)

    env: dict[str, str] = {}
    if config.max_workers is not None:
        env["BENCHMARK_MC_MAX_WORKERS"] = str(config.max_workers)

    raw_frames: list[pd.DataFrame] = []
    artifacts: list[dict[str, str]] = []

    with _temporary_env(env):
        for scenario_name in config.scenarios:
            scenario_cls = scenarios[scenario_name]
            scenario_dir = run_root / scenario_name
            scenario_dir.mkdir(parents=True, exist_ok=True)
            for estimator_name, (estimator_cls, explicit_params, family) in estimators.items():
                safe_estimator = estimator_name.replace("/", "_")
                out_dir = scenario_dir / safe_estimator
                out_dir.mkdir(parents=True, exist_ok=True)
                params, param_source = _params_for_pair(
                    config=config,
                    scenario_name=scenario_name,
                    estimator_name=estimator_name,
                    cls=estimator_cls,
                    explicit_params=explicit_params,
                )

                engine = MonteCarloEngine(
                    scenario_cls=scenario_cls,
                    estimator_cls=estimator_cls,
                    estimator_params=params,
                    n_runs=config.n_runs,
                    base_seed=config.base_seed,
                    capture_signals=config.capture_signals,
                )
                result = engine.run()
                summary = result.summary_df.copy()
                summary["scenario"] = scenario_name
                summary["estimator"] = estimator_name
                summary["family"] = family
                summary_path = out_dir / f"{scenario_name}__{safe_estimator}_summary.csv"
                summary.to_csv(summary_path, index=False)
                raw_frames.append(summary)

                signals_path = ""
                if config.capture_signals and not result.signals_df.empty:
                    signals = result.signals_df.copy()
                    signals["scenario"] = scenario_name
                    signals["estimator"] = estimator_name
                    signals_path_obj = out_dir / f"{scenario_name}__{safe_estimator}_signals.csv"
                    signals.to_csv(signals_path_obj, index=False)
                    signals_path = str(signals_path_obj)

                run_spec = {
                    "scenario": scenario_name,
                    "estimator": estimator_name,
                    "family": family,
                    "params": params,
                    "parameter_source": param_source,
                    "n_runs": config.n_runs,
                    "base_seed": config.base_seed,
                    "metric_profile": CANONICAL_METRIC_PROFILE,
                }
                spec_path = out_dir / "run_spec.json"
                spec_path.write_text(json.dumps(run_spec, indent=2), encoding="utf-8")
                artifacts.append(
                    {
                        "scenario": scenario_name,
                        "estimator": estimator_name,
                        "summary_csv": str(summary_path),
                        "signals_csv": signals_path,
                        "run_spec": str(spec_path),
                    }
                )

    raw = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
    aggregated = _aggregate(raw)
    raw_csv = run_root / "raw_run_records.csv"
    agg_csv = run_root / "aggregated_metrics.csv"
    raw.to_csv(raw_csv, index=False)
    aggregated.to_csv(agg_csv, index=False)

    manifest = platform_manifest()
    payload = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "generator": "openfreqbench-v2",
        },
        "run_configuration": {
            "run_id": config.run_id,
            "mode": config.mode,
            "metric_profile": CANONICAL_METRIC_PROFILE,
            "metrics_locked": True,
            "metrics": list(CANONICAL_METRIC_IDS),
            "metric_labels": METRIC_LABELS,
            "scenarios": config.scenarios,
            "estimators": list(estimators),
            "estimator_families": {name: fam for name, (_, _, fam) in estimators.items()},
            "n_mc_runs": config.n_runs,
            "base_seed": config.base_seed,
            "capture_signals": config.capture_signals,
            "parameter_policy": config.parameter_policy,
            "tuned_artifacts_dir": str(config.tuned_artifacts_dir) if config.tuned_artifacts_dir else None,
        },
        "reproducibility": build_reproducibility_manifest(ROOT, config.source_path),
        "platform_manifest": manifest,
        "raw_run_records": raw.to_dict(orient="records") if not raw.empty else [],
        "aggregated_metrics": aggregated.to_dict(orient="records") if not aggregated.empty else [],
        "artifacts": {
            "run_root": str(run_root),
            "raw_run_records_csv": str(raw_csv),
            "aggregated_metrics_csv": str(agg_csv),
            "per_pair": artifacts,
        },
    }
    report_json = run_root / "benchmark_report.json"
    report_json.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False), encoding="utf-8")
    return {
        "run_root": str(run_root),
        "report_json": str(report_json),
        "raw_csv": str(raw_csv),
        "aggregated_csv": str(agg_csv),
        "n_records": int(len(raw)),
        "n_pairs": int(len(artifacts)),
    }
