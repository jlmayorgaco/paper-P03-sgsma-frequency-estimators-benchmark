from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .registry import CANONICAL_METRIC_PROFILE, assert_canonical_metric_ids


class ConfigError(ValueError):
    """Raised when an OpenFreqBench YAML file violates the platform contract."""


@dataclass(frozen=True)
class EstimatorSelection:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CustomEstimatorSelection:
    name: str
    path: Path
    class_name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkRunConfig:
    run_id: str
    mode: str
    output_dir: Path
    scenarios: list[str]
    estimators: list[EstimatorSelection]
    custom_estimators: list[CustomEstimatorSelection]
    metric_profile: str
    metric_include: list[str]
    n_runs: int
    base_seed: int
    capture_signals: bool
    max_workers: int | None
    parameter_policy: str
    tuned_artifacts_dir: Path | None
    source_path: Path | None = None


def _as_list(value: Any, *, field_name: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    raise ConfigError(f"`{field_name}` must be a list.")


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _parse_estimators(items: list[Any]) -> list[EstimatorSelection]:
    out: list[EstimatorSelection] = []
    for item in items:
        if isinstance(item, str):
            out.append(EstimatorSelection(name=item))
            continue
        if not isinstance(item, dict):
            raise ConfigError("Each estimator entry must be a string or mapping.")
        name = str(item.get("name", "")).strip()
        if not name:
            raise ConfigError("Estimator entries require `name`.")
        params = item.get("params", {}) or {}
        if not isinstance(params, dict):
            raise ConfigError(f"Estimator `{name}` params must be a mapping.")
        out.append(EstimatorSelection(name=name, params=dict(params)))
    return out


def _parse_custom_estimators(items: list[Any], base_dir: Path) -> list[CustomEstimatorSelection]:
    out: list[CustomEstimatorSelection] = []
    for item in items:
        if not isinstance(item, dict):
            raise ConfigError("Each custom estimator entry must be a mapping.")
        name = str(item.get("name", "")).strip()
        path_raw = str(item.get("path", "")).strip()
        class_name = str(item.get("class", item.get("class_name", ""))).strip()
        if not name or not path_raw or not class_name:
            raise ConfigError("Custom estimators require `name`, `path`, and `class`.")
        params = item.get("params", {}) or {}
        if not isinstance(params, dict):
            raise ConfigError(f"Custom estimator `{name}` params must be a mapping.")
        path = Path(path_raw)
        if not path.is_absolute():
            path = base_dir / path
        out.append(
            CustomEstimatorSelection(
                name=name,
                path=path.resolve(),
                class_name=class_name,
                params=dict(params),
            )
        )
    return out


def _validate_metric_block(metrics: dict[str, Any]) -> tuple[str, list[str]]:
    forbidden = {"formulas", "definitions", "functions", "code", "python"}
    present_forbidden = sorted(forbidden.intersection(metrics))
    if present_forbidden:
        raise ConfigError(
            "Metric formulas are platform-owned and cannot be defined in YAML. "
            f"Remove: {present_forbidden}."
        )
    profile = str(metrics.get("profile", CANONICAL_METRIC_PROFILE)).strip()
    if profile != CANONICAL_METRIC_PROFILE:
        raise ConfigError(
            f"Unsupported metric profile {profile!r}. Use {CANONICAL_METRIC_PROFILE!r}."
        )
    include = [str(item) for item in _as_list(metrics.get("include", []), field_name="metrics.include")]
    assert_canonical_metric_ids(include)
    return profile, include


def load_config(path: Path) -> BenchmarkRunConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ConfigError("OpenFreqBench config must be a mapping.")
    return parse_config(data, source_path=path)


def parse_config(data: dict[str, Any], source_path: Path | None = None) -> BenchmarkRunConfig:
    if "metric_definitions" in data:
        raise ConfigError("Top-level `metric_definitions` is forbidden. Metrics are canonical.")

    base_dir = source_path.parent if source_path is not None else Path.cwd()
    run = data.get("run", {}) or {}
    benchmark = data.get("benchmark", {}) or {}
    metrics = data.get("metrics", {}) or {}
    if not isinstance(run, dict) or not isinstance(benchmark, dict) or not isinstance(metrics, dict):
        raise ConfigError("`run`, `benchmark`, and `metrics` must be mappings when present.")

    run_id = str(run.get("id", data.get("id", "openfreqbench-run"))).strip()
    mode = str(run.get("mode", "matrix")).strip().lower()
    if mode not in {"single", "compare", "matrix", "monte_carlo", "smoke"}:
        raise ConfigError(f"Unsupported run.mode={mode!r}.")

    output_dir = Path(str(run.get("output_dir", "artifacts/openfreqbench"))).expanduser()
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir

    scenarios = [str(item) for item in _as_list(benchmark.get("scenarios"), field_name="benchmark.scenarios")]
    estimators = _parse_estimators(_as_list(benchmark.get("estimators"), field_name="benchmark.estimators"))
    custom_estimators = _parse_custom_estimators(
        _as_list(benchmark.get("custom_estimators"), field_name="benchmark.custom_estimators"),
        base_dir=base_dir,
    )
    parameter_policy = str(benchmark.get("parameter_policy", "default")).strip().lower()
    if parameter_policy not in {"default", "explicit", "artifact_tuned"}:
        raise ConfigError("benchmark.parameter_policy must be one of: default, explicit, artifact_tuned.")
    tuned_raw = benchmark.get("tuned_artifacts_dir", None)
    tuned_artifacts_dir = None
    if tuned_raw is not None:
        tuned_artifacts_dir = Path(str(tuned_raw)).expanduser()
        if not tuned_artifacts_dir.is_absolute():
            tuned_artifacts_dir = base_dir / tuned_artifacts_dir
        tuned_artifacts_dir = tuned_artifacts_dir.resolve()
    if parameter_policy == "artifact_tuned" and tuned_artifacts_dir is None:
        raise ConfigError("benchmark.tuned_artifacts_dir is required when parameter_policy=artifact_tuned.")

    if not scenarios:
        raise ConfigError("At least one scenario is required.")
    if not estimators and not custom_estimators:
        raise ConfigError("At least one canonical or custom estimator is required.")

    metric_profile, metric_include = _validate_metric_block(metrics)
    n_runs = int(run.get("n_runs", 1))
    if n_runs < 1:
        raise ConfigError("run.n_runs must be >= 1.")
    base_seed = int(run.get("base_seed", 12345))
    capture_signals = _as_bool(run.get("capture_signals"), default=True)
    max_workers_raw = run.get("max_workers", None)
    max_workers = int(max_workers_raw) if max_workers_raw is not None else None
    if max_workers is not None and max_workers < 1:
        raise ConfigError("run.max_workers must be >= 1 when set.")

    return BenchmarkRunConfig(
        run_id=run_id,
        mode=mode,
        output_dir=output_dir.resolve(),
        scenarios=scenarios,
        estimators=estimators,
        custom_estimators=custom_estimators,
        metric_profile=metric_profile,
        metric_include=metric_include,
        n_runs=n_runs,
        base_seed=base_seed,
        capture_signals=capture_signals,
        max_workers=max_workers,
        parameter_policy=parameter_policy,
        tuned_artifacts_dir=tuned_artifacts_dir,
        source_path=source_path.resolve() if source_path is not None else None,
    )
