from __future__ import annotations

import csv
import importlib
import json
import platform
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import BenchmarkRunConfig
from .registry import CANONICAL_METRIC_IDS, estimator_spec_registry, scenario_registry
from .reproducibility import build_reproducibility_manifest, git_manifest, sha256_file, sha256_text

ARTIFACT_EXTENSIONS = {
    ".csv",
    ".json",
    ".yaml",
    ".yml",
    ".png",
    ".pdf",
    ".md",
    ".txt",
    ".log",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def module_availability(required: list[str] | None = None, optional: list[str] | None = None) -> list[dict[str, Any]]:
    required = required or ["numpy", "scipy", "pandas", "matplotlib", "yaml"]
    optional = optional or ["optuna", "numba", "sklearn", "torch", "andes", "opendssdirect"]
    rows: list[dict[str, Any]] = []
    for name in required + optional:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", None)
            rows.append(
                {
                    "module": name,
                    "required": name in required,
                    "status": "ok",
                    "version": version,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "module": name,
                    "required": name in required,
                    "status": "missing",
                    "version": None,
                    "detail": str(exc),
                }
            )
    return rows


def environment_report(root: Path, *, config_path: Path | None = None, source_root: Path | None = None) -> dict[str, Any]:
    repro = build_reproducibility_manifest(root, config_path, source_root=source_root)
    checks = module_availability()
    torch_ok = any(row["module"] == "torch" and row["status"] == "ok" for row in checks)
    checkpoint_count = len(repro.get("checkpoints", {}))
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "node": platform.node(),
        },
        "git": git_manifest(root),
        "modules": checks,
        "pi_gru": {
            "torch_available": bool(torch_ok),
            "checkpoint_available": bool(checkpoint_count),
            "checkpoint_count": int(checkpoint_count),
        },
        "reproducibility_manifest_sha256": repro.get("manifest_sha256"),
        "source_hashes": repro.get("source_hashes", {}),
        "checkpoints": repro.get("checkpoints", {}),
    }


def write_environment_report(
    root: Path,
    output_path: Path,
    *,
    config_path: Path | None = None,
    source_root: Path | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = environment_report(root, config_path=config_path, source_root=source_root)
    output_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    return output_path


def build_artifact_index(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    run_root = run_root.resolve()
    for path in sorted(run_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in ARTIFACT_EXTENSIONS:
            continue
        try:
            rel = path.relative_to(run_root).as_posix()
        except ValueError:
            rel = str(path)
        rows.append(
            {
                "relative_path": rel,
                "artifact_path": str(path),
                "type": path.suffix.lower().lstrip("."),
                "size_bytes": int(path.stat().st_size),
                "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                "sha256": sha256_file(path),
            }
        )
    return rows


def write_artifact_index(run_root: Path, output_path: Path | None = None) -> Path:
    output_path = output_path or (run_root / "artifact_index.csv")
    rows = build_artifact_index(run_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["relative_path", "artifact_path", "type", "size_bytes", "modified_utc", "sha256"]
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _report_command(report: dict[str, Any]) -> str:
    config = report.get("reproducibility", {}).get("config", {})
    config_path = config.get("path")
    if config_path:
        return f"openfreqbench run --config {config_path}"
    return "openfreqbench run --config <unknown>"


def write_paper_traceability(report_json: Path, output_path: Path | None = None) -> Path:
    report = json.loads(report_json.read_text(encoding="utf-8"))
    output_path = output_path or (report_json.parent / "paper_traceability.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aggregated_path = Path(report.get("artifacts", {}).get("aggregated_metrics_csv", ""))
    artifact_path = aggregated_path if aggregated_path.exists() else report_json
    artifact_hash = sha256_file(artifact_path)
    commit = report.get("reproducibility", {}).get("git", {}).get("commit")
    script = _report_command(report)
    rows: list[dict[str, Any]] = []
    for row in report.get("aggregated_metrics", []):
        scenario = str(row.get("scenario", ""))
        estimator = str(row.get("estimator", ""))
        for metric in CANONICAL_METRIC_IDS:
            mean_key = f"{metric}_mean"
            if mean_key not in row and metric not in row:
                continue
            value_key = mean_key if mean_key in row else metric
            rows.append(
                {
                    "claim": f"{scenario} / {estimator} / {metric} = {row.get(value_key)}",
                    "metric": metric,
                    "scenario": scenario,
                    "estimator": estimator,
                    "artifact_path": str(artifact_path),
                    "hash": artifact_hash,
                    "script": script,
                    "commit": commit or "",
                }
            )
    fieldnames = ["claim", "metric", "scenario", "estimator", "artifact_path", "hash", "script", "commit"]
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def write_evidence_manifest(
    run_root: Path,
    output_path: Path | None = None,
    *,
    source_report: Path | None = None,
) -> Path:
    output_path = output_path or (run_root / "evidence_manifest.json")
    report_path = source_report or (run_root / "benchmark_report.json")
    report_hash = sha256_file(report_path) if report_path.exists() else None
    artifact_rows = build_artifact_index(run_root)
    payload = {
        "schema_version": "openfreqbench-evidence-manifest-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root.resolve()),
        "source_report": str(report_path),
        "source_report_sha256": report_hash,
        "n_artifacts": len(artifact_rows),
        "artifacts": artifact_rows,
    }
    payload["manifest_sha256"] = sha256_text(json.dumps(payload, sort_keys=True, default=str))
    output_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    return output_path


def freeze_artifacts(
    run_root: Path,
    *,
    package_root: Path,
    source_root: Path | None = None,
    config_path: Path | None = None,
    make_zip: bool = False,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    if not run_root.exists():
        raise FileNotFoundError(f"Run root not found: {run_root}")
    env_path = write_environment_report(
        package_root,
        run_root / "environment_report.json",
        config_path=config_path,
        source_root=source_root,
    )
    report_path = run_root / "benchmark_report.json"
    trace_path = ""
    if report_path.exists():
        trace_path = str(write_paper_traceability(report_path))
    index_path = write_artifact_index(run_root)
    evidence_path = write_evidence_manifest(run_root, source_report=report_path if report_path.exists() else None)
    archive_manifest = {
        "schema_version": "openfreqbench-archive-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "environment_report": str(env_path),
        "artifact_index": str(index_path),
        "paper_traceability": trace_path,
        "evidence_manifest": str(evidence_path),
        "git": git_manifest(package_root),
    }
    zip_path = ""
    if make_zip:
        zip_file = run_root.with_suffix(".zip")
        with zipfile.ZipFile(zip_file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(run_root.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(run_root))
        zip_path = str(zip_file)
        archive_manifest["zip_path"] = zip_path
        archive_manifest["zip_sha256"] = sha256_file(zip_file)
    archive_manifest["archive_manifest_sha256"] = sha256_text(
        json.dumps(archive_manifest, sort_keys=True, default=str)
    )
    manifest_path = run_root / "archive_manifest.json"
    manifest_path.write_text(json.dumps(_json_safe(archive_manifest), indent=2), encoding="utf-8")
    return {
        "archive_manifest": str(manifest_path),
        "environment_report": str(env_path),
        "artifact_index": str(index_path),
        "paper_traceability": trace_path,
        "evidence_manifest": str(evidence_path),
        "zip_path": zip_path,
    }


def validate_tuned_artifacts(config: BenchmarkRunConfig) -> dict[str, Any]:
    if config.parameter_policy != "artifact_tuned":
        raise ValueError("Artifact validation requires benchmark.parameter_policy=artifact_tuned.")
    if config.tuned_artifacts_dir is None:
        raise ValueError("benchmark.tuned_artifacts_dir is required.")

    known_scenarios = scenario_registry()
    known_estimators = estimator_spec_registry(include_experimental=False)
    unknown_scenarios = [name for name in config.scenarios if name not in known_scenarios]
    unknown_estimators = [item.name for item in config.estimators if item.name not in known_estimators]

    present: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for scenario in config.scenarios:
        for estimator in [item.name for item in config.estimators]:
            safe = estimator.replace("/", "_")
            candidates = [
                config.tuned_artifacts_dir / scenario / estimator / "run_spec.json",
                config.tuned_artifacts_dir / scenario / safe / "run_spec.json",
            ]
            found = next((path for path in candidates if path.exists()), None)
            if found is None:
                missing.append({"scenario": scenario, "estimator": estimator})
                continue
            present.append(
                {
                    "scenario": scenario,
                    "estimator": estimator,
                    "run_spec": str(found),
                    "sha256": sha256_file(found),
                }
            )

    expected_pairs = len(config.scenarios) * len(config.estimators)
    status = "pass" if not unknown_scenarios and not unknown_estimators and not missing else "fail"
    required_labels = {"LKF", "LKF2", "PI-GRU"}
    configured_labels = {item.name for item in config.estimators}
    missing_required = sorted(required_labels - configured_labels)
    if len(config.scenarios) >= 32 and missing_required:
        status = "fail"
    return {
        "status": status,
        "run_id": config.run_id,
        "parameter_policy": config.parameter_policy,
        "tuned_artifacts_dir": str(config.tuned_artifacts_dir),
        "n_scenarios": len(config.scenarios),
        "n_estimators": len(config.estimators),
        "n_expected_pairs": expected_pairs,
        "n_present_pairs": len(present),
        "n_missing_pairs": len(missing),
        "unknown_scenarios": unknown_scenarios,
        "unknown_estimators": unknown_estimators,
        "missing_required_estimators": missing_required,
        "present": present,
        "missing": missing,
    }
