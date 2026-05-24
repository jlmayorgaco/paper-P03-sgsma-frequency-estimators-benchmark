from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        probe = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except Exception:
        return None
    if probe.returncode != 0:
        return None
    return probe.stdout.strip()


def git_manifest(root: Path) -> dict[str, Any]:
    # OpenFreqBench may live as a subdirectory inside a larger research repo.
    # The release gate therefore scopes dirtiness to the package root.
    short_status = _git(["status", "--short", "--", "."], root)
    return {
        "commit": _git(["rev-parse", "HEAD"], root),
        "branch": _git(["branch", "--show-current"], root),
        "dirty": bool(short_status),
        "status_short": short_status or "",
        "remote_origin": _git(["remote", "get-url", "origin"], root),
    }


def dependency_manifest(names: list[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def source_hash_manifest(root: Path) -> dict[str, dict[str, Any]]:
    tracked = [
        "src/openfreqbench/config.py",
        "src/openfreqbench/runner.py",
        "src/openfreqbench/registry.py",
        "src/openfreqbench/reports.py",
        "src/openfreqbench/reproducibility.py",
        "src/analysis/metrics.py",
        "src/analysis/monte_carlo_engine.py",
        "src/scenarios/base.py",
        "src/estimators/base.py",
        "src/pipelines/benchmark_definition.py",
    ]
    out: dict[str, dict[str, Any]] = {}
    for rel in tracked:
        path = root / rel
        if path.exists():
            out[rel] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    return out


def checkpoint_manifest(root: Path) -> dict[str, dict[str, Any]]:
    estimator_dir = root / "src" / "estimators"
    out: dict[str, dict[str, Any]] = {}
    for pattern in ("*.pt", "*.npz", "*.json"):
        for path in sorted(estimator_dir.glob(pattern)):
            if path.name.startswith("pi_gru"):
                rel = path.relative_to(root).as_posix()
                out[rel] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    return out


def config_manifest(config_path: Path | None) -> dict[str, Any]:
    if config_path is None or not config_path.exists():
        return {"path": str(config_path) if config_path else None, "sha256": None}
    return {
        "path": str(config_path),
        "sha256": sha256_file(config_path),
        "size_bytes": config_path.stat().st_size,
    }


def build_reproducibility_manifest(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    dependencies = [
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "optuna",
        "numba",
        "scikit-learn",
        "PyYAML",
        "torch",
        "andes",
        "opendssdirect.py",
    ]
    payload = {
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
        "dependencies": dependency_manifest(dependencies),
        "config": config_manifest(config_path),
        "source_hashes": source_hash_manifest(root),
        "checkpoints": checkpoint_manifest(root),
    }
    payload["manifest_sha256"] = sha256_text(json.dumps(payload, sort_keys=True, default=str))
    return payload
