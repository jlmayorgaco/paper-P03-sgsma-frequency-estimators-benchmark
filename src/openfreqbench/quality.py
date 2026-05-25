from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import load_config
from .registry import CANONICAL_METRIC_PROFILE, metric_registry, scenario_registry
from .reproducibility import build_reproducibility_manifest
from .runner import dry_run_manifest


def _check_file(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else 0}


def _run_pytest(root: Path) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": "python -m pytest tests -q -p no:cacheprovider",
        "returncode": int(proc.returncode),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def run_quality_gate(root: Path, *, run_tests: bool = True, release: bool = False) -> dict[str, Any]:
    required_files = [
        root / "README.md",
        root / "docs" / "ARCHITECTURE.md",
        root / "docs" / "JOURNAL_RESULTS_PROTOCOL.md",
        root / "docs" / "METHODS.md",
        root / "docs" / "VALIDATION.md",
        root / "docs" / "RESEARCHER_CONTRACT.md",
        root / "docs" / "SCIENTIFIC_READINESS.md",
        root / "docs" / "RELEASE.md",
        root / "docs" / "MVP2_RELEASE_NOTES.md",
        root / "docs" / "TUTORIALS.md",
        root / "docs" / "OUTPUT_SCHEMA.md",
        root / "docs" / "WEIGHTS.md",
        root / "schemas" / "benchmark_report.schema.json",
        root / "schemas" / "manifest.schema.json",
        root / "schemas" / "run_config.schema.json",
        root / "schemas" / "atlas_readiness.schema.json",
        root / "examples" / "README.md",
        root / ".github" / "workflows" / "ci.yml",
        root / "scripts" / "verify_local.ps1",
        root / "scripts" / "verify_local.sh",
        root / "scripts" / "verify_release.ps1",
        root / "scripts" / "verify_release.sh",
        root / "CONTRIBUTING.md",
        root / "CITATION.cff",
        root / "pyproject.toml",
    ]
    file_checks = [_check_file(path) for path in required_files]
    config_checks = []
    for cfg in sorted((root / "configs").glob("*.yaml")):
        if cfg.name.startswith("hypotheses"):
            continue
        try:
            parsed = load_config(cfg)
            plan = dry_run_manifest(parsed)
            config_checks.append({"path": str(cfg), "ok": True, "run_id": plan["run_id"]})
        except Exception as exc:
            config_checks.append({"path": str(cfg), "ok": False, "error": str(exc)})

    registry_check = {
        "scenario_count": len(scenario_registry()),
        "metric_count": len(metric_registry()),
        "metric_profile": CANONICAL_METRIC_PROFILE,
        "metric_profile_locked": CANONICAL_METRIC_PROFILE == "canonical-single-phase-v1",
    }
    pytest_result = _run_pytest(root) if run_tests else {"skipped": True}
    repro = build_reproducibility_manifest(root, None)
    release_checks = {
        "enabled": bool(release),
        "git_clean": not bool(repro.get("git", {}).get("dirty")),
        "remote_origin_set": bool(repro.get("git", {}).get("remote_origin")),
        "citation_present": (root / "CITATION.cff").exists(),
    }
    checks_ok = (
        all(item["exists"] and item["size_bytes"] > 0 for item in file_checks)
        and all(item["ok"] for item in config_checks)
        and registry_check["scenario_count"] >= 32
        and registry_check["metric_count"] >= 33
        and bool(registry_check["metric_profile_locked"])
        and (not run_tests or pytest_result.get("returncode") == 0)
        and (not release or all(bool(v) for k, v in release_checks.items() if k != "enabled"))
    )
    payload = {
        "status": "pass" if checks_ok else "fail",
        "files": file_checks,
        "configs": config_checks,
        "registry": registry_check,
        "release_checks": release_checks,
        "pytest": pytest_result,
        "reproducibility": repro,
    }
    payload["json_valid"] = bool(json.dumps(payload, default=str))
    return payload
