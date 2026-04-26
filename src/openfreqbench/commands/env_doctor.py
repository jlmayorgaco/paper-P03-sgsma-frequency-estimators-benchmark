from __future__ import annotations

import importlib
import json
import platform
import sys
from pathlib import Path


def _check_module(name: str) -> dict[str, str]:
    try:
        importlib.import_module(name)
        return {"name": name, "status": "ok"}
    except Exception as exc:
        return {"name": name, "status": "missing", "detail": str(exc)}


def env_doctor(_: object) -> int:
    root = Path(__file__).resolve().parents[3]
    required = ["numpy", "scipy", "pandas", "matplotlib", "optuna", "numba", "sklearn"]
    optional = ["torch", "andes"]

    checks = {
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": str(root),
        "required_modules": [_check_module(m) for m in required],
        "optional_modules": [_check_module(m) for m in optional],
        "paths": {
            "artifacts_dir_exists": (root / "artifacts" / "full_mc_benchmark").exists(),
            "paper_dir_exists": (root / "paper").exists(),
            "pi_gru_weights_exists": (root / "src" / "estimators" / "pi_gru_weights.pt").exists(),
        },
    }
    print(json.dumps(checks, indent=2))
    missing_required = any(item["status"] != "ok" for item in checks["required_modules"])
    return 1 if missing_required else 0
