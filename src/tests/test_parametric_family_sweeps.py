from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_parametric_sweep_plan_cli() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "pipelines.parametric_family_sweeps", "--plan-only"],
        cwd=ROOT / "src",
        check=True,
        capture_output=True,
        text=True,
    )
    stdout = proc.stdout
    assert "Families: 8" in stdout
    assert "Scenarios: 240" in stdout
    assert "IBR Harmonics Severity" in stdout
    assert "IEEE Magnitude Step" in stdout
