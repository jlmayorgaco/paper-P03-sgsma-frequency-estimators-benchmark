from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1]


def _run(module: str) -> None:
    print(f"\n[DUAL] Starting {module}", flush=True)
    subprocess.run([sys.executable, "-m", module], cwd=SRC, check=True)
    print(f"[DUAL] Finished {module}", flush=True)


def main() -> None:
    _run("pipelines.voltage_step_sweep_fixed_policy")
    _run("pipelines.voltage_step_sweep")


if __name__ == "__main__":
    main()
