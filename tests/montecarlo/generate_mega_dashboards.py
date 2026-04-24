from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from plotting.benchmark.generate_mega_dashboards import *  # noqa: F401,F403
from plotting.benchmark.generate_mega_dashboards import generate_benchmark_figures


if __name__ == "__main__":
    generate_benchmark_figures()
