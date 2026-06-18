from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    if not raw:
        return default
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


BENCHMARK_OUTPUT_DIR = _env_path("BENCHMARK_OUTPUT_DIR", ROOT / "artifacts" / "full_mc_benchmark")
PAPER_FIGURES_DIR = ROOT / "paper" / "Figures" / "Plots_And_Graphs"

FIGURE1_BASENAME = "Fig1_Scenarios_Final"
FIGURE2_BASENAME = "Fig2_Mega_Dashboard"
JSON_REPORT_NAME = "benchmark_full_report.json"

