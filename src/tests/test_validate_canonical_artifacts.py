from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipelines.validate_canonical_artifacts import validate_canonical_artifacts


def test_validate_canonical_artifacts_detects_missing_top_level(tmp_path: Path) -> None:
    result = validate_canonical_artifacts(tmp_path)
    assert not result.ok
    assert "global_metrics_report.csv" in result.missing_top_level
    assert "benchmark_full_report.json" in result.missing_top_level
