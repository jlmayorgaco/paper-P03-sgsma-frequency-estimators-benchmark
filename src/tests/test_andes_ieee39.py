from __future__ import annotations

import json
from pathlib import Path

from pipelines.andes_ieee39 import run_andes_ieee39


def test_andes_ieee39_runner_produces_manifest(tmp_path: Path) -> None:
    result = run_andes_ieee39(seed=123, output_dir=tmp_path)
    manifest_path = Path(result["manifest_path"])
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["network"] == "IEEE39"
    assert data["event_count"] == 4
    assert len(data["events"]) == 4
