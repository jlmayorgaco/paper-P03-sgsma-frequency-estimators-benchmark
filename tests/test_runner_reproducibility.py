from __future__ import annotations

import json

from openfreqbench.config import parse_config
from openfreqbench.runner import run_benchmark_config


def test_artifact_tuned_policy_loads_run_spec_and_writes_repro_manifest(tmp_path) -> None:
    tuned_dir = tmp_path / "tuned"
    spec_dir = tuned_dir / "IEEE_Single_SinWave" / "ZCD"
    spec_dir.mkdir(parents=True)
    (spec_dir / "run_spec.json").write_text(
        json.dumps({"params": {"nominal_f": 60.0}}),
        encoding="utf-8",
    )
    cfg = parse_config(
        {
            "run": {
                "id": "artifact-tuned-smoke",
                "mode": "single",
                "output_dir": str(tmp_path / "out"),
                "n_runs": 1,
                "base_seed": 12345,
                "max_workers": 1,
                "capture_signals": False,
            },
            "benchmark": {
                "parameter_policy": "artifact_tuned",
                "tuned_artifacts_dir": str(tuned_dir),
                "scenarios": ["IEEE_Single_SinWave"],
                "estimators": ["ZCD"],
            },
            "metrics": {"profile": "canonical-single-phase-v1"},
        }
    )

    result = run_benchmark_config(cfg)
    report = json.loads((tmp_path / "out" / "artifact-tuned-smoke" / "benchmark_report.json").read_text())
    run_spec = json.loads(
        (
            tmp_path
            / "out"
            / "artifact-tuned-smoke"
            / "IEEE_Single_SinWave"
            / "ZCD"
            / "run_spec.json"
        ).read_text()
    )
    assert result["n_records"] == 1
    assert report["run_configuration"]["parameter_policy"] == "artifact_tuned"
    assert "reproducibility" in report
    assert "manifest_sha256" in report["reproducibility"]
    assert run_spec["parameter_source"]["policy"] == "artifact_tuned"
    assert "tuned_artifact" in run_spec["parameter_source"]

