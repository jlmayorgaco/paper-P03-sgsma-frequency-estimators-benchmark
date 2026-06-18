from __future__ import annotations

import json

from openfreqbench.config import CustomEstimatorSelection, parse_config
from openfreqbench.runner import _load_custom_estimator, run_benchmark_config


def test_artifact_tuned_policy_loads_run_spec_and_writes_repro_manifest(tmp_path) -> None:
    tuned_dir = tmp_path / "tuned"
    spec_dir = tuned_dir / "IEEE_Single_SinWave" / "ZCD"
    spec_dir.mkdir(parents=True)
    (spec_dir / "run_spec.json").write_text(
        json.dumps({"best_params": {"nominal_f": 60.0}}),
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
            "metrics": {"profile": "canonical-single-phase-v1", "include": ["m1_rmse_hz"]},
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
    assert report["run_configuration"]["metric_include"] == ["m1_rmse_hz"]
    assert "m13_cpu_time_us" in report["run_configuration"]["metrics"]
    assert "reproducibility" in report
    assert "manifest_sha256" in report["reproducibility"]
    assert run_spec["parameter_source"]["policy"] == "artifact_tuned"
    assert "tuned_artifact" in run_spec["parameter_source"]
    assert run_spec["parameter_source"]["tuned_artifact"]["param_key"] == "best_params"


def test_custom_estimator_loader_uses_file_identity_for_same_stem(tmp_path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_path = first_dir / "plugin.py"
    second_path = second_dir / "plugin.py"
    first_path.write_text(
        "class MyEstimator:\n"
        "    def step(self, *_args, **_kwargs):\n"
        "        return 51.0\n",
        encoding="utf-8",
    )
    second_path.write_text(
        "class MyEstimator:\n"
        "    def step(self, *_args, **_kwargs):\n"
        "        return 61.0\n",
        encoding="utf-8",
    )

    first_cls = _load_custom_estimator(
        CustomEstimatorSelection(name="First", path=first_path, class_name="MyEstimator")
    )
    second_cls = _load_custom_estimator(
        CustomEstimatorSelection(name="Second", path=second_path, class_name="MyEstimator")
    )

    assert first_cls.__module__ != second_cls.__module__
    assert first_cls().step() == 51.0
    assert second_cls().step() == 61.0
