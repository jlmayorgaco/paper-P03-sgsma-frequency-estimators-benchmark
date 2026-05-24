from __future__ import annotations

import pytest

from openfreqbench.config import ConfigError, parse_config


def test_metric_formula_definitions_are_rejected() -> None:
    with pytest.raises(ConfigError):
        parse_config(
            {
                "run": {"id": "bad"},
                "benchmark": {"scenarios": ["IEEE_Single_SinWave"], "estimators": ["ZCD"]},
                "metrics": {
                    "profile": "canonical-single-phase-v1",
                    "formulas": {"m1_rmse_hz": "custom"},
                },
            }
        )


def test_quick_config_shape_is_valid() -> None:
    cfg = parse_config(
        {
            "run": {"id": "ok", "n_runs": 1},
            "benchmark": {"scenarios": ["IEEE_Single_SinWave"], "estimators": ["ZCD"]},
            "metrics": {"profile": "canonical-single-phase-v1", "include": ["m1_rmse_hz"]},
        }
    )
    assert cfg.run_id == "ok"
    assert cfg.scenarios == ["IEEE_Single_SinWave"]
    assert cfg.estimators[0].name == "ZCD"


def test_artifact_tuned_requires_artifact_dir() -> None:
    with pytest.raises(ConfigError):
        parse_config(
            {
                "run": {"id": "bad", "n_runs": 1},
                "benchmark": {
                    "parameter_policy": "artifact_tuned",
                    "scenarios": ["IEEE_Single_SinWave"],
                    "estimators": ["ZCD"],
                },
                "metrics": {"profile": "canonical-single-phase-v1"},
            }
        )


def test_artifact_tuned_config_shape_is_valid(tmp_path) -> None:
    tuned = tmp_path / "tuned"
    cfg = parse_config(
        {
            "run": {"id": "ok", "n_runs": 1},
            "benchmark": {
                "parameter_policy": "artifact_tuned",
                "tuned_artifacts_dir": str(tuned),
                "scenarios": ["IEEE_Single_SinWave"],
                "estimators": ["ZCD"],
            },
            "metrics": {"profile": "canonical-single-phase-v1"},
        }
    )
    assert cfg.parameter_policy == "artifact_tuned"
    assert cfg.tuned_artifacts_dir == tuned.resolve()
