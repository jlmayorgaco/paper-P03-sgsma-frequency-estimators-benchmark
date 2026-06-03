from __future__ import annotations

import json

import pytest

from openfreqbench.artifacts import freeze_artifacts, validate_tuned_artifacts
from openfreqbench.config import parse_config
from openfreqbench.contracts import ContractError, validate_estimator_contract, validate_scenario_contract
from openfreqbench.schemas import get_schema, validate_payload


class GoodEstimator:
    name = "Good"

    def step(self, z, t_s=None, memory=None):
        return 60.0, memory


class BadEstimator:
    name = "Bad"


class GoodScenario:
    SCENARIO_NAME = "GoodScenario"
    DEFAULT_PARAMS = {}

    @classmethod
    def get_name(cls):
        return cls.SCENARIO_NAME

    def run(self, seed=None):
        return {}


class BadScenario:
    SCENARIO_NAME = "BadScenario"
    DEFAULT_PARAMS = {}


def test_public_contract_validators_accept_minimal_extensions() -> None:
    validate_estimator_contract(GoodEstimator)
    validate_scenario_contract(GoodScenario)


def test_public_contract_validators_reject_bad_extensions() -> None:
    with pytest.raises(ContractError):
        validate_estimator_contract(BadEstimator)
    with pytest.raises(ContractError):
        validate_scenario_contract(BadScenario)


def test_schema_registry_validates_required_report_fields() -> None:
    schema = get_schema("benchmark-report")
    assert schema["title"] == "OpenFreqBench benchmark_report.json"
    errors = validate_payload("benchmark-report", {"metadata": {}})
    assert "Missing required field: run_configuration" in errors


def test_schema_registry_exposes_atlas_readiness() -> None:
    schema = get_schema("atlas-readiness")
    assert schema["title"] == "OpenFreqBench ATLAS readiness report"
    errors = validate_payload(
        "atlas-readiness",
        {
            "schema_version": "openfreqbench-atlas-readiness-v1",
            "method_version": "test",
            "status": "diagnostic",
            "scope": "subset",
            "paper_claims_allowed": False,
            "journal_claims_allowed": False,
            "summary": {},
            "issues": [],
        },
    )
    assert errors == []


def test_validate_tuned_artifacts_reports_missing_pairs(tmp_path) -> None:
    tuned_dir = tmp_path / "tuned"
    (tuned_dir / "IEEE_Single_SinWave" / "ZCD").mkdir(parents=True)
    (tuned_dir / "IEEE_Single_SinWave" / "ZCD" / "run_spec.json").write_text(
        json.dumps({"params": {}}),
        encoding="utf-8",
    )
    cfg = parse_config(
        {
            "run": {"id": "validate", "n_runs": 1},
            "benchmark": {
                "parameter_policy": "artifact_tuned",
                "tuned_artifacts_dir": str(tuned_dir),
                "scenarios": ["IEEE_Single_SinWave"],
                "estimators": ["ZCD", "IPDFT", "LKF", "LKF2", "PI-GRU"],
            },
            "metrics": {"profile": "canonical-single-phase-v1"},
        }
    )
    result = validate_tuned_artifacts(cfg)
    assert result["status"] == "fail"
    assert result["n_expected_pairs"] == 5
    assert result["n_missing_pairs"] == 4
    assert result["n_invalid_specs"] == 0


def test_validate_tuned_artifacts_accepts_best_params_and_rejects_empty_spec(tmp_path) -> None:
    tuned_dir = tmp_path / "tuned"
    (tuned_dir / "IEEE_Single_SinWave" / "ZCD").mkdir(parents=True)
    (tuned_dir / "IEEE_Single_SinWave" / "IPDFT").mkdir(parents=True)
    (tuned_dir / "IEEE_Single_SinWave" / "ZCD" / "run_spec.json").write_text(
        json.dumps({"best_params": {}}),
        encoding="utf-8",
    )
    (tuned_dir / "IEEE_Single_SinWave" / "IPDFT" / "run_spec.json").write_text(
        json.dumps({"tuning_meta": {}}),
        encoding="utf-8",
    )
    cfg = parse_config(
        {
            "run": {"id": "validate", "n_runs": 1},
            "benchmark": {
                "parameter_policy": "artifact_tuned",
                "tuned_artifacts_dir": str(tuned_dir),
                "scenarios": ["IEEE_Single_SinWave"],
                "estimators": ["ZCD", "IPDFT"],
            },
            "metrics": {"profile": "canonical-single-phase-v1"},
        }
    )

    result = validate_tuned_artifacts(cfg)

    assert result["status"] == "fail"
    assert result["n_present_pairs"] == 1
    assert result["present"][0]["param_key"] == "best_params"
    assert result["n_invalid_specs"] == 1


def test_freeze_artifacts_writes_traceability_and_index(tmp_path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    report = {
        "metadata": {},
        "run_configuration": {"run_id": "run", "metric_profile": "canonical-single-phase-v1"},
        "reproducibility": {"git": {"commit": "abc"}, "config": {"path": "config.yaml"}},
        "raw_run_records": [],
        "aggregated_metrics": [
            {
                "scenario": "IEEE_Single_SinWave",
                "estimator": "ZCD",
                "m1_rmse_hz_mean": 0.01,
            }
        ],
        "artifacts": {"aggregated_metrics_csv": str(run_root / "aggregated_metrics.csv")},
    }
    (run_root / "aggregated_metrics.csv").write_text("scenario,estimator,m1_rmse_hz_mean\ns,e,0.01\n")
    (run_root / "benchmark_report.json").write_text(json.dumps(report), encoding="utf-8")

    result = freeze_artifacts(run_root, package_root=tmp_path, source_root=tmp_path)

    assert (run_root / "artifact_index.csv").exists()
    assert (run_root / "paper_traceability.csv").exists()
    assert (run_root / "evidence_manifest.json").exists()
    assert result["paper_traceability"].endswith("paper_traceability.csv")
