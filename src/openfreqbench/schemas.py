from __future__ import annotations

from copy import deepcopy
from typing import Any

SCHEMAS: dict[str, dict[str, Any]] = {
    "benchmark-report": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://openfreqbench.org/schemas/benchmark-report-v1.json",
        "title": "OpenFreqBench benchmark_report.json",
        "type": "object",
        "required": [
            "metadata",
            "run_configuration",
            "reproducibility",
            "raw_run_records",
            "aggregated_metrics",
            "artifacts",
        ],
        "properties": {
            "metadata": {"type": "object"},
            "run_configuration": {
                "type": "object",
                "required": ["run_id", "metric_profile", "metrics_locked", "scenarios", "estimators"],
            },
            "reproducibility": {"type": "object"},
            "raw_run_records": {"type": "array", "items": {"type": "object"}},
            "aggregated_metrics": {"type": "array", "items": {"type": "object"}},
            "artifacts": {"type": "object"},
        },
    },
    "manifest": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://openfreqbench.org/schemas/manifest-v1.json",
        "title": "OpenFreqBench reproducibility or evidence manifest",
        "type": "object",
        "required": ["timestamp_utc"],
        "properties": {
            "timestamp_utc": {"type": "string"},
            "created_utc": {"type": "string"},
            "git": {"type": "object"},
            "source_hashes": {"type": "object"},
            "checkpoints": {"type": "object"},
            "manifest_sha256": {"type": "string"},
        },
    },
    "atlas-readiness": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://openfreqbench.org/schemas/atlas-readiness-v1.json",
        "title": "OpenFreqBench ATLAS readiness report",
        "type": "object",
        "required": [
            "schema_version",
            "method_version",
            "status",
            "scope",
            "paper_claims_allowed",
            "journal_claims_allowed",
            "summary",
            "issues",
        ],
        "properties": {
            "schema_version": {"type": "string"},
            "method_version": {"type": "string"},
            "created_utc": {"type": "string"},
            "status": {"type": "string"},
            "scope": {"type": "string"},
            "paper_claims_allowed": {"type": "boolean"},
            "journal_claims_allowed": {"type": "boolean"},
            "settings": {"type": "object"},
            "summary": {"type": "object"},
            "issues": {"type": "array"},
            "required_next_action": {"type": "string"},
        },
    },
    "run-config": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://openfreqbench.org/schemas/run-config-v1.json",
        "title": "OpenFreqBench YAML run config",
        "type": "object",
        "required": ["run", "benchmark", "metrics"],
        "properties": {
            "run": {
                "type": "object",
                "required": ["id", "n_runs", "base_seed"],
            },
            "benchmark": {
                "type": "object",
                "required": ["scenarios", "estimators"],
            },
            "metrics": {
                "type": "object",
                "required": ["profile"],
            },
        },
    },
}


def schema_names() -> list[str]:
    return sorted(SCHEMAS)


def get_schema(name: str) -> dict[str, Any]:
    if name not in SCHEMAS:
        raise KeyError(f"Unknown schema {name!r}. Known schemas: {schema_names()}")
    return deepcopy(SCHEMAS[name])


def validate_payload(name: str, payload: dict[str, Any]) -> list[str]:
    schema = get_schema(name)
    errors: list[str] = []
    required = schema.get("required", [])
    for field in required:
        if field not in payload:
            errors.append(f"Missing required field: {field}")
    properties = schema.get("properties", {})
    for field, spec in properties.items():
        if field not in payload:
            continue
        expected = spec.get("type")
        value = payload[field]
        if expected == "object" and not isinstance(value, dict):
            errors.append(f"Field {field} must be an object.")
        elif expected == "array" and not isinstance(value, list):
            errors.append(f"Field {field} must be an array.")
        elif expected == "string" and not isinstance(value, str):
            errors.append(f"Field {field} must be a string.")
        elif expected == "boolean" and not isinstance(value, bool):
            errors.append(f"Field {field} must be a boolean.")
    return errors
