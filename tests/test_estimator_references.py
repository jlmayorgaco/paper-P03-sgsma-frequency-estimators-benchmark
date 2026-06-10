from __future__ import annotations

import ast
from pathlib import Path

from estimators.references import ESTIMATOR_REFERENCE_KEYS, REFERENCE_REGISTRY
from pipelines.benchmark_definition import (
    ALL_ESTIMATOR_SPECS,
    build_estimator_registry_manifest,
)
from openfreqbench.registry import platform_manifest


ROOT = Path(__file__).resolve().parents[1]
ESTIMATOR_ROOT = ROOT / "src" / "estimators"
EXTRA_METHOD_MODULES = {"epll", "music_experimental"}


def _module_reference_keys(module_name: str) -> tuple[str, ...]:
    path = ESTIMATOR_ROOT / f"{module_name}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        target_names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        if "REFERENCE_KEYS" in target_names:
            value = ast.literal_eval(node.value)
            return tuple(value)
    raise AssertionError(f"{path} does not define REFERENCE_KEYS")


def test_estimator_reference_registry_covers_all_method_modules() -> None:
    module_names = {spec.module_name for spec in ALL_ESTIMATOR_SPECS} | EXTRA_METHOD_MODULES
    missing_from_registry = module_names - set(ESTIMATOR_REFERENCE_KEYS)
    assert not missing_from_registry

    for module_name in sorted(module_names):
        expected = ESTIMATOR_REFERENCE_KEYS[module_name]
        assert expected, module_name
        assert _module_reference_keys(module_name) == expected
        unknown = set(expected) - set(REFERENCE_REGISTRY)
        assert not unknown, f"{module_name} references unknown keys: {sorted(unknown)}"


def test_estimator_manifest_exports_reference_keys() -> None:
    manifest = build_estimator_registry_manifest()
    for section in ("active", "excluded"):
        for entry in manifest[section]:
            assert entry["reference_keys"], entry["module_name"]

    public_manifest = platform_manifest()
    for entry in public_manifest["estimators"]:
        assert entry["reference_keys"], entry["module_name"]
