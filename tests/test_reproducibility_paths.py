from __future__ import annotations

from openfreqbench.paths import SOURCE_ROOT
from openfreqbench.reproducibility import checkpoint_manifest, source_hash_manifest


def test_source_hash_manifest_works_from_source_root() -> None:
    hashes = source_hash_manifest(SOURCE_ROOT)
    assert "src/openfreqbench/runner.py" in hashes
    assert "src/analysis/metrics.py" in hashes


def test_checkpoint_manifest_uses_stable_public_keys() -> None:
    checkpoints = checkpoint_manifest(SOURCE_ROOT)
    assert "src/estimators/pi_gru_weights_hybrid.pt" in checkpoints


def test_manifest_helpers_accept_installed_layout(tmp_path) -> None:
    for rel in [
        "openfreqbench/config.py",
        "openfreqbench/runner.py",
        "openfreqbench/registry.py",
        "openfreqbench/reports.py",
        "openfreqbench/paths.py",
        "openfreqbench/reproducibility.py",
        "analysis/metrics.py",
        "analysis/monte_carlo_engine.py",
        "scenarios/base.py",
        "estimators/base.py",
        "pipelines/benchmark_definition.py",
    ]:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x = 1\n", encoding="utf-8")
    checkpoint = tmp_path / "estimators" / "pi_gru_weights_hybrid.pt"
    checkpoint.write_bytes(b"weights")

    hashes = source_hash_manifest(tmp_path)
    checkpoints = checkpoint_manifest(tmp_path)

    assert "src/openfreqbench/config.py" in hashes
    assert "src/openfreqbench/paths.py" in hashes
    assert "src/estimators/pi_gru_weights_hybrid.pt" in checkpoints
