from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
BASE_PATH = SRC / "estimators" / "base.py"
SPEC = importlib.util.spec_from_file_location("local_estimators_base", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load estimator base module from {BASE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
BaseFrequencyEstimator = MODULE.BaseFrequencyEstimator
MemoryStore = MODULE.MemoryStore


class _ToyEstimator(BaseFrequencyEstimator):
    name = "Toy"

    def _step(self, z: float, t_s: float | None, memory: MemoryStore) -> float:
        acc = memory.get("acc", 0.0) + float(z)
        memory["acc"] = acc
        memory["last_t"] = -1.0 if t_s is None else float(t_s)
        return acc


class _LegacyStepEstimator(BaseFrequencyEstimator):
    name = "LegacyStep"

    def step(self, z: float) -> float:  # type: ignore[override]
        return float(z) + 1.0


def test_memory_store_observes_peak_and_mean() -> None:
    mem = MemoryStore()
    mem["a"] = [1, 2, 3]
    mem.observe()
    mem["b"] = list(range(100))
    mem.observe()
    summary = mem.summary()
    assert int(summary["samples"]) == 2
    assert int(summary["peak_bytes"]) >= int(summary["current_bytes"])
    assert float(summary["mean_bytes"]) > 0.0


def test_estimator_step_runtime_summary_contract() -> None:
    est = _ToyEstimator()
    mem = MemoryStore()
    out = []
    for k in range(6):
        out.append(est.step(0.5, t_s=0.1 * k, memory=mem))
    assert out[-1] > out[0]
    rt = est.runtime_summary(mem)
    assert int(rt["runtime_steps"]) == 6
    assert float(rt["runtime_mean_step_us"]) >= 0.0
    assert int(rt["memory_key_count"]) >= 2


def test_legacy_step_is_auto_wrapped_into_standardized_step() -> None:
    est = _LegacyStepEstimator()
    mem = MemoryStore()
    y = est.step(2.0, t_s=0.1, memory=mem)
    assert y == 3.0
    rt = est.runtime_summary(mem)
    assert int(rt["runtime_steps"]) == 1
    assert int(rt["memory_peak_bytes"]) >= 0
