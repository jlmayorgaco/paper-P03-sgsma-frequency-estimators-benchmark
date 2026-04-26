from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis import monte_carlo_engine as mce_mod
from analysis.monte_carlo_engine import MonteCarloEngine


class _DummyScenarioResult:
    def __init__(self) -> None:
        self.name = "DummyScenario"
        self.t = np.array([0.0, 1e-4, 2e-4], dtype=float)
        self.v = np.array([0.0, 0.1, 0.2], dtype=float)
        self.f_true = np.array([60.0, 60.0, 60.0], dtype=float)


class _DummyScenario:
    @staticmethod
    def get_default_params() -> dict[str, float]:
        return {}

    @staticmethod
    def get_monte_carlo_space() -> dict[str, dict[str, float]]:
        return {}

    @staticmethod
    def run(**kwargs):  # noqa: ANN003
        return _DummyScenarioResult()


class _DummyEstimator:
    name = "DummyEstimator"

    def __init__(self, **kwargs):  # noqa: ANN003
        pass

    def step_vectorized(self, v):
        return np.asarray(v, dtype=float)


def test_monte_carlo_default_cost_reps_contract() -> None:
    engine = MonteCarloEngine(scenario_cls=_DummyScenario, estimator_cls=_DummyEstimator)
    assert engine.n_cost_reps == 20


def test_monte_carlo_uses_process_time_for_cpu_metric(monkeypatch) -> None:
    calls: list[float] = []

    def fake_process_time() -> float:
        calls.append(1.0)
        return float(len(calls)) * 1e-3

    monkeypatch.setattr(mce_mod.time, "process_time", fake_process_time)

    engine = MonteCarloEngine(
        scenario_cls=_DummyScenario,
        estimator_cls=_DummyEstimator,
        n_cost_reps=4,
    )
    cpu_time = engine._measure_exec_time(np.array([0.0, 0.1, 0.2], dtype=float))
    assert cpu_time > 0.0
    assert len(calls) == 8, "process_time must be called twice per repetition"
