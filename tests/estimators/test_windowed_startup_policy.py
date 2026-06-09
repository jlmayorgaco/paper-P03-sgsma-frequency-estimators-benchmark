import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from estimators.base import MemoryStore
from estimators.esprit import ESPRIT_Estimator
from estimators.koopman import Koopman_Estimator
from estimators.prony import Prony_Estimator


WINDOWED_ESTIMATORS = [
    (Prony_Estimator, {"nominal_f": 60.0, "n_cycles": 2.0, "order": 4}),
    (ESPRIT_Estimator, {"nominal_f": 60.0, "n_cycles": 1.0}),
    (Koopman_Estimator, {"nominal_f": 60.0, "n_cycles": 1.5}),
]


def _nominal_signal() -> tuple[np.ndarray, np.ndarray]:
    fs = 10000.0
    dt = 1.0 / fs
    t = np.arange(0.0, 0.08, dt)
    v = np.sin(2.0 * np.pi * 60.0 * t)
    return t, v


def _assert_startup_contract(f_hat: np.ndarray, latency_samples: int) -> None:
    finite = np.flatnonzero(np.isfinite(f_hat))
    assert finite.size > 0
    first_valid_samples = int(finite[0] + 1)

    assert first_valid_samples == latency_samples
    assert np.all(~np.isfinite(f_hat[: latency_samples - 1]))
    assert np.all(np.isfinite(f_hat[latency_samples - 1 :]))


@pytest.mark.parametrize("estimator_cls, params", WINDOWED_ESTIMATORS)
def test_windowed_estimators_report_latency_at_first_vectorized_output(estimator_cls, params):
    t, v = _nominal_signal()
    estimator = estimator_cls(**params, dt=float(t[1] - t[0]))

    f_hat = estimator.estimate(t, v)

    _assert_startup_contract(f_hat, estimator.structural_latency_samples())


@pytest.mark.parametrize("estimator_cls, params", WINDOWED_ESTIMATORS)
def test_windowed_estimators_report_latency_at_first_standardized_output(estimator_cls, params):
    t, v = _nominal_signal()
    estimator = estimator_cls(**params, dt=float(t[1] - t[0]))
    memory = MemoryStore()

    f_hat = np.array(
        [estimator.step(float(sample), float(t[idx]), memory) for idx, sample in enumerate(v)],
        dtype=float,
    )
    runtime = estimator.runtime_summary(memory)

    latency_samples = estimator.structural_latency_samples()
    _assert_startup_contract(f_hat, latency_samples)
    assert runtime["startup_valid_samples"] == latency_samples
    assert runtime["post_startup_invalid_rate"] == 0.0
