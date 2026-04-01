import os
import sys
import math

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "src")

if SRC not in sys.path:
    sys.path.insert(0, SRC)

from estimators.lkf import LKF_Estimator
from estimators.common import DT_DSP, F_NOM


def generate_sine(freq_hz=60.0, n_samples=6000, amplitude=1.0):
    t = np.arange(n_samples) * DT_DSP
    return amplitude * np.sin(2.0 * math.pi * freq_hz * t)


def test_lkf_instantiates():
    est = LKF_Estimator()
    assert est.name == "LKF"
    assert isinstance(est.structural_latency_samples(), int)
    assert est.structural_latency_samples() > 0


def test_lkf_step_returns_float():
    est = LKF_Estimator()
    out = est.step(0.0)
    assert isinstance(out, float)


def test_lkf_tracks_clean_60hz_sine():
    est = LKF_Estimator(q_val=1e-4, r_val=1e-2, smooth_win=10, f_nom=F_NOM)
    signal = generate_sine(freq_hz=60.0, n_samples=6000, amplitude=1.0)

    outputs = np.array([est.step(x) for x in signal], dtype=float)

    warmup = est.structural_latency_samples() + 200
    steady = outputs[warmup:]

    assert len(steady) > 100
    assert np.all(np.isfinite(steady))

    mean_est = float(np.mean(steady))
    std_est = float(np.std(steady))

    # Basic sanity for a clean 60 Hz sine
    assert abs(mean_est - F_NOM) < 1.0
    assert std_est < 2.0


def test_lkf_reset_restores_initial_behavior():
    est = LKF_Estimator()
    signal = generate_sine(freq_hz=60.0, n_samples=1000)

    for x in signal:
        est.step(x)

    est.reset()

    out = est.step(0.0)
    assert isinstance(out, float)
    assert 40.0 <= out <= 80.0 or out == F_NOM


def test_lkf_handles_small_noise_without_nan():
    rng = np.random.default_rng(42)
    est = LKF_Estimator(q_val=1e-4, r_val=1e-2, smooth_win=10, f_nom=F_NOM)

    signal = generate_sine(freq_hz=60.0, n_samples=4000)
    noisy_signal = signal + rng.normal(0.0, 0.01, size=signal.shape)

    outputs = np.array([est.step(x) for x in noisy_signal], dtype=float)

    assert np.all(np.isfinite(outputs))
    assert np.all(outputs >= 40.0)
    assert np.all(outputs <= 80.0)

    warmup = est.structural_latency_samples() + 200
    steady = outputs[warmup:]
    assert len(steady) > 100

    mean_est = float(np.mean(steady))
    assert abs(mean_est - F_NOM) < 2.0


def test_lkf_structural_latency_matches_formula():
    est = LKF_Estimator(smooth_win=10)
    expected = 10
    assert est.structural_latency_samples() == expected