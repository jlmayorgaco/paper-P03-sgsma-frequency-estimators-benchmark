import importlib.util
import sys
import numpy as np
import pytest
from pathlib import Path

HAS_TORCH = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if HAS_TORCH:
    from estimators.pi_gru import PI_GRU_Estimator
else:
    PI_GRU_Estimator = None

def test_pi_gru_instantiation():
    estimator = PI_GRU_Estimator(nominal_f=60.0, dt=1e-4)
    assert estimator.name == "PI-GRU"
    assert hasattr(estimator, 'model')
    assert estimator.window_len == 100

def test_pi_gru_latency():
    estimator = PI_GRU_Estimator()
    expected = (estimator.window_len // 2) + (estimator.ma_len // 2)
    assert estimator.structural_latency_samples() == expected

def test_pi_gru_stationary_tracking():
    fs = 10000.0
    dt = 1.0 / fs
    t = np.arange(0.0, 0.5, dt) 
    
    v = np.sin(2.0 * np.pi * 60.0 * t) + np.random.normal(0, 0.001, len(t))
    
    estimator = PI_GRU_Estimator(nominal_f=60.0, dt=dt)
    f_est = estimator.estimate(t, v)
    
    f_steady = f_est[500:]
    mae = np.mean(np.abs(f_steady - 60.0))
    
    assert mae < 0.10, f"Stationary tracking MAE is too high: {mae:.4f} Hz"

def test_pi_gru_step_vs_vectorized():
    fs = 10000.0
    dt = 1.0 / fs
    t = np.arange(0.0, 0.1, dt)
    v = np.sin(2.0 * np.pi * 60.0 * t) + np.random.normal(0, 0.005, len(t))
    
    estimator_vec = PI_GRU_Estimator(nominal_f=60.0, dt=dt)
    f_vec = estimator_vec.estimate(t, v)
    
    estimator_step = PI_GRU_Estimator(nominal_f=60.0, dt=dt)
    f_step = np.empty_like(t)
    for i in range(len(t)):
        f_step[i] = estimator_step.step(v[i])
        
    np.testing.assert_allclose(f_vec, f_step, atol=1e-6)
