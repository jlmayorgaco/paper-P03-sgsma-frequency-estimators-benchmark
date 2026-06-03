import sys
import numpy as np
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from estimators.music import MUSIC_Estimator

def test_music_steady_state():
    """¿MUSIC rastrea correctamente una frecuencia estacionaria?"""
    fs = 10000.0
    dt = 1.0 / fs
    t = np.arange(0, 0.2, dt)
    f_target = 61.5
    v = np.sin(2.0 * np.pi * f_target * t)
    
    music = MUSIC_Estimator(nominal_f=60.0, n_cycles=1.0)
    f_est = music.estimate(t, v)
    
    # Validamos que el estado final sea muy preciso (error < 0.01 Hz)
    final_f = f_est[-1]
    assert np.isclose(final_f, f_target, atol=0.01)

def test_music_noise_immunity():
    """¿Mantiene su super-resolución en presencia de ruido moderado?"""
    fs = 10000.0
    dt = 1.0 / fs
    t = np.arange(0, 0.2, dt)
    
    np.random.seed(42)
    noise = np.random.normal(0, 0.05, len(t))
    v = np.sin(2.0 * np.pi * 60.0 * t) + noise
    
    music = MUSIC_Estimator(nominal_f=60.0, n_cycles=1.0)
    f_est = music.estimate(t, v)
    
    assert np.isclose(np.mean(f_est[t > 0.1]), 60.0, atol=0.1)