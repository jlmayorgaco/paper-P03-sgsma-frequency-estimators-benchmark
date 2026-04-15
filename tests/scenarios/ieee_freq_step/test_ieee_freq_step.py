import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenarios.ieee_freq_step import IEEEFreqStepScenario
from estimators.common import F_NOM


def test_ieee_freq_step_default_params_present():
    params = IEEEFreqStepScenario.get_default_params()
    assert isinstance(params, dict)
    assert "duration_s" in params
    assert "freq_pre_hz" in params
    assert "freq_post_hz" in params
    assert "t_step_s" in params


def test_ieee_freq_step_monte_carlo_space_present():
    mc = IEEEFreqStepScenario.get_monte_carlo_space()
    assert isinstance(mc, dict)
    assert "freq_post_hz" in mc
    assert "t_step_s" in mc


def test_ieee_freq_step_name_matches():
    assert IEEEFreqStepScenario.get_name() == "IEEE_Freq_Step"


def test_ieee_freq_step_frequency_behavior():
    """Valida que la frecuencia cambie exactamente en t_step."""
    f_pre = 60.0
    f_post = 59.0
    t_step = 0.5
    
    sc = IEEEFreqStepScenario.run(
        freq_pre_hz=f_pre, freq_post_hz=f_post, t_step_s=t_step, noise_sigma=0.0
    )
    
    assert np.allclose(sc.f_true[sc.t < t_step - 0.01], f_pre)
    assert np.allclose(sc.f_true[sc.t > t_step + 0.01], f_post)


def test_ieee_freq_step_amplitude_is_constant():
    sc = IEEEFreqStepScenario.run(amplitude=1.0, noise_sigma=0.0)
    assert np.max(sc.v) <= 1.0 + 1e-6
    assert np.min(sc.v) >= -1.0 - 1e-6


def test_ieee_freq_step_no_nan():
    sc = IEEEFreqStepScenario.run()
    assert np.all(np.isfinite(sc.t))
    assert np.all(np.isfinite(sc.v))
    assert np.all(np.isfinite(sc.f_true))


def test_ieee_freq_step_reproducible():
    sc1 = IEEEFreqStepScenario.run(noise_sigma=0.01, seed=123)
    sc2 = IEEEFreqStepScenario.run(noise_sigma=0.01, seed=123)
    assert np.allclose(sc1.v, sc2.v)