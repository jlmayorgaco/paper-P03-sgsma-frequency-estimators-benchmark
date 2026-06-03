import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenarios.ieee_oob_interference import IEEEOOBInterferenceScenario
from estimators.common import F_NOM


def test_ieee_oob_default_params_present():
    params = IEEEOOBInterferenceScenario.get_default_params()
    assert isinstance(params, dict)
    assert "interf_freq_hz" in params
    assert "interf_amp_pu" in params


def test_ieee_oob_monte_carlo_space_present():
    mc = IEEEOOBInterferenceScenario.get_monte_carlo_space()
    assert isinstance(mc, dict)
    assert "interf_freq_hz" in mc
    assert "interf_amp_pu" in mc


def test_ieee_oob_name_matches():
    assert IEEEOOBInterferenceScenario.get_name() == "IEEE_OOB_Interference"


def test_ieee_oob_frequency_is_constant():
    sc = IEEEOOBInterferenceScenario.run(freq_hz=F_NOM)
    assert np.allclose(sc.f_true, F_NOM)


def test_ieee_oob_amplitude_bounds():
    """Valida que la amplitud máxima sea la suma de la fundamental y la interferencia."""
    amp_fund = 1.0
    amp_oob = 0.1
    sc = IEEEOOBInterferenceScenario.run(
        amplitude=amp_fund, interf_amp_pu=amp_oob, noise_sigma=0.0
    )
    
    expected_max = amp_fund + amp_oob
    assert np.max(sc.v) <= expected_max + 1e-3
    assert np.max(sc.v) > amp_fund # Asegura que la interferencia está presente


def test_ieee_oob_no_nan():
    sc = IEEEOOBInterferenceScenario.run()
    assert np.all(np.isfinite(sc.t))
    assert np.all(np.isfinite(sc.v))
    assert np.all(np.isfinite(sc.f_true))


def test_ieee_oob_reproducible():
    sc1 = IEEEOOBInterferenceScenario.run(noise_sigma=0.01, seed=123)
    sc2 = IEEEOOBInterferenceScenario.run(noise_sigma=0.01, seed=123)
    assert np.allclose(sc1.v, sc2.v)