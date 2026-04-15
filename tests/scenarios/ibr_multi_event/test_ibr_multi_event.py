import sys
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenarios.ibr_multi_event import IBRMultiEventScenario
from estimators.common import F_NOM


def test_ibr_multi_event_default_params_present():
    params = IBRMultiEventScenario.get_default_params()
    assert isinstance(params, dict)
    assert "amp_pre_pu" in params
    assert "amp_post_pu" in params
    assert "freq_post_hz" in params
    assert "phase_jump_rad" in params
    assert "t_event_s" in params


def test_ibr_multi_event_monte_carlo_space_present():
    mc = IBRMultiEventScenario.get_monte_carlo_space()
    assert isinstance(mc, dict)
    assert "amp_post_pu" in mc
    assert "freq_post_hz" in mc
    assert "phase_jump_rad" in mc
    assert "t_event_s" in mc


def test_ibr_multi_event_name_matches():
    assert IBRMultiEventScenario.get_name() == "IBR_Multi_Event"


def test_ibr_multi_event_frequency_drops_at_event():
    """Valida el escalón de frecuencia en el instante del evento."""
    f_pre = 60.0
    f_post = 58.5
    t_evt = 0.5
    
    sc = IBRMultiEventScenario.run(
        freq_pre_hz=f_pre, freq_post_hz=f_post, t_event_s=t_evt, noise_sigma=0.0
    )
    
    # Comprobamos un poco antes y un poco después del evento
    assert np.allclose(sc.f_true[sc.t < t_evt - 0.01], f_pre)
    assert np.allclose(sc.f_true[sc.t > t_evt + 0.01], f_post)


def test_ibr_multi_event_amplitude_sag_with_harmonics():
    """
    Valida el hueco de tensión (Sag) y que la amplitud general baje.
    Debido a que los armónicos tienen una fase inicial aleatoria, 
    pueden causar interferencia constructiva o destructiva en los picos de la onda.
    """
    amp_pre = 1.0
    amp_post = 0.5
    t_evt = 2
    sc = IBRMultiEventScenario.run(
        amp_pre_pu=amp_pre, amp_post_pu=amp_post, t_event_s=t_evt, noise_sigma=0.0
    )
    
    # Pre-falla: La fundamental es 1.0. Los armónicos suman máximo 0.06.
    # Por tanto, el pico debe estar entre 0.94 y 1.06 (dependiendo de la fase).
    v_pre = sc.v[sc.t < t_evt - 0.05]
    assert 0.93 <= np.max(v_pre) <= 1.07
    
    # Post-falla: La fundamental es 0.5. Los armónicos suman máximo 0.03.
    # Por tanto, el pico debe estar entre 0.47 y 0.53.
    v_post = sc.v[sc.t > t_evt + 0.05]
    assert 0.46 <= np.max(v_post) <= 0.54


def test_ibr_multi_event_no_nan():
    sc = IBRMultiEventScenario.run()
    assert np.all(np.isfinite(sc.t))
    assert np.all(np.isfinite(sc.v))
    assert np.all(np.isfinite(sc.f_true))


def test_ibr_multi_event_reproducible():
    sc1 = IBRMultiEventScenario.run(noise_sigma=0.01, seed=123)
    sc2 = IBRMultiEventScenario.run(noise_sigma=0.01, seed=123)
    assert np.allclose(sc1.v, sc2.v)