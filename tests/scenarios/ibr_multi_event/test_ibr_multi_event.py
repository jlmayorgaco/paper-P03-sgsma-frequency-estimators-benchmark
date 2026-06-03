import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenarios.ibr_multi_event import IBRMultiEventScenario


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
    f_pre = 60.0
    f_post = 58.5
    t_evt = 0.5

    sc = IBRMultiEventScenario.run(
        freq_pre_hz=f_pre,
        freq_post_hz=f_post,
        t_event_s=t_evt,
        noise_sigma=0.0,
    )

    pre = sc.f_true[sc.t < t_evt - 0.05]
    post = sc.f_true[sc.t > t_evt + 0.05]
    assert np.median(pre) > np.median(post)
    assert np.isclose(np.median(pre), f_pre, atol=0.2)
    assert np.isclose(np.median(post), f_post, atol=0.2)


def test_ibr_multi_event_amplitude_sag_with_harmonics():
    amp_pre = 1.0
    amp_post = 0.5
    t_evt = 2.0

    sc = IBRMultiEventScenario.run(
        amp_pre_pu=amp_pre,
        amp_post_pu=amp_post,
        t_event_s=t_evt,
        noise_sigma=0.0,
    )

    v_pre = sc.v[sc.t < t_evt - 0.05]
    v_post = sc.v[sc.t > t_evt + 0.05]

    pre_rms = float(np.sqrt(np.mean(v_pre**2)))
    post_rms = float(np.sqrt(np.mean(v_post**2)))

    assert post_rms < pre_rms
    assert np.isclose(post_rms / pre_rms, amp_post / amp_pre, atol=0.2)


def test_ibr_multi_event_no_nan():
    sc = IBRMultiEventScenario.run()
    assert np.all(np.isfinite(sc.t))
    assert np.all(np.isfinite(sc.v))
    assert np.all(np.isfinite(sc.f_true))


def test_ibr_multi_event_reproducible():
    sc1 = IBRMultiEventScenario.run(noise_sigma=0.01, seed=123)
    sc2 = IBRMultiEventScenario.run(noise_sigma=0.01, seed=123)
    assert np.allclose(sc1.v, sc2.v)
