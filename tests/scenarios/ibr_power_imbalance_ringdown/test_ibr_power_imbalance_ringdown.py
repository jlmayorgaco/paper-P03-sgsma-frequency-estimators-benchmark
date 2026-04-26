import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenarios.ibr_power_imbalance_ringdown import IBRPowerImbalanceRingdownScenario
from estimators.common import F_NOM


def test_ringdown_default_params_present():
    params = IBRPowerImbalanceRingdownScenario.get_default_params()
    assert isinstance(params, dict)
    assert "t_ramp_s" in params
    assert "rocof_hz_s" in params
    assert "t_event_s" in params
    assert "phase_jump_event_rad" in params
    assert "ring_jump_hz" in params
    assert "ring_freq_hz" in params
    assert "ring_tau_s" in params


def test_ringdown_monte_carlo_space_present():
    mc = IBRPowerImbalanceRingdownScenario.get_monte_carlo_space()
    assert isinstance(mc, dict)
    assert "phase_jump_event_rad" in mc
    assert "amp_fault_pu" in mc
    assert "ring_tau_s" in mc


def test_ringdown_name_matches():
    assert IBRPowerImbalanceRingdownScenario.get_name() == "IBR_Power_Imbalance_Ringdown"


def test_ringdown_frequency_ramp_behavior():
    t_ramp = 0.5
    t_evt = 1.0
    rocof = -2.0

    sc = IBRPowerImbalanceRingdownScenario.run(
        t_ramp_s=t_ramp,
        t_event_s=t_evt,
        rocof_hz_s=rocof,
        noise_sigma=0.0,
    )

    dt = t_evt - t_ramp
    f_expected = F_NOM + (rocof * dt)
    idx_pre_evt = np.searchsorted(sc.t, t_evt) - 1
    f_actual = sc.f_true[idx_pre_evt]
    assert np.isclose(f_actual, f_expected, atol=0.05)


def test_ringdown_frequency_oscillation_and_settling():
    t_evt = 1.0
    jump_hz = 1.5

    sc = IBRPowerImbalanceRingdownScenario.run(
        t_event_s=t_evt,
        ring_jump_hz=jump_hz,
        ring_tau_s=0.25,
        noise_sigma=0.0,
    )

    post_evt_mask = sc.t >= t_evt
    f_max_post = np.max(sc.f_true[post_evt_mask])
    assert f_max_post > F_NOM + 1.0

    f_final = sc.f_true[-1]
    assert np.isclose(f_final, F_NOM, atol=0.2)


def test_ringdown_amplitude_sag():
    amp_pre = 1.0
    amp_post = 0.8
    t_evt = 1.0

    sc = IBRPowerImbalanceRingdownScenario.run(
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


def test_ringdown_no_nan():
    sc = IBRPowerImbalanceRingdownScenario.run()
    assert np.all(np.isfinite(sc.t))
    assert np.all(np.isfinite(sc.v))
    assert np.all(np.isfinite(sc.f_true))


def test_ringdown_reproducible():
    sc1 = IBRPowerImbalanceRingdownScenario.run(noise_sigma=0.01, seed=123)
    sc2 = IBRPowerImbalanceRingdownScenario.run(noise_sigma=0.01, seed=123)
    assert np.allclose(sc1.v, sc2.v)
