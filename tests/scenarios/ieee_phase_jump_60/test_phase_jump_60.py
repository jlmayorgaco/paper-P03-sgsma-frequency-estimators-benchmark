import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenarios.ieee_phase_jump_60 import IEEEPhaseJump60Scenario
from estimators.common import F_NOM


def test_ieee_phase_jump_60_default_params_present():
    params = IEEEPhaseJump60Scenario.get_default_params()

    assert isinstance(params, dict)
    assert "duration_s" in params
    assert "freq_hz" in params
    assert "phase_rad" in params
    assert "amplitude" in params
    assert "phase_jump_rad" in params
    assert "t_jump_s" in params
    assert "noise_sigma" in params
    assert "seed" in params


def test_ieee_phase_jump_60_monte_carlo_space_present():
    mc = IEEEPhaseJump60Scenario.get_monte_carlo_space()

    assert isinstance(mc, dict)
    assert "phase_jump_rad" in mc
    assert "t_jump_s" in mc
    assert "noise_sigma" in mc


def test_ieee_phase_jump_60_name_matches():
    assert IEEEPhaseJump60Scenario.get_name() == "IEEE_Phase_Jump_60"


def test_ieee_phase_jump_60_shapes_match():
    sc = IEEEPhaseJump60Scenario.run()

    assert len(sc.t) == len(sc.v)
    assert len(sc.t) == len(sc.f_true)
    assert len(sc.t) > 0


def test_ieee_phase_jump_60_time_is_monotonic():
    sc = IEEEPhaseJump60Scenario.run()
    dt = np.diff(sc.t)

    assert np.all(dt > 0.0)


def test_ieee_phase_jump_60_frequency_is_constant():
    sc = IEEEPhaseJump60Scenario.run(freq_hz=F_NOM)
    # En un salto de fase puro, la frecuencia rotacional subyacente no cambia
    assert np.allclose(sc.f_true, F_NOM)


def test_ieee_phase_jump_60_amplitude_is_bounded_without_noise():
    sc = IEEEPhaseJump60Scenario.run(amplitude=1.0, noise_sigma=0.0)

    # La amplitud máxima debe permanecer acotada por la amplitud nominal
    assert np.max(sc.v) <= 1.0 + 1e-6
    assert np.min(sc.v) >= -1.0 - 1e-6


def test_ieee_phase_jump_60_no_nan():
    sc = IEEEPhaseJump60Scenario.run()

    assert np.all(np.isfinite(sc.t))
    assert np.all(np.isfinite(sc.v))
    assert np.all(np.isfinite(sc.f_true))


def test_ieee_phase_jump_60_metadata_present():
    sc = IEEEPhaseJump60Scenario.run()

    assert sc.name == "IEEE_Phase_Jump_60"
    assert isinstance(sc.meta, dict)
    assert "description" in sc.meta
    assert "parameters" in sc.meta
    assert "purpose" in sc.meta


def test_ieee_phase_jump_60_mean_is_near_zero():
    # En 1.5 segundos (90 ciclos exactos a 60 Hz), aun con un salto de fase grande,
    # la señal debe seguir teniendo media cercana a cero.
    sc = IEEEPhaseJump60Scenario.run(noise_sigma=0.0)
    mean_val = float(np.mean(sc.v))

    assert abs(mean_val) < 1e-2


def test_ieee_phase_jump_60_reproducible_with_seed():
    sc1 = IEEEPhaseJump60Scenario.run(noise_sigma=0.01, seed=123)
    sc2 = IEEEPhaseJump60Scenario.run(noise_sigma=0.01, seed=123)

    assert np.allclose(sc1.v, sc2.v)
    assert np.allclose(sc1.t, sc2.t)
    assert np.allclose(sc1.f_true, sc2.f_true)


def test_ieee_phase_jump_60_different_seed_changes_noise():
    sc1 = IEEEPhaseJump60Scenario.run(noise_sigma=0.01, seed=123)
    sc2 = IEEEPhaseJump60Scenario.run(noise_sigma=0.01, seed=456)

    assert not np.allclose(sc1.v, sc2.v)