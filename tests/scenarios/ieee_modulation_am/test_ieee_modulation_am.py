import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenarios.ieee_modulation_am import IEEEModulationAMScenario
from estimators.common import F_NOM


def test_ieee_modulation_am_default_params_present():
    params = IEEEModulationAMScenario.get_default_params()

    assert isinstance(params, dict)
    assert "duration_s" in params
    assert "freq_nom_hz" in params
    assert "phase_rad" in params
    assert "amplitude" in params
    assert "kx" in params
    assert "fm_hz" in params
    assert "noise_sigma" in params
    assert "seed" in params


def test_ieee_modulation_am_monte_carlo_space_present():
    mc = IEEEModulationAMScenario.get_monte_carlo_space()

    assert isinstance(mc, dict)
    assert "fm_hz" in mc
    assert "kx" in mc
    assert "noise_sigma" in mc


def test_ieee_modulation_am_name_matches():
    assert IEEEModulationAMScenario.get_name() == "IEEE_Modulation_AM"


def test_ieee_modulation_am_shapes_match():
    sc = IEEEModulationAMScenario.run()

    assert len(sc.t) == len(sc.v)
    assert len(sc.t) == len(sc.f_true)
    assert len(sc.t) > 0


def test_ieee_modulation_am_time_is_monotonic():
    sc = IEEEModulationAMScenario.run()
    dt = np.diff(sc.t)

    assert np.all(dt > 0.0)


def test_ieee_modulation_am_frequency_is_constant():
    """
    En el escenario AM puro, la frecuencia verdadera debe permanecer constante
    en freq_nom_hz para todo el tiempo.
    """
    f_nom = 60.0
    sc = IEEEModulationAMScenario.run(
        duration_s=2.0,
        freq_nom_hz=f_nom,
        kx=0.1,
        fm_hz=2.0,
        noise_sigma=0.0,
    )

    assert np.allclose(sc.f_true, f_nom, atol=1e-12)
    assert np.isclose(np.max(sc.f_true), f_nom, atol=1e-12)
    assert np.isclose(np.min(sc.f_true), f_nom, atol=1e-12)


def test_ieee_modulation_am_envelope_upper_bound():
    """
    Valida que la envolvente AM limite correctamente el voltaje máximo:
    |v(t)| <= amplitude * (1 + kx)
    """
    amp_base = 1.0
    kx = 0.1
    sc = IEEEModulationAMScenario.run(
        duration_s=2.0,
        amplitude=amp_base,
        kx=kx,
        noise_sigma=0.0,
    )

    expected_peak = amp_base * (1.0 + kx)

    assert np.max(sc.v) <= expected_peak + 1e-3
    assert np.min(sc.v) >= -expected_peak - 1e-3
    assert np.isclose(np.max(np.abs(sc.v)), expected_peak, atol=1e-3)


def test_ieee_modulation_am_no_nan():
    sc = IEEEModulationAMScenario.run()

    assert np.all(np.isfinite(sc.t))
    assert np.all(np.isfinite(sc.v))
    assert np.all(np.isfinite(sc.f_true))


def test_ieee_modulation_am_metadata_present():
    sc = IEEEModulationAMScenario.run()

    assert sc.name == "IEEE_Modulation_AM"
    assert isinstance(sc.meta, dict)
    assert "description" in sc.meta
    assert "parameters" in sc.meta
    assert "purpose" in sc.meta
    assert "dynamics" in sc.meta


def test_ieee_modulation_am_reproducible_with_seed():
    sc1 = IEEEModulationAMScenario.run(noise_sigma=0.01, seed=123)
    sc2 = IEEEModulationAMScenario.run(noise_sigma=0.01, seed=123)

    assert np.allclose(sc1.v, sc2.v)
    assert np.allclose(sc1.t, sc2.t)
    assert np.allclose(sc1.f_true, sc2.f_true)


def test_ieee_modulation_am_different_seed_changes_noise():
    sc1 = IEEEModulationAMScenario.run(noise_sigma=0.01, seed=123)
    sc2 = IEEEModulationAMScenario.run(noise_sigma=0.01, seed=456)

    assert not np.allclose(sc1.v, sc2.v)


def test_ieee_modulation_am_validate_params_rejects_invalid_duration():
    try:
        IEEEModulationAMScenario.run(duration_s=0.0)
        assert False, "Expected ValueError for duration_s <= 0"
    except ValueError as e:
        assert "duration_s must be > 0" in str(e)


def test_ieee_modulation_am_validate_params_rejects_invalid_fm():
    try:
        IEEEModulationAMScenario.run(fm_hz=0.0)
        assert False, "Expected ValueError for fm_hz <= 0"
    except ValueError as e:
        assert "fm_hz must be > 0" in str(e) or "Modulation frequency fm_hz must be > 0" in str(e)


def test_ieee_modulation_am_validate_params_rejects_negative_noise():
    try:
        IEEEModulationAMScenario.run(noise_sigma=-1e-3)
        assert False, "Expected ValueError for noise_sigma < 0"
    except ValueError as e:
        assert "noise_sigma must be >= 0" in str(e)