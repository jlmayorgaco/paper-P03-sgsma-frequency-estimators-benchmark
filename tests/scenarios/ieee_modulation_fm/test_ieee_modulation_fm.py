import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenarios.ieee_modulation_fm import IEEEModulationFMScenario
from estimators.common import F_NOM


def test_ieee_modulation_fm_default_params_present():
    params = IEEEModulationFMScenario.get_default_params()

    assert isinstance(params, dict)
    assert "duration_s" in params
    assert "freq_nom_hz" in params
    assert "phase_rad" in params
    assert "amplitude" in params
    assert "ka" in params
    assert "fm_hz" in params
    assert "noise_sigma" in params
    assert "seed" in params


def test_ieee_modulation_fm_monte_carlo_space_present():
    mc = IEEEModulationFMScenario.get_monte_carlo_space()

    assert isinstance(mc, dict)
    assert "fm_hz" in mc
    assert "ka" in mc
    assert "noise_sigma" in mc


def test_ieee_modulation_fm_name_matches():
    assert IEEEModulationFMScenario.get_name() == "IEEE_Modulation_FM"


def test_ieee_modulation_fm_shapes_match():
    sc = IEEEModulationFMScenario.run()

    assert len(sc.t) == len(sc.v)
    assert len(sc.t) == len(sc.f_true)
    assert len(sc.t) > 0


def test_ieee_modulation_fm_time_is_monotonic():
    sc = IEEEModulationFMScenario.run()
    dt = np.diff(sc.t)

    assert np.all(dt > 0.0)


def test_ieee_modulation_fm_amplitude_is_bounded():
    """
    Como la amplitud es constante, el voltaje debe quedar acotado por
    +/- amplitude en ausencia de ruido.
    """
    amp = 1.0
    sc = IEEEModulationFMScenario.run(
        duration_s=2.0,
        amplitude=amp,
        ka=0.1,
        fm_hz=2.0,
        noise_sigma=0.0,
    )

    assert np.max(sc.v) <= amp + 1e-3
    assert np.min(sc.v) >= -amp - 1e-3
    assert np.isclose(np.max(np.abs(sc.v)), amp, atol=1e-3)


def test_ieee_modulation_fm_frequency_bounds():
    """
    Valida que la desviación máxima de frecuencia por PM/FM coincida con:
    delta_f = ka * fm
    dado que:
    f_true = f_nom - ka * fm * sin(2*pi*fm*t)
    """
    f_nom = 60.0
    ka = 0.1
    fm = 2.0
    sc = IEEEModulationFMScenario.run(
        duration_s=2.0,
        freq_nom_hz=f_nom,
        ka=ka,
        fm_hz=fm,
        noise_sigma=0.0,
    )

    expected_max_f = f_nom + (ka * fm)
    expected_min_f = f_nom - (ka * fm)

    assert np.isclose(np.max(sc.f_true), expected_max_f, atol=1e-3)
    assert np.isclose(np.min(sc.f_true), expected_min_f, atol=1e-3)


def test_ieee_modulation_fm_frequency_mean_is_nominal():
    """
    La frecuencia instantánea oscila alrededor de la nominal, por lo que
    su promedio temporal debe ser aproximadamente freq_nom_hz.
    """
    f_nom = 60.0
    sc = IEEEModulationFMScenario.run(
        duration_s=2.0,
        freq_nom_hz=f_nom,
        ka=0.1,
        fm_hz=2.0,
        noise_sigma=0.0,
    )

    assert np.isclose(np.mean(sc.f_true), f_nom, atol=1e-3)


def test_ieee_modulation_fm_no_nan():
    sc = IEEEModulationFMScenario.run()

    assert np.all(np.isfinite(sc.t))
    assert np.all(np.isfinite(sc.v))
    assert np.all(np.isfinite(sc.f_true))


def test_ieee_modulation_fm_metadata_present():
    sc = IEEEModulationFMScenario.run()

    assert sc.name == "IEEE_Modulation_FM"
    assert isinstance(sc.meta, dict)
    assert "description" in sc.meta
    assert "parameters" in sc.meta
    assert "purpose" in sc.meta
    assert "dynamics" in sc.meta


def test_ieee_modulation_fm_reproducible_with_seed():
    sc1 = IEEEModulationFMScenario.run(noise_sigma=0.01, seed=123)
    sc2 = IEEEModulationFMScenario.run(noise_sigma=0.01, seed=123)

    assert np.allclose(sc1.v, sc2.v)
    assert np.allclose(sc1.t, sc2.t)
    assert np.allclose(sc1.f_true, sc2.f_true)


def test_ieee_modulation_fm_different_seed_changes_noise():
    sc1 = IEEEModulationFMScenario.run(noise_sigma=0.01, seed=123)
    sc2 = IEEEModulationFMScenario.run(noise_sigma=0.01, seed=456)

    assert not np.allclose(sc1.v, sc2.v)


def test_ieee_modulation_fm_validate_params_rejects_invalid_duration():
    try:
        IEEEModulationFMScenario.run(duration_s=0.0)
        assert False, "Expected ValueError for duration_s <= 0"
    except ValueError as e:
        assert "duration_s must be > 0" in str(e)


def test_ieee_modulation_fm_validate_params_rejects_invalid_fm():
    try:
        IEEEModulationFMScenario.run(fm_hz=0.0)
        assert False, "Expected ValueError for fm_hz <= 0"
    except ValueError as e:
        assert "fm_hz must be > 0" in str(e) or "Modulation frequency fm_hz must be > 0" in str(e)


def test_ieee_modulation_fm_validate_params_rejects_negative_noise():
    try:
        IEEEModulationFMScenario.run(noise_sigma=-1e-3)
        assert False, "Expected ValueError for noise_sigma < 0"
    except ValueError as e:
        assert "noise_sigma must be >= 0" in str(e)