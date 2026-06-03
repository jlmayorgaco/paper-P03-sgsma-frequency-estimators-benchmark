import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenarios.ieee_modulation import IEEEModulationScenario
from estimators.common import F_NOM


def test_ieee_modulation_default_params_present():
    params = IEEEModulationScenario.get_default_params()

    assert isinstance(params, dict)
    assert "duration_s" in params
    assert "freq_nom_hz" in params
    assert "phase_rad" in params
    assert "amplitude" in params
    assert "kx" in params
    assert "ka" in params
    assert "fm_hz" in params
    assert "noise_sigma" in params
    assert "seed" in params


def test_ieee_modulation_monte_carlo_space_present():
    mc = IEEEModulationScenario.get_monte_carlo_space()

    assert isinstance(mc, dict)
    assert "fm_hz" in mc
    assert "kx" in mc
    assert "ka" in mc
    assert "noise_sigma" in mc


def test_ieee_modulation_name_matches():
    assert IEEEModulationScenario.get_name() == "IEEE_Modulation"


def test_ieee_modulation_shapes_match():
    sc = IEEEModulationScenario.run()

    assert len(sc.t) == len(sc.v)
    assert len(sc.t) == len(sc.f_true)
    assert len(sc.t) > 0


def test_ieee_modulation_time_is_monotonic():
    sc = IEEEModulationScenario.run()
    dt = np.diff(sc.t)

    assert np.all(dt > 0.0)


def test_ieee_modulation_amplitude_envelope():
    """
    Valida que la modulación AM (kx) limite correctamente los picos de voltaje.
    El voltaje máximo debe ser amplitude * (1 + kx).
    """
    amp_base = 1.0
    kx = 0.1
    sc = IEEEModulationScenario.run(
        duration_s=2.0, amplitude=amp_base, kx=kx, noise_sigma=0.0
    )
    
    expected_max_v = amp_base * (1.0 + kx)
    expected_min_v = -amp_base * (1.0 + kx)

    assert np.isclose(np.max(sc.v), expected_max_v, atol=1e-3)
    assert np.isclose(np.min(sc.v), expected_min_v, atol=1e-3)


def test_ieee_modulation_frequency_bounds():
    """
    Valida que la desviación máxima de frecuencia por PM coincida con
    la relación matemática: delta_f = ka * fm
    """
    f_nom = 60.0
    ka = 0.1
    fm = 2.0
    sc = IEEEModulationScenario.run(
        duration_s=2.0, freq_nom_hz=f_nom, ka=ka, fm_hz=fm, noise_sigma=0.0
    )
    
    expected_max_f = f_nom + (ka * fm)
    expected_min_f = f_nom - (ka * fm)
    
    assert np.isclose(np.max(sc.f_true), expected_max_f, atol=1e-3)
    assert np.isclose(np.min(sc.f_true), expected_min_f, atol=1e-3)


def test_ieee_modulation_no_nan():
    sc = IEEEModulationScenario.run()

    assert np.all(np.isfinite(sc.t))
    assert np.all(np.isfinite(sc.v))
    assert np.all(np.isfinite(sc.f_true))


def test_ieee_modulation_metadata_present():
    sc = IEEEModulationScenario.run()

    assert sc.name == "IEEE_Modulation"
    assert isinstance(sc.meta, dict)
    assert "description" in sc.meta
    assert "parameters" in sc.meta
    assert "purpose" in sc.meta
    assert "dynamics" in sc.meta


def test_ieee_modulation_reproducible_with_seed():
    sc1 = IEEEModulationScenario.run(noise_sigma=0.01, seed=123)
    sc2 = IEEEModulationScenario.run(noise_sigma=0.01, seed=123)

    assert np.allclose(sc1.v, sc2.v)
    assert np.allclose(sc1.t, sc2.t)
    assert np.allclose(sc1.f_true, sc2.f_true)


def test_ieee_modulation_different_seed_changes_noise():
    sc1 = IEEEModulationScenario.run(noise_sigma=0.01, seed=123)
    sc2 = IEEEModulationScenario.run(noise_sigma=0.01, seed=456)

    assert not np.allclose(sc1.v, sc2.v)