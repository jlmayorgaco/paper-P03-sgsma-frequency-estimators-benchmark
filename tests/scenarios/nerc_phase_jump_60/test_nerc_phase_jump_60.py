import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenarios.nerc_phase_jump_60 import NERCPhaseJump60Scenario
from estimators.common import F_NOM


def estimate_thd_percent(signal: np.ndarray) -> float:
    """
    Estimación aproximada del THD usando FFT.
    Para el escenario NERC, debe detectar los armónicos 5to y 7mo.
    """
    x = np.asarray(signal, dtype=float)
    x = x - np.mean(x)

    n = len(x)
    if n < 16:
        return 0.0

    window = np.hanning(n)
    xw = x * window
    spectrum = np.fft.rfft(xw)
    mag = np.abs(spectrum)

    if len(mag) < 3:
        return 0.0

    mag[0] = 0.0
    k1 = int(np.argmax(mag))
    fundamental = mag[k1]

    if fundamental <= 1e-12:
        return 0.0

    harmonic_energy = 0.0
    max_h = min(10, (len(mag) - 1) // max(k1, 1))
    for h in range(2, max_h + 1):
        kh = h * k1
        if kh < len(mag):
            harmonic_energy += mag[kh] ** 2

    thd = np.sqrt(harmonic_energy) / fundamental
    return float(thd * 100.0)


def test_nerc_phase_jump_60_default_params_present():
    params = NERCPhaseJump60Scenario.get_default_params()

    assert isinstance(params, dict)
    assert "duration_s" in params
    assert "freq_hz" in params
    assert "phase_rad" in params
    assert "amplitude" in params
    assert "phase_jump_rad" in params
    assert "t_jump_s" in params
    assert "noise_sigma" in params
    assert "seed" in params


def test_nerc_phase_jump_60_monte_carlo_space_present():
    mc = NERCPhaseJump60Scenario.get_monte_carlo_space()

    assert isinstance(mc, dict)
    assert "phase_jump_rad" in mc
    assert "t_jump_s" in mc
    assert "noise_sigma" in mc


def test_nerc_phase_jump_60_name_matches():
    assert NERCPhaseJump60Scenario.get_name() == "NERC_Phase_Jump_60"


def test_nerc_phase_jump_60_shapes_match():
    sc = NERCPhaseJump60Scenario.run()

    assert len(sc.t) == len(sc.v)
    assert len(sc.t) == len(sc.f_true)
    assert len(sc.t) > 0


def test_nerc_phase_jump_60_time_is_monotonic():
    sc = NERCPhaseJump60Scenario.run()
    dt = np.diff(sc.t)

    assert np.all(dt > 0.0)


def test_nerc_phase_jump_60_frequency_is_constant():
    sc = NERCPhaseJump60Scenario.run(freq_hz=F_NOM)
    # A pesar del ruido y el salto de fase, el vector f_true subyacente sigue en 60 Hz
    assert np.allclose(sc.f_true, F_NOM)


def test_nerc_phase_jump_60_amplitude_with_harmonics():
    sc = NERCPhaseJump60Scenario.run(amplitude=1.0, noise_sigma=0.0)
    
    # La amplitud máxima ya no es 1.0. Es 1.0 + 0.04 (5to) + 0.02 (7mo) + 0.005 (inter) = 1.065
    assert np.max(sc.v) <= 1.07
    assert np.max(sc.v) > 1.0  # Aseguramos que los armónicos sumen a los picos
    assert np.min(sc.v) >= -1.07


def test_nerc_phase_jump_60_no_nan():
    sc = NERCPhaseJump60Scenario.run()

    assert np.all(np.isfinite(sc.t))
    assert np.all(np.isfinite(sc.v))
    assert np.all(np.isfinite(sc.f_true))


def test_nerc_phase_jump_60_mean_is_near_zero():
    sc = NERCPhaseJump60Scenario.run(noise_sigma=0.0)
    mean_val = float(np.mean(sc.v))

    assert abs(mean_val) < 1e-2


def test_nerc_phase_jump_60_thd_is_correct():
    """
    Valida que la inyección armónica sume un THD aproximado al límite de IEEE 519.
    Raíz cuadrada de la suma de cuadrados (0.04^2 + 0.02^2) = 4.472%
    Para medirlo limpiamente con FFT, apagamos el salto de fase temporalmente.
    """
    sc = NERCPhaseJump60Scenario.run(phase_jump_rad=0.0, noise_sigma=0.0)
    thd_percent = estimate_thd_percent(sc.v)

    # El THD estacionario debe estar entre 4.0% y 5.0%
    assert 4.0 < thd_percent < 5.0


def test_nerc_phase_jump_60_reproducible_with_seed():
    sc1 = NERCPhaseJump60Scenario.run(noise_sigma=0.01, seed=123)
    sc2 = NERCPhaseJump60Scenario.run(noise_sigma=0.01, seed=123)

    assert np.allclose(sc1.v, sc2.v)
    assert np.allclose(sc1.t, sc2.t)
    assert np.allclose(sc1.f_true, sc2.f_true)


def test_nerc_phase_jump_60_different_seed_changes_noise():
    sc1 = NERCPhaseJump60Scenario.run(noise_sigma=0.01, seed=123)
    sc2 = NERCPhaseJump60Scenario.run(noise_sigma=0.01, seed=456)

    assert not np.allclose(sc1.v, sc2.v)