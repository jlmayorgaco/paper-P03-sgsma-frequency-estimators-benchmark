import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenarios.ieee_mag_step import IEEEMagStepScenario
from estimators.common import F_NOM


def estimate_thd_percent(signal: np.ndarray) -> float:
    """
    Estimación aproximada del THD usando FFT, solo para validación (sanity-check).
    Para una onda senoidal pura sin ruido, el THD debería ser cercano a cero.
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


def test_ieee_mag_step_default_params_present():
    params = IEEEMagStepScenario.get_default_params()

    assert isinstance(params, dict)
    assert "duration_s" in params
    assert "amp_pre_pu" in params
    assert "amp_post_pu" in params
    assert "t_step_s" in params
    assert "freq_hz" in params
    assert "phase_rad" in params
    assert "noise_sigma" in params
    assert "seed" in params


def test_ieee_mag_step_monte_carlo_space_present():
    mc = IEEEMagStepScenario.get_monte_carlo_space()

    assert isinstance(mc, dict)
    assert "amp_post_pu" in mc
    assert "t_step_s" in mc
    assert "phase_rad" in mc
    assert "noise_sigma" in mc


def test_ieee_mag_step_name_matches():
    assert IEEEMagStepScenario.get_name() == "IEEE_Mag_Step"


def test_ieee_mag_step_shapes_match():
    sc = IEEEMagStepScenario.run()

    assert len(sc.t) == len(sc.v)
    assert len(sc.t) == len(sc.f_true)
    assert len(sc.t) > 0


def test_ieee_mag_step_time_is_monotonic():
    sc = IEEEMagStepScenario.run()
    dt = np.diff(sc.t)

    assert np.all(dt > 0.0)


def test_ieee_mag_step_frequency_is_constant():
    sc = IEEEMagStepScenario.run(freq_hz=F_NOM)
    # En este escenario la frecuencia NO debe cambiar
    assert np.allclose(sc.f_true, F_NOM)


def test_ieee_mag_step_amplitude_is_bounded_without_noise():
    sc = IEEEMagStepScenario.run(amp_pre_pu=1.0, amp_post_pu=1.1, noise_sigma=0.0)
    
    # La máxima amplitud debería ser exactamente amp_post_pu
    assert np.max(sc.v) <= 1.1 + 1e-6
    assert np.min(sc.v) >= -1.1 - 1e-6


def test_ieee_mag_step_amplitude_changes_at_t_step():
    """Prueba específica para validar que el escalón ocurre donde y como debe."""
    t_step = 0.5
    sc = IEEEMagStepScenario.run(amp_pre_pu=1.0, amp_post_pu=1.5, t_step_s=t_step, noise_sigma=0.0)
    
    # Antes del escalón, la amplitud máxima es 1.0
    v_pre = sc.v[sc.t < (t_step - 0.05)]
    assert np.max(np.abs(v_pre)) <= 1.0 + 1e-6
    
    # Después del escalón, la amplitud máxima es 1.5
    v_post = sc.v[sc.t > (t_step + 0.05)]
    assert np.max(np.abs(v_post)) >= 1.5 - 1e-6


def test_ieee_mag_step_no_nan():
    sc = IEEEMagStepScenario.run()

    assert np.all(np.isfinite(sc.t))
    assert np.all(np.isfinite(sc.v))
    assert np.all(np.isfinite(sc.f_true))


def test_ieee_mag_step_metadata_present():
    sc = IEEEMagStepScenario.run()

    assert sc.name == "IEEE_Mag_Step"
    assert isinstance(sc.meta, dict)
    assert "description" in sc.meta
    assert "parameters" in sc.meta
    assert "purpose" in sc.meta


def test_ieee_mag_step_rms_is_reasonable():
    # Probamos sin escalón para verificar el RMS base
    sc = IEEEMagStepScenario.run(amp_pre_pu=1.0, amp_post_pu=1.0, noise_sigma=0.0)
    rms = float(np.sqrt(np.mean(sc.v ** 2)))

    assert abs(rms - (1.0 / np.sqrt(2.0))) < 1e-2


def test_ieee_mag_step_mean_is_near_zero():
    sc = IEEEMagStepScenario.run(noise_sigma=0.0)
    mean_val = float(np.mean(sc.v))

    assert abs(mean_val) < 1e-3


def test_ieee_mag_step_thd_is_near_zero():
    # Lo probamos sin salto para no generar artefactos de alta frecuencia por el escalón
    sc = IEEEMagStepScenario.run(amp_pre_pu=1.0, amp_post_pu=1.0, noise_sigma=0.0)
    thd_percent = estimate_thd_percent(sc.v)

    assert thd_percent < 1.0


def test_ieee_mag_step_reproducible_with_seed():
    sc1 = IEEEMagStepScenario.run(noise_sigma=0.01, seed=123)
    sc2 = IEEEMagStepScenario.run(noise_sigma=0.01, seed=123)

    assert np.allclose(sc1.v, sc2.v)
    assert np.allclose(sc1.t, sc2.t)
    assert np.allclose(sc1.f_true, sc2.f_true)


def test_ieee_mag_step_different_seed_changes_noise():
    sc1 = IEEEMagStepScenario.run(noise_sigma=0.01, seed=123)
    sc2 = IEEEMagStepScenario.run(noise_sigma=0.01, seed=456)

    assert not np.allclose(sc1.v, sc2.v)