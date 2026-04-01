import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenarios.ieee_single_sinwave import IEEESingleSinWaveScenario
from estimators.common import F_NOM


def estimate_thd_percent(signal: np.ndarray) -> float:
    """
    Rough THD estimate using FFT, for sanity-check purposes only.
    For a clean single sine wave, THD should be near zero.
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


def test_ieee_single_sinwave_default_params_present():
    params = IEEESingleSinWaveScenario.get_default_params()

    assert isinstance(params, dict)
    assert "duration_s" in params
    assert "amplitude" in params
    assert "freq_hz" in params
    assert "phase_rad" in params
    assert "noise_sigma" in params
    assert "seed" in params


def test_ieee_single_sinwave_monte_carlo_space_present():
    mc = IEEESingleSinWaveScenario.get_monte_carlo_space()

    assert isinstance(mc, dict)
    assert "amplitude" in mc
    assert "freq_hz" in mc
    assert "phase_rad" in mc
    assert "noise_sigma" in mc


def test_ieee_single_sinwave_name_matches():
    assert IEEESingleSinWaveScenario.get_name() == "IEEE_Single_SinWave"


def test_ieee_single_sinwave_shapes_match():
    sc = IEEESingleSinWaveScenario.run()

    assert len(sc.t) == len(sc.v)
    assert len(sc.t) == len(sc.f_true)
    assert len(sc.t) > 0


def test_ieee_single_sinwave_time_is_monotonic():
    sc = IEEESingleSinWaveScenario.run()
    dt = np.diff(sc.t)

    assert np.all(dt > 0.0)


def test_ieee_single_sinwave_frequency_is_constant():
    sc = IEEESingleSinWaveScenario.run(freq_hz=F_NOM)

    assert np.allclose(sc.f_true, F_NOM)


def test_ieee_single_sinwave_amplitude_is_bounded_without_noise():
    sc = IEEESingleSinWaveScenario.run(amplitude=1.0, noise_sigma=0.0)

    assert np.max(sc.v) <= 1.0 + 1e-6
    assert np.min(sc.v) >= -1.0 - 1e-6


def test_ieee_single_sinwave_no_nan():
    sc = IEEESingleSinWaveScenario.run()

    assert np.all(np.isfinite(sc.t))
    assert np.all(np.isfinite(sc.v))
    assert np.all(np.isfinite(sc.f_true))


def test_ieee_single_sinwave_metadata_present():
    sc = IEEESingleSinWaveScenario.run()

    assert sc.name == "IEEE_Single_SinWave"
    assert isinstance(sc.meta, dict)
    assert "description" in sc.meta
    assert "parameters" in sc.meta
    assert "purpose" in sc.meta


def test_ieee_single_sinwave_rms_is_reasonable():
    sc = IEEESingleSinWaveScenario.run(amplitude=1.0, noise_sigma=0.0)
    rms = float(np.sqrt(np.mean(sc.v ** 2)))

    assert abs(rms - (1.0 / np.sqrt(2.0))) < 1e-2


def test_ieee_single_sinwave_mean_is_near_zero():
    sc = IEEESingleSinWaveScenario.run(amplitude=1.0, noise_sigma=0.0)
    mean_val = float(np.mean(sc.v))

    assert abs(mean_val) < 1e-3


def test_ieee_single_sinwave_thd_is_near_zero():
    sc = IEEESingleSinWaveScenario.run(amplitude=1.0, noise_sigma=0.0)
    thd_percent = estimate_thd_percent(sc.v)

    assert thd_percent < 1.0


def test_ieee_single_sinwave_reproducible_with_seed():
    sc1 = IEEESingleSinWaveScenario.run(noise_sigma=0.01, seed=123)
    sc2 = IEEESingleSinWaveScenario.run(noise_sigma=0.01, seed=123)

    assert np.allclose(sc1.v, sc2.v)
    assert np.allclose(sc1.t, sc2.t)
    assert np.allclose(sc1.f_true, sc2.f_true)


def test_ieee_single_sinwave_different_seed_changes_noise():
    sc1 = IEEESingleSinWaveScenario.run(noise_sigma=0.01, seed=123)
    sc2 = IEEESingleSinWaveScenario.run(noise_sigma=0.01, seed=456)

    assert not np.allclose(sc1.v, sc2.v)