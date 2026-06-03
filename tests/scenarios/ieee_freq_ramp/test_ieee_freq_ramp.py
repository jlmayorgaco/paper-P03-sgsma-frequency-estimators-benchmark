import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenarios.ieee_freq_ramp import IEEEFreqRampScenario
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


def test_ieee_freq_ramp_default_params_present():
    params = IEEEFreqRampScenario.get_default_params()

    assert isinstance(params, dict)
    assert "duration_s" in params
    assert "freq_nom_hz" in params
    assert "rocof_hz_s" in params
    assert "t_start_s" in params
    assert "freq_cap_hz" in params
    assert "phase_rad" in params
    assert "amplitude" in params
    assert "noise_sigma" in params
    assert "seed" in params


def test_ieee_freq_ramp_monte_carlo_space_present():
    mc = IEEEFreqRampScenario.get_monte_carlo_space()

    assert isinstance(mc, dict)
    assert "rocof_hz_s" in mc
    assert "t_start_s" in mc
    assert "phase_rad" in mc
    assert "noise_sigma" in mc


def test_ieee_freq_ramp_name_matches():
    assert IEEEFreqRampScenario.get_name() == "IEEE_Freq_Ramp"


def test_ieee_freq_ramp_shapes_match():
    sc = IEEEFreqRampScenario.run()

    assert len(sc.t) == len(sc.v)
    assert len(sc.t) == len(sc.f_true)
    assert len(sc.t) > 0


def test_ieee_freq_ramp_time_is_monotonic():
    sc = IEEEFreqRampScenario.run()
    dt = np.diff(sc.t)

    assert np.all(dt > 0.0)


def test_ieee_freq_ramp_frequency_behavior():
    """
    Verifica las tres fases de la frecuencia:
    1. Constante antes de la rampa.
    2. Rampa lineal con la pendiente correcta.
    3. Saturación en el techo de frecuencia.
    """
    f_nom = 60.0
    rocof = 3.0
    t_start = 0.5
    f_cap = 61.5
    
    sc = IEEEFreqRampScenario.run(
        duration_s=2.0, 
        freq_nom_hz=f_nom, 
        rocof_hz_s=rocof, 
        t_start_s=t_start, 
        freq_cap_hz=f_cap, 
        noise_sigma=0.0
    )
    
    # 1. Pre-rampa: debe ser f_nom
    f_pre = sc.f_true[sc.t < (t_start - 0.01)]
    assert np.allclose(f_pre, f_nom)
    
    # 2. Durante la rampa (ej. a t=0.75s)
    # f(0.75) = 60.0 + 3.0 * (0.75 - 0.5) = 60.75 Hz
    idx_mid = np.searchsorted(sc.t, 0.75)
    assert np.isclose(sc.f_true[idx_mid], 60.75, atol=1e-4)
    
    # 3. Post-rampa (saturación): el cap se alcanza a t = 0.5 + (61.5-60.0)/3.0 = 1.0s
    f_post = sc.f_true[sc.t > 1.05]
    assert np.allclose(f_post, f_cap)


def test_ieee_freq_ramp_amplitude_is_bounded_without_noise():
    sc = IEEEFreqRampScenario.run(amplitude=1.0, noise_sigma=0.0)
    
    assert np.max(sc.v) <= 1.0 + 1e-6
    assert np.min(sc.v) >= -1.0 - 1e-6


def test_ieee_freq_ramp_no_nan():
    sc = IEEEFreqRampScenario.run()

    assert np.all(np.isfinite(sc.t))
    assert np.all(np.isfinite(sc.v))
    assert np.all(np.isfinite(sc.f_true))


def test_ieee_freq_ramp_metadata_present():
    sc = IEEEFreqRampScenario.run()

    assert sc.name == "IEEE_Freq_Ramp"
    assert isinstance(sc.meta, dict)
    assert "description" in sc.meta
    assert "parameters" in sc.meta
    assert "purpose" in sc.meta


def test_ieee_freq_ramp_rms_is_reasonable():
    # La modulación de frecuencia no altera el valor RMS de la envolvente de amplitud
    sc = IEEEFreqRampScenario.run(amplitude=1.0, noise_sigma=0.0)
    rms = float(np.sqrt(np.mean(sc.v ** 2)))

    assert abs(rms - (1.0 / np.sqrt(2.0))) < 1e-2


def test_ieee_freq_ramp_mean_is_near_zero():
    sc = IEEEFreqRampScenario.run(noise_sigma=0.0)
    mean_val = float(np.mean(sc.v))
    assert abs(mean_val) < 1e-2


def test_ieee_freq_ramp_thd_is_near_zero():
    # Para validar el THD de la señal base, la corremos con RoCoF = 0.0
    # ya que un FM sweep (Chirp) genera bandas laterales que el FFT interpreta erróneamente como armónicos.
    sc = IEEEFreqRampScenario.run(rocof_hz_s=0.0, noise_sigma=0.0)
    thd_percent = estimate_thd_percent(sc.v)

    assert thd_percent < 1.0


def test_ieee_freq_ramp_reproducible_with_seed():
    sc1 = IEEEFreqRampScenario.run(noise_sigma=0.01, seed=123)
    sc2 = IEEEFreqRampScenario.run(noise_sigma=0.01, seed=123)

    assert np.allclose(sc1.v, sc2.v)
    assert np.allclose(sc1.t, sc2.t)
    assert np.allclose(sc1.f_true, sc2.f_true)


def test_ieee_freq_ramp_different_seed_changes_noise():
    sc1 = IEEEFreqRampScenario.run(noise_sigma=0.01, seed=123)
    sc2 = IEEEFreqRampScenario.run(noise_sigma=0.01, seed=456)

    assert not np.allclose(sc1.v, sc2.v)