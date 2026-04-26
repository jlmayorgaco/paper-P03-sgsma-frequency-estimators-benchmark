import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenarios.ibr_chamorro_playback import IBRChamorroPlaybackScenario
from estimators.common import FS_DSP


def test_playback_default_params_present():
    params = IBRChamorroPlaybackScenario.get_default_params()
    assert isinstance(params, dict)
    assert "csv_filename" in params
    assert "time_col" in params
    assert "voltage_col" in params
    assert "normalize_pu" in params


def test_playback_monte_carlo_space_present():
    mc = IBRChamorroPlaybackScenario.get_monte_carlo_space()
    assert isinstance(mc, dict)
    assert "noise_sigma" in mc


def test_playback_name_matches():
    assert IBRChamorroPlaybackScenario.get_name() == "IBR_Chamorro_Playback"


def test_playback_time_is_monotonic_and_aligned_to_fs():
    sc = IBRChamorroPlaybackScenario.run()

    assert sc.t[0] == 0.0
    dt_actual = sc.t[1] - sc.t[0]
    dt_expected = 1.0 / FS_DSP
    assert np.isclose(dt_actual, dt_expected, rtol=1e-6)
    assert np.all(np.diff(sc.t) > 0.0)


def test_playback_auto_normalization_works():
    sc = IBRChamorroPlaybackScenario.run(normalize_pu=True, noise_sigma=0.0)

    t_mask = sc.t < 0.05
    v_early = sc.v[t_mask]

    if len(v_early) > 0:
        max_v = np.max(np.abs(v_early))
        assert 0.95 <= max_v <= 1.05


def test_playback_no_nan():
    sc = IBRChamorroPlaybackScenario.run()
    assert np.all(np.isfinite(sc.t))
    assert np.all(np.isfinite(sc.v))
    assert np.all(np.isfinite(sc.f_true))


def test_playback_frequency_is_within_physical_bounds():
    sc = IBRChamorroPlaybackScenario.run(noise_sigma=0.0)

    f_median = np.median(sc.f_true)
    assert 40.0 <= f_median <= 80.0

    valid_mask = (sc.t > 0.1) & (sc.t < (sc.t[-1] - 0.1))
    f_clean = sc.f_true[valid_mask]
    assert np.all(np.isfinite(f_clean))


def test_playback_reproducible_with_noise():
    sc1 = IBRChamorroPlaybackScenario.run(noise_sigma=0.01, seed=123)
    sc2 = IBRChamorroPlaybackScenario.run(noise_sigma=0.01, seed=123)
    sc3 = IBRChamorroPlaybackScenario.run(noise_sigma=0.01, seed=456)

    assert np.allclose(sc1.v, sc2.v)
    assert not np.allclose(sc1.v, sc3.v)
