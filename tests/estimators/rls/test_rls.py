import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from estimators.rls import RLS_Estimator


def test_rls_fixed_nominal():
    """Valida que el RLS clásico sea estable en estado estacionario."""
    fs = 10000.0
    t = np.arange(0.0, 1.0, 1.0 / fs)
    v = np.sin(2.0 * np.pi * 60.0 * t)

    estimator = RLS_Estimator(
        nominal_f=60.0,
        is_vff=False,
        lambda_fixed=0.995,
        output_smoothing=0.03,
        p0=25.0,
    )
    f_est = estimator.estimate(t, v)

    f_clean = f_est[t > 0.15]
    abs_err = np.abs(f_clean - 60.0)

    assert np.isclose(np.mean(f_clean), 60.0, atol=0.05)
    assert np.percentile(abs_err, 95) < 0.10
    assert np.max(abs_err) < 0.12


def test_vff_rls_step_is_stable_and_convergent():
    """
    Valida que el VFF-RLS no diverja y converja razonablemente tras un escalón.

    No exigimos que sea mejor que el RLS fijo, porque esa superioridad no ha
    quedado demostrada de forma consistente en este benchmark.
    """
    fs = 10000.0
    t = np.arange(0.0, 1.5, 1.0 / fs)

    f_true = np.where(t < 0.5, 60.0, 62.0)
    phase = np.cumsum(2.0 * np.pi * f_true / fs)
    v = np.sin(phase)

    estimator = RLS_Estimator(
        nominal_f=60.0,
        is_vff=True,
        alpha_vff=0.20,
        lambda_min=0.90,
        lambda_max=0.9995,
        vff_beta=0.02,
        output_smoothing=0.03,
        p0=25.0,
    )
    f_est = estimator.estimate(t, v)

    assert np.all(np.isfinite(f_est))

    f_post = f_est[t > 1.0]
    abs_err = np.abs(f_post - 62.0)

    assert np.isclose(np.mean(f_post), 62.0, atol=0.12)
    assert np.percentile(abs_err, 95) < 0.25