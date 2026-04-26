from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


ESTIMATOR_SPECS: tuple[tuple[str, str, str], ...] = (
    ("ckf", "ckf", "CKF_Estimator"),
    ("sr_ukf", "sr_ukf", "SR_UKF_Estimator"),
    ("adaptive_ekf", "adaptive_ekf", "Adaptive_EKF_Estimator"),
    ("imm_ekf_ukf", "imm_ekf_ukf", "IMM_EKF_UKF_Estimator"),
    ("hinf_frequency_kf", "hinf_frequency_kf", "Hinf_KF_Estimator"),
    ("wls_ipdft", "wls_ipdft", "WLS_IPDFT_Estimator"),
    ("quinn_fernandes", "quinn_fernandes", "Quinn_Fernandes_Estimator"),
    (
        "jacobsen_interpolated_dft",
        "jacobsen_interpolated_dft",
        "Jacobsen_Interpolated_DFT_Estimator",
    ),
    ("sliding_least_squares", "sliding_least_squares", "Sliding_Least_Squares_Estimator"),
    ("music_experimental", "music_experimental", "MUSIC_Experimental_Estimator"),
    ("matrix_pencil", "matrix_pencil", "Matrix_Pencil_Estimator"),
    ("hilbert_phase_derivative", "hilbert_phase_derivative", "Hilbert_Phase_Derivative_Estimator"),
)


def _load_estimator(module_name: str, class_name: str) -> type:
    module = importlib.import_module(f"src.estimators.{module_name}")
    return getattr(module, class_name)


def _step_signal(fs: float = 10_000.0, duration_s: float = 0.25) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.arange(0.0, duration_s, 1.0 / fs)
    f_true = np.where(t < 0.10, 60.0, 61.0)
    phase = np.cumsum(2.0 * np.pi * f_true / fs)
    v = np.sin(phase)
    return t, v, f_true


@pytest.mark.parametrize(("key", "module_name", "class_name"), ESTIMATOR_SPECS)
def test_experimental_estimator_output_contract(key: str, module_name: str, class_name: str) -> None:
    estimator_cls = _load_estimator(module_name, class_name)
    _, v, _ = _step_signal()

    est = estimator_cls()
    f_hat = np.asarray(est.step_vectorized(v), dtype=float)

    assert f_hat.shape == v.shape, f"{key}: output shape mismatch"
    assert np.all(np.isfinite(f_hat)), f"{key}: non-finite outputs"
    assert np.std(f_hat) > 0.0, f"{key}: degenerate constant output"


@pytest.mark.parametrize(("key", "module_name", "class_name"), ESTIMATOR_SPECS)
def test_experimental_step_matches_vectorized(key: str, module_name: str, class_name: str) -> None:
    estimator_cls = _load_estimator(module_name, class_name)
    _, v, _ = _step_signal()

    est_vec = estimator_cls()
    y_vec = np.asarray(est_vec.step_vectorized(v), dtype=float)

    est_seq = estimator_cls()
    y_seq = np.asarray([float(est_seq.step(float(x))) for x in v], dtype=float)

    assert y_seq.shape == y_vec.shape, f"{key}: sequential/vectorized shape mismatch"
    assert np.allclose(y_seq, y_vec, atol=1e-12), f"{key}: sequential/vectorized mismatch"


@pytest.mark.parametrize(("key", "module_name", "class_name"), ESTIMATOR_SPECS)
def test_experimental_estimator_tracks_frequency_step(key: str, module_name: str, class_name: str) -> None:
    estimator_cls = _load_estimator(module_name, class_name)
    t, v, f_true = _step_signal(duration_s=0.35)

    est = estimator_cls()
    y = np.asarray(est.step_vectorized(v), dtype=float)

    pre_mask = (t > 0.03) & (t < 0.09)
    post_mask = t > 0.22
    pre_mean = float(np.mean(y[pre_mask]))
    post_mean = float(np.mean(y[post_mask]))

    assert abs(pre_mean - 60.0) < 0.6, f"{key}: poor pre-step tracking ({pre_mean:.3f} Hz)"
    assert abs(post_mean - 61.0) < 0.6, f"{key}: poor post-step tracking ({post_mean:.3f} Hz)"
    assert post_mean > pre_mean + 0.2, f"{key}: estimator did not respond clearly to step"
