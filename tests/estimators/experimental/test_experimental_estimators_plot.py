from __future__ import annotations

import csv
import importlib
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def _step_signal(fs: float = 10_000.0, duration_s: float = 0.30) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.arange(0.0, duration_s, 1.0 / fs)
    f_true = np.where(t < 0.10, 60.0, 61.0)
    phase = np.cumsum(2.0 * np.pi * f_true / fs)
    rng = np.random.default_rng(7)
    v = np.sin(phase) + rng.normal(0.0, 0.01, size=t.shape[0])
    return t, v, f_true


def _rmse(y: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, dtype=float) - np.asarray(y_true, dtype=float)) ** 2)))


@pytest.mark.parametrize(("key", "module_name", "class_name"), ESTIMATOR_SPECS)
def test_experimental_estimator_plot_and_parameter_sweep(
    key: str,
    module_name: str,
    class_name: str,
) -> None:
    estimator_cls = _load_estimator(module_name, class_name)
    t, v, f_true = _step_signal()

    defaults = dict(estimator_cls.default_params())
    default_gain = float(defaults.get("gain", 0.0))
    gain_grid = sorted(
        {
            0.0,
            default_gain * -1.0,
            default_gain * -0.5,
            default_gain,
            default_gain * 0.5,
            default_gain * 1.5,
            default_gain * 2.0,
        }
    )

    baseline_est = estimator_cls(**defaults)
    baseline_hat = np.asarray(baseline_est.step_vectorized(v), dtype=float)
    baseline_rmse = _rmse(baseline_hat, f_true)

    rows: list[dict[str, float]] = []
    best_rmse = float("inf")
    best_gain = default_gain
    best_hat = baseline_hat.copy()

    for gain in gain_grid:
        params = dict(defaults)
        params["gain"] = float(gain)
        est = estimator_cls(**params)
        f_hat = np.asarray(est.step_vectorized(v), dtype=float)
        rmse_val = _rmse(f_hat, f_true)
        mae_val = float(np.mean(np.abs(f_hat - f_true)))

        rows.append({"gain": float(gain), "rmse": rmse_val, "mae": mae_val})
        if rmse_val < best_rmse:
            best_rmse = rmse_val
            best_gain = float(gain)
            best_hat = f_hat.copy()

    out_dir = ROOT / "tests" / "estimators" / key / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{key}_gain_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["gain", "rmse", "mae"])
        writer.writeheader()
        writer.writerows(rows)

    txt_path = out_dir / f"{key}_best_params.txt"
    txt_path.write_text(
        "\n".join(
            [
                f"estimator={class_name}",
                f"baseline_gain={default_gain}",
                f"best_gain={best_gain}",
                f"baseline_rmse={baseline_rmse:.8f}",
                f"best_rmse={best_rmse:.8f}",
            ]
        ),
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=180)
    ax.plot(t, f_true, label="True frequency", color="black", linestyle="--", linewidth=1.8)
    ax.plot(t, baseline_hat, label="Default params", color="tab:orange", alpha=0.9, linewidth=1.3)
    ax.plot(t, best_hat, label="Best gain sweep", color="tab:blue", linewidth=1.8)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title(f"{class_name}: default vs tuned (gain sweep)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    plot_path = out_dir / f"{key}_tuned_vs_default.png"
    fig.savefig(plot_path)
    plt.close(fig)

    assert plot_path.exists(), f"{key}: plot artifact missing"
    assert csv_path.exists(), f"{key}: sweep CSV missing"
    assert txt_path.exists(), f"{key}: best-params summary missing"
    assert best_rmse <= baseline_rmse + 1e-12, f"{key}: tuning did not improve/stabilize RMSE"
