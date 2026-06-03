import sys
from pathlib import Path
from itertools import product
import csv

import numpy as np

import matplotlib
matplotlib.use("Agg")  # Evita errores de backend en Windows/CI
import matplotlib.pyplot as plt

# Rutas
ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from estimators.zcd import ZCDEstimator
from estimators.ekf import EKF_Estimator


def test_ekf_vs_zcd_visual_diagnostic_rmse_sweep():
    out_dir = ROOT / "tests" / "estimators" / "ekf" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    fs = 10000.0
    dt = 1.0 / fs
    t = np.arange(0.0, 0.30, dt)

    t_step = 0.10
    f_pre = 50.0
    f_post = 52.0

    f_true = np.where(t < t_step, f_pre, f_post)
    phase = np.cumsum(2.0 * np.pi * f_true * dt)

    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 0.005, size=len(t))
    v = np.sin(phase) + noise

    # Warm-up JIT para que el tiempo del sweep sea más estable
    ekf_warmup = EKF_Estimator(
        nominal_f=50.0,
        q_dc=1e-6,
        q_alpha=1e-4,
        q_beta=1e-3,
        q_omega=2.5,
        r_meas=1e-3,
        output_smoothing=0.01,
    )
    _ = ekf_warmup.estimate(t[:64], v[:64])

    # Grid de búsqueda: suficientemente amplio, pero todavía razonable para pytest
    q_dc_grid = np.logspace(-7, 2, num=5) 
    q_alpha_grid = np.logspace(-6, -2, num=3) 
    q_beta_grid = np.logspace(-6, -2, num=3) 
    q_omega_grid = np.logspace(-4, 2, num=4) 
    r_meas_grid = np.logspace(-5, 0, num=3) 
    output_smoothing_grid = [0.0, 0.005, 0.01, 0.02, 0.05]

    results = []
    best_rmse = np.inf
    best_params = None
    best_f_ekf = None

    for q_dc, q_alpha, q_beta, q_omega, r_meas, output_smoothing in product(
        q_dc_grid,
        q_alpha_grid,
        q_beta_grid,
        q_omega_grid,
        r_meas_grid,
        output_smoothing_grid,
    ):
        ekf = EKF_Estimator(
            nominal_f=50.0,
            q_dc=q_dc,
            q_alpha=q_alpha,
            q_beta=q_beta,
            q_omega=q_omega,
            r_meas=r_meas,
            output_smoothing=output_smoothing,
        )

        f_ekf = ekf.estimate(t, v)

        rmse_full = float(np.sqrt(np.mean((f_ekf - f_true) ** 2)))
        rmse_post = float(np.sqrt(np.mean((f_ekf[t > t_step] - f_true[t > t_step]) ** 2)))
        mae_full = float(np.mean(np.abs(f_ekf - f_true)))

        row = {
            "q_dc": q_dc,
            "q_alpha": q_alpha,
            "q_beta": q_beta,
            "q_omega": q_omega,
            "r_meas": r_meas,
            "output_smoothing": output_smoothing,
            "rmse_full": rmse_full,
            "rmse_post": rmse_post,
            "mae_full": mae_full,
        }
        results.append(row)

        if rmse_full < best_rmse:
            best_rmse = rmse_full
            best_params = row
            best_f_ekf = f_ekf.copy()

    assert best_params is not None
    assert best_f_ekf is not None

    # Guardar resultados completos del sweep
    csv_path = out_dir / "ekf_rmse_sweep_results.csv"
    fieldnames = [
        "q_dc",
        "q_alpha",
        "q_beta",
        "q_omega",
        "r_meas",
        "output_smoothing",
        "rmse_full",
        "rmse_post",
        "mae_full",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Guardar resumen del mejor
    best_txt_path = out_dir / "ekf_best_params_rmse.txt"
    best_txt_path.write_text(
        "\n".join([
            "Best EKF parameters by RMSE",
            f"q_dc={best_params['q_dc']}",
            f"q_alpha={best_params['q_alpha']}",
            f"q_beta={best_params['q_beta']}",
            f"q_omega={best_params['q_omega']}",
            f"r_meas={best_params['r_meas']}",
            f"output_smoothing={best_params['output_smoothing']}",
            f"rmse_full={best_params['rmse_full']:.6f}",
            f"rmse_post={best_params['rmse_post']:.6f}",
            f"mae_full={best_params['mae_full']:.6f}",
        ]),
        encoding="utf-8",
    )

    # Calcular ZCD una sola vez para el plot final
    zcd = ZCDEstimator(nominal_f=50.0)
    f_zcd = zcd.estimate(t, v)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
    })

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    plt.subplots_adjust(hspace=0.25)

    axes[0].plot(
        t,
        v,
        linewidth=1.0,
        color="teal",
        alpha=0.85,
        label="Voltage Signal (with mild noise)",
    )
    axes[0].axvline(
        t_step,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
        label="Freq Step (50→52 Hz)",
    )
    axes[0].set_ylabel("Amplitude [p.u.]")
    axes[0].set_title("Input Signal")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower right")

    axes[1].plot(
        t,
        f_true,
        linewidth=2.0,
        color="black",
        linestyle="--",
        alpha=0.6,
        label="True Frequency",
    )
    axes[1].plot(
        t,
        f_zcd,
        linewidth=1.5,
        color="orange",
        alpha=0.85,
        label="ZCD (Legacy Hardware)",
    )
    axes[1].plot(
        t,
        best_f_ekf,
        linewidth=2.0,
        color="crimson",
        label="EKF (best RMSE over sweep)",
    )
    axes[1].axvline(
        t_step,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
    )

    axes[1].set_ylabel("Frequency [Hz]")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_title("Dynamic Tracking: ZCD Latency vs. Best-RMSE EKF")
    axes[1].set_ylim(48.0, 54.5)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="lower right")

    info_text = (
        f"Best RMSE = {best_params['rmse_full']:.4f} Hz\n"
        f"q_dc={best_params['q_dc']}, q_alpha={best_params['q_alpha']}\n"
        f"q_beta={best_params['q_beta']}, q_omega={best_params['q_omega']}\n"
        f"r_meas={best_params['r_meas']}, smooth={best_params['output_smoothing']}"
    )
    axes[1].text(
        0.02,
        0.98,
        info_text,
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    plot_path = out_dir / "ekf_vs_zcd_best_rmse_sweep.png"
    fig.savefig(plot_path)
    plt.close(fig)

    assert plot_path.exists()
    assert csv_path.exists()
    assert best_txt_path.exists()

    print(f"\n[DIAGNOSTIC] Plot saved at: {plot_path}")
    print(f"[DIAGNOSTIC] Sweep CSV saved at: {csv_path}")
    print(f"[DIAGNOSTIC] Best params saved at: {best_txt_path}")
    print(f"[DIAGNOSTIC] Best RMSE = {best_params['rmse_full']:.6f} Hz")