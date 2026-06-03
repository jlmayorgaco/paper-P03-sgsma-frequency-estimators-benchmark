import sys
from pathlib import Path
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


def test_ekf_vs_zcd_visual_diagnostic():
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

    zcd = ZCDEstimator(nominal_f=50.0)
    f_zcd = zcd.estimate(t, v)

    ekf = EKF_Estimator(
        nominal_f=50.0,
        q_dc=1e-6,
        q_alpha=1e-4,
        q_beta=10e-4,
        q_omega=25e-1,
        r_meas=1e-3,
        output_smoothing=0.01,
    )
    f_ekf = ekf.estimate(t, v)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
    })

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    plt.subplots_adjust(hspace=0.2)

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
        f_ekf,
        linewidth=2.0,
        color="crimson",
        label="EKF (explicit frequency state)",
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
    axes[1].set_title("Dynamic Tracking: ZCD Latency vs. EKF Smooth Tracking")
    axes[1].set_ylim(48.0, 54.5)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="lower right")

    plot_path = out_dir / "ekf_vs_zcd_diagnostic.png"
    fig.savefig(plot_path)
    plt.close(fig)

    assert plot_path.exists()
    print(f"\n[DIAGNOSTIC] Plot saved at: {plot_path}")