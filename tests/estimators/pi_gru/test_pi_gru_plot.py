import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HAS_TORCH = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from estimators.pi_gru import PI_GRU_Estimator
from estimators.zcd import ZCDEstimator


def test_pi_gru_vs_zcd_visual_diagnostic():
    out_dir = ROOT / "tests" / "estimators" / "pi_gru" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    fs = 10000.0
    dt = 1.0 / fs
    t = np.arange(0.0, 0.30, dt)

    t_step = 0.10
    f_pre = 60.0
    f_post = 62.5

    f_true = np.where(t < t_step, f_pre, f_post)
    phase = np.cumsum(2.0 * np.pi * f_true * dt)

    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 0.002, size=len(t))
    v = np.sin(phase) + noise

    zcd = ZCDEstimator(nominal_f=60.0)
    f_zcd = zcd.estimate(t, v)

    pi_gru = PI_GRU_Estimator(
        nominal_f=60.0,
        dt=dt,
        weights_filename="pi_gru_weights.pt",
        show_progress=False,
    )
    f_pi_gru = pi_gru.estimate(t, v)

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
        label="Freq Step (60→62.5 Hz)",
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
        f_pi_gru,
        linewidth=2.0,
        color="crimson",
        label="PI-GRU (pretrained model)",
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
    axes[1].set_title("Dynamic Tracking: ZCD Latency vs. PI-GRU Response")
    axes[1].set_ylim(58.0, 64.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="lower right")

    plot_path = out_dir / "pi_gru_vs_zcd_diagnostic.png"
    fig.savefig(plot_path)
    plt.close(fig)

    assert plot_path.exists()
    print(f"\n[DIAGNOSTIC] Plot saved at: {plot_path}")
