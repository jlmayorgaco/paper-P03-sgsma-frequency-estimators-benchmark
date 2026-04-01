import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from estimators.lkf import LKF_Estimator
from estimators.common import DT_DSP, F_NOM


def generate_sine(freq_hz=60.0, n_samples=5000, amplitude=1.0):
    t = np.arange(n_samples) * DT_DSP
    v = amplitude * np.sin(2.0 * math.pi * freq_hz * t)
    return t, v


def test_lkf_diagnostic_plot_and_csv():
    out_dir = ROOT / "tests" / "lkf" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    est = LKF_Estimator(q_val=1e-4, r_val=1e-2, smooth_win=10, f_nom=F_NOM)

    t, signal = generate_sine(freq_hz=60.0, n_samples=5000, amplitude=1.0)
    freq_hat = np.array([est.step(x) for x in signal], dtype=float)
    freq_true = np.full_like(freq_hat, F_NOM, dtype=float)

    # Save CSV
    df = pd.DataFrame(
        {
            "t_s": t,
            "v_pu": signal,
            "f_true_hz": freq_true,
            "f_hat_hz": freq_hat,
        }
    )
    csv_path = out_dir / "lkf_diagnostic.csv"
    df.to_csv(csv_path, index=False)

    # IEEE-ish plotting style
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "lines.linewidth": 1.0,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "savefig.bbox": "tight",
        }
    )

    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.2), sharex=True)

    # Top: voltage
    axes[0].plot(t, signal, label="Input voltage")
    axes[0].set_ylabel("v(t) [pu]")
    axes[0].set_title("LKF Diagnostic")
    axes[0].legend(loc="upper right")

    # Bottom: frequency estimate
    axes[1].plot(t, freq_true, linestyle="--", label="True 60 Hz")
    axes[1].plot(t, freq_hat, label="LKF estimate")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("f(t) [Hz]")
    axes[1].set_ylim(55, 65)
    axes[1].set_xlim(0.0, 0.01)
    axes[1].legend(loc="upper right")

    png_path = out_dir / "lkf_diagnostic.png"
    fig.savefig(png_path)
    plt.close(fig)

    # Basic artifact checks
    assert csv_path.exists()
    assert png_path.exists()
    assert len(df) == 5000
    assert np.all(np.isfinite(freq_hat))