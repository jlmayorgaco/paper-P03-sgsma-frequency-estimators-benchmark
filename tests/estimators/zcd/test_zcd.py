import sys
from pathlib import Path
import numpy as np

import matplotlib
# FORZAR BACKEND SIN VENTANA (Evita el error de _tkinter en Windows)
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# Rutas
ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from estimators.zcd import ZCDEstimator

def test_zcd_plot_visual_diagnostic():
    out_dir = ROOT / "tests" / "estimators" / "zcd" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    fs = 10000.0
    t = np.arange(0, 0.2, 1/fs)

    f_true = np.where(t < 0.1, 50.0, 60.0)
    phase = np.cumsum(f_true) * (1/fs) * 2 * np.pi
    v = np.sin(phase)

    estimator = ZCDEstimator(nominal_f=50.0)
    f_est = estimator.estimate(t, v)

    plt.rcParams.update({
        "font.family": "serif", 
        "font.size": 10, 
        "figure.dpi": 200, 
        "savefig.bbox": "tight"
    })
    
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    plt.subplots_adjust(hspace=0.2)

    axes[0].plot(t, v, linewidth=1.5, color='teal', label="Voltage Signal")
    axes[0].axvline(0.1, color='gray', linestyle='--', linewidth=1, alpha=0.7, label="Freq Jump")
    
    crossings = (v[:-1] <= 0.0) & (v[1:] > 0.0)
    idx_cross = np.where(crossings)[0]
    axes[0].plot(t[idx_cross], v[idx_cross], 'ro', markersize=4, label="Positive Zero Crossings")
    
    axes[0].set_ylabel("Amplitude [p.u.]")
    axes[0].set_title("ZCD Input Signal (50 Hz -> 60 Hz Step)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(t, f_true, linewidth=1.5, color='gray', linestyle='--', label="True Frequency")
    axes[1].plot(t, f_est, linewidth=1.5, color='crimson', label="ZCD Estimate")
    axes[1].axvline(0.1, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    
    axes[1].set_ylabel("Frequency [Hz]")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_title("Zero-Crossing Detector Response (ZOH Behavior)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper left")

    plot_path = out_dir / "zcd_step_diagnostic.png"
    fig.savefig(plot_path)
    plt.close(fig)