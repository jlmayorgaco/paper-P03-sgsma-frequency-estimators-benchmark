import sys
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Rutas
ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from estimators.pll import PLL_Estimator
from estimators.epll import EPLL_Estimator


def test_epll_vs_pll_visual_diagnostic():
    out_dir = ROOT / "tests" / "estimators" / "epll" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    fs = 10000.0
    dt = 1.0 / fs
    t = np.arange(0.0, 0.40, dt)

    t_event = 0.10
    
    # 1. Generamos Escalón de Frecuencia (50 -> 52 Hz)
    f_true = np.where(t < t_event, 50.0, 52.0)
    phase = np.cumsum(2.0 * np.pi * f_true * dt)
    
    # 2. Generamos Hueco de Tensión Simultáneo (Sag de 1.0 p.u a 0.6 p.u)
    amp_true = np.where(t < t_event, 1.0, 0.6)

    # Señal final
    v = amp_true * np.sin(phase)

    # Instanciamos el Basic PLL (con sintonización conservadora previa)
    pll_basic = PLL_Estimator(
        nominal_f=50.0, settle_time=0.08, output_smoothing=0.005
    )
    f_pll = pll_basic.estimate(t, v)

    # Instanciamos el Enhanced PLL
    epll = EPLL_Estimator(nominal_f=50.0)
    f_epll = epll.estimate(t, v)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
    })

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    plt.subplots_adjust(hspace=0.2)

    # --- Plot Señal de Entrada ---
    axes[0].plot(
        t, v, linewidth=1.0, color="teal", alpha=0.85, 
        label="Voltaje (Escalón Freq. + Voltage Sag)"
    )
    axes[0].axvline(t_event, color="gray", linestyle="--", alpha=0.7)
    axes[0].set_ylabel("Amplitud [p.u.]")
    axes[0].set_title("Input Signal: Frequency Step (50→52Hz) & Amplitude Sag (1.0→0.6 p.u.)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower right")

    # --- Plot Seguimiento de Frecuencia ---
    axes[1].plot(
        t, f_true, linewidth=2.0, color="black", linestyle="--", alpha=0.6,
        label="True Frequency"
    )
    axes[1].plot(
        t, f_pll, linewidth=1.5, color="crimson", alpha=0.7,
        label="Basic Multiplier PLL"
    )
    axes[1].plot(
        t, f_epll, linewidth=2.0, color="dodgerblue",
        label="Enhanced PLL (EPLL)"
    )
    axes[1].axvline(t_event, color="gray", linestyle="--", alpha=0.7)

    axes[1].set_ylabel("Frecuencia [Hz]")
    axes[1].set_xlabel("Tiempo [s]")
    axes[1].set_title("Dynamic Tracking: EPLL vs Basic PLL")
    axes[1].set_ylim(48.0, 54.5)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="lower right")

    plot_path = out_dir / "epll_vs_pll_diagnostic.png"
    fig.savefig(plot_path)
    plt.close(fig)

    assert plot_path.exists()
    print(f"\n[DIAGNOSTIC] Plot saved at: {plot_path}")