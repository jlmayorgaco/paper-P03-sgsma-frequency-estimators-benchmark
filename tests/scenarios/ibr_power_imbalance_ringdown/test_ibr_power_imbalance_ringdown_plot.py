import sys
from pathlib import Path

import matplotlib
# FORZAR BACKEND SIN VENTANA (Evita el error de _tkinter en Windows)
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenarios.ibr_power_imbalance_ringdown import IBRPowerImbalanceRingdownScenario

def test_ringdown_plot_and_csv():
    out_dir = ROOT / "tests" / "scenarios" / "ibr_power_imbalance_ringdown" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    sc = IBRPowerImbalanceRingdownScenario.run(noise_sigma=0.0, seed=42)

    df = pd.DataFrame({"t_s": sc.t, "v_pu": sc.v, "f_true_hz": sc.f_true})
    df.to_csv(out_dir / "ibr_ringdown.csv", index=False)

    plt.rcParams.update({"font.family": "serif", "font.size": 8, "figure.dpi": 300, "savefig.bbox": "tight"})
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.5), sharex=True)

    axes[0].plot(sc.t, sc.v, linewidth=0.6, color='purple')
    axes[0].set_ylabel("Voltage [pu]")
    axes[0].set_title("IBR Power Imbalance & Control Ring-down")
    axes[0].axvline(x=1.0, color="black", linestyle=":", alpha=0.7)
    axes[0].annotate("Sag + Phase Jump", xy=(1.0, 0.5), xytext=(1.1, 0.8),
                     arrowprops=dict(arrowstyle="->", color="black"), fontsize=8)

    axes[1].plot(sc.t, sc.f_true, linewidth=1.5, color='red')
    axes[1].axvline(x=0.5, color="black", linestyle=":", alpha=0.5)
    axes[1].axvline(x=1.0, color="black", linestyle=":", alpha=0.7)
    
    # Coordenadas ajustadas para el sistema base de 50 Hz
    axes[1].annotate("1. Neg ROCOF", xy=(0.75, 49.5), xytext=(0.55, 49.0), fontsize=8)
    axes[1].annotate("2. PID Ring-down", xy=(1.1, 51.0), xytext=(1.3, 51.2),
                     arrowprops=dict(arrowstyle="->", color="black"), fontsize=8)

    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Frequency [Hz]")
    axes[1].set_ylim(58.5, 62.0) # Ajustado para 50 Hz

    fig.savefig(out_dir / "ibr_ringdown.png")
    plt.close(fig)