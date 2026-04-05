import sys
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenarios.ieee_phase_jump_20 import IEEEPhaseJump20Scenario
from estimators.common import F_NOM

def test_ieee_phase_jump_20_plot_and_csv():
    out_dir = ROOT / "tests" / "scenarios" / "ieee_phase_jump_20" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generamos el escenario con los parámetros normativos
    t_jump = 0.7
    phase_jump = math.pi / 9.0  # +20 grados

    sc = IEEEPhaseJump20Scenario.run(
        duration_s=1.5,
        freq_hz=F_NOM,
        phase_jump_rad=phase_jump,
        t_jump_s=t_jump,
        noise_sigma=0.0, # Sin ruido para ver la fractura de fase limpia
        seed=42,
    )

    df = pd.DataFrame({"t_s": sc.t, "v_pu": sc.v, "f_true_hz": sc.f_true})
    csv_path = out_dir / "ieee_phase_jump_20.csv"
    df.to_csv(csv_path, index=False)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })

    fig, axes = plt.subplots(2, 1, figsize=(4.0, 4.2), sharex=True)

    # Gráfico de Voltaje con Zoom Extremo en el salto
    axes[0].plot(sc.t, sc.v, linewidth=1.0, color='darkblue', marker='.', markersize=1.5)
    axes[0].set_ylabel("Voltage [pu]")
    axes[0].set_title("IEEE 1547 Phase Jump Limit (+20°)")
    
    # Anotación del salto
    axes[0].axvline(x=t_jump, color="red", linestyle=":", alpha=0.8)
    axes[0].annotate(
        r"$\Delta \phi = +20^\circ$", 
        xy=(t_jump, 0.5), 
        xytext=(t_jump - 0.015, 0.8),
        arrowprops=dict(arrowstyle="->", color="red", lw=1.0),
        color="red", fontsize=8, ha="center"
    )
    
    # Zoom de ventana temporal: 1.5 ciclos antes y después del salto
    axes[0].set_xlim(t_jump - 0.025, t_jump + 0.025)
    axes[0].set_ylim(-1.2, 1.2)

    # Gráfico de Frecuencia Verdadera
    axes[1].plot(sc.t, sc.f_true, linewidth=1.5, color='red', label="True Frequency")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Frequency [Hz]")
    axes[1].set_ylim(59.5, 60.5)
    axes[1].legend(loc="upper right")

    png_path = out_dir / "ieee_phase_jump_20.png"
    fig.savefig(png_path)
    plt.close(fig)

    assert csv_path.exists()
    assert png_path.exists()