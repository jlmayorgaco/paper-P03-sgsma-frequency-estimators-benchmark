import sys
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenarios.nerc_phase_jump_60 import NERCPhaseJump60Scenario
from estimators.common import F_NOM

def test_nerc_phase_jump_60_plot_and_csv():
    out_dir = ROOT / "tests" / "scenarios" / "nerc_phase_jump_60" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    t_jump = 0.7
    phase_jump = math.pi / 3.0  # +60 grados

    sc = NERCPhaseJump60Scenario.run(
        duration_s=1.5,
        freq_hz=F_NOM,
        phase_jump_rad=phase_jump,
        t_jump_s=t_jump,
        noise_sigma=0.005,  # Usamos el ruido elevado estándar del escenario para ver su "suciedad"
        seed=42,
    )

    df = pd.DataFrame({"t_s": sc.t, "v_pu": sc.v, "f_true_hz": sc.f_true})
    csv_path = out_dir / "nerc_phase_jump_60.csv"
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

    # Gráfico de Voltaje
    axes[0].plot(sc.t, sc.v, linewidth=1.0, color='darkred', marker='.', markersize=1.0, alpha=0.8)
    axes[0].set_ylabel("Voltage [pu]")
    axes[0].set_title("NERC Phase Jump (+60°) + Harmonics")
    
    # Anotación del salto
    axes[0].axvline(x=t_jump, color="black", linestyle=":", alpha=0.8)
    axes[0].annotate(
        r"$\Delta \phi = +60^\circ$", 
        xy=(t_jump, 0.5), 
        xytext=(t_jump - 0.015, 0.8),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
        color="black", fontsize=8, ha="center"
    )
    
    axes[0].set_xlim(t_jump - 0.025, t_jump + 0.025)
    axes[0].set_ylim(-1.3, 1.3)

    # Gráfico de Frecuencia Verdadera
    axes[1].plot(sc.t, sc.f_true, linewidth=1.5, color='black', label="True Frequency")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Frequency [Hz]")
    axes[1].set_ylim(59.5, 60.5)
    axes[1].legend(loc="upper right")

    png_path = out_dir / "nerc_phase_jump_60.png"
    fig.savefig(png_path)
    plt.close(fig)

    assert csv_path.exists()
    assert png_path.exists()