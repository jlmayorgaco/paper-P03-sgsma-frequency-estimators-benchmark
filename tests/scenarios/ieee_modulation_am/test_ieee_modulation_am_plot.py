import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenarios.ieee_modulation_am import IEEEModulationAMScenario
from estimators.common import F_NOM

def test_ieee_modulation_am_plot_and_csv():
    out_dir = ROOT / "tests" / "scenarios" / "ieee_modulation_am" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    sc = IEEEModulationAMScenario.run(
        duration_s=2.0,
        freq_nom_hz=F_NOM,
        kx=0.1,         # 10% de modulación AM
        fm_hz=2.0,      # Modulación a 2 Hz
        noise_sigma=0.0,# Limpio para la gráfica
        seed=42,
    )

    df = pd.DataFrame({"t_s": sc.t, "v_pu": sc.v, "f_true_hz": sc.f_true})
    csv_path = out_dir / "ieee_modulation_am.csv"
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

    fig, axes = plt.subplots(2, 1, figsize=(6.0, 4.2), sharex=True)

    # Gráfico de Voltaje (se verá la envolvente AM)
    axes[0].plot(sc.t, sc.v, linewidth=0.5, color='darkblue')
    axes[0].set_ylabel("Voltage [pu]")
    axes[0].set_title(f"{IEEEModulationAMScenario.get_name()} (Pure AM at 2.0 Hz)")
    axes[0].set_ylim(-1.3, 1.3)

    # Gráfico de Frecuencia (Línea RECTA perfecta)
    axes[1].plot(sc.t, sc.f_true, linewidth=1.5, color='red', label="True Frequency")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Frequency [Hz]")
    # Forzamos los mismos límites que la prueba combinada para que la comparación sea justa
    axes[1].set_ylim(59.7, 60.3) 
    axes[1].legend(loc="upper right")

    png_path = out_dir / "ieee_modulation_am.png"
    fig.savefig(png_path)
    plt.close(fig)

    assert csv_path.exists()
    assert png_path.exists()