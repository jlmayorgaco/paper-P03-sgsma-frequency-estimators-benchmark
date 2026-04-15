import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenarios.ieee_freq_ramp import IEEEFreqRampScenario
from estimators.common import F_NOM

def test_ieee_freq_ramp_plot_and_csv():
    out_dir = ROOT / "tests" / "scenarios" / "ieee_freq_ramp" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    sc = IEEEFreqRampScenario.run(
        duration_s=1.5,
        freq_nom_hz=F_NOM,
        rocof_hz_s=3.0,     # +3.0 Hz/s
        t_start_s=0.3,
        freq_cap_hz=61.5,
        noise_sigma=0.0,    # Sin ruido para ver la matemática perfecta
        seed=42,
    )

    df = pd.DataFrame({"t_s": sc.t, "v_pu": sc.v, "f_true_hz": sc.f_true})
    csv_path = out_dir / "ieee_freq_ramp.csv"
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

    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.2), sharex=True)

    # Gráfico de Voltaje
    axes[0].plot(sc.t, sc.v, linewidth=0.5, color='darkblue')
    axes[0].set_ylabel("Voltage [pu]")
    axes[0].set_title(IEEEFreqRampScenario.get_name() + " (+3.0 Hz/s)")

    # Gráfico de Frecuencia
    axes[1].plot(sc.t, sc.f_true, linewidth=1.5, color='red', label="True Freq")
    
    # Anotaciones
    axes[1].axvline(x=0.3, color="black", linestyle=":", alpha=0.5)
    axes[1].annotate(
        r"RoCoF $= +3.0$ Hz/s", 
        xy=(0.55, 60.75), 
        xytext=(0.7, 60.2),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
        fontsize=7, ha="center"
    )

    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Frequency [Hz]")
    axes[1].set_ylim(59.5, 62.0)
    axes[1].legend(loc="lower right")

    png_path = out_dir / "ieee_freq_ramp.png"
    fig.savefig(png_path)
    plt.close(fig)

    assert csv_path.exists()
    assert png_path.exists()