import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenarios.ieee_mag_step import IEEEMagStepScenario
from estimators.common import F_NOM

def test_ieee_mag_step_plot_and_csv():
    out_dir = ROOT / "tests" / "scenarios" / "ieee_mag_step" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    sc = IEEEMagStepScenario.run(
        duration_s=1.5,
        freq_hz=F_NOM,
        phase_rad=0.0,
        amp_pre_pu=1.0,
        amp_post_pu=1.1, # Salto del +10%
        t_step_s=0.5,    # El salto ocurre a los 0.5s
        noise_sigma=0.0,
        seed=42,
    )

    df = pd.DataFrame(
        {
            "t_s": sc.t,
            "v_pu": sc.v,
            "f_true_hz": sc.f_true,
        }
    )

    csv_path = out_dir / "ieee_mag_step.csv"
    df.to_csv(csv_path, index=False)

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

    # -----------------------------------------------------
    # PLOT DEL VOLTAJE (axes[0])
    # -----------------------------------------------------
    axes[0].plot(sc.t, sc.v, label="v(t)")
    axes[0].set_ylabel("Voltage [pu]")
    axes[0].set_title(IEEEMagStepScenario.get_name())
    
    # 1. Línea roja punteada en t_step
    t_step = 0.5
    axes[0].axvline(x=t_step, color="red", linestyle=":", linewidth=1.2, alpha=0.8)

    # 2. Anotación con flecha y etiqueta (t_0 y Delta V)
    axes[0].annotate(
        r"$t_0 = 0.5$ s" + "\n" + r"$\Delta V = +10\%$", 
        xy=(t_step, 1.05),             # Punto donde apunta la flecha
        xytext=(t_step - 0.035, 1.25), # Posición del texto
        color="red",
        fontsize=7,
        ha="center",
        va="center",
        arrowprops=dict(arrowstyle="->", color="red", lw=1.0)
    )

    axes[0].set_xlim(0.4, 0.6)
    # Ampliamos un poco el eje Y para que la anotación no se corte
    axes[0].set_ylim(-1.15, 1.45) 
    axes[0].legend(loc="upper right")

    # -----------------------------------------------------
    # PLOT DE FRECUENCIA (axes[1])
    # -----------------------------------------------------
    axes[1].plot(sc.t, sc.f_true, label="True frequency")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Frequency [Hz]")
    axes[1].set_xlim(0.4, 0.6)
    axes[1].set_ylim(59.5, 60.5)
    axes[1].legend(loc="upper right")

    png_path = out_dir / "ieee_mag_step.png"
    fig.savefig(png_path)
    plt.close(fig)

    assert csv_path.exists()
    assert png_path.exists()
    assert len(df) == len(sc.t)
    assert np.all(np.isfinite(sc.v))
    assert np.all(np.isfinite(sc.f_true))