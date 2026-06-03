import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenarios.ieee_single_sinwave import IEEESingleSinWaveScenario
from estimators.common import F_NOM

def test_ieee_single_sinwave_plot_and_csv():
    out_dir = ROOT / "tests" / "scenarios" / "ieee_single_sinwave" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    sc = IEEESingleSinWaveScenario.run(
        duration_s=1.5,
        amplitude=1.0,
        freq_hz=F_NOM,
        phase_rad=0.0,
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

    csv_path = out_dir / "ieee_single_sinwave.csv"
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

    axes[0].plot(sc.t, sc.v, label="v(t)")
    axes[0].set_ylabel("Voltage [pu]")
    axes[0].set_title(IEEESingleSinWaveScenario.get_name())
    axes[0].set_xlim(0.0, 0.1)
    axes[0].legend(loc="upper right")

    axes[1].plot(sc.t, sc.f_true, label="True frequency")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Frequency [Hz]")
    axes[1].set_xlim(0.0, 0.1)
    axes[1].set_ylim(59.5, 60.5)
    axes[1].legend(loc="upper right")

    png_path = out_dir / "ieee_single_sinwave.png"
    fig.savefig(png_path)
    plt.close(fig)

    assert csv_path.exists()
    assert png_path.exists()
    assert len(df) == len(sc.t)
    assert np.all(np.isfinite(sc.v))
    assert np.all(np.isfinite(sc.f_true))