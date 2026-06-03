import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenarios.ieee_oob_interference import IEEEOOBInterferenceScenario

def test_ieee_oob_interference_plot_and_csv():
    out_dir = ROOT / "tests" / "scenarios" / "ieee_oob_interference" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    sc = IEEEOOBInterferenceScenario.run(
        interf_freq_hz=30.0, interf_amp_pu=0.1, noise_sigma=0.0, seed=42
    )

    df = pd.DataFrame({"t_s": sc.t, "v_pu": sc.v, "f_true_hz": sc.f_true})
    df.to_csv(out_dir / "ieee_oob_interference.csv", index=False)

    plt.rcParams.update({"font.family": "serif", "font.size": 8, "figure.dpi": 300, "savefig.bbox": "tight"})
    fig, axes = plt.subplots(2, 1, figsize=(6.0, 4.2), sharex=True)

    axes[0].plot(sc.t, sc.v, linewidth=0.5, color='darkmagenta')
    axes[0].set_ylabel("Voltage [pu]")
    axes[0].set_title("IEEE 1547 Out-of-Band (OOB) Interference (30 Hz, 10%)")
    
    # Zoom para ver claramente la deformación de la onda causada por los 30 Hz
    axes[0].set_xlim(0.0, 0.2) 

    axes[1].plot(sc.t, sc.f_true, linewidth=1.5, color='black', label="True Frequency")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Frequency [Hz]")
    axes[1].set_ylim(59.5, 60.5)
    axes[1].set_xlim(0.0, 0.2)
    axes[1].legend(loc="upper right")

    fig.savefig(out_dir / "ieee_oob_interference.png")
    plt.close(fig)