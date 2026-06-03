import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenarios.ieee_freq_step import IEEEFreqStepScenario

def test_ieee_freq_step_plot_and_csv():
    out_dir = ROOT / "tests" / "scenarios" / "ieee_freq_step" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    sc = IEEEFreqStepScenario.run(
        freq_pre_hz=60.0, freq_post_hz=59.0, t_step_s=0.5, noise_sigma=0.0, seed=42
    )

    df = pd.DataFrame({"t_s": sc.t, "v_pu": sc.v, "f_true_hz": sc.f_true})
    df.to_csv(out_dir / "ieee_freq_step.csv", index=False)

    plt.rcParams.update({"font.family": "serif", "font.size": 8, "figure.dpi": 300, "savefig.bbox": "tight"})
    fig, axes = plt.subplots(2, 1, figsize=(6.0, 4.2), sharex=True)

    axes[0].plot(sc.t, sc.v, linewidth=0.5, color='darkblue')
    axes[0].set_ylabel("Voltage [pu]")
    axes[0].set_title("IEEE 1547 Frequency Step (-1.0 Hz)")
    axes[0].axvline(x=0.5, color="red", linestyle=":", alpha=0.5)

    axes[1].plot(sc.t, sc.f_true, linewidth=1.5, color='red', label="True Frequency")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Frequency [Hz]")
    axes[1].set_ylim(58.5, 60.5)
    axes[1].legend(loc="upper right")

    fig.savefig(out_dir / "ieee_freq_step.png")
    plt.close(fig)