import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from estimators.pll import PLL_Estimator
from estimators.sogi_fll import SOGI_FLL_Estimator

def test_sogi_fll_step_and_jump_diagnostic():
    out_dir = ROOT / "tests" / "estimators" / "sogi_fll" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    fs = 10000.0
    dt = 1.0 / fs
    t = np.arange(0.0, 6.0, dt)  # 6 second simulation

    t_step = 1.0
    t_jump = 4.0
    
    # 1. Frequency Step Event (60 -> 62 Hz at t=1s)
    f_true = np.where(t < t_step, 60.0, 62.0)
    # Phase must be integrated when frequency is variable
    phase_base = np.cumsum(2.0 * np.pi * f_true * dt)
    
    # 2. Phase Jump Event (+45 degrees at t=4s)
    phase_jump = np.where(t < t_jump, 0.0, np.pi/4)
    
    # Final combined signal
    v = np.sin(phase_base + phase_jump)

    # Initialize estimators at the new 60Hz nominal frequency
    pll = PLL_Estimator(nominal_f=60.0)
    f_pll = pll.estimate(t, v)

    fll = SOGI_FLL_Estimator(nominal_f=60.0, gamma=50.0)
    f_fll = fll.estimate(t, v)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
    })

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    plt.subplots_adjust(hspace=0.25)

    # --- Plot Input Signal ---
    axes[0].plot(t, v, color="teal", linewidth=1.0, alpha=0.8, label="Voltage Signal")
    axes[0].axvline(t_step, color="gray", linestyle="--", alpha=0.7)
    axes[0].axvline(t_jump, color="gray", linestyle="--", alpha=0.7)
    axes[0].set_ylabel("Amplitude [p.u.]")
    axes[0].set_title("Grid Events: Freq Step (60→62Hz) at 1s & +45° Phase Jump at 4s")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower right")

    # --- Plot Frequency Tracking ---
    axes[1].plot(t, f_true, 'k--', linewidth=2.0, alpha=0.6, label="True Frequency")
    axes[1].plot(t, f_pll, color="crimson", linewidth=1.5, alpha=0.8, label="PLL (Phase-Locked Loop)")
    axes[1].plot(t, f_fll, color="dodgerblue", linewidth=2.0, label="SOGI-FLL")
    
    axes[1].axvline(t_step, color="gray", linestyle="--", alpha=0.7)
    axes[1].axvline(t_jump, color="gray", linestyle="--", alpha=0.7)

    axes[1].set_ylabel("Frequency [Hz]")
    axes[1].set_xlabel("Time [s]")
    # Expanded y-limits to capture the 60-62Hz range and the PLL's massive error spike
    axes[1].set_ylim(58.0, 68.0) 
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    plot_path = out_dir / "fll_step_and_jump.png"
    fig.savefig(plot_path)
    plt.close(fig)

    assert plot_path.exists()
    print(f"\n[DIAGNOSTIC] Plot saved at: {plot_path}")