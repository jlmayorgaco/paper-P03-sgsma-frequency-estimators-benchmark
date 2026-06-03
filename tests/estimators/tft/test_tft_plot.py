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
from estimators.tft import TFT_Estimator

def test_tft_latency_tradeoff_diagnostic():
    out_dir = ROOT / "tests" / "estimators" / "tft" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    fs = 10000.0
    dt = 1.0 / fs
    t = np.arange(0.0, 0.6, dt)

    t_step = 0.1
    
    # Perfil: Escalón severo de 60 -> 63 Hz
    f_true = np.where(t < t_step, 60.0, 63.0)
    phase = np.cumsum(2.0 * np.pi * f_true * dt)
    v = np.sin(phase)

    # Basic PLL (Reactivo)
    pll = PLL_Estimator(nominal_f=60.0)
    f_pll = pll.estimate(t, v)

    # TFT (Basado en Ventana, N_c = 2 ciclos)
    tft = TFT_Estimator(nominal_f=60.0, n_cycles=2.0)
    f_tft = tft.estimate(t, v)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
    })

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    plt.subplots_adjust(hspace=0.25)

    axes[0].plot(t, v, color="teal", linewidth=1.0, alpha=0.8, label="Voltage Signal")
    axes[0].axvline(t_step, color="gray", linestyle="--", alpha=0.7)
    axes[0].set_ylabel("Amplitude [p.u.]")
    axes[0].set_title("Window-Based Method Assessment: Frequency Step (60→63Hz)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower right")

    axes[1].plot(t, f_true, 'k--', linewidth=2.0, alpha=0.6, label="True Frequency")
    axes[1].plot(t, f_pll, color="crimson", linewidth=1.5, alpha=0.8, label="PLL (Feedback Loop)")
    axes[1].plot(t, f_tft, color="mediumpurple", linewidth=2.0, label="TFT (2-Cycle Window)")
    
    axes[1].axvline(t_step, color="gray", linestyle="--", alpha=0.7)

    # Anotación visual de la latencia estructural
    axes[1].annotate(
        "Structural Latency\n(~1 cycle lag)",
        xy=(t_step + 0.016, 61.5), xytext=(t_step + 0.08, 60.5),
        arrowprops=dict(facecolor='black', arrowstyle='->', alpha=0.6),
        fontsize=9, color="black"
    )

    axes[1].set_ylabel("Frequency [Hz]")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylim(59.0, 64.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="lower right")

    plot_path = out_dir / "tft_latency_tradeoff.png"
    fig.savefig(plot_path)
    plt.close(fig)

    assert plot_path.exists()
    print(f"\n[DIAGNOSTIC] Plot saved at: {plot_path}")