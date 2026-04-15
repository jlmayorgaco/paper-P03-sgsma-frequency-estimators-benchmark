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

# Asumo que tu SOGI-PLL estándar está guardado así (ajusta el import si es distinto)
from estimators.sogi_pll import SOGIPLLEstimator 
from estimators.type3_sogi_pll import Type3_SOGI_PLL_Estimator

def test_type3_rocof_tracking_diagnostic():
    out_dir = ROOT / "tests" / "estimators" / "type3_sogi_pll" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    fs = 10000.0
    dt = 1.0 / fs
    t = np.arange(0.0, 2.5, dt)

    t_ramp_start = 0.5
    rocof_rate = -10.0  # Caída severa de -10 Hz/s (típico de fallas en microredes)
    
    # Construcción de la rampa de frecuencia
    f_true = np.ones_like(t) * 60.0
    ramp_mask = t > t_ramp_start
    f_true[ramp_mask] = 60.0 + rocof_rate * (t[ramp_mask] - t_ramp_start)
    
    # La fase DEBE ser la integral matemática de la frecuencia variable
    phase = np.cumsum(2.0 * np.pi * f_true * dt)
    v = np.sin(phase)

    # 1. SOGI-PLL Clásico (Sistema Tipo 2)
    sogi_type2 = SOGIPLLEstimator(nominal_f=60.0)
    f_type2 = sogi_type2.estimate(t, v)

    # 2. SOGI-PLL Avanzado (Sistema Tipo 3)
    sogi_type3 = Type3_SOGI_PLL_Estimator(nominal_f=60.0)
    f_type3 = sogi_type3.estimate(t, v)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
    })

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    plt.subplots_adjust(hspace=0.25)

    axes[0].plot(t, v, color="teal", linewidth=0.8, alpha=0.7, label="Voltage Signal")
    axes[0].axvline(t_ramp_start, color="gray", linestyle="--", alpha=0.7)
    axes[0].set_ylabel("Amplitude [p.u.]")
    axes[0].set_title("Low-Inertia Event: Severe RoCoF Ramp (-20 Hz/s)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower left")

    axes[1].plot(t, f_true, 'k--', linewidth=2.5, alpha=0.6, label="True Frequency Ramp")
    axes[1].plot(t, f_type2, color="crimson", linewidth=1.5, alpha=0.8, label="Standard SOGI-PLL (Type-2 Steady-State Error)")
    axes[1].plot(t, f_type3, color="dodgerblue", linewidth=2.0, label="Type-3 SOGI-PLL (Zero Tracking Error)")
    
    axes[1].axvline(t_ramp_start, color="gray", linestyle="--", alpha=0.7)

    axes[1].set_ylabel("Frequency [Hz]")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylim(20.0, 62.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="lower left")

    plot_path = out_dir / "type3_rocof_tracking.png"
    fig.savefig(plot_path)
    plt.close(fig)

    assert plot_path.exists()
    print(f"\n[DIAGNOSTIC] Plot saved at: {plot_path}")