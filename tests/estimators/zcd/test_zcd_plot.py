import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Rutas
ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from estimators.zcd import ZCDEstimator

def test_zcd_plot_visual_diagnostic():
    # 1. Crear directorio de artefactos
    out_dir = ROOT / "tests" / "estimators" / "zcd" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Configuración de la señal (50 Hz, igual a tu escenario Chamorro)
    fs = 10000.0
    t = np.arange(0, 0.2, 1/fs) # 200 ms
    
    # Señal con salto de frecuencia a la mitad
    # 50 Hz durante los primeros 0.1s, luego salta a 60 Hz
    f_true = np.where(t < 0.1, 50.0, 60.0)
    phase = np.cumsum(f_true) * (1/fs) * 2 * np.pi
    v = np.sin(phase)

    # 3. Estimación
    estimator = ZCDEstimator(nominal_f=50.0)
    f_est = estimator.estimate(t, v)

    # 4. Configuración estética de Matplotlib
    plt.rcParams.update({
        "font.family": "serif", 
        "font.size": 10, 
        "figure.dpi": 200, 
        "savefig.bbox": "tight"
    })
    
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    plt.subplots_adjust(hspace=0.2)

    # --- PANEL 1: VOLTAJE ---
    axes[0].plot(t, v, linewidth=1.5, color='teal', label="Voltage Signal")
    axes[0].axvline(0.1, color='gray', linestyle='--', linewidth=1, alpha=0.7, label="Freq Jump")
    
    # Marcar los cruces por cero positivos para ver dónde mide el ZCD
    crossings = (v[:-1] <= 0.0) & (v[1:] > 0.0)
    idx_cross = np.where(crossings)[0]
    axes[0].plot(t[idx_cross], v[idx_cross], 'ro', markersize=4, label="Positive Zero Crossings")
    
    axes[0].set_ylabel("Amplitude [p.u.]")
    axes[0].set_title("ZCD Input Signal (50 Hz -> 60 Hz Step)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    # --- PANEL 2: FRECUENCIA ---
    axes[1].plot(t, f_true, linewidth=1.5, color='gray', linestyle='--', label="True Frequency")
    axes[1].plot(t, f_est, linewidth=1.5, color='crimson', label="ZCD Estimate")
    axes[1].axvline(0.1, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    
    axes[1].set_ylabel("Frequency [Hz]")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_title("Zero-Crossing Detector Response (ZOH Behavior)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper left")

    # Guardar
    plot_path = out_dir / "zcd_step_diagnostic.png"
    fig.savefig(plot_path)
    plt.close(fig)
    
    print(f"\n[DIAGNOSTIC] Plot saved at: {plot_path}")