import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Configuración de rutas
ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenarios.ibr_chamorro_playback import IBRChamorroPlaybackScenario

def test_playback_plot_and_csv():
    # 1. Crear directorio de artefactos
    out_dir = ROOT / "tests" / "scenarios" / "ibr_chamorro_playback" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Ejecutar el escenario (limpio de ruido extra)
    sc = IBRChamorroPlaybackScenario.run(noise_sigma=0.0, normalize_pu=True)

    # 3. Guardar CSV procesado para auditoría
    df = pd.DataFrame({
        "t_s": sc.t, 
        "v_pu": sc.v, 
        "f_true_hz": sc.f_true
    })
    df.to_csv(out_dir / "ibr_chamorro_processed.csv", index=False)

    # 4. Configuración estética de Matplotlib
    plt.rcParams.update({
        "font.family": "serif", 
        "font.size": 9, 
        "figure.dpi": 200, 
        "savefig.bbox": "tight"
    })
    
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [1, 1, 0.8]})
    plt.subplots_adjust(hspace=0.4)

    # --- PANEL 1: VOLTAJE ---
    axes[0].plot(sc.t, sc.v, linewidth=0.7, color='teal', label="Voltage Va (p.u.)")
    axes[0].set_ylabel("Amplitude [p.u.]")
    axes[0].set_title("Chamorro Playback: Normalized Voltage Signal")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    # --- PANEL 2: FRECUENCIA (GROUND TRUTH) ---
    # Ignoramos bordes para que la escala del eje Y sea útil
    mask = (sc.t > 0.1) & (sc.t < (sc.t[-1] - 0.1))
    axes[1].plot(sc.t[mask], sc.f_true[mask], linewidth=1.5, color='crimson', label="f_true (Hilbert Offline)")
    axes[1].set_ylabel("Frequency [Hz]")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_title("Extracted Reference Frequency")
    axes[1].grid(True, alpha=0.3)
    
    # Ajuste dinámico de límites en Y
    f_mean = np.median(sc.f_true[mask])
    axes[1].set_ylim(f_mean - 5, f_mean + 5)
    axes[1].legend(loc="upper right")

    # --- PANEL 3: ANÁLISIS ESPECTRAL (FFT) ---
    # Para ver si hay componentes raras (como los 75Hz)
    from scipy.fft import fft, fftfreq
    N = len(sc.v)
    T = sc.t[1] - sc.t[0]
    vf = fft(sc.v)
    xf = fftfreq(N, T)[:N//2]
    axes[2].semilogy(xf, 2.0/N * np.abs(vf[0:N//2]), color='black', linewidth=0.8)
    axes[2].set_xlim(0, 150) # Foco en la zona de 60Hz y armónicos bajos
    axes[2].set_ylabel("Magnitude (Log)")
    axes[2].set_xlabel("Frequency [Hz]")
    axes[2].set_title("Power Spectrum Density (FFT)")
    axes[2].grid(True, alpha=0.3, which='both')

    # Guardar y cerrar
    plot_path = out_dir / "ibr_chamorro_playback_diagnostic.png"
    fig.savefig(plot_path)
    plt.close(fig)
    
    print(f"\n[DIAGNOSTIC] Plot saved at: {plot_path}")
    print(f"[DIAGNOSTIC] Median Frequency: {f_mean:.4f} Hz")