import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

# Configuración de rutas
ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from estimators.esprit import ESPRIT_Estimator
from estimators.ipdft import IPDFT_Estimator 

def test_esprit_vs_ipdft_resolution():
    out_dir = ROOT / "tests" / "estimators" / "esprit" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Frecuencia de muestreo del DSP (10 kHz)
    fs = 10000.0 
    t = np.arange(0.0, 0.4, 1.0/fs)
    
    # Escenario: Escalón de frecuencia de 60 a 62 Hz en t=0.1s
    f_true = np.where(t < 0.1, 60.0, 62.0)
    phase = np.cumsum(2.0 * np.pi * f_true / fs)
    v = np.sin(phase)

    # 1. ESPRIT (2 ciclos)
    # El ESPRIT opera directamente sobre la tasa de entrada (10 kHz)
    start = time.time()
    esprit = ESPRIT_Estimator(nominal_f=60.0, n_cycles=1.0)
    f_esprit = esprit.estimate(t, v)
    t_esprit = time.time() - start

    # 2. IPDFT (2 ciclos) 
    # IMPORTANTE: decim=1 porque 'v' ya está a la frecuencia del DSP (10 kHz)
    start = time.time()
    ipdft = IPDFT_Estimator(nominal_f=60.0, cycles=2.0, decim=1) 
    f_ipdft = ipdft.estimate(t, v)
    t_ipdft = time.time() - start

    # Graficación de resultados
    plt.figure(figsize=(10, 6))
    plt.plot(t, f_true, 'k--', alpha=0.6, label="True Freq")
    plt.plot(t, f_ipdft, color="forestgreen", linewidth=1.5, label=f"IpDFT (Time: {t_ipdft:.3f}s)")
    plt.plot(t, f_esprit, color="darkviolet", linewidth=2, label=f"ESPRIT (Time: {t_esprit:.3f}s)")
    
    plt.title("Window Methods: High-Resolution ESPRIT vs IpDFT")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [s]")
    plt.ylim(59, 63)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Guardar artefacto
    plot_path = out_dir / "esprit_comparison.png"
    plt.savefig(plot_path)
    plt.close()
    
    print(f"\n[BENCHMARK RESULTS]")
    print(f"ESPRIT Execution Time: {t_esprit:.4f}s")
    print(f"IpDFT Execution Time:  {t_ipdft:.4f}s")
    print(f"Plot saved at: {plot_path}")

if __name__ == "__main__":
    test_esprit_vs_ipdft_resolution()