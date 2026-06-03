import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from estimators.esprit import ESPRIT_Estimator
from estimators.prony import Prony_Estimator

def test_prony_noise_sensitivity_diagnostic():
    out_dir = ROOT / "tests" / "estimators" / "prony" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    fs = 10000.0
    t = np.arange(0.0, 0.3, 1.0/fs)
    
    # Escalón de frecuencia de 60 a 62 Hz con RUIDO BLANCO (SNR ~40dB)
    f_true = np.where(t < 0.1, 60.0, 62.0)
    phase = np.cumsum(2.0 * np.pi * f_true / fs)
    
    # Inyectamos ruido para mostrar cómo Prony sufre
    np.random.seed(42)
    noise = np.random.normal(0, 0.00000000000000000000001, len(t))
    v = np.sin(phase) + noise

    # ESPRIT (Subespacios = Inmune al ruido AWGN)
    start = time.time()
    esprit = ESPRIT_Estimator(nominal_f=60.0, n_cycles=1.0)
    f_esprit = esprit.estimate(t, v)
    t_esprit = time.time() - start

    # Prony (AR = Muy sensible al ruido)
    start = time.time()
    prony = Prony_Estimator(nominal_f=60.0, n_cycles=1.0, order=6) # Orden 6 para intentar filtrar el ruido
    f_prony = prony.estimate(t, v)
    t_prony = time.time() - start

    plt.figure(figsize=(10, 6))
    plt.plot(t, f_true, 'k--', alpha=0.5, label="True Freq")
    plt.plot(t, f_prony, color="darkorange", linewidth=1.5, alpha=0.8, label=f"Prony (Time: {t_prony:.2f}s)")
    plt.plot(t, f_esprit, color="darkviolet", linewidth=2.0, alpha=0.9, label=f"ESPRIT (Time: {t_esprit:.2f}s)")
    
    plt.title("Parametric Methods under Noise: Prony vs ESPRIT")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [s]")
    plt.ylim(59, 63)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plot_path = out_dir / "prony_vs_esprit_noise.png"
    plt.savefig(plot_path)
    plt.close()
    
    print(f"\n[BENCHMARK] Prony: {t_prony:.3f}s | ESPRIT: {t_esprit:.3f}s")