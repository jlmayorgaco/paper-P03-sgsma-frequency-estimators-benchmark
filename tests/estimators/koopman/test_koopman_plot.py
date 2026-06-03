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
from estimators.koopman import Koopman_Estimator
from estimators.ipdft import IPDFT_Estimator  # Importamos IpDFT

def test_koopman_vs_esprit_transient():
    out_dir = ROOT / "tests" / "estimators" / "koopman" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    fs = 10000.0
    t = np.arange(0.0, 0.4, 1.0/fs)
    
    # Escenario: Distorsión AM-FM severa (Típico en faltas de baja inercia)
    # La frecuencia cae a 58 Hz oscilando, y la amplitud decae.
    f_true = np.where(t < 0.1, 60.0, 60.0 - 2.0 * np.sin(10 * np.pi * (t - 0.1)))
    f_true[t > 0.2] = 58.0
    
    phase = np.cumsum(2.0 * np.pi * f_true / fs)
    amp = np.where(t < 0.1, 1.0, 1.0 - 0.5 * (t - 0.1)) # Caída de tensión
    
    v = amp * np.sin(phase) + np.random.normal(0, 0.00000000000000005, len(t))

    # 1. IpDFT (Baseline clásico - Rápido pero asume estacionariedad)
    start = time.time()
    ipdft = IPDFT_Estimator(nominal_f=60.0, cycles=1.5, decim=1)
    f_ipdft = ipdft.estimate(t, v)
    t_ipdft = time.time() - start

    # 2. ESPRIT (Modelo estricto de subespacios)
    start = time.time()
    esprit = ESPRIT_Estimator(nominal_f=60.0, n_cycles=1.5)
    f_esprit = esprit.estimate(t, v)
    t_esprit = time.time() - start

    # 3. Koopman / EDMD (Modelo dinámico no lineal)
    start = time.time()
    rk = Koopman_Estimator(nominal_f=60.0, n_cycles=1.5)
    f_koopman = rk.estimate(t, v)
    t_koopman = time.time() - start

    plt.figure(figsize=(12, 7))
    plt.plot(t, f_true, 'k--', alpha=0.6, label="True Dynamic Freq")
    
    # Añadimos IpDFT en verde
    plt.plot(t, f_ipdft, color="forestgreen", linewidth=1.5, alpha=0.7, label=f"IpDFT (Time: {t_ipdft:.3f}s)")
    plt.plot(t, f_esprit, color="darkviolet", linewidth=1.5, alpha=0.8, label=f"ESPRIT (Time: {t_esprit:.3f}s)")
    plt.plot(t, f_koopman, color="teal", linewidth=2.5, alpha=0.9, label=f"Koopman RK-DPMU (Time: {t_koopman:.3f}s)")
    
    plt.title("Non-Stationary Transient: Koopman vs ESPRIT vs IpDFT")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [s]")
    plt.ylim(57.5, 60.5)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plot_path = out_dir / "koopman_vs_esprit_dynamic.png"
    plt.savefig(plot_path)
    plt.close()
    
    print(f"\n[BENCHMARK] IpDFT: {t_ipdft:.3f}s | ESPRIT: {t_esprit:.3f}s | Koopman: {t_koopman:.3f}s")

if __name__ == "__main__":
    test_koopman_vs_esprit_transient()