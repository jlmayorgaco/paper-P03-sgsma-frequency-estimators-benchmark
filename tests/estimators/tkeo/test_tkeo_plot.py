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

from estimators.tkeo import TKEO_Estimator
from estimators.sogi_pll import SOGIPLLEstimator

def test_tkeo_noise_sensitivity_diagnostic():
    out_dir = ROOT / "tests" / "estimators" / "tkeo" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    fs = 10000.0
    t = np.arange(0.0, 0.4, 1.0/fs)
    f_true = 60.0
    
    # Señal con ruido según escenario A-C del paper (sigma=0.001)
    rng = np.random.default_rng(42)
    v = np.sin(2.0 * np.pi * f_true * t) + rng.normal(0, 0.001, len(t))

    # TKEO: Latencia P3 pero sensible
    tkeo = TKEO_Estimator(nominal_f=60.0, output_smoothing=0.005)
    f_tkeo = tkeo.estimate(t, v)

    # SOGI-PLL para comparar robustez
    sogi = SOGIPLLEstimator(nominal_f=60.0)
    f_sogi = sogi.estimate(t, v)

    plt.figure(figsize=(10, 5))
    plt.plot(t, f_tkeo, color="orange", label="TKEO (Noise Amplification)")
    plt.plot(t, f_sogi, color="crimson", label="SOGI-PLL (Filtered)")
    plt.axhline(60.0, color="black", linestyle="--", alpha=0.5)
    
    plt.title("TKEO Failure Mode: Noise Amplification ")
    plt.ylabel("Frequency [Hz]")
    plt.ylim(55, 65)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(out_dir / "tkeo_noise_fail.png")
    plt.close()