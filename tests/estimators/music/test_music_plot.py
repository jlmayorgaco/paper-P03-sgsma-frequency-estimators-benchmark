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
from estimators.music import MUSIC_Estimator

def test_music_vs_esprit_diagnostic():
    out_dir = ROOT / "tests" / "estimators" / "music" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    fs = 10000.0
    t = np.arange(0.0, 0.3, 1.0/fs)
    
    # Escalón de frecuencia: 60 -> 63 Hz
    f_true = np.where(t < 0.1, 60.0, 63.0)
    phase = np.cumsum(2.0 * np.pi * f_true / fs)
    v = np.sin(phase)

    # ESPRIT
    start = time.time()
    esprit = ESPRIT_Estimator(nominal_f=60.0, n_cycles=1.0)
    f_esprit = esprit.estimate(t, v)
    t_esprit = time.time() - start

    # MUSIC
    start = time.time()
    music = MUSIC_Estimator(nominal_f=60.0, n_cycles=1.0)
    f_music = music.estimate(t, v)
    t_music = time.time() - start

    plt.figure(figsize=(10, 6))
    plt.plot(t, f_true, 'k--', alpha=0.5, label="True Freq")
    plt.plot(t, f_esprit, color="darkviolet", linewidth=2.5, alpha=0.7, label=f"ESPRIT (Time: {t_esprit:.2f}s)")
    plt.plot(t, f_music, color="crimson", linewidth=1.5, label=f"MUSIC (Time: {t_music:.2f}s)")
    
    plt.title("Subspace Methods: MUSIC vs ESPRIT")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [s]")
    plt.ylim(59, 64)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plot_path = out_dir / "music_vs_esprit.png"
    plt.savefig(plot_path)
    plt.close()
    
    print(f"\n[BENCHMARK] ESPRIT: {t_esprit:.3f}s | MUSIC: {t_music:.3f}s")