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

from estimators.ekf import EKF_Estimator
from estimators.ra_ekf import RAEKF_Estimator

def test_ra_ekf_islanding_diagnostic():
    out_dir = ROOT / "tests" / "estimators" / "ra_ekf" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    fs = 10000.0
    dt = 1.0 / fs
    t = np.arange(0.0, 0.50, dt)

    t_jump = 0.20
    f_true = 60.0
    
    # Simulación de Isla Compuesta: Salto brutal de +60° instantáneo
    base_phase = 2.0 * np.pi * f_true * t
    phase_jump = np.where(t < t_jump, 0.0, np.pi/3.0) 
    v = np.sin(base_phase + phase_jump)

    # 1. EKF Clásico (Modelo base de estado)
    ekf = EKF_Estimator(nominal_f=60.0)
    f_ekf = ekf.estimate(t, v)

    # 2. RA-EKF (Innovación escalada y Gating activo)
    # Calibrado empíricamente para el salto
    ra_ekf = RAEKF_Estimator(nominal_f=60.0, sigma_v=0.01)
    f_ra = ra_ekf.estimate(t, v)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
    })

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    plt.subplots_adjust(hspace=0.25)

    axes[0].plot(t, v, color="teal", linewidth=1.0, alpha=0.8, label="Voltage Signal")
    axes[0].axvline(t_jump, color="gray", linestyle="--", alpha=0.7)
    axes[0].set_ylabel("Amplitude [p.u.]")
    axes[0].set_title("Composite Islanding Event: Instantaneous +60° Phase Jump")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, np.full_like(t, 60.0), 'k--', linewidth=2.0, alpha=0.6, label="Ref")
    axes[1].plot(t, f_ekf, color="crimson", linewidth=1.5, alpha=0.7, linestyle="--", label="Standard EKF (False Trip)")
    axes[1].plot(t, f_ra, color="darkred", linewidth=2.5, label="RA-EKF (Event-Gating Immunity)")
    
    # Banda de disparo de seguridad (+/- 0.5 Hz típica)
    axes[1].fill_between(t, 59.5, 60.5, color='green', alpha=0.1, label="\u00B10.5 Hz Safe Band")
    axes[1].axvline(t_jump, color="gray", linestyle="--", alpha=0.7)

    axes[1].set_ylabel("Frequency [Hz]")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylim(58.0, 64.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    plot_path = out_dir / "ra_ekf_islanding.png"
    fig.savefig(plot_path)
    plt.close(fig)

    assert plot_path.exists()