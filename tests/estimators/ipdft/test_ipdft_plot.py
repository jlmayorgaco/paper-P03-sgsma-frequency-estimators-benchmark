import sys
from pathlib import Path
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

# Rutas
ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from estimators.ipdft import IPDFT_Estimator

def test_ipdft_visual_diagnostic():
    out_dir = ROOT / "tests" / "estimators" / "ipdft" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    fs_dsp = 10_000.0  # DSP rate — what the estimator actually works at
    dt = 1.0 / fs_dsp
    t = np.arange(0.0, 0.20, dt)

    t_step = 0.08
    f_pre = 50.0
    f_post = 72.5  # Salto masivo de +22.5 Hz

    f_true = np.where(t < t_step, f_pre, f_post)
    phase = np.cumsum(2.0 * np.pi * f_true * dt)

    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 0.002, size=len(t))
    v = np.sin(phase) + noise

    estimator = IPDFT_Estimator(nominal_f=50.0, cycles=2.0)
    
    # Usar estimate() para que el estimador lea el 'dt' real de 1us
    f_ipdft = estimator.estimate(t, v)

    latency_samples = estimator.structural_latency_samples()
    latency_s = latency_samples / fs_dsp
    t_visible = t_step + latency_s

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
    })

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    plt.subplots_adjust(hspace=0.2)

    axes[0].plot(
        t,
        v,
        linewidth=1.0,
        color="teal",
        alpha=0.85,
        label="Voltage Signal (with mild noise)",
    )
    axes[0].axvline(
        t_step,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        alpha=0.8,
        label="Frequency Step (50→72.5 Hz)",
    )
    axes[0].set_ylabel("Amplitude [p.u.]")
    axes[0].set_title("Input Signal")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower right")

    axes[1].plot(
        t,
        f_true,
        linewidth=2.0,
        color="black",
        linestyle="--",
        alpha=0.6,
        label="True Frequency",
    )
    axes[1].plot(
        t,
        f_ipdft,
        linewidth=2.0,
        color="crimson",
        label="IpDFT (Hann + Parabolic Interpolation)",
    )
    axes[1].axvline(
        t_step,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        alpha=0.8,
        label="Event Time",
    )
    axes[1].axvline(
        t_visible,
        color="darkorange",
        linestyle=":",
        linewidth=1.5,
        alpha=0.9,
        label=f"Structural Latency ≈ {1e3 * latency_s:.1f} ms",
    )

    axes[1].set_ylabel("Frequency [Hz]")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_title("IpDFT Dynamic Response: Tracking an Extreme Step")
    
    # Ampliar el eje Y para que no recorte el pico de 72.5 Hz
    axes[1].set_ylim(48.0, 75.0) 
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="lower right")

    plot_path = out_dir / "ipdft_visual_diagnostic_12.png"
    fig.savefig(plot_path)
    
    print(f"\n[DIAGNOSTIC] Intentando abrir la ventana interactiva...")
    
    # BLOCK=TRUE fuerza a la terminal a pausarse hasta que cierres la imagen
    plt.show(block=True) 
    
    plt.close(fig)

    assert plot_path.exists()
    print(f"[DIAGNOSTIC] Plot guardado físicamente en: {plot_path}")

# ¡LLAMADA DIRECTA SIN CONDICIONALES!
test_ipdft_visual_diagnostic()