import sys
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")  # Evita errores de backend en Windows/CI
import matplotlib.pyplot as plt

# Rutas
ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from estimators.ipdft import IPDFT_Estimator
from estimators.pi_gru import PI_GRU_Estimator  # Added GRU Import


def test_estimators_visual_diagnostic():
    out_dir = ROOT / "tests" / "estimators" / "pi_gru" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Setup Signal (Generando a 10 kHz directamente para estabilizar PI-GRU)
    fs_phys = 10_000.0 
    dt = 1.0 / fs_phys
    t = np.arange(0.0, 0.20, dt)

    t_step = 0.08
    f_pre = 60.0   # Ajustado a 60 Hz
    f_post = 62.5  # Salto de 2.5 Hz

    f_true = np.where(t < t_step, f_pre, f_post)
    phase = np.cumsum(2.0 * np.pi * f_true * dt)

    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 0.002, size=len(t))
    v = np.sin(phase) + noise

    # 2. IpDFT Inference
    print("[...] Procesando IpDFT...")
    # decim=1 porque la señal ya viene en 10 kHz
    ipdft = IPDFT_Estimator(nominal_f=60.0, cycles=2.0, decim=1)
    f_ipdft = ipdft.step_vectorized(v)

    latency_samples = ipdft.structural_latency_samples()
    latency_s = latency_samples / fs_phys
    t_visible = t_step + latency_s

    # 3. PI-GRU Inference
    print("[...] Procesando PI-GRU...")
    # dt ahora es 1e-4, coincidiendo con la fase de entrenamiento
    gru = PI_GRU_Estimator(nominal_f=60.0, dt=dt)
    f_gru = gru.estimate(t, v)

    # 4. Rendering
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
    })

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    plt.subplots_adjust(hspace=0.2)

    # Top Plot: Signal
    axes[0].plot(
        t, v,
        linewidth=1.0, color="teal", alpha=0.85,
        label="Voltage Signal (with mild noise)"
    )
    axes[0].axvline(
        t_step, color="gray", linestyle="--", linewidth=1.0, alpha=0.8,
        label="Frequency Step (60→62.5 Hz)" # Etiqueta actualizada
    )
    axes[0].set_ylabel("Amplitude [p.u.]")
    axes[0].set_title("Input Signal")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower right")

    # Bottom Plot: Frequencies
    axes[1].plot(
        t, f_true,
        linewidth=2.0, color="black", linestyle="--", alpha=0.6,
        label="True Frequency"
    )
    axes[1].plot(
        t, f_ipdft,
        linewidth=2.0, color="crimson",
        label="IpDFT (Hann + Parabolic Interpolation)"
    )
    axes[1].plot(
        t, f_gru,
        linewidth=2.0, color="coral",
        label="PI-GRU (PIDRE Trained)"
    )
    
    axes[1].axvline(
        t_step, color="gray", linestyle="--", linewidth=1.0, alpha=0.8,
        label="Event Time"
    )
    axes[1].axvline(
        t_visible, color="darkorange", linestyle=":", linewidth=1.5, alpha=0.9,
        label=f"IpDFT Latency ≈ {1e3 * latency_s:.1f} ms"
    )

    axes[1].set_ylabel("Frequency [Hz]")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_title("Estimator Dynamic Response vs. True Frequency")
    axes[1].set_ylim(58.5, 64.0) # Límites ajustados para 60Hz
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="lower right")

    plot_path = out_dir / "combined_visual_diagnostic2.png"
    fig.savefig(plot_path)
    plt.close(fig)

    assert plot_path.exists()
    print(f"\n[DIAGNOSTIC] Plot saved at: {plot_path}")


if __name__ == "__main__":
    test_estimators_visual_diagnostic()