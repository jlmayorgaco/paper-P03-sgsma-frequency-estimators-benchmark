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

from estimators.rls import RLS_Estimator


def test_rls_vs_vff_diagnostic():
    out_dir = ROOT / "tests" / "estimators" / "rls" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    fs = 10000.0
    dt = 1.0 / fs
    t = np.arange(0.0, 1.0, dt)

    t_step = 0.20

    # Escalón severo pero limpio: 60 -> 65 Hz
    f_true = np.where(t < t_step, 60.0, 65.0)
    phase = np.cumsum(2.0 * np.pi * f_true * dt)
    v = np.sin(phase)

    # RLS clásico: baseline principal
    rls_fixed = RLS_Estimator(
        nominal_f=60.0,
        is_vff=False,
        lambda_fixed=0.999,
        output_smoothing=0.03,
        p0=25.0,
    )
    f_fixed = rls_fixed.estimate(t, v)

    # VFF-RLS: variante adaptativa exploratoria
    rls_vff = RLS_Estimator(
        nominal_f=60.0,
        is_vff=True,
        alpha_vff=0.20,
        lambda_min=0.90,
        lambda_max=0.9995,
        vff_beta=0.02,
        output_smoothing=0.03,
        p0=25.0,
    )
    f_vff = rls_vff.estimate(t, v)

    # Validaciones numéricas mínimas
    assert np.all(np.isfinite(f_fixed))
    assert np.all(np.isfinite(f_vff))

    final_mask = t > 0.80
    f_fixed_final = f_fixed[final_mask]
    f_vff_final = f_vff[final_mask]

    # El fixed debe converger bien
    assert np.isclose(np.mean(f_fixed_final), 65.0, atol=0.08)

    # El VFF debe ser estable y converger razonablemente, sin exigir superioridad
    assert np.isclose(np.mean(f_vff_final), 65.0, atol=0.20)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "figure.dpi": 200,
            "savefig.bbox": "tight",
        }
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    plt.subplots_adjust(hspace=0.25)

    axes[0].plot(
        t,
        v,
        color="teal",
        linewidth=1.0,
        alpha=0.85,
        label="Voltage Signal",
    )
    axes[0].axvline(
        t_step,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
        label="Frequency Step",
    )
    axes[0].set_ylabel("Amplitude [p.u.]")
    axes[0].set_title("RLS Frequency Tracking Under Severe 60→65 Hz Step")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower right")

    axes[1].plot(
        t,
        f_true,
        "k--",
        linewidth=2.0,
        alpha=0.65,
        label="True Frequency",
    )
    axes[1].plot(
        t,
        f_fixed,
        color="crimson",
        linewidth=2.0,
        alpha=0.85,
        label=r"Fixed-$\lambda$ RLS ($\lambda=0.999$)",
    )
    axes[1].plot(
        t,
        f_vff,
        color="dodgerblue",
        linewidth=2.0,
        label=r"VFF-RLS (exploratory)",
    )
    axes[1].axvline(
        t_step,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
    )

    axes[1].set_ylabel("Frequency [Hz]")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylim(59.0, 66.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="lower right")

    plot_path = out_dir / "rls_vff_adaptation.png"
    fig.savefig(plot_path)
    plt.close(fig)

    assert plot_path.exists()
    print(f"\n[DIAGNOSTIC] Plot saved at: {plot_path}")