from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = REPO_ROOT / "tests" / "montecarlo" / "outputs"
DEFAULT_OUTPUT_DIR = PAPER_ROOT / "outputs"
DEFAULT_FIGURE_DIR = PAPER_ROOT / "Figures" / "Plots_And_Graphs"
FIG_BASENAME = "Fig1_Scenarios_Wide"


IEEE_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 7.0,
    "axes.titlesize": 7.0,
    "axes.labelsize": 6.5,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,
    "figure.dpi": 300,
}


def _load_series(source_dir: Path, folder: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    csv_path = source_dir / folder / f"{folder}_scenario.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing scenario CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    return (
        df["t_s"].to_numpy(),
        df["v_pu"].to_numpy(),
        df["f_true_hz"].to_numpy(),
    )


def _add_ibr_noise(
    t: np.ndarray,
    v: np.ndarray,
    h5: float = 0.02,
    h7: float = 0.01,
    inter_f: float = 180.0,
    inter_amp: float = 0.003,
    sigma: float = 0.001,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v_noisy = v.copy()
    phase_nom = 2.0 * np.pi * 60.0 * t
    v_noisy += h5 * np.sin(5.0 * phase_nom)
    v_noisy += h7 * np.sin(7.0 * phase_nom)
    v_noisy += inter_amp * np.sin(2.0 * np.pi * inter_f * t)
    v_noisy += rng.normal(0.0, sigma, size=len(t))
    return v_noisy


def _save_outputs(fig: plt.Figure, output_dir: Path, figure_dir: Path, publish_final: bool) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    out_png = output_dir / f"{FIG_BASENAME}.png"
    out_pdf = output_dir / f"{FIG_BASENAME}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)

    figure_png = figure_dir / out_png.name
    figure_pdf = figure_dir / out_pdf.name
    figure_png.write_bytes(out_png.read_bytes())
    figure_pdf.write_bytes(out_pdf.read_bytes())

    if publish_final:
        final_png = figure_dir / "Fig1_Scenarios_Final.png"
        final_pdf = figure_dir / "Fig1_Scenarios_Final.pdf"
        final_out_png = output_dir / "Fig1_Scenarios_Final.png"
        final_out_pdf = output_dir / "Fig1_Scenarios_Final.pdf"
        for src, dst in (
            (out_png, final_out_png),
            (out_pdf, final_out_pdf),
            (out_png, final_png),
            (out_pdf, final_pdf),
        ):
            dst.write_bytes(src.read_bytes())

    return out_png, out_pdf


def build_figure(
    source_dir: Path,
    output_dir: Path,
    figure_dir: Path,
    publish_final: bool,
) -> tuple[Path, Path]:
    t_a, v_a, f_a = _load_series(source_dir, "IEEE_Mag_Step")
    t_b, v_b, f_b = _load_series(source_dir, "IEEE_Freq_Ramp")
    t_c, v_c, f_c = _load_series(source_dir, "IEEE_Modulation")
    t_d, v_d, f_d = _load_series(source_dir, "IEEE_Phase_Jump_60")
    t_e, v_e, f_e = _load_series(source_dir, "IBR_Multi_Event")
    t_f, v_f, f_f = _load_series(source_dir, "IBR_Power_Imbalance_Ringdown")

    rng_a = np.random.default_rng(101)
    rng_b = np.random.default_rng(102)
    rng_c = np.random.default_rng(103)
    v_a_plot = _add_ibr_noise(t_a, v_a, h5=0.005, h7=0.005, seed=1) + rng_a.normal(0.0, 0.01, len(v_a))
    v_b_plot = _add_ibr_noise(t_b, v_b, h5=0.001, h7=0.002, seed=2) + rng_b.normal(0.0, 0.01, len(v_b))
    v_c_plot = _add_ibr_noise(t_c, v_c, h5=0.001, h7=0.002, seed=3) + rng_c.normal(0.0, 0.01, len(v_c))
    v_d_plot = _add_ibr_noise(t_d, v_d, h5=0.0001, h7=0.01, seed=4)
    v_e_plot = _add_ibr_noise(t_e, v_e, h5=0.0001, h7=0.01, seed=5)
    v_f_plot = _add_ibr_noise(t_f, v_f, h5=0.0001, h7=0.01, seed=6)

    t_a_ms = t_a * 1000.0
    t_b_ms = t_b * 1000.0
    t_c_ms = t_c * 1000.0
    t_d_ms = t_d * 1000.0
    t_e_ms = t_e * 1000.0
    t_f_ms = t_f * 1000.0

    multi_zoom = (2450.0, 2550.0) if t_e_ms.max() >= 2550.0 else (450.0, 550.0)
    multi_event_1 = 1000.0 if t_e_ms.max() >= 2500.0 else 500.0
    multi_event_2 = 2500.0 if t_e_ms.max() >= 2500.0 else 1000.0
    multi_full_xlim = (0.0, 5000.0) if t_e_ms.max() >= 5000.0 else (0.0, float(t_e_ms.max()))
    ring_full_xlim = (0.0, 2000.0) if t_f_ms.max() >= 2000.0 else (0.0, float(t_f_ms.max()))

    with plt.rc_context(IEEE_RC):
        fig, axs = plt.subplots(
            6,
            2,
            figsize=(8.15, 4.70),
            gridspec_kw={"hspace": 0.52, "wspace": 0.20},
        )
        bbox = dict(boxstyle="square,pad=0.1", fc="white", alpha=0.8, ec="none")

        def row(
            i: int,
            t_ms: np.ndarray,
            v_plot: np.ndarray,
            f_true: np.ndarray,
            left_title: str,
            freq_ylim: tuple[float, float],
            left_xlim: tuple[float, float],
            right_title: str,
            right_xlim: tuple[float, float] | None = None,
        ) -> None:
            left_color = "purple" if i in (4, 5) else "b"
            axs[i, 0].plot(t_ms, v_plot, left_color, lw=0.5)
            axs[i, 0].set_xlim(left_xlim)
            axs[i, 0].set_yticks([])
            axs[i, 0].set_title(left_title, fontsize=7, fontweight="bold", pad=2)
            axs[i, 0].set_ylabel("V [pu]", rotation=0, labelpad=5, fontsize=6)

            axs[i, 1].plot(t_ms, f_true, "r", lw=0.8)
            axs[i, 1].set_ylim(freq_ylim)
            if right_xlim is not None:
                axs[i, 1].set_xlim(right_xlim)
            axs[i, 1].set_title(right_title, fontsize=7, pad=2)
            axs[i, 1].set_ylabel("f [Hz]", labelpad=1)
            if i == 5:
                axs[i, 1].set_xlabel("Time [ms]", labelpad=1)

        row(0, t_a_ms, v_a_plot, f_a, "(A) Step (+10%)", (59.5, 60.5), (450.0, 550.0), "Freq (Ideal)", right_xlim=(0.0, 1500.0))
        row(1, t_b_ms, v_b_plot, f_b, "(B) Ramp (+5Hz/s)", (59.0, 65.0), (0.0, 1000.0), "Freq (+5Hz/s)", right_xlim=(0.0, 1000.0))
        row(2, t_c_ms, v_c_plot, f_c, "(C) Modulation (2Hz)", (59.4, 60.6), (0.0, 1000.0), "Freq (FM)", right_xlim=(0.0, 1500.0))
        row(
            3,
            t_d_ms,
            v_d_plot,
            f_d,
            "(D) Islanding (Jump)",
            (50.0, 80.0),
            (680.0, 720.0),
            "Dirac Impulse",
            right_xlim=(0.0, 1500.0),
        )
        row(
            4,
            t_e_ms,
            v_e_plot,
            f_e,
            "(E) Multi-Evt (Noise)",
            (57.5, 60.5) if t_e_ms.max() < 2500.0 else (50.0, 65.0),
            multi_zoom,
            "Full Profile",
            right_xlim=multi_full_xlim,
        )
        row(
            5,
            t_f_ms,
            v_f_plot,
            f_f,
            "(F) Ringdown",
            (59.0, 61.1),
            (900.0, 1200.0),
            "Ring-down",
            right_xlim=ring_full_xlim,
        )

        mag_pre = (t_a_ms >= 450.0) & (t_a_ms <= 500.0)
        mag_post = (t_a_ms >= 500.0) & (t_a_ms <= 550.0)
        axs[0, 0].plot(t_a_ms[mag_pre], v_a_plot[mag_pre], color="navy", lw=0.70)
        axs[0, 0].plot(t_a_ms[mag_post] - 50.0, v_a_plot[mag_post], color="royalblue", lw=0.75, linestyle="--", alpha=0.95)
        axs[0, 0].text(452.0, 0.96, "1.0 pu", fontsize=5.8, color="navy", ha="left", va="bottom", bbox=bbox)
        axs[0, 0].text(483.0, 1.06, "1.1 pu", fontsize=5.8, color="royalblue", ha="left", va="bottom", bbox=bbox)

        axs[3, 0].annotate(
            "Jump",
            xy=(700.0, v_d_plot[(np.abs(t_d_ms - 700.0)).argmin()]),
            xytext=(689.0, 0.6),
            textcoords="data",
            arrowprops=dict(arrowstyle="->", color="r"),
            color="r",
            fontsize=6,
        )
        axs[3, 1].arrow(700.0, 60.0, 0.0, 15.0, head_width=20.0, color="red")

        axs[4, 0].annotate(
            "Jump" if t_e_ms.max() < 2500.0 else r"Jump $+80^\circ$",
            xy=(multi_event_1, v_e_plot[(np.abs(t_e_ms - multi_event_1)).argmin()]),
            xytext=((multi_zoom[0] + 10.0), 0.6),
            arrowprops=dict(arrowstyle="->", color="k"),
            fontsize=6,
            bbox=bbox,
        )

        axs[4, 1].axvline(multi_event_1, color="k", linestyle=":", lw=0.7)
        axs[4, 1].axvline(multi_event_2, color="k", linestyle=":", lw=0.7)
        if t_e_ms.max() >= 2500.0:
            axs[4, 1].text(multi_event_1 + 20.0, 61.5, r"1. +40$^\circ$ Jump", fontsize=6, ha="left", color="k", bbox=bbox)
            axs[4, 1].text(1500.0, 56.5, "2. Neg. Ramp", fontsize=6, ha="center", color="k", bbox=bbox)
            axs[4, 1].text(3000.0, 61.0, "3. Ring-down", fontsize=6, ha="center", color="b", bbox=bbox)
        else:
            axs[4, 1].text(multi_event_1 + 20.0, 60.15, "1. Jump", fontsize=6, ha="left", color="k", bbox=bbox)
            axs[4, 1].text(900.0, 58.9, "2. Neg. Ramp", fontsize=6, ha="center", color="k", bbox=bbox)
            axs[4, 1].text(1550.0, 58.15, "3. Hold @ 58 Hz", fontsize=6, ha="center", color="k", bbox=bbox)

        axs[5, 1].axvline(500.0, color="k", linestyle=":", lw=0.7)
        axs[5, 1].axvline(1000.0, color="k", linestyle=":", lw=0.7)
        axs[5, 1].text(640.0, 59.15, "1. Ramp", fontsize=6, ha="left", color="k", bbox=bbox)
        axs[5, 1].text(1120.0, 60.88, "2. Ring-down", fontsize=6, ha="left", color="k", bbox=bbox)

        for i in range(5):
            axs[i, 0].set_xlabel("")
            axs[i, 1].set_xlabel("")

        plt.subplots_adjust(hspace=0.42, wspace=0.15, top=0.985, bottom=0.04)

        out_png, out_pdf = _save_outputs(fig, output_dir, figure_dir, publish_final)
        plt.close(fig)

    return out_png, out_pdf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Figure 1 in the Paper_reviewed style using tests/montecarlo/outputs."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--publish-final", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_png, out_pdf = build_figure(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        publish_final=args.publish_final,
    )
    print(f"[OK] Figure 1 PNG -> {out_png}")
    print(f"[OK] Figure 1 PDF -> {out_pdf}")
    if args.publish_final:
        print("[OK] Fig1_Scenarios_Final.* updated in outputs and Figures/Plots_And_Graphs")


if __name__ == "__main__":
    main()
