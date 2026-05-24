from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
for path in (str(SRC), str(ROOT)):
    while path in sys.path:
        sys.path.remove(path)
sys.path.insert(0, str(SRC))
sys.path.insert(1, str(ROOT))

from scenarios.ieee_mag_step import IEEEMagStepScenario


DEFAULT_ARTIFACT_DIR = ROOT / "artifacts" / "amplitude_step_v10_rls_diagnostic_21steps_with30"
SIGNAL_PARAM_COLUMNS = [
    "run_idx",
    "scenario_name",
    "duration_s",
    "freq_hz",
    "phase_rad",
    "amp_pre_pu",
    "amp_post_pu",
    "t_step_s",
    "noise_sigma",
    "seed",
]

AMPLITUDE_STEP_REGIONS = (
    ("Voltage PMU", 1.0, 10.0, "#66BB6A"),
    ("Grid stress", 10.0, 25.0, "#DCE775"),
    ("IBR normal", 25.0, 100.0, "#FDD835"),
    ("IBR stress", 100.0, 500.0, "#FFB74D"),
    ("Mega stress", 500.0, 1000.0, "#EF5350"),
)


def _step_token(step_percent: float) -> str:
    return f"{step_percent:g}".replace(".", "p")


def _format_step(value: float, _pos: int | None = None) -> str:
    return f"{value:g}"


def _load_manifest(artifact_dir: Path) -> dict[str, Any]:
    manifest_path = artifact_dir / "scenario_manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _load_signal_parameters(artifact_dir: Path) -> pd.DataFrame:
    summary_paths = sorted(artifact_dir.rglob("*__*_summary.csv"))
    if not summary_paths:
        raise FileNotFoundError(f"No per-estimator summary CSV files found under {artifact_dir}")

    frames: list[pd.DataFrame] = []
    for path in summary_paths:
        df = pd.read_csv(path, usecols=lambda col: col in SIGNAL_PARAM_COLUMNS)
        missing = sorted(set(SIGNAL_PARAM_COLUMNS) - set(df.columns))
        if missing:
            raise ValueError(f"{path} is missing signal parameter columns: {missing}")
        frames.append(df)

    params = pd.concat(frames, ignore_index=True)
    params = params.drop_duplicates(
        subset=["scenario_name", "run_idx", "seed", "amp_pre_pu", "amp_post_pu", "t_step_s"]
    ).reset_index(drop=True)
    params["step_percent"] = 100.0 * (params["amp_post_pu"].astype(float) / params["amp_pre_pu"].astype(float) - 1.0)
    params["step_percent"] = params["step_percent"].round(6)
    params = params.sort_values(["step_percent", "run_idx"]).reset_index(drop=True)
    return params


def _generate_signal(row: pd.Series, *, noise: bool = True):
    seed = int(row["seed"]) if noise else 0
    noise_sigma = float(row["noise_sigma"]) if noise else 0.0
    return IEEEMagStepScenario.run(
        duration_s=float(row["duration_s"]),
        freq_hz=float(row["freq_hz"]),
        phase_rad=float(row["phase_rad"]),
        amp_pre_pu=float(row["amp_pre_pu"]),
        amp_post_pu=float(row["amp_post_pu"]),
        t_step_s=float(row["t_step_s"]),
        noise_sigma=noise_sigma,
        seed=seed,
    )


def _reference_row(step_percent: float) -> pd.Series:
    return pd.Series(
        {
            "run_idx": 0,
            "scenario_name": f"Sweep_AmplitudeStep_{_step_token(step_percent)}pct",
            "duration_s": 1.8,
            "freq_hz": 60.0,
            "phase_rad": 0.0,
            "amp_pre_pu": 1.0,
            "amp_post_pu": 1.0 + float(step_percent) / 100.0,
            "t_step_s": 0.5,
            "noise_sigma": 0.0,
            "seed": 0,
            "step_percent": float(step_percent),
        }
    )


def _shade_regions(ax: plt.Axes, *, y: float | None = None) -> None:
    ymin, ymax = ax.get_ylim()
    for label, lo, hi, color in AMPLITUDE_STEP_REGIONS:
        ax.axvspan(lo, hi, color=color, alpha=0.10, linewidth=0)
        if y is not None:
            x = math.sqrt(lo * hi)
            ax.text(x, y, label, ha="center", va="bottom", fontsize=7, rotation=0, color="#263238")
    ax.set_ylim(ymin, ymax)


def plot_reference_grid(params: pd.DataFrame, out_dir: Path) -> list[Path]:
    steps = sorted(params["step_percent"].unique().astype(float).tolist())
    if not steps:
        return []

    ncols = 3
    nrows = int(math.ceil(len(steps) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.0, 2.55 * nrows), sharex=True)
    axes_arr = np.atleast_1d(axes).ravel()

    for ax, step in zip(axes_arr, steps):
        row = _reference_row(step)
        sc = _generate_signal(row, noise=False)
        t0 = float(row["t_step_s"])
        tau = sc.t - t0
        mask = (tau >= -0.04) & (tau <= 0.04)
        amp_pre = float(row["amp_pre_pu"])
        amp_post = float(row["amp_post_pu"])
        ax.plot(tau[mask] * 1000.0, sc.v[mask], color="#111111", linewidth=0.85)
        ax.axvline(0.0, color="#455A64", linestyle="--", linewidth=0.8)
        for amp in (amp_pre, -amp_pre, amp_post, -amp_post):
            ax.axhline(amp, color="#607D8B", linestyle=":", linewidth=0.6, alpha=0.75)
        ylim = max(1.15 * amp_post, 1.25)
        ax.set_ylim(-ylim, ylim)
        ax.set_title(f"+{step:g}%  ({amp_pre:g}->{amp_post:g} pu)", loc="left", fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.22, linewidth=0.6)

    for ax in axes_arr[len(steps):]:
        ax.axis("off")

    fig.suptitle("Amplitude-Step Input Waveforms: Noise-Free Reference per Step", fontsize=14, fontweight="bold")
    fig.supxlabel("Time relative to step [ms]")
    fig.supylabel("Input signal x(t) [pu]")
    fig.tight_layout(rect=(0.025, 0.02, 1.0, 0.975))

    png = out_dir / "amplitude_step_reference_waveforms_grid.png"
    pdf = out_dir / "amplitude_step_reference_waveforms_grid.pdf"
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def plot_parameter_coverage(params: pd.DataFrame, out_dir: Path, manifest: dict[str, Any]) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.0), sharex=True)
    ax_amp, ax_t, ax_noise, ax_phase = axes.ravel()
    steps = sorted(params["step_percent"].unique().astype(float).tolist())

    grouped = params.groupby("step_percent", sort=True)
    amp_post = grouped["amp_post_pu"].median()
    ax_amp.plot(amp_post.index, amp_post.values, marker="o", linewidth=1.1, color="#111111")
    ax_amp.set_ylabel("Post-step amplitude [pu]")
    ax_amp.set_title("Amplitude Schedule", loc="left", fontweight="bold")

    ax_t.boxplot(
        [grouped.get_group(step)["t_step_s"].to_numpy(dtype=float) for step in steps],
        positions=steps,
        widths=[max(step * 0.08, 0.25) for step in steps],
        manage_ticks=False,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#90CAF9", "alpha": 0.55, "linewidth": 0.8},
        medianprops={"color": "#0D47A1", "linewidth": 1.1},
    )
    ax_t.set_ylabel("Step time [s]")
    ax_t.set_title("MC Step-Time Jitter", loc="left", fontweight="bold")

    ax_noise.boxplot(
        [grouped.get_group(step)["noise_sigma"].to_numpy(dtype=float) for step in steps],
        positions=steps,
        widths=[max(step * 0.08, 0.25) for step in steps],
        manage_ticks=False,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#A5D6A7", "alpha": 0.55, "linewidth": 0.8},
        medianprops={"color": "#1B5E20", "linewidth": 1.1},
    )
    ax_noise.set_ylabel("AWGN sigma [pu]")
    ax_noise.set_title("MC Noise Coverage", loc="left", fontweight="bold")

    for step, df_step in grouped:
        phase = np.mod(df_step["phase_rad"].to_numpy(dtype=float), 2.0 * math.pi)
        ax_phase.scatter(
            np.full_like(phase, float(step), dtype=float),
            phase,
            s=8,
            color="#5E35B1",
            alpha=0.45,
            linewidths=0,
        )
    ax_phase.set_ylabel("Initial phase [rad]")
    ax_phase.set_title("Phase Stratification", loc="left", fontweight="bold")
    ax_phase.set_ylim(-0.05, 2.0 * math.pi + 0.05)

    step_levels = manifest.get("step_levels_percent") or steps
    for ax in axes.ravel():
        ax.set_xscale("log")
        ax.set_xlim(max(min(step_levels) / 1.15, 0.5), max(step_levels) * 1.15)
        ax.xaxis.set_major_formatter(FuncFormatter(_format_step))
        ax.set_xlabel("Amplitude step [%]")
        ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
        _shade_regions(ax)

    fig.suptitle("Amplitude-Step Signal Coverage Used by the Current Artifact", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.95))
    png = out_dir / "amplitude_step_signal_parameter_coverage.png"
    pdf = out_dir / "amplitude_step_signal_parameter_coverage.pdf"
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def plot_mc_overlay_pdf(params: pd.DataFrame, out_dir: Path) -> list[Path]:
    pdf = out_dir / "amplitude_step_mc_signal_overlay_all_steps.pdf"
    preview_png = out_dir / "amplitude_step_mc_signal_overlay_preview.png"
    steps = sorted(params["step_percent"].unique().astype(float).tolist())
    if not steps:
        return []

    with PdfPages(pdf) as pages:
        for page_idx, step in enumerate(steps):
            df_step = params[np.isclose(params["step_percent"], step)].sort_values("run_idx")
            fig, axes = plt.subplots(2, 1, figsize=(11.0, 7.2), sharex=True, gridspec_kw={"height_ratios": [2.1, 1.0]})
            ax_sig, ax_freq = axes
            amp_pre = float(df_step["amp_pre_pu"].median())
            amp_post = float(df_step["amp_post_pu"].median())

            for _, row in df_step.iterrows():
                sc = _generate_signal(row, noise=True)
                tau = sc.t - float(row["t_step_s"])
                mask = (tau >= -0.045) & (tau <= 0.045)
                ax_sig.plot(tau[mask] * 1000.0, sc.v[mask], color="#263238", alpha=0.18, linewidth=0.65)
                ax_freq.plot(tau[mask] * 1000.0, sc.f_true[mask], color="#1565C0", alpha=0.18, linewidth=0.65)

            ax_sig.axvline(0.0, color="#B71C1C", linestyle="--", linewidth=0.9, label="Aligned step instant")
            for amp in (amp_pre, -amp_pre, amp_post, -amp_post):
                ax_sig.axhline(amp, color="#607D8B", linestyle=":", linewidth=0.8, alpha=0.8)
            y_lim = max(1.18 * amp_post, 1.25)
            ax_sig.set_ylim(-y_lim, y_lim)
            ax_sig.set_ylabel("x(t) [pu]")
            ax_sig.set_title(
                f"MC input traces, +{step:g}% amplitude step ({len(df_step)} runs)",
                loc="left",
                fontweight="bold",
            )
            ax_sig.grid(True, alpha=0.25, linewidth=0.6)
            ax_sig.legend(loc="upper right", fontsize=8)

            ax_freq.axvline(0.0, color="#B71C1C", linestyle="--", linewidth=0.9)
            ax_freq.set_ylim(59.94, 60.06)
            ax_freq.set_ylabel("f_true [Hz]")
            ax_freq.set_xlabel("Time relative to each run's step [ms]")
            ax_freq.set_title("Frequency contract: amplitude-only event", loc="left", fontweight="bold")
            ax_freq.grid(True, alpha=0.25, linewidth=0.6)

            fig.tight_layout()
            pages.savefig(fig)
            if page_idx == min(9, len(steps) - 1):
                fig.savefig(preview_png, dpi=240)
            plt.close(fig)

    return [pdf, preview_png]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot all amplitude-step input signals recorded by a sweep artifact.")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    artifact_dir = args.artifact_dir.resolve()
    out_dir = (args.out_dir or artifact_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(artifact_dir)
    params = _load_signal_parameters(artifact_dir)

    outputs: list[Path] = []
    outputs.extend(plot_reference_grid(params, out_dir))
    outputs.extend(plot_parameter_coverage(params, out_dir, manifest))
    outputs.extend(plot_mc_overlay_pdf(params, out_dir))

    index_path = out_dir / "amplitude_step_signal_plot_index.csv"
    pd.DataFrame(
        {
            "n_signal_runs": [int(len(params))],
            "n_steps": [int(params["step_percent"].nunique())],
            "min_step_percent": [float(params["step_percent"].min())],
            "max_step_percent": [float(params["step_percent"].max())],
            "source_artifact_dir": [str(artifact_dir)],
        }
    ).to_csv(index_path, index=False)
    outputs.append(index_path)

    print("[SIGNAL_PLOTS] Generated:")
    for path in outputs:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
