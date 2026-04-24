from __future__ import annotations

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipelines.paths import FIGURE1_BASENAME


ROOT = Path(__file__).resolve().parents[3]
LEGACY_OUTPUT_DIR = ROOT / "tests" / "montecarlo" / "outputs"


def _candidate_roots(base_results_dir: Path) -> list[Path]:
    roots = [base_results_dir]
    if LEGACY_OUTPUT_DIR != base_results_dir:
        roots.append(LEGACY_OUTPUT_DIR)
    return roots


def _resolve_scenario_csv(base_results_dir: Path, folder_spec: str | list[str]) -> tuple[Path, str, Path]:
    """
    Resolve the scenario CSV path for the paper-facing overview figure.

    The canonical output root is searched first. If the local workspace only
    contains a partial canonical artifact set, the function falls back to the
    legacy compatibility outputs so the scenario-overview figure remains
    reproducible and visually complete.
    """
    folders = [folder_spec] if isinstance(folder_spec, str) else list(folder_spec)
    for root in _candidate_roots(base_results_dir):
        for folder in folders:
            csv_path = root / folder / f"{folder}_scenario.csv"
            if csv_path.exists():
                return root, folder, csv_path
    tried = ", ".join(folders)
    searched = ", ".join(str(root) for root in _candidate_roots(base_results_dir))
    raise FileNotFoundError(
        f"[MEGA1] Missing scenario CSV. Tried folders: {tried}. Searched roots: {searched}"
    )


def plot_megadashboard1(
    base_results_dir: Path,
    ieee_rc: dict,
    ieee_full_w: float,
    ieee_page_h: float,
) -> None:
    """
    Generate Figure 1: ground-truth scenario overview used in the paper.

    The layout intentionally matches the established legacy version so the
    paper-facing figure remains stable across workflow changes.
    """
    print("\n[MEGA1] Building scenario signal overview from CSVs ...")

    base_results_dir = Path(base_results_dir)
    base_results_dir.mkdir(parents=True, exist_ok=True)

    dw = 0.125

    GRID_SPECS = [
        dict(
            col0=dict(
                folder="IEEE_Mag_Step",
                title="(0,0) Mag Step (+10%) Voltage",
                col="v_pu",
                color="darkblue",
                ylabel="V [pu]",
                xlim=(0.5 - dw, 0.5 + dw),
                ylim=(-1.35, 1.35),
                event_t=0.5,
            ),
            col1=dict(
                folder="IEEE_Mag_Step",
                title="(0,1) Mag Step Frequency",
                col="f_true_hz",
                color="darkred",
                ylabel="f [Hz]",
                event_t=0.5,
                xlim=(0.0, 1.5),
                ylim=(59.95, 60.05),
            ),
        ),
        dict(
            col0=dict(
                folder="IEEE_Freq_Ramp",
                title="(1,0) Ramp (+5Hz/s) Voltage",
                col="v_pu",
                color="darkblue",
                ylabel="V [pu]",
                xlim=(0.10, 0.50),
                ylim=(-1.25, 1.25),
                event_t=0.3,
            ),
            col1=dict(
                folder="IEEE_Freq_Ramp",
                title="(1,1) Ramp Frequency",
                col="f_true_hz",
                color="darkred",
                ylabel="f [Hz]",
                event_t=0.3,
                xlim=(0.0, 1.5),
                ylim=(59.8, 62.0),
            ),
        ),
        dict(
            col0=dict(
                folder="IEEE_Modulation",
                title="(2,0) Modulation (2Hz) Voltage",
                col="v_pu",
                color="darkblue",
                ylabel="V [pu]",
                xlim=(0.5 - dw, 0.5 + dw),
                ylim=(-1.5, 1.5),
            ),
            col1=dict(
                folder="IEEE_Modulation",
                title="(2,1) Modulation Frequency",
                col="f_true_hz",
                color="darkred",
                ylabel="f [Hz]",
                xlim=(0.0, 1.5),
                ylim=(59.00, 61.00),
            ),
        ),
        dict(
            col0=dict(
                folder="IEEE_Phase_Jump_60",
                title="(3,0) Islanding Jump Voltage",
                col="v_pu",
                color="darkblue",
                ylabel="V [pu]",
                xlim=(0.5 - dw, 0.5 + dw),
                ylim=(-1.25, 1.25),
                event_t=0.5,
            ),
            col1=dict(
                folder="IEEE_Phase_Jump_60",
                title="(3,1) Islanding Jump Frequency",
                col="f_true_hz",
                color="darkred",
                ylabel="f [Hz]",
                event_t=0.5,
                xlim=(0.0, 1.5),
                ylim=(59.5, 60.5),
            ),
        ),
        dict(
            col0=dict(
                folder="IBR_Multi_Event",
                title="(4,0) Multi-Evt Voltage (First Jump)",
                col="v_pu",
                color="indigo",
                ylabel="V [pu]",
                xlim=(0.5 - dw, 0.5 + dw),
                ylim=(-1.25, 1.25),
                event_t=[0.5, 1.0],
            ),
            col1=dict(
                folder="IBR_Multi_Event",
                title="(4,1) Multi-Evt Frequency Profile",
                col="f_true_hz",
                color="darkred",
                ylabel="f [Hz]",
                event_t=[0.5, 1.0],
                xlim=(0.0, 2.0),
                ylim=(56.0, 62.0),
            ),
        ),
        dict(
            col0=dict(
                folder="IBR_Power_Imbalance_Ringdown",
                title="(5,0) Ringdown Voltage",
                col="v_pu",
                color="indigo",
                ylabel="V [pu]",
                xlim=(0.5 - dw, 0.5 + dw),
                ylim=(-1.5, 1.5),
                event_t=0.5,
            ),
            col1=dict(
                folder="IBR_Power_Imbalance_Ringdown",
                title="(5,1) Ringdown Frequency Profile",
                col="f_true_hz",
                color="darkred",
                ylabel="f [Hz]",
                event_t=0.5,
                xlim=(0.0, 2.0),
                ylim=(59.0, 61.0),
            ),
        ),
    ]

    # Make Fig. 1 wider while preserving the same height so the paper can use
    # more horizontal space without stretching the content.
    fig_w = ieee_full_w * 1.43
    fig_h = ieee_page_h * 0.60

    source_roots_used: set[Path] = set()

    with plt.rc_context(ieee_rc):
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = gridspec.GridSpec(
            len(GRID_SPECS),
            2,
            figure=fig,
            hspace=0.85,
            wspace=0.20,
            left=0.08,
            right=0.98,
            top=0.92,
            bottom=0.08,
        )

        for ridx, row_spec in enumerate(GRID_SPECS):
            for cidx, col_key in enumerate(["col0", "col1"]):
                spec = row_spec[col_key]
                ax = fig.add_subplot(gs[ridx, cidx])

                root_used, folder_used, csv_path = _resolve_scenario_csv(base_results_dir, spec["folder"])
                source_roots_used.add(root_used)

                df_scen = pd.read_csv(csv_path)
                t = df_scen["t_s"].values
                y = df_scen[spec["col"]].values

                ax.plot(t, y, lw=0.75, color=spec["color"], rasterized=True)
                ax.set_ylabel(spec["ylabel"], labelpad=2, fontsize=6.5)
                ax.set_title(spec["title"], fontsize=7.5, fontweight="bold", pad=3)

                if "xlim" in spec:
                    ax.set_xlim(spec["xlim"][0], spec["xlim"][1])
                else:
                    ax.set_xlim(float(t[0]), float(t[-1]))

                if "ylim" in spec:
                    ax.set_ylim(spec["ylim"][0], spec["ylim"][1])

                if folder_used == "IEEE_Mag_Step" and spec["col"] == "v_pu":
                    mask_pre = (t >= (0.5 - dw)) & (t <= 0.5)
                    mask_post = (t > 0.5) & (t <= (0.5 + dw))
                    if mask_pre.any() and mask_post.any():
                        max_pre = np.max(y[mask_pre])
                        max_post = np.max(y[mask_post])
                        ax.axhline(max_pre, color="#333", linestyle="--", linewidth=0.5, alpha=0.8)
                        ax.axhline(max_post, color="#333", linestyle="--", linewidth=0.5, alpha=0.8)

                if "event_t" in spec:
                    events = spec["event_t"] if isinstance(spec["event_t"], list) else [spec["event_t"]]
                    for ev in events:
                        ax.axvline(ev, color="black", linestyle="--", linewidth=1.0, alpha=0.6)

                ax.grid(True, alpha=0.3, lw=0.4)
                ax.tick_params(axis="both", which="major", labelsize=6)
                ax.set_xlabel("Time [s]", labelpad=2, fontsize=7)

        fig.text(
            0.5,
            0.97,
            "IBR Frequency Estimator Benchmark - Ground-Truth Scenario Overview",
            ha="center",
            va="top",
            fontsize=8.5,
            fontweight="bold",
            color="#111",
        )

        out_png = base_results_dir / f"{FIGURE1_BASENAME}.png"
        out_pdf = base_results_dir / f"{FIGURE1_BASENAME}.pdf"
        legacy_png = base_results_dir / "megadashboard1.png"

        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")
        fig.savefig(legacy_png, dpi=300, bbox_inches="tight")
        plt.close(fig)

    if any(root != base_results_dir for root in source_roots_used):
        for root in sorted(source_roots_used):
            print(f"    [MEGA1] Source root: {root}")
    print(f"    {FIGURE1_BASENAME} saved -> {out_png}")
