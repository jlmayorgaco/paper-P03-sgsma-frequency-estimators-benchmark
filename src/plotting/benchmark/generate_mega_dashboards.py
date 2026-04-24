"""Canonical orchestrator for benchmark figures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
LEGACY_OUTPUT_DIR = ROOT / "tests" / "montecarlo" / "outputs"

from pipelines.paths import BENCHMARK_OUTPUT_DIR, FIGURE2_BASENAME, JSON_REPORT_NAME
from plotting.benchmark.mega_dashboard1 import plot_megadashboard1
from plotting.benchmark.mega_dashboard2_p00 import md2_subplot_00
from plotting.benchmark.mega_dashboard2_p01 import md2_subplot_01
from plotting.benchmark.mega_dashboard2_p10 import md2_subplot_10
from plotting.benchmark.mega_dashboard2_p11 import md2_subplot_11
from plotting.benchmark.mega_dashboard2_p20 import md2_subplot_20
from plotting.benchmark.mega_dashboard2_p21 import md2_subplot_21
from plotting.benchmark.mega_dashboard2_p30 import md2_subplot_30
from plotting.benchmark.mega_dashboard2_p31 import md2_subplot_31

BASE_RESULTS_DIR = BENCHMARK_OUTPUT_DIR
DATA_RESULTS_DIR = BENCHMARK_OUTPUT_DIR

_IEEE_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 11.0,
    "axes.labelsize": 11.0,
    "axes.titlesize": 12.0,
    "xtick.labelsize": 10.0,
    "ytick.labelsize": 10.0,
    "legend.fontsize": 10.0,
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.4,
}

_IEEE_FULL_W = 7.16

_ESTIMATOR_FAMILIES = {
    "ZCD": "Loop-based",
    "PLL": "Loop-based",
    "SOGI-PLL": "Loop-based",
    "SOGI-FLL": "Loop-based",
    "Type-3 SOGI-PLL": "Loop-based",
    "EKF": "Model-based",
    "RA-EKF": "Model-based",
    "UKF": "Model-based",
    "IPDFT": "Window-based",
    "TFT": "Window-based",
    "Prony": "Window-based",
    "ESPRIT": "Window-based",
    "TKEO": "Adaptive",
    "RLS": "Adaptive",
    "Koopman (RK-DPMU)": "Data-driven",
}

_FAMILY_PALETTE = {
    "Model-based": "#1565C0",
    "Loop-based": "#2E7D32",
    "Window-based": "#E65100",
    "Adaptive": "#6A1B9A",
    "Data-driven": "#B71C1C",
}

TRACKING_ESTIMATORS = ["EKF", "SOGI-PLL", "IPDFT", "PLL", "RLS", "Koopman (RK-DPMU)"]

COLORS_6 = {
    "EKF": "#1565C0",
    "SOGI-PLL": "#2E7D32",
    "IPDFT": "#E65100",
    "PLL": "#8D6E63",
    "RLS": "#6A1B9A",
    "Koopman (RK-DPMU)": "#B71C1C",
}


def _candidate_roots() -> list[Path]:
    roots = [Path(DATA_RESULTS_DIR)]
    if LEGACY_OUTPUT_DIR not in roots:
        roots.append(LEGACY_OUTPUT_DIR)
    return roots


def _resolve_trace_path(scenario: str, estimator: str) -> Path:
    for root in _candidate_roots():
        trace_path = root / scenario / estimator / f"{scenario}__{estimator}_signals.csv"
        if trace_path.exists():
            return trace_path
    searched = ", ".join(str(root) for root in _candidate_roots())
    raise FileNotFoundError(
        f"[!] Missing trace for scenario={scenario}, estimator={estimator}. Searched roots: {searched}"
    )


def load_estimator_trace(scenario: str, estimator: str):
    trace_path = _resolve_trace_path(scenario, estimator)
    df = pd.read_csv(trace_path)
    df0 = df[df["run_idx"] == 0].copy()
    if df0.empty:
        raise ValueError(f"[!] No run_idx=0 in {trace_path.name}")
    return df0["t_s"].values, df0["f_hat_hz"].values


def orchestrate_dashboard2(
    df_global: pd.DataFrame,
    df_raw: pd.DataFrame | None = None,
) -> None:
    print("\n[MEGA2] Orchestrating Reordered 4x2 Dashboard...")

    avg_rmse = df_global.groupby("estimator")["m1_rmse_hz_mean"].mean()
    avg_cpu = df_global.groupby("estimator")["m13_cpu_time_us_mean"].mean()
    avg_trip = df_global.groupby("estimator")["m5_trip_risk_s_mean"].mean()
    avg_mae = df_global.groupby("estimator")["m2_mae_hz_mean"].mean()
    common_estimators = avg_rmse.index.intersection(avg_cpu.index)

    data_bundle = {
        "df_global": df_global,
        "df_raw": df_raw,
        "avg_rmse": avg_rmse,
        "avg_mae": avg_mae,
        "avg_cpu": avg_cpu,
        "avg_trip": avg_trip,
        "common_estimators": common_estimators,
        "_ESTIMATOR_FAMILIES": _ESTIMATOR_FAMILIES,
        "_FAMILY_PALETTE": _FAMILY_PALETTE,
        "BASE_RESULTS_DIR": DATA_RESULTS_DIR,
        "TRACKING_ESTIMATORS": TRACKING_ESTIMATORS,
        "COLORS_6": COLORS_6,
        "load_estimator_trace": load_estimator_trace,
    }

    with plt.rc_context(_IEEE_RC):
        fig = plt.figure(figsize=(_IEEE_FULL_W * 2.10, 9.8 * 1.2705 * 1.364), constrained_layout=True)
        fig.set_constrained_layout_pads(w_pad=0.4 / 72, h_pad=1.0 / 72)

        gs = gridspec.GridSpec(4, 2, figure=fig, wspace=0.024, hspace=0.040)
        ax_d = fig.add_subplot(gs[0, 0])
        ax_e = fig.add_subplot(gs[0, 1])
        ax_g = fig.add_subplot(gs[1, 0])
        ax_h = fig.add_subplot(gs[1, 1])
        ax_c = fig.add_subplot(gs[2, 0])
        ax_f = fig.add_subplot(gs[2, 1])
        ax_a = fig.add_subplot(gs[3, 0])
        ax_b = fig.add_subplot(gs[3, 1])

        md2_subplot_00(ax_d, data_bundle)
        md2_subplot_01(ax_e, data_bundle)
        md2_subplot_10(ax_g, data_bundle)
        md2_subplot_11(ax_h, data_bundle)
        md2_subplot_20(ax_c, data_bundle)
        md2_subplot_21(ax_f, data_bundle)
        md2_subplot_30(ax_a, data_bundle)
        md2_subplot_31(ax_b, data_bundle)

        for ax in (ax_e, ax_h, ax_f, ax_b):
            ax.set_ylabel("")

        ax_d.set_ylabel("Frequency [Hz]", fontsize=7, labelpad=2)
        ax_g.set_ylabel("Frequency [Hz]", fontsize=7, labelpad=2)

        out_png = BASE_RESULTS_DIR / f"{FIGURE2_BASENAME}.png"
        out_pdf = BASE_RESULTS_DIR / f"{FIGURE2_BASENAME}.pdf"
        out_svg = BASE_RESULTS_DIR / f"{FIGURE2_BASENAME}.svg"
        out_eps = BASE_RESULTS_DIR / f"{FIGURE2_BASENAME}.eps"
        legacy = BASE_RESULTS_DIR / "megadashboard2_reordered.png"
        fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.005)
        fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.005)
        fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.005)
        fig.savefig(out_eps, bbox_inches="tight", pad_inches=0.005)
        fig.savefig(legacy, dpi=300, bbox_inches="tight", pad_inches=0.005)
        plt.close(fig)

    print(f"[OK] Dashboard saved: {out_png}")


def generate_benchmark_figures(base_results_dir: Path | None = None) -> None:
    global BASE_RESULTS_DIR, DATA_RESULTS_DIR
    if base_results_dir is not None:
        BASE_RESULTS_DIR = Path(base_results_dir)
    BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_RESULTS_DIR = BASE_RESULTS_DIR

    plot_megadashboard1(BASE_RESULTS_DIR, _IEEE_RC, _IEEE_FULL_W, 11.0)

    report_path = DATA_RESULTS_DIR / "global_metrics_report.csv"
    if not report_path.exists() and (LEGACY_OUTPUT_DIR / "global_metrics_report.csv").exists():
        DATA_RESULTS_DIR = LEGACY_OUTPUT_DIR
        report_path = DATA_RESULTS_DIR / "global_metrics_report.csv"
        print(f"[WARN] Canonical benchmark summary missing. Reading dashboard data from legacy outputs: {DATA_RESULTS_DIR}")
    if not report_path.exists():
        raise FileNotFoundError(f"[!] Missing global_metrics_report.csv under: {DATA_RESULTS_DIR}")
    df_global = pd.read_csv(report_path)

    df_raw = None
    report_json = DATA_RESULTS_DIR / JSON_REPORT_NAME
    if report_json.exists():
        with open(report_json, encoding="utf-8") as f:
            full = json.load(f)
        rr = full.get("raw_run_records", [])
        if rr:
            df_raw = pd.DataFrame(rr)
            if "scenario" not in df_raw.columns and "scenario_name" in df_raw.columns:
                df_raw = df_raw.rename(columns={"scenario_name": "scenario"})
            print(f"[OK] Loaded {len(df_raw)} raw MC records for boxplot.")
    else:
        print("[WARN] benchmark_full_report.json not found - boxplot shows placeholder.")

    orchestrate_dashboard2(df_global, df_raw)


if __name__ == "__main__":
    generate_benchmark_figures()
