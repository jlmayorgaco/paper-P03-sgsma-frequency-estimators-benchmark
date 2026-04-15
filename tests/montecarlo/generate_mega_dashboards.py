"""
Standalone Dashboard Generator for IBR Frequency Estimator Benchmark.
Reads pre-aggregated CSV artifacts and Phase 1 scenario traces to generate
publication-ready Mega Dashboards 1 and 2 without re-simulating.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

# ── Paths & Configuration ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]  # Adjust if placing in a different directory
BASE_RESULTS_DIR = ROOT / "tests" / "montecarlo" / "outputs"

_IEEE_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.4,
    "axes.linewidth": 0.6,
    "lines.linewidth": 0.8,
    "patch.linewidth": 0.5,
    "mathtext.fontset": "stix",
}

_IEEE_FULL_W = 7.16    # two-column text width, inches
_IEEE_PAGE_H = 11.0    # letter-size page height, inches

_ESTIMATOR_FAMILIES = {
    "ZCD": "Loop-based", "PLL": "Loop-based", "SOGI-PLL": "Loop-based",
    "SOGI-FLL": "Loop-based", "Type-3 SOGI-PLL": "Loop-based",
    "EKF": "Model-based", "RA-EKF": "Model-based", "UKF": "Model-based",
    "IPDFT": "Window-based", "TFT": "Window-based", "Prony": "Window-based",
    "ESPRIT": "Window-based", "TKEO": "Adaptive", "RLS": "Adaptive",
    "Koopman": "Data-driven",
}

_FAMILY_PALETTE = {
    "Model-based": "#1565C0", "Loop-based": "#2E7D32",
    "Window-based": "#E65100", "Adaptive": "#6A1B9A", "Data-driven": "#B71C1C",
}

# ═══════════════════════════════════════════════════════════════════════════════
# Mega Dashboard 1 — Scenario Signal Overview (Reads Phase 1 CSVs)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_megadashboard1() -> None:
    print("\n[MEGA1] Building scenario signal overview from CSVs ...")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import pandas as pd
    import numpy as np

    # Set symmetric zoom window half-width (e.g., +/- 0.125s around the event) for general plots
    dw = 0.125 

    # Complete explicit control over title, xlim, and ylim for every plot
    GRID_SPECS = [
        # Row 0: Mag Step. Event at 0.5s. 
        dict(
            col0=dict(folder="IEEE_Mag_Step", title="(0,0) Mag Step (+10%) Voltage", col="v_pu", color="darkblue", ylabel="V [pu]", xlim=(0.5 - dw, 0.5 + dw), ylim=(-1.35, 1.35), event_t=0.5),
            col1=dict(folder="IEEE_Mag_Step", title="(0,1) Mag Step Frequency", col="f_true_hz", color="darkred", ylabel="f [Hz]", event_t=0.5, xlim=(0.0, 1.5), ylim=(59.95, 60.05))
        ),
        # Row 1: Ramp. Event at 0.3s. 
        dict(
            col0=dict(folder="IEEE_Freq_Ramp", title="(1,0) Ramp (+5Hz/s) Voltage", col="v_pu", color="darkblue", ylabel="V [pu]", xlim=(0.10, 0.50), ylim=(-1.25, 1.25), event_t=0.3),
            col1=dict(folder="IEEE_Freq_Ramp", title="(1,1) Ramp Frequency", col="f_true_hz", color="darkred", ylabel="f [Hz]", event_t=0.3, xlim=(0.0, 1.5), ylim=(59.8, 62.0))
        ),
        # Row 2: Modulation. Continuous.
        dict(
            col0=dict(folder="IEEE_Modulation", title="(2,0) Modulation (2Hz) Voltage", col="v_pu", color="darkblue", ylabel="V [pu]", xlim=(0.5 - dw, 0.5 + dw), ylim=(-1.5, 1.5)),
            col1=dict(folder="IEEE_Modulation", title="(2,1) Modulation Frequency", col="f_true_hz", color="darkred", ylabel="f [Hz]", xlim=(0.0, 1.5), ylim=(59.00, 61.00)) 
        ),
        # Row 3: Islanding Jump. Event at 0.5s. 
        dict(
            col0=dict(folder="IEEE_Phase_Jump_60", title="(3,0) Islanding Jump Voltage", col="v_pu", color="darkblue", ylabel="V [pu]", xlim=(0.5 - dw, 0.5 + dw), ylim=(-1.25, 1.25), event_t=0.5),
            col1=dict(folder="IEEE_Phase_Jump_60", title="(3,1) Islanding Jump Frequency", col="f_true_hz", color="darkred", ylabel="f [Hz]", event_t=0.5, xlim=(0.0, 1.5), ylim=(59.5, 60.5))
        ),
        # Row 4: Multi-Event. Events at 0.5s and 1.0s. 
        # ALIGNED: col1 xlim set to (0.0, 1.5) to align 0.5s perfectly with all plots above.
        dict(
            col0=dict(folder="IBR_Multi_Event", title="(4,0) Multi-Evt Voltage (First Jump)", col="v_pu", color="indigo", ylabel="V [pu]", xlim=(0.5 - dw, 0.5 + dw), ylim=(-1.25, 1.25), event_t=[0.5, 1.0]),
            col1=dict(folder="IBR_Multi_Event", title="(4,1) Multi-Evt Frequency Profile", col="f_true_hz", color="darkred", ylabel="f [Hz]", event_t=[0.5, 1.0], xlim=(0.0, 2), ylim=(56.0, 62.0))
        ),
    ]

    nrows = len(GRID_SPECS)
    
    # 100% IEEE double width size, and the height 50% of the page height
    fig_w = _IEEE_FULL_W 
    fig_h = _IEEE_PAGE_H * 0.50

    with plt.rc_context(_IEEE_RC):
        fig = plt.figure(figsize=(fig_w, fig_h))
        # Increased hspace heavily to accommodate the X-labels on every single row
        gs = gridspec.GridSpec(nrows, 2, figure=fig, hspace=0.85, wspace=0.20,
                               left=0.08, right=0.98, top=0.90, bottom=0.08)

        for ridx, row_spec in enumerate(GRID_SPECS):
            for cidx, col_key in enumerate(["col0", "col1"]):
                spec = row_spec[col_key]
                ax = fig.add_subplot(gs[ridx, cidx])
                
                csv_path = BASE_RESULTS_DIR / spec["folder"] / f"{spec['folder']}_scenario.csv"
                if not csv_path.exists():
                    t = np.linspace(0, 1.5, 1000)
                    y = np.sin(2 * np.pi * 60 * t) if spec["col"] == "v_pu" else np.full_like(t, 60.0)
                else:
                    df_scen = pd.read_csv(csv_path)
                    t = df_scen["t_s"].values
                    y = df_scen[spec["col"]].values

                # Base Plot
                ax.plot(t, y, lw=0.75, color=spec["color"], rasterized=True)
                ax.set_ylabel(spec["ylabel"], labelpad=2, fontsize=6.5)
                ax.set_title(spec["title"], fontsize=7.5, fontweight="bold", pad=3)

                # Explicit Axis Limits
                if "xlim" in spec:
                    ax.set_xlim(spec["xlim"][0], spec["xlim"][1])
                else:
                    ax.set_xlim(float(t[0]), float(t[-1]))
                    
                if "ylim" in spec:
                    ax.set_ylim(spec["ylim"][0], spec["ylim"][1])

                # --- Horizontal tracking lines for Mag Step Peaks ---
                if spec["folder"] == "IEEE_Mag_Step" and spec["col"] == "v_pu":
                    mask_pre = (t >= (0.5 - dw)) & (t <= 0.5)
                    mask_post = (t > 0.5) & (t <= (0.5 + dw))
                    
                    if mask_pre.any() and mask_post.any():
                        max_pre = np.max(y[mask_pre])
                        max_post = np.max(y[mask_post])
                        
                        # Draw horizontal lines at the peak values
                        ax.axhline(max_pre, color="#333", linestyle="--", linewidth=0.5, alpha=0.8)
                        ax.axhline(max_post, color="#333", linestyle="--", linewidth=0.5, alpha=0.8)

                # Vertical Dashed Line for Transition Events
                if "event_t" in spec:
                    events = spec["event_t"] if isinstance(spec["event_t"], list) else [spec["event_t"]]
                    for ev in events:
                        ax.axvline(ev, color="black", linestyle="--", linewidth=1.0, alpha=0.6)

                # Grid & Formatting
                ax.grid(True, alpha=0.3, lw=0.4)
                ax.tick_params(axis='both', which='major', labelsize=6)
                
                # Enable Time Axis Label Formatting for ALL plots
                ax.set_xlabel("Time [s]", labelpad=2, fontsize=7)
                ax.tick_params(labelbottom=True)

        # Header Title
        fig.text(0.5, 0.96, "IBR Frequency Estimator Benchmark — Detailed Scenario Overview", 
                 ha="center", va="top", fontsize=8.5, fontweight="bold", color="#111")

        out = BASE_RESULTS_DIR / "megadashboard1.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        
    print(f"    megadashboard1 saved → {out.relative_to(ROOT) if hasattr(out, 'relative_to') else out}")





# ═══════════════════════════════════════════════════════════════════════════════
# Mega Dashboard 2 — Estimator Performance Summary (Reads global_metrics_report)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_megadashboard2(df_global: pd.DataFrame) -> None:
    print("\n[MEGA2] Building estimator performance mega-dashboard from global metrics ...")

    def safe_pivot(col: str) -> "pd.DataFrame | None":
        if col not in df_global.columns: return None
        piv = df_global.pivot_table(index="estimator", columns="scenario", values=col, aggfunc="mean")
        return piv if not piv.empty else None

    def fam_clr(est: str) -> str:
        return _FAMILY_PALETTE.get(_ESTIMATOR_FAMILIES.get(est, "Adaptive"), "#757575")

    def sc_short(name: str) -> str:
        return (name
                .replace("IEEE_Single_SinWave", "SinWave")
                .replace("IEEE_Mag_Step", "MagStep")
                .replace("IEEE_Freq_Ramp", "FreqRamp")
                .replace("IEEE_Freq_Step", "FreqStep")
                .replace("IEEE_Modulation_AM", "Modul_AM")
                .replace("IEEE_Modulation_FM", "Modul_FM")
                .replace("IEEE_Modulation", "Modul.")
                .replace("IEEE_OOB_Interference", "OOB")
                .replace("IEEE_Phase_Jump_20", "PJ20°")
                .replace("IEEE_Phase_Jump_60", "PJ60°")
                .replace("NERC_Phase_Jump_60", "NERC\nPJ60°")
                .replace("IBR_Multi_Event", "IBR\nMulti")
                .replace("IBR_Power_Imbalance_Ringdown", "IBR\nRing.")
                .replace("IBR_Harmonics_Small", "Harm\nSm")
                .replace("IBR_Harmonics_Medium", "Harm\nMed")
                .replace("IBR_Harmonics_Large", "Harm\nLg"))

    def avg_col(col: str) -> "pd.Series":
        if col not in df_global.columns: return pd.Series(dtype=float)
        return df_global.groupby("estimator")[col].mean()

    piv_rmse  = safe_pivot("m1_rmse_hz_mean")
    piv_trip  = safe_pivot("m5_trip_risk_s_mean")
    piv_peak  = safe_pivot("m3_max_peak_hz_mean")

    avg_rmse   = avg_col("m1_rmse_hz_mean").sort_values()
    avg_cpu    = avg_col("m13_cpu_time_us_mean")
    avg_settle = avg_col("m8_settling_time_s_mean").sort_values()

    def _draw_heatmap(ax, piv, cmap, label, annotate=True):
        if piv is None or piv.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", fontsize=7, color="#888")
            return
        data = piv.values.astype(float)
        finite = data[np.isfinite(data) & (data > 0)]
        if len(finite) == 0: return
        vmin, vmax = max(finite.min() / 2.0, 1e-5), min(finite.max() * 2.0, 50.0)
        clipped = np.clip(data, vmin, vmax)
        im = ax.imshow(clipped, aspect="auto", cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
        xl = [sc_short(c) for c in piv.columns]
        ax.set_xticks(range(len(xl)))
        ax.set_xticklabels(xl, rotation=38, ha="right", fontsize=4.8)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index, fontsize=5.0)
        cb = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.88, aspect=22)
        cb.set_label(label, fontsize=5.5)
        cb.ax.tick_params(labelsize=4.5)
        if annotate:
            thresh = np.nanpercentile(finite, 72)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    val = data[i, j]
                    if np.isfinite(val):
                        tc = "w" if val >= thresh else "k"
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=3.8, color=tc, fontweight="bold")

    fig_h = _IEEE_PAGE_H * 0.70

    with plt.rc_context(_IEEE_RC):
        fig = plt.figure(figsize=(_IEEE_FULL_W, fig_h))
        gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.78, wspace=0.24, left=0.08, right=0.98, top=0.930, bottom=0.055)

        # (0,0) RMSE heatmap
        ax00 = fig.add_subplot(gs[0, 0])
        _draw_heatmap(ax00, piv_rmse, "RdYlGn_r", "RMSE [Hz]")
        ax00.set_title("RMSE [Hz] — Scenario × Estimator", fontsize=7, fontweight="bold", pad=3)

        # (0,1) Trip-risk heatmap
        ax01 = fig.add_subplot(gs[0, 1])
        _draw_heatmap(ax01, piv_trip, "YlOrRd", r"$T_\mathrm{trip}$ [s]")
        ax01.set_title(r"Trip Risk $T_\mathrm{trip}$ [s]", fontsize=7, fontweight="bold", pad=3)

        # (1,0) CPU vs RMSE scatter
        ax10 = fig.add_subplot(gs[1, 0])
        if not avg_rmse.empty and not avg_cpu.empty:
            common = avg_rmse.index.intersection(avg_cpu.index)
            seen_fam = set()
            for est in common:
                fam = _ESTIMATOR_FAMILIES.get(est, "Adaptive")
                clr, lbl = _FAMILY_PALETTE.get(fam, "#757575"), fam if fam not in seen_fam else "_"
                seen_fam.add(fam)
                ax10.scatter(avg_cpu[est], avg_rmse[est], s=22, color=clr, edgecolors="white", lw=0.4, zorder=4, label=lbl)
                ax10.annotate(est, (avg_cpu[est], avg_rmse[est]), xytext=(3, 2), textcoords="offset points", fontsize=4.0, color=clr)
            ax10.axhline(0.1, color="#B71C1C", ls="--", lw=0.7, alpha=0.75, label="IEC 100 mHz")
            ax10.set_xscale("log"); ax10.set_yscale("log")
            ax10.set_xlabel(r"Avg CPU [$\mu$s / sample]", labelpad=1)
            ax10.set_ylabel("Avg RMSE [Hz]", labelpad=1)
            ax10.legend(loc="upper left", fontsize=5.5, handlelength=0.9, borderpad=0.3)
        ax10.set_title("Accuracy–Latency Trade-off (mean across all scenarios)", fontsize=7, fontweight="bold", pad=3)

        # (1,1) Max-peak heatmap
        ax11 = fig.add_subplot(gs[1, 1])
        _draw_heatmap(ax11, piv_peak, "RdPu", "Max Peak [Hz]", annotate=False)
        ax11.set_title("Max Peak Error [Hz]", fontsize=7, fontweight="bold", pad=3)

        # (2,0) RMSE grouped bar
        ax20 = fig.add_subplot(gs[2, 0])
        rmse_col = "m1_rmse_hz_mean"
        if rmse_col in df_global.columns:
            sc_list = sorted(df_global["scenario"].unique())
            est_list = df_global.groupby("estimator")[rmse_col].mean().sort_values().index.tolist()
            x20 = np.arange(len(sc_list))
            w = max(0.05, 0.72 / max(len(est_list), 1))
            cmap20 = plt.get_cmap("tab20", len(est_list))
            for i, est in enumerate(est_list):
                vals = [float(df_global[(df_global["estimator"]==est) & (df_global["scenario"]==sc)][rmse_col].mean()) if len(df_global[(df_global["estimator"]==est) & (df_global["scenario"]==sc)]) else np.nan for sc in sc_list]
                ax20.bar(x20 + (i - len(est_list) / 2.0 + 0.5) * w, vals, width=w * 0.88, color=cmap20(i), label=est, alpha=0.87, edgecolor="none")
            ax20.set_xticks(x20)
            ax20.set_xticklabels([sc_short(s) for s in sc_list], rotation=30, ha="right", fontsize=4.8)
            ax20.set_yscale("log")
            ax20.set_ylabel("RMSE [Hz]", labelpad=1)
            ax20.legend(loc="upper right", fontsize=3.8, ncol=2, handlelength=0.6, borderpad=0.2, labelspacing=0.12)
        ax20.set_title("RMSE by Estimator per Scenario", fontsize=7, fontweight="bold", pad=3)

        # (2,1) Settling time horizontal bar
        ax21 = fig.add_subplot(gs[2, 1])
        if not avg_settle.empty:
            ax21.barh(range(len(avg_settle)), avg_settle.values, color=[fam_clr(e) for e in avg_settle.index], edgecolor="none", height=0.65)
            ax21.set_yticks(range(len(avg_settle)))
            ax21.set_yticklabels(avg_settle.index, fontsize=5.2)
            ax21.set_xlabel("Avg Settling Time [s]", labelpad=1)
            ax21.invert_yaxis()
        for fam, clr in _FAMILY_PALETTE.items():
            ax21.barh(0, 0, color=clr, label=fam, height=0)
        ax21.legend(loc="lower right", fontsize=5, handlelength=0.7, borderpad=0.3)
        ax21.set_title("Average Settling Time by Estimator", fontsize=7, fontweight="bold", pad=3)

        # (3,0) Global RMSE ranking
        ax30 = fig.add_subplot(gs[3, 0])
        if not avg_rmse.empty:
            ax30.barh(range(len(avg_rmse)), avg_rmse.values, color=[fam_clr(e) for e in avg_rmse.index], edgecolor="none", height=0.65)
            ax30.set_yticks(range(len(avg_rmse)))
            ax30.set_yticklabels(avg_rmse.index, fontsize=5.2)
            ax30.set_xscale("log")
            ax30.set_xlabel("Mean RMSE [Hz] — log scale", labelpad=1)
            ax30.axvline(0.1, color="#B71C1C", ls="--", lw=0.7, alpha=0.8)
            ax30.invert_yaxis()
            for i, (est, val) in enumerate(avg_rmse.items()):
                if np.isfinite(val): ax30.text(val * 1.07, i, f"{val:.3f}", va="center", fontsize=4.0, color="#333")
        for fam, clr in _FAMILY_PALETTE.items():
            ax30.barh(0, 0, color=clr, label=fam, height=0)
        ax30.legend(loc="lower right", fontsize=5, handlelength=0.7, borderpad=0.3)
        ax30.set_title("Estimator Ranking — Mean RMSE (all scenarios)", fontsize=7, fontweight="bold", pad=3)

        # (3,1) Composite score
        ax31 = fig.add_subplot(gs[3, 1])
        score_cols = [c for c in ("m1_rmse_hz_mean", "m3_max_peak_hz_mean", "m5_trip_risk_s_mean", "m13_cpu_time_us_mean") if c in df_global.columns]
        if score_cols:
            agg = df_global.groupby("estimator")[score_cols].mean()
            normed = pd.DataFrame(index=agg.index)
            for c in score_cols:
                mn, mx = agg[c].min(), agg[c].max()
                normed[c] = (agg[c] - mn) / (mx - mn + 1e-12)
            score = normed.mean(axis=1).sort_values()
            ax31.barh(range(len(score)), score.values, color=[fam_clr(e) for e in score.index], edgecolor="none", height=0.65)
            ax31.set_yticks(range(len(score)))
            ax31.set_yticklabels(score.index, fontsize=5.2)
            ax31.set_xlabel("Composite Score (lower = better)", labelpad=1)
            ax31.invert_yaxis()
            for i, val in enumerate(score.values): ax31.text(val + 0.005, i, f"#{i+1}", va="center", fontsize=4.0, color="#333")
        for fam, clr in _FAMILY_PALETTE.items():
            ax31.barh(0, 0, color=clr, label=fam, height=0)
        ax31.legend(loc="lower right", fontsize=5, handlelength=0.7, borderpad=0.3)
        ax31.set_title("Composite Performance Score", fontsize=7, fontweight="bold", pad=3)

        fig.text(0.5, 0.962, "IBR Frequency Estimator Benchmark — Performance Analysis", ha="center", va="top", fontsize=8, fontweight="bold")
        fig.text(0.5, 0.946, r"$N_\mathrm{MC}=30$ runs per estimator/scenario  |  $f_s=10\,\mathrm{kHz}$  |  dual-rate physics simulation (1 MHz $\rightarrow$ 10 kHz)", ha="center", va="top", fontsize=5.5, color="#555")

        out = BASE_RESULTS_DIR / "megadashboard2.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
    print(f"    megadashboard2 saved → {out}")

if __name__ == "__main__":
    report_path = BASE_RESULTS_DIR / "global_metrics_report.csv"
    
    # 1. Generate Mega Dashboard 1 from Scenario CSV traces
    plot_megadashboard1()

    # 2. Generate Mega Dashboard 2 from aggregate metrics CSV
    if report_path.exists():
        df_global = pd.read_csv(report_path)
        plot_megadashboard2(df_global)
    else:
        print(f"[!] {report_path.name} not found. Ensure Phase 2 completed successfully.")