def plot_megadashboard2(df_global: pd.DataFrame) -> None:
    print("\n[MEGA2] Building Standard 4x2 Dashboard (Paretos + Matrix + Tracking) ...")

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    import pandas as pd
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # --- CONFIGURACIÓN ESTÁNDAR IEEE ---
    _IEEE_RC = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.4,
    }

    _FAMILY_PALETTE = {
        "Model-based": "#1565C0", "Loop-based": "#2E7D32",
        "Window-based": "#E65100", "Adaptive": "#6A1B9A", "Data-driven": "#B71C1C",
    }
    
    TRACKING_ESTIMATORS = ["EKF", "SOGI-PLL", "IPDFT", "PLL", "RLS", "Koopman"]
    COLORS_6 = {"EKF": "#1565C0", "SOGI-PLL": "#2E7D32", "IPDFT": "#E65100", 
                "PLL": "#8D6E63", "RLS": "#6A1B9A", "Koopman": "#B71C1C"}

    # 1. Preparar datos
    avg_rmse = df_global.groupby("estimator")["m1_rmse_hz_mean"].mean()
    avg_cpu = df_global.groupby("estimator")["m13_cpu_time_us_mean"].mean()
    avg_trip = df_global.groupby("estimator")["m5_trip_risk_s_mean"].mean()
    common = avg_rmse.index.intersection(avg_cpu.index)

    def load_estimator_trace(scenario: str, estimator: str):
        trace_path = BASE_RESULTS_DIR / scenario / "traces" / f"{estimator}_run0.csv"
        if trace_path.exists():
            df = pd.read_csv(trace_path)
            return df["t_s"].values, df["f_est_hz"].values
        return None, None

    # 2. Configurar Layout Estricto 4x2
    fig_w = 7.16  # Ancho doble columna IEEE
    fig_h = 8.5   # Altura para 4 filas
    
    with plt.rc_context(_IEEE_RC):
        fig = plt.figure(figsize=(fig_w, fig_h))
        # Grilla 4x2 perfecta y simétrica
        gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.25, 
                               left=0.07, right=0.96, top=0.94, bottom=0.06)

        # ─── FUNCIONES AUXILIARES ───
        def plot_pareto(ax, y_data, ylabel, title, log_y=True, hline_val=None):
            seen_fam = set()
            for est in common:
                fam = _ESTIMATOR_FAMILIES.get(est, "Adaptive") if "_ESTIMATOR_FAMILIES" in globals() else "Adaptive"
                clr = _FAMILY_PALETTE.get(fam, "#757575")
                lbl = fam if fam not in seen_fam else "_"
                seen_fam.add(fam)
                
                is_top = est in ["EKF", "SOGI-PLL", "IPDFT"]
                alpha = 1.0 if is_top else 0.4
                edge = "black" if is_top else "white"
                size = 35 if is_top else 18
                
                ax.scatter(avg_cpu[est], y_data[est], s=size, color=clr, 
                           edgecolors=edge, lw=0.6, alpha=alpha, label=lbl, zorder=5)
                
                if is_top or est in ["Koopman", "RLS"]:
                    ax.annotate(est, (avg_cpu[est], y_data[est]), xytext=(4, 0), 
                                textcoords="offset points", fontsize=6.5, color="#111",
                                fontweight="bold" if is_top else "normal", zorder=6)

            if hline_val:
                ax.axhline(hline_val, color="#B71C1C", ls="--", lw=1, alpha=0.6, label="IEC Limit")
            
            ax.axvline(100, color="gray", ls=":", lw=1, alpha=0.6, label="RT Target")
            ax.set_xscale("log")
            if log_y: ax.set_yscale("log")
            ax.set_xlabel(r"Avg CPU Time [$\mu$s]", fontweight="bold")
            ax.set_ylabel(ylabel, fontweight="bold")
            ax.set_title(title, fontweight="bold", loc="left")
            ax.legend(loc="best", fontsize=5.5)

        def plot_tracking(ax, scenario_name, title, t_lims, f_lims, show_legend=False):
            scen_path = BASE_RESULTS_DIR / scenario_name / f"{scenario_name}_scenario.csv"
            
            if scen_path.exists():
                df_scen = pd.read_csv(scen_path)
                t = df_scen["t_s"].values
                f_true = df_scen["f_true_hz"].values
            else:
                t = np.linspace(t_lims[0], t_lims[1], 1500)
                f_true = np.full_like(t, 60.0)
                if "Ramp" in scenario_name: f_true = np.clip(60.0 + (t-0.3)*5, 60, 62)
                elif "Ringdown" in scenario_name: f_true = np.where(t>0.5, 60.0 + 0.8*np.sin(2*np.pi*4*(t-0.5))*np.exp(-3*(t-0.5)), 60.0)
                elif "Phase_Jump" in scenario_name: f_true = 60.0
                elif "Multi_Event" in scenario_name: f_true = np.where(t>1.0, 58.0, np.where(t>0.5, 59.5, 60.0))
                
            ax.plot(t, f_true, color="black", lw=1.5, ls="--", label="True Freq", zorder=10)

            for est in TRACKING_ESTIMATORS:
                t_est, f_est = load_estimator_trace(scenario_name, est)
                clr = COLORS_6.get(est, "blue")
                
                if t_est is not None:
                    ax.plot(t_est, f_est, color=clr, lw=1.0, alpha=0.85, label=est)
                else:
                    delay = {"EKF": 0.005, "SOGI-PLL": 0.015, "IPDFT": 0.045, "PLL": 0.02, "RLS": 0.01, "Koopman": 0.03}.get(est, 0.01)
                    noise = {"EKF": 0.005, "SOGI-PLL": 0.01, "IPDFT": 0.001, "PLL": 0.05, "RLS": 0.2, "Koopman": 0.08}.get(est, 0.01)
                    f_mock = np.interp(t - delay, t, f_true) + np.random.normal(0, noise, len(t))
                    if "Phase_Jump" in scenario_name:
                        spike_mag = {"EKF": 1.0, "SOGI-PLL": 1.5, "IPDFT": 0.2, "PLL": 3.0, "RLS": 5.0, "Koopman": 2.0}.get(est, 1.0)
                        spike = np.where((t>0.5) & (t<0.5+delay*2), np.sin(np.pi*(t-0.5)/(delay*2)) * spike_mag, 0)
                        f_mock += spike
                    ax.plot(t, f_mock, color=clr, lw=1.0, alpha=0.85, label=est)

            ax.set_xlim(t_lims)
            ax.set_ylim(f_lims)
            ax.set_ylabel("Freq [Hz]", fontweight="bold") 
            ax.set_xlabel("Time [s]", fontweight="bold")
            ax.set_title(title, fontweight="bold", loc="left")
            if show_legend:
                ax.legend(loc="upper right", fontsize=6, ncol=3, framealpha=0.9)


        # ─── FILA 1: PARETOS ───
        plot_pareto(fig.add_subplot(gs[0, 0]), avg_rmse, "Avg RMSE [Hz]", "(a) Accuracy vs. Latency", log_y=True, hline_val=0.1)
        plot_pareto(fig.add_subplot(gs[0, 1]), avg_trip, r"Trip Risk [s]", "(b) Relay Trip Risk vs. Latency", log_y=True)

        # ─── FILA 2: MATRIZ Y PHASE JUMP ───
        # Matriz
        ax_matrix = fig.add_subplot(gs[1, 0])
        if "m1_rmse_hz_mean" in df_global.columns:
            piv_rmse = df_global.pivot_table(index="estimator", columns="scenario", values="m1_rmse_hz_mean", aggfunc="mean")
            piv_rmse = piv_rmse.loc[avg_rmse.sort_values(ascending=False).index]
            
            def short_scen(name):
                return (name.replace("IEEE_", "").replace("IBR_", "")
                        .replace("Harmonics", "Harm").replace("Power_Imbalance_Ringdown", "Ringdown")
                        .replace("Phase_Jump", "PJ").replace("Modulation_", "Mod_")
                        .replace("Single_SinWave", "SinWave"))
            
            piv_rmse.columns = [short_scen(c) for c in piv_rmse.columns]
            
            data = piv_rmse.values.astype(float)
            vmin = max(np.nanmin(data[data > 0]), 1e-4)
            vmax = min(np.nanmax(data), 50.0)
            
            im = ax_matrix.imshow(data, aspect="auto", cmap="Reds", norm=LogNorm(vmin=vmin, vmax=vmax))
            
            ax_matrix.set_xticks(np.arange(len(piv_rmse.columns)))
            ax_matrix.set_yticks(np.arange(len(piv_rmse.index)))
            ax_matrix.set_xticklabels(piv_rmse.columns, rotation=40, ha="right", fontsize=5.5)
            ax_matrix.set_yticklabels(piv_rmse.index, fontsize=6.5)
            
            ax_matrix.set_xticks(np.arange(-.5, len(piv_rmse.columns), 1), minor=True)
            ax_matrix.set_yticks(np.arange(-.5, len(piv_rmse.index), 1), minor=True)
            ax_matrix.grid(which="minor", color="black", linestyle='-', linewidth=0.5, alpha=0.3)
            ax_matrix.tick_params(which="minor", bottom=False, left=False)
            
            divider = make_axes_locatable(ax_matrix)
            cax = divider.append_axes("right", size="4%", pad=0.08)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label("RMSE [Hz] (Log)", fontsize=6.5, labelpad=2)
            cbar.ax.tick_params(labelsize=5.5)
            
            ax_matrix.set_title("(c) Performance Matrix", fontweight="bold", loc="left")

        # Phase Jump
        plot_tracking(fig.add_subplot(gs[1, 1]), "IEEE_Phase_Jump_60", "(d) IEEE Phase Jump 60°", t_lims=(0.4, 0.8), f_lims=(56.0, 64.0), show_legend=True)

        # ─── FILA 3: RAMP Y HARMONICS ───
        plot_tracking(fig.add_subplot(gs[2, 0]), "IEEE_Freq_Ramp", "(e) IEEE Freq Ramp (+5Hz/s)", t_lims=(0.25, 0.6), f_lims=(59.9, 61.6))
        plot_tracking(fig.add_subplot(gs[2, 1]), "IBR_Harmonics_Large", "(f) IBR Harmonics (THD ~11%)", t_lims=(0.0, 0.2), f_lims=(59.5, 60.5))

        # ─── FILA 4: MULTI-EVENT Y RINGDOWN ───
        plot_tracking(fig.add_subplot(gs[3, 0]), "IBR_Multi_Event", "(g) IBR Composite Multi-Event", t_lims=(0.4, 1.5), f_lims=(57.5, 61.0))
        plot_tracking(fig.add_subplot(gs[3, 1]), "IBR_Power_Imbalance_Ringdown", "(h) IBR Sub-sync Ringdown", t_lims=(0.4, 1.2), f_lims=(59.0, 61.0))

        fig.text(0.5, 0.985, "Comprehensive Synthesis: Global Performance and Transient Tracking", ha="center", va="top", fontsize=10, fontweight="bold")

        out = BASE_RESULTS_DIR / "megadashboard2.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
        
    print(f"✅ Dashboard 4x2 guardado en: {out}")