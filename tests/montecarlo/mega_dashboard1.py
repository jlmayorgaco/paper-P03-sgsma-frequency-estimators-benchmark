import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

def plot_megadashboard1(
    base_results_dir: Path, 
    ieee_rc: dict, 
    ieee_full_w: float, 
    ieee_page_h: float
) -> None:
    """
    Generates Mega Dashboard 1 (Scenario Signal Overview).
    
    Args:
        base_results_dir: Path to the outputs directory containing scenario subfolders.
        ieee_rc: Dictionary with matplotlib rcParams for IEEE styling.
        ieee_full_w: Target figure width in inches.
        ieee_page_h: Target figure height in inches.
    """
    print("\n[MEGA1] Building scenario signal overview from CSVs ...")

    # Asegurar que sea un objeto Path por si acaso se pasa como string
    base_results_dir = Path(base_results_dir)

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
        dict(
            col0=dict(folder="IBR_Multi_Event", title="(4,0) Multi-Evt Voltage (First Jump)", col="v_pu", color="indigo", ylabel="V [pu]", xlim=(0.5 - dw, 0.5 + dw), ylim=(-1.25, 1.25), event_t=[0.5, 1.0]),
            col1=dict(folder="IBR_Multi_Event", title="(4,1) Multi-Evt Frequency Profile", col="f_true_hz", color="darkred", ylabel="f [Hz]", event_t=[0.5, 1.0], xlim=(0.0, 2), ylim=(56.0, 62.0))
        ),
        # Row 5: Ringdown Event. Event at 0.5s. 
        dict(
            col0=dict(folder="IBR_Power_Imbalance_Ringdown", title="(5,0) Ringdown Voltage", col="v_pu", color="indigo", ylabel="V [pu]", xlim=(0.5 - dw, 0.5 + dw), ylim=(-1.5, 1.5), event_t=0.5),
            col1=dict(folder="IBR_Power_Imbalance_Ringdown", title="(5,1) Ringdown Frequency Profile", col="f_true_hz", color="darkred", ylabel="f [Hz]", event_t=0.5, xlim=(0.0, 2), ylim=(59.0, 61.0))
        ),
    ]

    nrows = len(GRID_SPECS)
    
    # Altura ajustada para 6 filas usando los inputs de la función
    fig_w = ieee_full_w 
    fig_h = ieee_page_h * 0.60

    with plt.rc_context(ieee_rc):
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = gridspec.GridSpec(nrows, 2, figure=fig, hspace=0.85, wspace=0.20,
                               left=0.08, right=0.98, top=0.92, bottom=0.08)

        for ridx, row_spec in enumerate(GRID_SPECS):
            for cidx, col_key in enumerate(["col0", "col1"]):
                spec = row_spec[col_key]
                ax = fig.add_subplot(gs[ridx, cidx])
                
                # Se utiliza el base_results_dir que entra por parámetro
                csv_path = base_results_dir / spec["folder"] / f"{spec['folder']}_scenario.csv"
                
                if not csv_path.exists():
                    t = np.linspace(0, 2.0, 2000)
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

                # Horizontal tracking lines for Mag Step Peaks
                if spec["folder"] == "IEEE_Mag_Step" and spec["col"] == "v_pu":
                    mask_pre = (t >= (0.5 - dw)) & (t <= 0.5)
                    mask_post = (t > 0.5) & (t <= (0.5 + dw))
                    if mask_pre.any() and mask_post.any():
                        max_pre, max_post = np.max(y[mask_pre]), np.max(y[mask_post])
                        ax.axhline(max_pre, color="#333", linestyle="--", linewidth=0.5, alpha=0.8)
                        ax.axhline(max_post, color="#333", linestyle="--", linewidth=0.5, alpha=0.8)

                # Vertical Dashed Line for Transition Events
                if "event_t" in spec:
                    events = spec["event_t"] if isinstance(spec["event_t"], list) else [spec["event_t"]]
                    for ev in events: 
                        ax.axvline(ev, color="black", linestyle="--", linewidth=1.0, alpha=0.6)

                ax.grid(True, alpha=0.3, lw=0.4)
                ax.tick_params(axis='both', which='major', labelsize=6)
                ax.set_xlabel("Time [s]", labelpad=2, fontsize=7)

        fig.text(0.5, 0.97, "IBR Frequency Estimator Benchmark — Detailed Scenario Overview", 
                 ha="center", va="top", fontsize=8.5, fontweight="bold", color="#111")

        # Guardar usando el directorio dinámico
        out = base_results_dir / "megadashboard1.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        
    print(f"    megadashboard1 saved -> {out}")