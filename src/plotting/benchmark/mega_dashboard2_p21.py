"""Subplot (f): harmonic steady-state comparison with a tight frequency scale."""

from plotting.benchmark.mega_dashboard2_time_utils import plot_clean_tracking_panel


_P21_ESTIMATORS = ["EKF", "IPDFT", "Prony", "UKF", "SOGI-PLL"]


def md2_subplot_21(ax, data_bundle):
    plot_clean_tracking_panel(
        ax,
        base_results_dir=data_bundle["BASE_RESULTS_DIR"],
        scenario="IBR_Harmonics_Large",
        estimators=_P21_ESTIMATORS,
        panel_title="(f) IBR Harmonics",
        align_col=None,
        align_value=0.0,
        t_window=(0.10, 0.20),
        event_label="Steady-state THD stress",
        legend_loc="upper right",
        legend_ncol=2,
        show_ylabel=False,
        min_y_span=0.10,
        xlabel="Time [s]",
        show_event_line=False,
        y_limits=(59.90, 60.03),
    )
