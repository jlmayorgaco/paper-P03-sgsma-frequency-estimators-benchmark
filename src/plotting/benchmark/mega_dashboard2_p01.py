"""Subplot (b): aligned ramp-onset comparison without uncertainty ribbons."""

from plotting.benchmark.mega_dashboard2_time_utils import plot_clean_tracking_panel


_P01_ESTIMATORS = ["SOGI-PLL", "Koopman (RK-DPMU)", "IPDFT", "EKF", "UKF"]


def md2_subplot_01(ax, data_bundle):
    plot_clean_tracking_panel(
        ax,
        base_results_dir=data_bundle["BASE_RESULTS_DIR"],
        scenario="IEEE_Freq_Ramp",
        estimators=_P01_ESTIMATORS,
        panel_title="(b) Freq. Ramp",
        align_col="t_start_s",
        align_value=0.0,
        t_window=(-0.05, 0.20),
        event_label="Ramp onset",
        legend_loc="upper left",
        legend_ncol=3,
        show_ylabel=False,
        min_y_span=0.9,
    )
