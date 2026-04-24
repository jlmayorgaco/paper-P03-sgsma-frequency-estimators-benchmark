"""Subplot (d): ringdown recovery aligned at reconnection."""

from plotting.benchmark.mega_dashboard2_time_utils import plot_clean_tracking_panel


_P11_ESTIMATORS = ["SOGI-PLL", "RA-EKF", "UKF", "IPDFT"]


def md2_subplot_11(ax, data_bundle):
    plot_clean_tracking_panel(
        ax,
        base_results_dir=data_bundle["BASE_RESULTS_DIR"],
        scenario="IBR_Power_Imbalance_Ringdown",
        estimators=_P11_ESTIMATORS,
        panel_title="(d) Sub-sync. Ringdown",
        align_col="t_event_s",
        align_value=1.0,
        t_window=(-0.18, 0.62),
        event_label="Reconnection and ringdown onset",
        legend_loc="upper right",
        legend_ncol=2,
        show_ylabel=False,
        min_y_span=1.2,
        y_limits=(58.95, 60.35),
    )
