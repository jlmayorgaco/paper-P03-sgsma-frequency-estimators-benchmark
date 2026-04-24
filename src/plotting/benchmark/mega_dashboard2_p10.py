"""Subplot (c): composite IBR event aligned at disturbance onset."""

from plotting.benchmark.mega_dashboard2_time_utils import plot_clean_tracking_panel


_P10_ESTIMATORS = ["SOGI-PLL", "EKF", "UKF", "IPDFT"]


def md2_subplot_10(ax, data_bundle):
    plot_clean_tracking_panel(
        ax,
        base_results_dir=data_bundle["BASE_RESULTS_DIR"],
        scenario="IBR_Multi_Event",
        estimators=_P10_ESTIMATORS,
        panel_title="(c) IBR Multi-Event",
        align_col="t_event_s",
        align_value=0.5,
        t_window=(-0.03, 0.85),
        event_label="Composite disturbance onset",
        legend_loc="upper right",
        legend_ncol=2,
        show_ylabel=True,
        min_y_span=1.4,
    )
