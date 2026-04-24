"""Subplot (a): aligned phase-jump response with a clean representative set."""

from plotting.benchmark.mega_dashboard2_time_utils import plot_clean_tracking_panel


_P00_ESTIMATORS = ["SOGI-FLL", "SOGI-PLL", "EKF", "UKF", "IPDFT"]


def md2_subplot_00(ax, data_bundle):
    plot_clean_tracking_panel(
        ax,
        base_results_dir=data_bundle["BASE_RESULTS_DIR"],
        scenario="IEEE_Phase_Jump_60",
        estimators=_P00_ESTIMATORS,
        panel_title="(a) Phase Jump 60 deg",
        align_col="t_jump_s",
        align_value=0.0,
        t_window=(-0.06, 0.24),
        event_label="Phase discontinuity",
        legend_loc="upper right",
        legend_ncol=3,
        show_ylabel=True,
        min_y_span=0.9,
    )
