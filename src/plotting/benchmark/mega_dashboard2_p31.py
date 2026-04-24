"""
Subplot (h): Trip risk vs. CPU latency with Pareto highlighting.
"""
import numpy as np
from matplotlib.lines import Line2D


_PARETO_FILL = "#E8F5E9"
_PARETO_EDGE = "#1B5E20"
_RT_COLOR = "#C62828"
_TRIP_FLOOR = 5e-4


def _pareto_mask(x, y):
    n = len(x)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dominates = (x[j] <= x[i] and y[j] <= y[i]) and (x[j] < x[i] or y[j] < y[i])
            if dominates:
                mask[i] = False
                break
    return mask


def _pareto_step_arrays(x, y, x_min):
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    step_x = np.r_[x_min, x_sorted]
    step_y = np.r_[y_sorted[0], y_sorted]
    return step_x, step_y


def _repulsion_offsets(log_xy, base=18):
    center = np.mean(log_xy, axis=0)
    offsets = []
    n = len(log_xy)
    for i in range(n):
        radial = log_xy[i] - center
        dx, dy = 1.2 * radial[0], 1.2 * radial[1]
        for j in range(n):
            if i == j:
                continue
            rx = log_xy[i, 0] - log_xy[j, 0]
            ry = log_xy[i, 1] - log_xy[j, 1]
            d2 = rx * rx + ry * ry + 1e-4
            dx += 1.7 * rx / d2
            dy += 1.7 * ry / d2
        if abs(dx) + abs(dy) < 1e-9:
            dx = 1.0 if (i % 2 == 0) else -1.0
            dy = 0.7 if (i % 3 == 0) else -0.7
        mag = np.sqrt(dx * dx + dy * dy) or 1.0
        ux, uy = dx / mag, dy / mag
        offsets.append((ux * base, uy * base))
    return offsets


def md2_subplot_31(ax, data_bundle):
    df_global = data_bundle["df_global"]
    estimator_families = data_bundle["_ESTIMATOR_FAMILIES"]
    family_palette = data_bundle["_FAMILY_PALETTE"]

    grp = df_global.groupby("estimator").agg(
        trip_mean=("m5_trip_risk_s_mean", "mean"),
        cpu_mean=("m13_cpu_time_us_mean", "mean"),
        trip_std=("m5_trip_risk_s_std", "mean"),
        cpu_std=("m13_cpu_time_us_std", "mean"),
    ).dropna()

    ests = grp.index.tolist()
    cpu_arr = grp["cpu_mean"].to_numpy(dtype=float)
    trip_raw = grp["trip_mean"].to_numpy(dtype=float)
    trip_arr = np.where(trip_raw > 0.0, trip_raw, _TRIP_FLOOR)
    cpu_serr = grp["cpu_std"].to_numpy(dtype=float)
    trip_serr = grp["trip_std"].to_numpy(dtype=float)
    at_floor = trip_raw <= 0.0

    pareto = _pareto_mask(cpu_arr, trip_arr)

    for i, est in enumerate(ests):
        fam = estimator_families.get(est, "Adaptive")
        clr = family_palette.get(fam, "#757575")
        is_pareto = bool(pareto[i])
        marker = "v" if at_floor[i] else "o"

        trip_lo = min(trip_serr[i], 0.9 * trip_arr[i]) if not at_floor[i] else 0.0
        cpu_lo = min(cpu_serr[i], 0.9 * cpu_arr[i])

        if not at_floor[i]:
            ax.errorbar(
                cpu_arr[i],
                trip_arr[i],
                xerr=[[cpu_lo], [cpu_serr[i]]],
                yerr=[[trip_lo], [trip_serr[i]]],
                fmt="none",
                ecolor=clr,
                elinewidth=0.45,
                capsize=1.5,
                capthick=0.45,
                alpha=0.16,
                zorder=2,
            )
        ax.scatter(
            cpu_arr[i],
            trip_arr[i],
            s=34 if is_pareto else 24,
            color=clr,
            marker=marker,
            edgecolors="none",
            alpha=0.82,
            zorder=4,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(100, color=_RT_COLOR, ls="--", lw=0.8, alpha=0.7, zorder=2)

    x_min, _ = ax.get_xlim()
    y_min, _ = ax.get_ylim()
    pareto_x = cpu_arr[pareto]
    pareto_y = trip_arr[pareto]
    step_x, step_y = _pareto_step_arrays(pareto_x, pareto_y, x_min)
    ax.fill_between(step_x, y_min, step_y, step="post", color=_PARETO_FILL, alpha=0.14, zorder=1)
    ax.scatter(
        pareto_x,
        pareto_y,
        s=70,
        facecolors="none",
        edgecolors=_PARETO_EDGE,
        marker="D",
        linewidths=1.0,
        zorder=5,
    )

    if at_floor.any():
        ax.axhline(_TRIP_FLOOR, color="#aaaaaa", ls=":", lw=0.6, zorder=1)
        ax.text(x_min, _TRIP_FLOOR * 1.18, "trip ~ 0", fontsize=7.7, color="#888888", va="bottom", zorder=3)

    log_xy = np.column_stack([np.log10(cpu_arr), np.log10(trip_arr)])
    offsets = _repulsion_offsets(log_xy)
    x_log = log_xy[:, 0]
    y_log = log_xy[:, 1]
    x_span = max(float(np.max(x_log) - np.min(x_log)), 1e-9)
    y_span = max(float(np.max(y_log) - np.min(y_log)), 1e-9)
    for i, est in enumerate(ests):
        short = est.replace(" (RK-DPMU)", "").replace("Type-3 ", "T3-")
        fam = estimator_families.get(est, "Adaptive")
        clr = family_palette.get(fam, "#444444")
        dx, dy = offsets[i]
        x_frac = float((x_log[i] - np.min(x_log)) / x_span)
        y_frac = float((y_log[i] - np.min(y_log)) / y_span)
        if y_frac > 0.82:
            dy = -max(abs(dy), 18)
        if y_frac < 0.14:
            dy = max(abs(dy), 14)
        if x_frac < 0.18:
            dx = max(abs(dx), 16)
        if x_frac > 0.84:
            dx = -max(abs(dx), 16)
        if x_frac < 0.24 and y_frac > 0.70:
            dx = max(abs(dx), 18)
            dy = -max(abs(dy), 18)
        ax.annotate(
            short,
            (cpu_arr[i], trip_arr[i]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.3,
            color=clr,
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy >= 0 else "top",
            fontweight="bold" if pareto[i] else "normal",
            bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="none", alpha=0.78),
            arrowprops=dict(arrowstyle="-", color=clr, lw=0.45, alpha=0.40, shrinkA=2, shrinkB=2),
            zorder=6,
        )

    family_handles = []
    seen_families = set()
    for est in ests:
        fam = estimator_families.get(est, "Adaptive")
        if fam in seen_families:
            continue
        seen_families.add(fam)
        family_handles.append(
            Line2D([0], [0], marker="o", linestyle="none", markersize=5.2,
                   markerfacecolor=family_palette.get(fam, "#757575"), markeredgecolor="none", label=fam)
        )

    family_handles.extend([
        Line2D([0], [0], marker="D", linestyle="none", markersize=5.6,
               markerfacecolor="none", markeredgecolor=_PARETO_EDGE, label="Pareto-optimal"),
        Line2D([0], [0], color=_RT_COLOR, ls="--", lw=0.8, label="RT limit"),
    ])

    ax.tick_params(axis="both", which="both", labelsize=9.5)
    ax.set_xlabel(r"CPU Time [$\mu$s/sample]", fontsize=10.5, labelpad=2)
    ax.set_title("(h) Trip Risk vs. Latency", fontweight="bold", fontsize=11.5, loc="left", pad=5)
    ax.grid(True, which="major", ls="-", alpha=0.15, zorder=0)
    ax.grid(True, which="minor", ls=":", alpha=0.08, zorder=0)
    ax.legend(
        handles=family_handles,
        loc="upper right",
        fontsize=8.2,
        ncol=2,
        frameon=True,
        framealpha=0.92,
        edgecolor="#cccccc",
        handletextpad=0.4,
        borderpad=0.42,
        labelspacing=0.24,
        columnspacing=0.9,
    )
