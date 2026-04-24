"""
Subplot (g): MAE vs. CPU Latency — Pareto frontier.
K-means (k=3) cluster regions in log-space. All estimator names shown.
MC error bars (±1σ). Log-safe lower bound on error bars.
"""
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull


_N_CLUSTERS   = 3
# Light tints: one per cluster (blue / green / amber)
_C_FILL = ["#E3F2FD", "#F1F8E9", "#FFF8E1"]
_C_EDGE = ["#90CAF9", "#A5D6A7", "#FFE082"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _hull_patch(ax, pts_log, fill, edge, expand=0.15):
    """Draw padded convex-hull (or ellipse for tiny clusters) in log-space."""
    n = len(pts_log)
    if n == 0:
        return
    centroid = pts_log.mean(axis=0)

    if n <= 2:
        # pad a circle / capsule around the 1-2 pts
        r = 0.40
        theta = np.linspace(0, 2 * np.pi, 80)
        if n == 1:
            xs = 10 ** (centroid[0] + r * np.cos(theta))
            ys = 10 ** (centroid[1] + r * np.sin(theta))
        else:
            # capsule along the line between the two pts
            mid = pts_log.mean(axis=0)
            half = pts_log[1] - pts_log[0]
            perp = np.array([-half[1], half[0]])
            perp /= (np.linalg.norm(perp) or 1)
            arc1 = mid + half + r * np.column_stack([np.cos(theta), np.sin(theta)])
            arc2 = mid - half + r * np.column_stack([np.cos(theta), np.sin(theta)])
            outline = np.vstack([arc1, arc2[::-1]])
            xs, ys = 10 ** outline[:, 0], 10 ** outline[:, 1]
        ax.fill(xs, ys, color=fill, alpha=0.28, zorder=1, lw=0)
        ax.plot(xs, ys, color=edge, lw=0.55, alpha=0.60, ls="--", zorder=1)
        return

    try:
        hull = ConvexHull(pts_log)
    except Exception:
        return

    verts = pts_log[hull.vertices]
    # expand outward from centroid in log-space
    expanded = centroid + (1 + expand) * (verts - centroid)
    # dense edge interpolation so boundary looks smooth on log axes
    outline = []
    nv = len(expanded)
    for i in range(nv):
        outline.extend(np.linspace(expanded[i], expanded[(i + 1) % nv], 20))
    outline = 10 ** np.array(outline)
    ax.fill(outline[:, 0], outline[:, 1], color=fill, alpha=0.28, zorder=1, lw=0)
    ax.plot(outline[:, 0], outline[:, 1], color=edge, lw=0.55, alpha=0.60,
            ls="--", zorder=1)


def _repulsion_offsets(log_xy, base=13):
    """Point-repulsion label offsets (pts) in log-space coordinates."""
    n = len(log_xy)
    offsets = []
    for i in range(n):
        dx, dy = 0.0, 0.0
        for j in range(n):
            if i == j:
                continue
            rx = log_xy[i, 0] - log_xy[j, 0]
            ry = log_xy[i, 1] - log_xy[j, 1]
            d2 = rx * rx + ry * ry + 1e-4
            dx += rx / d2
            dy += ry / d2
        mag = np.sqrt(dx * dx + dy * dy) or 1.0
        offsets.append((dx / mag * base, dy / mag * base))
    return offsets


# ── main subplot ──────────────────────────────────────────────────────────────

def md2_subplot_30(ax, data_bundle):
    df_global          = data_bundle["df_global"]
    _ESTIMATOR_FAMILIES = data_bundle["_ESTIMATOR_FAMILIES"]
    _FAMILY_PALETTE     = data_bundle["_FAMILY_PALETTE"]

    grp = df_global.groupby("estimator").agg(
        mae_mean =("m2_mae_hz_mean",      "mean"),
        cpu_mean =("m13_cpu_time_us_mean", "mean"),
        mae_std  =("m2_mae_hz_std",        "mean"),
        cpu_std  =("m13_cpu_time_us_std",  "mean"),
    ).dropna()

    ests     = grp.index.tolist()
    cpu_arr  = grp["cpu_mean"].values
    mae_arr  = grp["mae_mean"].values * 1000   # → mHz
    cpu_serr = grp["cpu_std"].values
    mae_serr = grp["mae_std"].values  * 1000

    # ── K-means in log-space ──────────────────────────────────────────────────
    log_xy = np.column_stack([np.log10(cpu_arr), np.log10(mae_arr)])
    km     = KMeans(n_clusters=_N_CLUSTERS, random_state=42, n_init=10)
    km_labels = km.fit_predict(log_xy)

    for k in range(_N_CLUSTERS):
        mask = km_labels == k
        _hull_patch(ax, log_xy[mask], _C_FILL[k], _C_EDGE[k])

    # ── Scatter + error bars ──────────────────────────────────────────────────
    _HIGHLIGHT = {"EKF", "RA-EKF", "SOGI-PLL", "IPDFT", "Koopman (RK-DPMU)"}
    seen_fam   = set()

    for i, est in enumerate(ests):
        fam  = _ESTIMATOR_FAMILIES.get(est, "Adaptive")
        clr  = _FAMILY_PALETTE.get(fam, "#757575")
        lbl  = fam if fam not in seen_fam else "_nolegend_"
        seen_fam.add(fam)

        is_hi = est in _HIGHLIGHT
        size  = 52 if is_hi else 22
        alpha = 1.0 if is_hi else 0.65
        edge  = "black" if is_hi else "none"
        zord  = 6  if is_hi else 4

        # Log-safe error bar lower bounds
        mae_lo = min(mae_serr[i], 0.9 * mae_arr[i])
        cpu_lo = min(cpu_serr[i], 0.9 * cpu_arr[i])

        ax.errorbar(cpu_arr[i], mae_arr[i],
                    xerr=[[cpu_lo], [cpu_serr[i]]],
                    yerr=[[mae_lo], [mae_serr[i]]],
                    fmt="none", ecolor=clr, elinewidth=0.55,
                    capsize=1.8, capthick=0.55, alpha=0.50, zorder=zord - 1)
        ax.scatter(cpu_arr[i], mae_arr[i], s=size, color=clr,
                   edgecolors=edge, linewidths=0.7, alpha=alpha,
                   label=lbl, zorder=zord)

    # ── Annotate ALL estimators ───────────────────────────────────────────────
    offsets = _repulsion_offsets(log_xy)
    for i, est in enumerate(ests):
        short = (est.replace(" (RK-DPMU)", "")
                    .replace("Type-3 ", "T3-"))
        fam   = _ESTIMATOR_FAMILIES.get(est, "Adaptive")
        is_hi = est in _HIGHLIGHT
        ax.annotate(short, (cpu_arr[i], mae_arr[i]),
                    xytext=offsets[i], textcoords="offset points",
                    fontsize=5.0,
                    color=_FAMILY_PALETTE.get(fam, "#444"),
                    fontweight="bold" if is_hi else "normal",
                    zorder=8)

    # ── Cluster centroid labels ───────────────────────────────────────────────
    for k in range(_N_CLUSTERS):
        mask = km_labels == k
        cx   = 10 ** log_xy[mask, 0].mean()
        cy   = 10 ** log_xy[mask, 1].mean()
        ax.text(cx, cy, f"C{k+1}", fontsize=5.5, color=_C_EDGE[k],
                ha="center", va="center", fontweight="bold",
                alpha=0.70, zorder=5)

    # ── Styling ───────────────────────────────────────────────────────────────
    ax.axvline(100, color="#c62828", ls="--", lw=0.8, alpha=0.7,
               label="RT limit", zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.tick_params(axis="both", which="both", labelsize=6)
    ax.set_xlabel(r"CPU Time [$\mu$s/sample]", fontsize=7, labelpad=2)
    ax.set_ylabel("Mean MAE [mHz]", fontsize=7, labelpad=2)
    ax.set_title("(g) Accuracy vs. Latency", fontweight="bold",
                 fontsize=8, loc="left", pad=3)
    ax.grid(True, which="major", ls="-", alpha=0.15, zorder=0)
    ax.grid(True, which="minor", ls=":", alpha=0.08, zorder=0)

    ax.legend(loc="upper right", fontsize=5.5, frameon=True,
              framealpha=0.92, edgecolor="#ccc",
              handletextpad=0.4, borderpad=0.45, labelspacing=0.28)
