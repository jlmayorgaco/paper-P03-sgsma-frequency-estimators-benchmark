"""
Subplot (a): Phase Jump 60° — focused comparison.
Best MC run per estimator (line-only) for selected estimators.
For visualization, each selected run is shifted so its phase-jump aligns at t=0.5 s.
"""
import numpy as np
import pandas as pd


# Focus set (full color)
_P00_FOCUS = ["EKF", "UKF", "RA-EKF", "SOGI-FLL", "IPDFT"]
# Context set (faded)
_P00_CONTEXT = ["RLS", "Koopman (RK-DPMU)", "PLL"]

_P00_COLORS = {
    "EKF":               "#1565C0",
    "UKF":               "#1E88E5",
    "RA-EKF":            "#42A5F5",
    "SOGI-FLL":          "#2E7D32",
    "IPDFT":             "#E65100",
    "RLS":               "#6A1B9A",
    "Koopman (RK-DPMU)": "#B71C1C",
    "PLL":               "#8D6E63",
}

_P00_ALIGN_EVENT_T = 0.5
_P00_FALLBACK_JUMP_T = 0.7


def _load_best_mc_trace(base_dir, scenario, estimator, t_lo, t_hi):
    path = base_dir / scenario / estimator / f"{scenario}__{estimator}_signals.csv"
    if not path.exists():
        return None, None, None
    df = pd.read_csv(path)
    w = df[(df["t_s"] >= t_lo) & (df["t_s"] <= t_hi)].copy()
    if w.empty:
        return None, None, None

    run_rmse = (
        w.groupby("run_idx")
        .apply(lambda g: float(np.sqrt(np.mean((g["f_hat_hz"] - g["f_true_hz"]) ** 2))))
        .sort_values()
    )
    if run_rmse.empty:
        return None, None, None

    best_run_idx = int(run_rmse.index[0])
    best_rmse = float(run_rmse.iloc[0])
    best = w[w["run_idx"] == best_run_idx].sort_values("t_s")
    if "t_jump_s" in best.columns and not best["t_jump_s"].isna().all():
        jump_t = float(best["t_jump_s"].iloc[0])
    else:
        jump_t = _P00_FALLBACK_JUMP_T

    t_aligned = best["t_s"].values - jump_t + _P00_ALIGN_EVENT_T
    return t_aligned, best["f_hat_hz"].values, (best_run_idx, best_rmse, jump_t)


def md2_subplot_00(ax, data_bundle):
    BASE_RESULTS_DIR = data_bundle["BASE_RESULTS_DIR"]

    scenario_name = "IEEE_Phase_Jump_60"
    EVENT_T = _P00_ALIGN_EVENT_T
    T_LO, T_HI   = 0.4, 0.8
    Y_LO,  Y_HI  = 59.75, 62.5

    scen_path = BASE_RESULTS_DIR / scenario_name / f"{scenario_name}_scenario.csv"
    if not scen_path.exists():
        raise FileNotFoundError(f"[!] Missing: {scen_path}")

    df_scen = pd.read_csv(scen_path)
    t_tr = df_scen["t_s"].values
    f_tr = df_scen["f_true_hz"].values
    mask_tr = (t_tr >= T_LO) & (t_tr <= T_HI)
    ax.plot(t_tr[mask_tr], f_tr[mask_tr],
            color="black", lw=1.4, ls="--", label="True", zorder=10)

    for est in (_P00_CONTEXT + _P00_FOCUS):
        clr   = _P00_COLORS[est]
        short = "Koopman" if "Koopman" in est else est
        is_focus = est in _P00_FOCUS

        line_alpha = 0.92 if is_focus else 0.35
        line_width = 1.00 if is_focus else 0.85

        t_best, f_best, best_meta = _load_best_mc_trace(
            BASE_RESULTS_DIR, scenario_name, est, T_LO, T_HI
        )
        if t_best is None:
            continue

        mask = (t_best >= T_LO) & (t_best <= T_HI)
        f_c = np.clip(f_best, Y_LO, Y_HI)
        run_idx, _, jump_t = best_meta
        ax.plot(t_best[mask], f_c[mask],
                color=clr, lw=line_width, alpha=line_alpha,
                label=f"{short} [run {run_idx}, jump {jump_t:.3f}s]",
                zorder=3, clip_on=True)

    ax.axvline(EVENT_T, color="#555", ls=":", lw=0.8, alpha=0.6)

    ax.set_xlim(T_LO, T_HI)
    ax.set_ylim(Y_LO, Y_HI)
    ax.set_xlabel("Time [s]", fontsize=7, labelpad=2)
    ax.set_ylabel("Frequency [Hz]", fontsize=7, labelpad=2)
    ax.set_title("(a) Phase Jump 60°", fontweight="bold", fontsize=8,
                 loc="left", pad=3)
    ax.tick_params(labelsize=6)

    # Local legend with selected estimators (best MC run per estimator)
    ax.legend(loc="upper right", fontsize=5.5, ncol=1, frameon=True,
              framealpha=0.92, edgecolor="#ccc",
              handlelength=1.2, labelspacing=0.22, borderpad=0.4)
