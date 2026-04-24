"""
Subplot (d): IBR Sub-synchronous Ringdown — power imbalance onset.
MC band (mean ± 1σ fill_between, α=0.20). No local legend (global strip).
"""
import numpy as np
import pandas as pd


def _load_mc_band(base_dir, scenario, estimator):
    path = base_dir / scenario / estimator / f"{scenario}__{estimator}_signals.csv"
    if not path.exists():
        return None, None, None
    df = pd.read_csv(path)
    grp = df.groupby("t_s")["f_hat_hz"]
    t   = grp.mean().index.values
    mu  = grp.mean().values
    sig = grp.std(ddof=1).fillna(0).values
    return t, mu, sig


def md2_subplot_11(ax, data_bundle):
    load_estimator_trace = data_bundle["load_estimator_trace"]
    BASE_RESULTS_DIR     = data_bundle["BASE_RESULTS_DIR"]
    TRACKING_ESTIMATORS  = data_bundle["TRACKING_ESTIMATORS"]
    COLORS_6             = data_bundle["COLORS_6"]

    scenario_name = "IBR_Power_Imbalance_Ringdown"
    EVENT_T       = 1.0
    T_LO, T_HI   = 0.80, 1.70
    Y_LO,  Y_HI  = 57.5, 65.0

    scen_path = BASE_RESULTS_DIR / scenario_name / f"{scenario_name}_scenario.csv"
    df_scen = pd.read_csv(scen_path)
    t_tr = df_scen["t_s"].values
    f_tr = df_scen["f_true_hz"].values
    mask_tr = (t_tr >= T_LO) & (t_tr <= T_HI)
    ax.plot(t_tr[mask_tr], f_tr[mask_tr],
            color="black", lw=1.4, ls="--", zorder=10)

    for est in TRACKING_ESTIMATORS:
        clr   = COLORS_6.get(est, "#607D8B")
        short = "Koopman" if "Koopman" in est else est

        t_mc, mu_mc, sig_mc = _load_mc_band(BASE_RESULTS_DIR, scenario_name, est)
        if t_mc is not None:
            mask = (t_mc >= T_LO) & (t_mc <= T_HI)
            mu_c = np.clip(mu_mc, Y_LO, Y_HI)
            ax.fill_between(t_mc[mask],
                            np.clip(mu_mc[mask] - sig_mc[mask], Y_LO, Y_HI),
                            np.clip(mu_mc[mask] + sig_mc[mask], Y_LO, Y_HI),
                            color=clr, alpha=0.20, linewidth=0, zorder=2)
            ax.plot(t_mc[mask], mu_c[mask],
                    color=clr, lw=0.95, alpha=0.90, label=short,
                    zorder=3, clip_on=True)
        else:
            t_e, f_e = load_estimator_trace(scenario_name, est)
            mask = (t_e >= T_LO) & (t_e <= T_HI)
            ax.plot(t_e[mask], np.clip(f_e[mask], Y_LO, Y_HI),
                    color=clr, lw=0.95, alpha=0.90, label=short,
                    clip_on=True)

    ax.axvline(EVENT_T, color="#555", ls=":", lw=0.8, alpha=0.6)
    ax.set_xlim(T_LO, T_HI)
    ax.set_ylim(Y_LO, Y_HI)
    ax.set_xlabel("Time [s]", fontsize=7, labelpad=2)
    ax.set_title("(d) Sub-sync. Ringdown", fontweight="bold", fontsize=8,
                 loc="left", pad=3)
    ax.tick_params(labelsize=6)
