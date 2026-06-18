"""Fast single-pass ringdown plot: run all 18 estimators (current FIXED code,
default params) on the clean ringdown signal, decimated 2x for speed, and save
one grid figure. No tuning, no MC, no ProcessPool -> avoids the numba-cache race
and finishes in well under a minute.
"""
from __future__ import annotations
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pipelines.benchmark_definition import ACTIVE_ESTIMATOR_SPECS, load_active_estimators

SCN = REPO / "artifacts" / "full_mc_benchmark" / "IBR_Power_Imbalance_Ringdown" / "IBR_Power_Imbalance_Ringdown_scenario.csv"
OUT = REPO / "scripts" / "out"; OUT.mkdir(exist_ok=True)
WARM = 750  # 0.15 s at 5 kHz (after 2x decimation)


def main():
    df = pd.read_csv(SCN)
    t = df["t_s"].to_numpy(float)[::4]
    v = df["v_pu"].to_numpy(float)[::4]
    ft = df["f_true_hz"].to_numpy(float)[::4]
    classes = load_active_estimators()
    rows = []
    for s in ACTIVE_ESTIMATOR_SPECS:
        cls = classes.get(s.label)
        if cls is None:
            continue
        try:
            fh = np.asarray(cls().estimate(t, v), float)
        except Exception as e:
            print(f"{s.label}: ERROR {e}", flush=True); continue
        fin = np.isfinite(fh)
        rmse = float(np.sqrt(np.nanmean((fh[WARM:] - ft[WARM:]) ** 2))) if len(fh) > WARM else float("nan")
        lo, hi = (float(np.nanmin(fh[WARM:])), float(np.nanmax(fh[WARM:]))) if fin.any() else (np.nan, np.nan)
        rows.append((s.label, t, ft, fh, rmse, lo, hi))
        print(f"{s.label:18s} RMSE={rmse:7.3f}  [{lo:6.1f},{hi:6.1f}]", flush=True)

    n = len(rows); cols = 5; rr = (n + cols - 1) // cols
    fig, axes = plt.subplots(rr, cols, figsize=(3.0 * cols, 2.2 * rr))
    for i, (lab, t, ft, fh, rmse, lo, hi) in enumerate(rows):
        ax = axes.flat[i]
        ax.plot(t, ft, color="black", lw=1.3, zorder=4)
        ax.plot(t, fh, color="#196B24", lw=1.0, alpha=0.9, zorder=3)
        ax.set_ylim(55, 65); ax.tick_params(labelsize=6)
        bad = (not np.isfinite(rmse)) or rmse > 2 or lo < 50 or hi > 70
        ax.set_title(lab, fontsize=8, color=("#B23A48" if bad else "black"))
        ax.annotate(f"{rmse:.2g} Hz\n[{lo:.1f},{hi:.1f}]", xy=(0.5, 0.04), xycoords="axes fraction",
                    ha="center", va="bottom", fontsize=6, color=("#B23A48" if bad else "#196B24"),
                    fontweight="bold", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    for j in range(n, rr * cols):
        axes.flat[j].set_visible(False)
    fig.suptitle("Ringdown -- all 18 estimators, FIXED code (default params) -- true (black) vs estimate (green)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "quick_ringdown_plot.png", dpi=130)
    print(f"\nwrote {OUT / 'quick_ringdown_plot.png'} ({n} estimators)", flush=True)


if __name__ == "__main__":
    main()
