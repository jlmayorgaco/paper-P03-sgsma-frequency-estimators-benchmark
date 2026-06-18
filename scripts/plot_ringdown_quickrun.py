"""Plot the ringdown quick-run results (real pipeline, tuned per scenario) for
visual inspection: a grid of true vs estimated frequency for every estimator,
with steady-state RMSE (warm-up excluded) annotated.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "artifacts" / "full_mc_ringdown_test" / "IBR_Power_Imbalance_Ringdown"
OUT = REPO / "scripts" / "out"
OUT.mkdir(exist_ok=True)
WARM = 1500  # 0.15 s at 10 kHz, matches benchmark metrics


def main():
    est_dirs = sorted([d for d in RUN.iterdir() if d.is_dir()])
    panels = []
    for d in est_dirs:
        est = d.name
        sig = list(d.glob("*_signals.csv"))
        if not sig:
            continue
        df = pd.read_csv(sig[0])
        d0 = df[df["run_idx"] == df["run_idx"].min()]
        t = d0["t_s"].to_numpy(float); ft = d0["f_true_hz"].to_numpy(float); fh = d0["f_hat_hz"].to_numpy(float)
        fin = np.isfinite(fh)
        rmse = float(np.sqrt(np.nanmean((fh[WARM:] - ft[WARM:]) ** 2))) if len(fh) > WARM else float("nan")
        lo, hi = (float(np.nanmin(fh[WARM:])), float(np.nanmax(fh[WARM:]))) if len(fh) > WARM and fin.any() else (np.nan, np.nan)
        panels.append((est, t, ft, fh, rmse, lo, hi))

    n = len(panels); cols = 5; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.2 * rows))
    for i, (est, t, ft, fh, rmse, lo, hi) in enumerate(panels):
        ax = axes.flat[i]
        ax.plot(t, ft, color="black", lw=1.3, zorder=4)
        ax.plot(t, fh, color="#196B24", lw=1.0, alpha=0.9, zorder=3)
        ax.set_ylim(55, 65); ax.tick_params(labelsize=6)
        bad = (not np.isfinite(rmse)) or rmse > 2 or lo < 50 or hi > 70
        ax.set_title(f"{est}", fontsize=8, color=("#B23A48" if bad else "black"))
        ax.annotate(f"RMSE {rmse:.2g} Hz\n[{lo:.1f},{hi:.1f}]", xy=(0.5, 0.04),
                    xycoords="axes fraction", ha="center", va="bottom", fontsize=6,
                    color=("#B23A48" if bad else "#196B24"), fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    for j in range(n, rows * cols):
        axes.flat[j].set_visible(False)
    fig.suptitle("Ringdown quick-run (real pipeline, tuned per scenario) -- true (black) vs estimate (green)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "ringdown_quickrun.png", dpi=130)
    print(f"wrote {OUT / 'ringdown_quickrun.png'} ({n} estimators)")
    print(f"\n{'estimator':18s} {'RMSE':>8s}  range")
    for est, t, ft, fh, rmse, lo, hi in sorted(panels, key=lambda p: (p[4] if np.isfinite(p[4]) else 1e9)):
        print(f"{est:18s} {rmse:8.3f}  [{lo:.1f},{hi:.1f}]")


if __name__ == "__main__":
    main()
