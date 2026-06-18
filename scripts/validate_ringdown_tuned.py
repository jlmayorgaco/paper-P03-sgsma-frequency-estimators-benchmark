"""Re-validate the ringdown with each estimator's TUNED best_params (from the
benchmark run_spec.json), not defaults. This is the apples-to-apples check:
the full_mc_benchmark tunes params per scenario via Optuna, so a fair integrity
test must use those same tuned params on the clean signal.
"""
from __future__ import annotations
import sys, json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipelines.benchmark_definition import ACTIVE_ESTIMATOR_SPECS, load_active_estimators

BASE = REPO / "artifacts" / "full_mc_benchmark" / "IBR_Power_Imbalance_Ringdown"
OUT = REPO / "scripts" / "out"
OUT.mkdir(exist_ok=True)


def main():
    sc = pd.read_csv(BASE / "IBR_Power_Imbalance_Ringdown_scenario.csv")
    t = sc["t_s"].to_numpy(float); v = sc["v_pu"].to_numpy(float); ft = sc["f_true_hz"].to_numpy(float)
    classes = load_active_estimators()

    print(f"Clean ringdown: n={len(t)} fs={1/(t[1]-t[0]):.0f}Hz f_true=[{ft.min():.1f},{ft.max():.1f}]\n")
    print(f"{'estimator':18s} {'RMSE':>8s} {'corr':>6s} {'range':>16s} {'verdict':>14s}  tuned?")
    print("-" * 80)

    rows, series = [], {}
    for s in ACTIVE_ESTIMATOR_SPECS:
        cls = classes.get(s.label)
        if cls is None:
            print(f"{s.label:18s} NO CLASS"); continue
        rs = BASE / s.label / "run_spec.json"
        bp, tuned = {}, False
        if rs.exists():
            try:
                bp = json.load(open(rs)).get("best_params", {}) or {}
                tuned = bool(bp)
            except Exception:
                bp = {}
        try:
            est = cls(**bp)
            fh = np.asarray(est.estimate(t, v), float)
        except Exception as e:
            print(f"{s.label:18s} ERROR {type(e).__name__}: {str(e)[:34]}"); continue
        fin = np.isfinite(fh)
        rmse = float(np.sqrt(np.nanmean((fh[fin] - ft[fin]) ** 2))) if fin.any() else float("nan")
        corr = float(np.corrcoef(ft[fin], fh[fin])[0, 1]) if fin.sum() > 2 and np.nanstd(fh[fin]) > 1e-9 else float("nan")
        lo, hi = (float(np.nanmin(fh)), float(np.nanmax(fh))) if fin.any() else (float("nan"), float("nan"))
        if not np.isfinite(rmse) or rmse > 5 or lo < 0 or hi > 200:
            vd = "DIVERGE"
        elif rmse < 0.4:
            vd = "OK"
        elif corr > 0.5:
            vd = "tracks"
        else:
            vd = "weak"
        print(f"{s.label:18s} {rmse:8.3f} {corr:+6.2f} [{lo:7.1f},{hi:6.1f}] {vd:>14s}  {'yes' if tuned else 'DEFAULT'}")
        series[s.label] = (t, ft, np.where(fin, fh, np.nan))

    labels = list(series)
    cols = 5; rrows = (len(labels) + cols - 1) // cols
    fig, axes = plt.subplots(rrows, cols, figsize=(3.0 * cols, 2.1 * rrows))
    for i, label in enumerate(labels):
        ax = axes.flat[i]; ta, tr, fh = series[label]
        ax.plot(ta, tr, color="black", lw=1.2); ax.plot(ta, fh, color="#196B24", lw=1.0, alpha=0.9)
        ax.set_ylim(55, 65); ax.set_title(label, fontsize=8); ax.tick_params(labelsize=6)
    for j in range(len(labels), rrows * cols): axes.flat[j].set_visible(False)
    fig.suptitle("Ringdown with TUNED best_params (clean signal) -- integrity check",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "validate_ringdown_tuned.png", dpi=130)
    print(f"\nwrote {OUT / 'validate_ringdown_tuned.png'}")


if __name__ == "__main__":
    main()
