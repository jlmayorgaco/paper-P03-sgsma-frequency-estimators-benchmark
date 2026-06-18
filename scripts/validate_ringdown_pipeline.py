"""End-to-end validation: run the IBR ringdown scenario through the REAL
MonteCarloEngine (with the T-100 sample-rate fix) for all 18 estimators, n=5.
Confirms code integrity before re-generating the full benchmark.

Produces:
  - console table: RMSE, peak err, corr(true,hat), invalid-rate, verdict
  - figure scripts/out/validate_ringdown_pipeline.png (grid, true vs hat, run 0)
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.monte_carlo_engine import MonteCarloEngine
from pipelines.benchmark_definition import ACTIVE_ESTIMATOR_SPECS, load_active_estimators
from scenarios.ibr_power_imbalance_ringdown import IBRPowerImbalanceRingdownScenario

OUT = REPO / "scripts" / "out"
OUT.mkdir(exist_ok=True)
N_RUNS = 5


def verdict(rmse, corr, lo, hi, invalid):
    if invalid > 0.02 or not np.isfinite(rmse) or rmse > 5 or lo < 0 or hi > 200:
        return "BROKEN/DIVERGE"
    if abs(hi - lo) < 0.05:
        return "FLAT (low-bw)"
    if corr > 0.5:
        return "TRACKS"
    if rmse < 0.4:
        return "low-bw avg"
    return "weak"


def main():
    classes = load_active_estimators()
    specs = {s.label: s for s in ACTIVE_ESTIMATOR_SPECS}
    rows, series = [], {}

    for label in specs:
        cls = classes.get(label)
        if cls is None:
            rows.append((label, "NO CLASS", float("nan"), float("nan"), float("nan"), 1.0)); continue
        try:
            eng = MonteCarloEngine(
                scenario_cls=IBRPowerImbalanceRingdownScenario,
                estimator_cls=cls, n_runs=N_RUNS, base_seed=42,
                n_cost_reps=1, enforce_standardized_step=True, capture_signals=True,
            )
            res = eng.run()
        except Exception as e:
            rows.append((label, f"ERROR {type(e).__name__}", float("nan"), float("nan"), float("nan"), 1.0))
            continue

        sdf = getattr(res, "signals_df", None)
        if sdf is None or sdf.empty:
            rows.append((label, "NO SIGNALS", float("nan"), float("nan"), float("nan"), 1.0)); continue
        d0 = sdf[sdf["run_idx"] == sdf["run_idx"].min()]
        tr = d0["f_true_hz"].to_numpy(float); ht = d0["f_hat_hz"].to_numpy(float)
        fin = np.isfinite(ht); invalid = 1.0 - fin.mean()
        rmse = float(np.sqrt(np.nanmean((ht - tr) ** 2)))
        peak = float(np.nanmax(np.abs(ht - tr)))
        corr = float(np.corrcoef(tr[fin], ht[fin])[0, 1]) if fin.sum() > 2 and np.nanstd(ht[fin]) > 1e-9 else float("nan")
        lo, hi = float(np.nanmin(ht)), float(np.nanmax(ht))
        rows.append((label, verdict(rmse, corr, lo, hi, invalid), rmse, peak, corr, invalid))
        series[label] = (d0["t_s"].to_numpy(float), tr, np.where(fin, ht, np.nan), lo, hi)

    print(f"\n{'estimator':20s} {'verdict':16s} {'RMSE':>8s} {'peak':>8s} {'corr':>6s} {'inval%':>7s} {'range':>16s}")
    print("-" * 92)
    for label, v, rmse, peak, corr, inval in sorted(rows, key=lambda r: (r[1], r[0])):
        rng = f"[{series[label][3]:.1f},{series[label][4]:.1f}]" if label in series else "n/a"
        print(f"{label:20s} {v:16s} {rmse:8.3f} {peak:8.2f} {corr:+6.2f} {inval*100:6.1f}% {rng:>16s}")

    labels = [l for l in specs if l in series]
    cols = 5; rrows = (len(labels) + cols - 1) // cols
    fig, axes = plt.subplots(rrows, cols, figsize=(3.0 * cols, 2.1 * rrows))
    for i, label in enumerate(labels):
        ax = axes.flat[i]; ta, tr, ht, lo, hi = series[label]
        ax.plot(ta, tr, color="black", lw=1.2); ax.plot(ta, ht, color="#B23A48", lw=1.0, alpha=0.9)
        ax.set_ylim(55, 65); ax.set_title(label, fontsize=8); ax.tick_params(labelsize=6)
    for j in range(len(labels), rrows * cols): axes.flat[j].set_visible(False)
    fig.suptitle(f"Pipeline validation: ringdown through MonteCarloEngine (n={N_RUNS}, run 0)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "validate_ringdown_pipeline.png", dpi=130)
    print(f"\nwrote {OUT / 'validate_ringdown_pipeline.png'}")


if __name__ == "__main__":
    main()
