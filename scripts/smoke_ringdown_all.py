"""Smoke test: run all 18 estimators (current code) on the CLEAN ringdown
scenario signal and report whether each tracks the true frequency. This
validates the estimators independently of the (possibly stale) full_mc_benchmark
artifacts. Produces:
  - console table: std, range, corr(true,hat), RMSE, verdict
  - figure scripts/out/smoke_ringdown_all.png (4x5 grid, true vs hat)
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

SCN = REPO / "artifacts" / "full_mc_benchmark" / "IBR_Power_Imbalance_Ringdown" / \
    "IBR_Power_Imbalance_Ringdown_scenario.csv"
OUT = REPO / "scripts" / "out"
OUT.mkdir(exist_ok=True)


def main():
    sc = pd.read_csv(SCN)
    t = sc["t_s"].to_numpy(float)
    v = sc["v_pu"].to_numpy(float)
    ftrue = sc["f_true_hz"].to_numpy(float)
    dt = float(t[1] - t[0])
    print(f"Ringdown scenario: n={len(t)}, dt={dt:.2e}s, fs={1/dt:.0f}Hz, "
          f"f_true range=[{ftrue.min():.2f},{ftrue.max():.2f}]\n")

    classes = load_active_estimators()
    specs = {s.label: s for s in ACTIVE_ESTIMATOR_SPECS}

    rows = []
    series = {}
    for label, spec in specs.items():
        cls = classes.get(label)
        if cls is None:
            rows.append((label, "NO CLASS", 0, 0, 0, float("nan"), float("nan")))
            continue
        try:
            est = cls()
            # prefer estimate(t,v); fall back to per-sample step()
            if hasattr(est, "estimate"):
                fhat = np.asarray(est.estimate(t, v), float)
            else:
                fhat = np.array([est.step(float(z)) for z in v], float)
        except Exception as e:
            rows.append((label, f"ERROR {type(e).__name__}", 0, 0, 0, float("nan"), float("nan")))
            continue

        if len(fhat) != len(ftrue):
            m = min(len(fhat), len(ftrue))
            fhat, fa, ta = fhat[:m], ftrue[:m], t[:m]
        else:
            fa, ta = ftrue, t
        finite = np.isfinite(fhat)
        invalid = 1.0 - finite.mean()
        fh = np.where(finite, fhat, np.nan)
        rmse = float(np.sqrt(np.nanmean((fh - fa) ** 2)))
        corr = float(np.corrcoef(fa[finite], fh[finite])[0, 1]) if finite.sum() > 2 and np.nanstd(fh) > 1e-9 else float("nan")
        lo, hi = float(np.nanmin(fh)), float(np.nanmax(fh))
        std = float(np.nanstd(fh))
        # verdict
        if invalid > 0.02 or not np.isfinite(rmse) or rmse > 5:
            verdict = "BROKEN/DIVERGE"
        elif abs(lo - hi) < 0.05:
            verdict = "FLAT (no track)"
        elif corr > 0.5:
            verdict = "tracks"
        elif rmse < 0.4:
            verdict = "low-bw (averages)"
        else:
            verdict = "weak"
        rows.append((label, verdict, std, lo, hi, corr, rmse))
        series[label] = (ta, fa, fh)

    # console table
    print(f"{'estimator':20s} {'verdict':16s} {'std':>6s} {'range':>16s} {'corr':>6s} {'RMSE':>7s}")
    print("-" * 78)
    for label, verdict, std, lo, hi, corr, rmse in sorted(rows, key=lambda r: (r[1], r[0])):
        rng = f"[{lo:.1f},{hi:.1f}]" if np.isfinite(lo) else "n/a"
        print(f"{label:20s} {verdict:16s} {std:6.3f} {rng:>16s} {corr:+6.2f} {rmse:7.3f}")

    # figure grid
    labels = [l for l in specs if l in series]
    n = len(labels); cols = 5; rrows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rrows, cols, figsize=(3.0 * cols, 2.1 * rrows))
    for i, label in enumerate(labels):
        ax = axes.flat[i]
        ta, fa, fh = series[label]
        ax.plot(ta, fa, color="black", lw=1.2, label="true")
        ax.plot(ta, fh, color="#B23A48", lw=1.0, alpha=0.9, label="hat")
        ax.set_ylim(55, 65); ax.set_title(label, fontsize=8)
        ax.tick_params(labelsize=6)
    for j in range(len(labels), rrows * cols):
        axes.flat[j].set_visible(False)
    fig.suptitle("Smoke test: all estimators (current code) on CLEAN ringdown signal",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "smoke_ringdown_all.png", dpi=130)
    print(f"\nwrote {OUT / 'smoke_ringdown_all.png'}")


if __name__ == "__main__":
    main()
