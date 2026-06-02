"""Inspect RA-EKF traces vs EKF/UKF on the scenarios where RA-EKF was flagged
non-physical, to tell a brief transient overshoot from a real divergence.
Prints, per scenario, WHEN/where RA-EKF leaves [45,75] Hz and for how long.
Saves a comparison figure.
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

from pipelines.benchmark_definition import load_active_estimators
MC = REPO / "artifacts" / "full_mc_benchmark"
OUT = REPO / "scripts" / "out"; OUT.mkdir(exist_ok=True)

SCN = ["IEEE_Freq_Ramp_10Hzs", "IEEE_Freq_Step", "IEEE_Mag_Step_25pct", "IBR_Multi_Event"]
ESTS = ["RA-EKF", "EKF", "UKF"]


def load(scn, max_s=2.0):
    df = pd.read_csv(MC / scn / f"{scn}_scenario.csv")
    t = df["t_s"].to_numpy(float); k = t <= max_s
    return t[k], df["v_pu"].to_numpy(float)[k], df["f_true_hz"].to_numpy(float)[k]


def tuned(scn, label):
    rs = MC / scn / label / "run_spec.json"
    if rs.exists():
        try: return json.load(open(rs)).get("best_params", {}) or {}
        except Exception: return {}
    return {}


def main():
    classes = load_active_estimators()
    fig, axes = plt.subplots(len(SCN), 1, figsize=(9, 2.4 * len(SCN)))
    for si, scn in enumerate(SCN):
        t, v, ft = load(scn)
        ax = axes[si]
        ax.plot(t, ft, color="black", lw=1.4, label="true", zorder=5)
        print(f"\n=== {scn} ===")
        for est in ESTS:
            cls = classes.get(est)
            if cls is None: continue
            fh = np.asarray(cls(**tuned(scn, est)).estimate(t, v), float)
            col = {"RA-EKF": "#B23A48", "EKF": "#1F5C8B", "UKF": "#196B24"}[est]
            ax.plot(t, fh, lw=1.0, alpha=0.85, color=col, label=est)
            oob = (fh < 45) | (fh > 75)
            frac = oob.mean() * 100
            if oob.any():
                tt = t[oob]
                print(f"  {est:7s} out-of-band {frac:5.2f}% of samples, "
                      f"t in [{tt.min():.3f},{tt.max():.3f}]s, "
                      f"fmin={np.nanmin(fh):.1f} fmax={np.nanmax(fh):.1f}, "
                      f"first-event-region={'YES' if tt.min() < 1.1 else 'no'}")
            else:
                print(f"  {est:7s} fully physical [{np.nanmin(fh):.1f},{np.nanmax(fh):.1f}]")
        ax.set_ylim(40, 80); ax.set_title(scn, fontsize=9); ax.legend(fontsize=7, ncol=4)
        ax.axhline(45, color="gray", ls=":", lw=0.6); ax.axhline(75, color="gray", ls=":", lw=0.6)
    fig.suptitle("RA-EKF vs EKF/UKF traces (tuned, clean signal)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(OUT / "inspect_ra_ekf.png", dpi=130)
    print(f"\nwrote {OUT / 'inspect_ra_ekf.png'}")


if __name__ == "__main__":
    main()
