"""Senior-review audit: run every estimator with its TUNED best_params on the
CLEAN signal of each key scenario, and report a estimator x scenario matrix of
RMSE + a physical-sanity verdict. Flags any estimator that produces
non-physical output (f<45 or f>75 Hz, NaN/Inf, or RMSE>5) so we know exactly
which code paths are broken BEFORE re-generating the benchmark.
"""
from __future__ import annotations
import sys, json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import numpy as np
import pandas as pd

from pipelines.benchmark_definition import ACTIVE_ESTIMATOR_SPECS, load_active_estimators

MC = REPO / "artifacts" / "full_mc_benchmark"
OUT = REPO / "scripts" / "out"
OUT.mkdir(exist_ok=True)

# representative scenarios spanning the stress families
SCENARIOS = [
    "IEEE_Freq_Ramp_10Hzs",
    "IEEE_Freq_Step",
    "IEEE_Mag_Step_25pct",
    "IEEE_Phase_Jump_60",
    "IEEE_Modulation_FM",
    "IBR_Harmonics_Large",
    "IBR_Power_Imbalance_Ringdown",
    "IBR_Multi_Event",
]


def load_scn(scn, max_s=1.2):
    # Cap to the first max_s seconds (the event window) to keep spectral methods
    # tractable; this is a sanity audit, not the full benchmark.
    p = MC / scn / f"{scn}_scenario.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    t = df["t_s"].to_numpy(float)
    keep = t <= max_s
    return t[keep], df["v_pu"].to_numpy(float)[keep], df["f_true_hz"].to_numpy(float)[keep]


def tuned_params(scn, label):
    rs = MC / scn / label / "run_spec.json"
    if rs.exists():
        try:
            return json.load(open(rs)).get("best_params", {}) or {}
        except Exception:
            return {}
    return {}


def evaluate(fh, ft, fs=10000.0):
    # Exclude the 0.15 s warm-up window, exactly like the benchmark metrics
    # (metrics.py: baseline_samples = int(0.15*fs)), so we judge STEADY-STATE
    # behaviour, not cold-start transients.
    w = int(0.15 * fs)
    if len(fh) > w + 2:
        fh = fh[w:]; ft = ft[w:]
    fin = np.isfinite(fh)
    if not fin.any():
        return float("nan"), float("nan"), float("nan"), float("nan"), "NaN-ALL"
    rmse = float(np.sqrt(np.nanmean((fh[fin] - ft[fin]) ** 2)))
    lo, hi = float(np.nanmin(fh)), float(np.nanmax(fh))
    invalid = 1.0 - fin.mean()
    # physical-sanity verdict (independent of accuracy)
    if invalid > 0.02:
        flag = "INVALID"
    elif lo < 45 or hi > 75 or not np.isfinite(rmse):
        flag = "NON-PHYSICAL"
    elif rmse > 5:
        flag = "DIVERGE"
    else:
        flag = "ok"
    return rmse, lo, hi, invalid, flag


def main():
    classes = load_active_estimators()
    labels = [s.label for s in ACTIVE_ESTIMATOR_SPECS]

    # cache scenarios
    scn_data = {scn: load_scn(scn) for scn in SCENARIOS}
    scn_data = {k: v for k, v in scn_data.items() if v is not None}
    scns = list(scn_data)

    print("PHYSICAL-SANITY AUDIT (tuned params, clean signal)", flush=True)
    print("flag legend: ok | NON-PHYSICAL (f<45|>75) | DIVERGE (rmse>5) | INVALID (NaN/Inf)\n", flush=True)
    hdr = f"{'estimator':18s}" + "".join(f"{s.split('_')[-1][:9]:>11s}" for s in scns)
    print(hdr, flush=True); print("-" * len(hdr), flush=True)

    problems = []
    for label in labels:
        cls = classes.get(label)
        if cls is None:
            print(f"{label:18s}  NO CLASS"); continue
        cells = []
        for scn in scns:
            t, v, ft = scn_data[scn]
            bp = tuned_params(scn, label)
            try:
                est = cls(**bp)
                fh = np.asarray(est.estimate(t, v), float)
                if len(fh) != len(ft):
                    m = min(len(fh), len(ft)); fh, ftc = fh[:m], ft[:m]
                else:
                    ftc = ft
                rmse, lo, hi, inv, flag = evaluate(fh, ftc)
            except Exception as e:
                rmse, flag = float("nan"), f"ERR"
                problems.append((label, scn, f"exception {type(e).__name__}: {str(e)[:40]}"))
            mark = {"ok": "", "NON-PHYSICAL": "!", "DIVERGE": "X", "INVALID": "#", "NaN-ALL": "#", "ERR": "E"}.get(flag, "?")
            cells.append(f"{rmse:8.2f}{mark:>1s}" if np.isfinite(rmse) else f"{'--':>8s}{mark:>1s}")
            if flag not in ("ok",):
                problems.append((label, scn, f"{flag} rmse={rmse:.2f} range=[{lo:.1f},{hi:.1f}]" if np.isfinite(rmse) else flag))
        print(f"{label:18s}" + "".join(f"{c:>11s}" for c in cells), flush=True)

    print("\n=== FLAGGED CODE PATHS (estimator x scenario) ===")
    if not problems:
        print("  none -- all physical")
    for label, scn, msg in problems:
        print(f"  {label:16s} @ {scn:30s} {msg}")


if __name__ == "__main__":
    main()
