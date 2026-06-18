import sys, json
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import numpy as np, pandas as pd
from estimators.tkeo import TKEO_Estimator
from estimators.koopman import Koopman_Estimator

MC = REPO / "artifacts" / "full_mc_benchmark"
ES = 1500

def test(cls, name, scn):
    sc = pd.read_csv(MC / scn / f"{scn}_scenario.csv")
    t = sc.t_s.to_numpy(); v = sc.v_pu.to_numpy(); ft = sc.f_true_hz.to_numpy()
    try:
        bp = json.load(open(MC / scn / name / "run_spec.json", encoding="utf-8")).get("best_params", {})
    except Exception:
        bp = {}
    try:
        fhT = np.asarray(cls(**bp).estimate(t, v), float)
        fhD = np.asarray(cls().estimate(t, v), float)
        rT = np.sqrt(np.nanmean((fhT[ES:]-ft[ES:])**2)); rD = np.sqrt(np.nanmean((fhD[ES:]-ft[ES:])**2))
        print(f"{name[:7]:7s} @ {scn:24s} TUNED={rT:6.2f}[{np.nanmin(fhT[ES:]):.0f},{np.nanmax(fhT[ES:]):.0f}]  DEFAULT={rD:6.2f}[{np.nanmin(fhD[ES:]):.0f},{np.nanmax(fhD[ES:]):.0f}]", flush=True)
    except Exception as e:
        print(f"{name} @ {scn} ERROR {e}", flush=True)

test(TKEO_Estimator, "TKEO", "IBR_Multi_Event")
test(TKEO_Estimator, "TKEO", "IEEE_Freq_Ramp_10Hzs")
test(Koopman_Estimator, "Koopman (RK-DPMU)", "IBR_Power_Imbalance_Ringdown")
test(Koopman_Estimator, "Koopman (RK-DPMU)", "IEEE_Freq_Ramp_10Hzs")
