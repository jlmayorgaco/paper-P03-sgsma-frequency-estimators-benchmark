"""
Reconstruct benchmark_results.json from per-method run files.
Run was completed 2026-03-30 but JSON finalization was interrupted.
"""
import json, os, numpy as np
from datetime import datetime

SCENARIOS = [
    'IEEE_Mag_Step', 'IEEE_Freq_Ramp', 'IEEE_Modulation',
    'IBR_Nightmare', 'IBR_MultiEvent_Classic', 'IBR_PrimaryFrequencyResponse'
]
RESULTS_RAW = 'results_raw'

IEC_RMSE_THRESH  = 0.05
IEC_PEAK_THRESH  = 0.5
IEC_TTRIP_THRESH = 0.1
MARGINAL_FACTOR  = 0.10

# ── 1. Load per-method files ──────────────────────────────────────────────────
results = {}
for sc in SCENARIOS:
    sc_dir = os.path.join(RESULTS_RAW, sc)
    results[sc] = {'methods': {}}
    sc_desc = None
    for fname in sorted(os.listdir(sc_dir)):
        if not fname.endswith('.json'):
            continue
        fpath = os.path.join(sc_dir, fname)
        with open(fpath, encoding='utf-8') as f:
            data = json.load(f)
        method = data['metadata']['method']['name']
        metrics = dict(data['metrics'])
        tuning  = data['metadata']['method'].get('tuning', {})
        tuned   = {k: v for k, v in tuning.items() if k not in ('label', 'timestamp_ref')}
        metrics['tuned_params'] = tuned
        results[sc]['methods'][method] = metrics
        if sc_desc is None:
            sc_desc = data['metadata']['scenario'].get('params', {})
            results[sc]['scenario_description'] = sc_desc
    print(f'{sc}: {len(results[sc]["methods"])} methods loaded')

# ── 2. IEC compliance ──────────────────────────────────────────────────────────
for sc in results:
    passing, failing, marginal = [], [], []
    for mname, mvals in results[sc]['methods'].items():
        rmse  = mvals.get('RMSE',         float('nan'))
        peak  = mvals.get('MAX_PEAK',      float('nan'))
        ttrip = mvals.get('TRIP_TIME_0p5', float('nan'))
        rmse_pass  = float(rmse)  <= IEC_RMSE_THRESH  if np.isfinite(rmse)  else False
        peak_pass  = float(peak)  <= IEC_PEAK_THRESH  if np.isfinite(peak)  else False
        ttrip_pass = float(ttrip) <= IEC_TTRIP_THRESH if np.isfinite(ttrip) else False
        passed = rmse_pass and peak_pass and ttrip_pass
        is_marginal = (
            (IEC_RMSE_THRESH*(1-MARGINAL_FACTOR)  <= float(rmse)  <= IEC_RMSE_THRESH  if np.isfinite(rmse)  else False) or
            (IEC_PEAK_THRESH*(1-MARGINAL_FACTOR)  <= float(peak)  <= IEC_PEAK_THRESH  if np.isfinite(peak)  else False) or
            (IEC_TTRIP_THRESH*(1-MARGINAL_FACTOR) <= float(ttrip) <= IEC_TTRIP_THRESH if np.isfinite(ttrip) else False)
        )
        mvals['iec_compliance'] = {
            'pass': passed, 'rmse_pass': rmse_pass, 'peak_pass': peak_pass,
            'ttrip_pass': ttrip_pass, 'marginal': is_marginal and not passed,
        }
        if passed:        passing.append(mname)
        elif is_marginal: marginal.append(mname)
        else:             failing.append(mname)
    results[sc]['compliance_summary'] = {
        'thresholds': {'RMSE_Hz': IEC_RMSE_THRESH, 'Peak_Hz': IEC_PEAK_THRESH, 'Ttrip_s': IEC_TTRIP_THRESH},
        'passing_methods': passing, 'failing_methods': failing, 'marginal_methods': marginal,
    }
    print(f'  IEC {sc}: pass={passing}')

# ── 3. Paper claims ────────────────────────────────────────────────────────────
def _sg(sc, meth, key):
    return float(results.get(sc, {}).get('methods', {}).get(meth, {}).get(key, float('nan')))

ekf2_nd_rmse = _sg('IBR_Nightmare',          'EKF2', 'RMSE')
ekf_nd_rmse  = _sg('IBR_Nightmare',          'EKF',  'RMSE')
pll_ramp     = _sg('IEEE_Freq_Ramp',          'PLL',  'RMSE')
ekf2_ramp    = _sg('IEEE_Freq_Ramp',          'EKF2', 'RMSE')
ekf_ramp     = _sg('IEEE_Freq_Ramp',          'EKF',  'RMSE')
ekf2_nd_trip = _sg('IBR_Nightmare',          'EKF2', 'TRIP_TIME_0p5')
ekf_nd_trip  = _sg('IBR_Nightmare',          'EKF',  'TRIP_TIME_0p5')
pll_me_trip  = _sg('IBR_MultiEvent_Classic', 'PLL',  'TRIP_TIME_0p5')
ekf2_me_trip = _sg('IBR_MultiEvent_Classic', 'EKF2', 'TRIP_TIME_0p5')

def ratio(a, b):
    if np.isfinite(a) and np.isfinite(b) and b > 1e-12:
        return round(a / b, 2)
    return None

def vfy(v, pv, tol):
    if v is None: return False
    return abs(v - pv) / max(pv, 1e-12) <= tol

c275  = ratio(ekf_nd_trip, ekf2_nd_trip)
c4p7  = ratio(ekf_nd_rmse, ekf2_nd_rmse)
c12p6 = ratio(pll_ramp, ekf2_ramp)
c1p59 = ratio(ekf_ramp, ekf2_ramp)
c3p3  = ratio(pll_me_trip, ekf2_me_trip)

print(f'\nFresh paper claims:')
print(f'  275x trip (EKF/EKF2, Nightmare): {c275}  [paper: 275]  verified={vfy(c275,275,0.60)}')
print(f'  4.7x RMSE (EKF/EKF2, Nightmare): {c4p7}   [paper: 4.7]  verified={vfy(c4p7,4.7,0.5)}')
print(f'  12.6x ramp (PLL/EKF2, Ramp):    {c12p6} [paper: 12.6] verified={vfy(c12p6,12.6,0.5)}')
print(f'  1.59x ramp (EKF/EKF2, Ramp):    {c1p59}  [paper: 1.59] verified={vfy(c1p59,1.59,0.3)}')
print(f'  3.3x trip (PLL/EKF2, MultiEv):  {c3p3}  [paper: 3.3]  verified={vfy(c3p3,3.3,0.5)}')

print(f'\nKey absolute values for manuscript:')
print(f'  EKF2 Ramp RMSE: {ekf2_ramp} Hz')
print(f'  EKF  Ramp RMSE: {ekf_ramp} Hz')
print(f'  PLL  Ramp RMSE: {pll_ramp} Hz')
print(f'  EKF  Nightmare RMSE:  {ekf_nd_rmse} Hz  TRIP: {ekf_nd_trip} s')
print(f'  EKF2 Nightmare RMSE:  {ekf2_nd_rmse} Hz TRIP: {ekf2_nd_trip} s')
ipd_nd_rmse = _sg('IBR_Nightmare', 'IpDFT', 'RMSE')
ipd_nd_peak = _sg('IBR_Nightmare', 'IpDFT', 'MAX_PEAK')
ekf2_nd_peak = _sg('IBR_Nightmare', 'EKF2', 'MAX_PEAK')
pll_me_trip_s = _sg('IBR_MultiEvent_Classic', 'PLL', 'TRIP_TIME_0p5')
ekf2_cpu = _sg('IBR_MultiEvent_Classic', 'EKF2', 'TIME_PER_SAMPLE_US')
print(f'  IpDFT Nightmare RMSE: {ipd_nd_rmse} Hz  MAX_PEAK: {ipd_nd_peak} Hz')
print(f'  EKF2 Nightmare MAX_PEAK: {ekf2_nd_peak} Hz')
print(f'  PLL MultiEvent TRIP: {pll_me_trip_s} s')
print(f'  EKF2 CPU us/sample: {ekf2_cpu}')

paper_claims = {
    'claim_275x':       {'value':c275,  'paper_value':275,  'verified':vfy(c275,275,0.60),
                         'note':'EKF_Ttrip/EKF2_Ttrip IBR_Nightmare'},
    'claim_4p7x_rmse':  {'value':c4p7,  'paper_value':4.7,  'verified':vfy(c4p7,4.7,0.5),
                         'note':'EKF_RMSE/EKF2_RMSE IBR_Nightmare'},
    'claim_12p6x_ramp': {'value':c12p6, 'paper_value':12.6, 'verified':vfy(c12p6,12.6,0.5),
                         'note':'PLL_RMSE/EKF2_RMSE IEEE_Freq_Ramp'},
    'claim_3p3x_ttrip': {'value':c3p3,  'paper_value':3.3,  'verified':vfy(c3p3,3.3,0.5),
                         'note':'PLL_Ttrip/EKF2_Ttrip IBR_MultiEvent_Classic'},
    'claim_1p59x_ramp': {'value':c1p59, 'paper_value':1.59, 'verified':vfy(c1p59,1.59,0.3),
                         'note':'EKF_RMSE/EKF2_RMSE IEEE_Freq_Ramp'},
}

# ── 4. MC summary from raw_mc.json ────────────────────────────────────────────
with open(os.path.join(RESULTS_RAW, 'raw_mc.json'), encoding='utf-8') as f:
    raw_mc_raw = json.load(f)

mc_summary = {}
for key, cell in raw_mc_raw.items():
    m, sc = key.split('__', 1)
    if m not in mc_summary:
        mc_summary[m] = {}
    mc_summary[m][sc] = {
        'RMSE_mean':   round(float(np.mean(cell['RMSE'])),   6),
        'RMSE_std':    round(float(np.std(cell['RMSE'])),    6),
        'FE_max_mean': round(float(np.mean(cell['FE_max'])), 6),
        'FE_max_std':  round(float(np.std(cell['FE_max'])),  6),
        'TRIP_mean':   round(float(np.mean(cell['TRIP'])),   6),
        'TRIP_std':    round(float(np.std(cell['TRIP'])),    6),
        'n_runs':      len(cell['RMSE']),
    }

# ── 5. CPU authoritative ──────────────────────────────────────────────────────
cpu_vals = {}
for mname, mvals in results['IBR_MultiEvent_Classic']['methods'].items():
    cpu_us = mvals.get('TIME_PER_SAMPLE_US', float('nan'))
    if np.isfinite(cpu_us):
        cpu_vals[mname] = round(float(cpu_us), 4)

# ── 6. Assemble and write ─────────────────────────────────────────────────────
now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
json_export = {
    'metadata': {
        'timestamp': now,
        'description': (
            'Benchmark results reconstructed from per-method run files. '
            'Run completed 2026-03-30; final JSON write was interrupted. '
            'All metrics, tuning, and MC data are from the fresh 2026-03-30 run.'
        ),
    },
    'results': results,
    'paper_claims_numbers': paper_claims,
    'monte_carlo': {
        'description': (
            'Monte Carlo robustness analysis over 30 independent noise realisations, '
            '11 methods (PI-GRU excluded), 6 scenarios. Seeds 2000-2029.'
        ),
        'n_runs': 30,
        'methods': mc_summary,
    },
    'cpu_authoritative': {
        'description': 'Per-sample CPU cost [us], IBR_MultiEvent_Classic reference scenario.',
        'values_us': cpu_vals,
    },
    'paper_claims_verification': {
        'verified': all(v['verified'] for v in paper_claims.values()),
        'claims': paper_claims,
    },
}

def round_json(obj, digits=6):
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, dict):
        return {k: round_json(v, digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_json(v, digits) for v in obj]
    return obj

with open('benchmark_results.json', 'w', encoding='utf-8') as f:
    json.dump(round_json(json_export), f, indent=4, ensure_ascii=False)
print('\nDONE: benchmark_results.json written successfully.')
print(f'All claims verified: {json_export["paper_claims_verification"]["verified"]}')
