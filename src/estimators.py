#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from collections import deque
import math

from ekf2 import EKF2  # Asegúrate de tener ekf2.py en el mismo directorio

# Intento opcional de importar PyTorch para el estimador PI-GRU
try:
    import torch
except ImportError:  # pragma: no cover - entorno sin torch
    torch = None

# =============================================================
# 0. CONFIGURACIÓN GLOBAL (SOLO NÚMEROS, SIN PLOTTING)
# =============================================================
SEED = 42
np.random.seed(SEED)

# Block F: Bayesian tuning flag
# Set True to replace grid search with LHS + differential_evolution in tune_* functions.
# Set False to use original grid search (reproducible, faster for small grids).
USE_BAYESIAN_TUNING = False

FS_PHYSICS = 1000000.0   # 1 MHz (Physics/Ground Truth)
FS_DSP = 10000.0         # 10 kHz (IED/Relay Sampling Rate)
RATIO = int(FS_PHYSICS / FS_DSP)
DT_DSP = 1.0 / FS_DSP
F_NOM = 60.0

# Umbrales de error para métricas
SETTLING_THRESHOLD = 0.2   # Hz
TRIP_THRESHOLD = 0.5       # Hz


# =============================================================
# 1. SCENARIO GENERATOR (With IEEE + IBR Multi-Event)
# =============================================================
def get_test_signals(seed=None):
    # REPRO-1: Accept seed parameter for reproducible signal generation.
    # np.random.seed() is called at module level (SEED=42) but re-calling here
    # ensures each MC run that passes a distinct seed gets independent noise.
    if seed is not None:
        np.random.seed(seed)
    # Escenarios cortos (1.5 s)
    t = np.arange(0, 1.5, 1.0 / FS_PHYSICS)
    n = len(t)
    signals = {}
    
    # --- A: VOLTAGE MAGNITUDE STEP (IEEE Std 60255-118-1 / C37.118.1-2011) -------
    # Physical motivation
    # -------------------
    # Models an abrupt voltage sag or swell at an IBR plant terminal caused by
    # sudden load connection, capacitor bank switching, or fast MPPT power injection
    # from a grid-following inverter (typical step time < 1 ms).  The frequency
    # remains constant at 60 Hz; only the amplitude envelope is perturbed.
    # This isolates amplitude-to-frequency cross-coupling in estimators: any estimator
    # whose frequency estimate deviates at t=0.5 s has a known systematic error that
    # must be reported per IEC 60255-118-1 §5.3.4.
    # -------------------------------------------------------------------------
    f_a = np.ones(n) * 60.0
    amp_a = np.ones(n)
    amp_a[t > 0.5] = 1.1  # +10% amplitude step (unit-amplitude baseline)
    phi_a = 2.0 * np.pi * 60.0 * t   # analytic phase: exact for constant frequency
    v_a = amp_a * np.sin(phi_a) + np.random.normal(0, 0.001, n)

    meta_a = {
        "description": "Scenario A — IEEE Voltage Magnitude Step (+10% at t=0.5s)",
        "standard": (
            "IEEE Std 60255-118-1:2018 §5.3.4 amplitude-step test; "
            "IEEE C37.118.1-2011 §5.4.2 (PMU compliance, magnitude transient)."
        ),
        "physical_motivation": (
            "Represents abrupt voltage sag/swell from fast IBR power injection or "
            "capacitor switching in a distribution-tied PV/wind grid. Frequency is "
            "held constant to isolate amplitude-frequency cross-coupling errors."
        ),
        "ibr_relevance": (
            "Grid-following inverters (GFL-IBR) inject current at near-unity PF "
            "with sub-millisecond amplitude transients on connect/disconnect. "
            "Compliant with IEEE 1547-2018 Category I/II/III amplitude limits."
        ),
        "parameters": "+10% amplitude step at t=0.5 s; f = 60.0 Hz (constant).",
        "harmonics": "None (pure 60 Hz fundamental + noise).",
        "noise": "Gaussian white noise sigma=0.001 pu (SNR ≈ 57 dB at 1 pu amplitude).",
        "noise_sigma": 0.001,
        "dynamics": "Piecewise constant amplitude: 1.0 pu → 1.1 pu at t=0.5 s. Phase C0-continuous.",
        "exceeds_standard": (
            "This is a compliance-level baseline test (not a beyond-standard stress). "
            "Serves as a reference point for amplitude-immunity benchmarking."
        ),
    }
    signals["IEEE_Mag_Step"] = (t, v_a, f_a, meta_a)

    # --- B: FREQUENCY RAMP (IEEE 1547-2018 Category III / IEC 60255-118-1) --------
    # Physical motivation
    # -------------------
    # Models sustained frequency deviation from a generation-loss event in a
    # low-inertia IBR-dominated grid.  Reduced system inertia (H < 2 s typical of
    # IBR-heavy grids, vs H = 4–6 s in synchronous-machine-dominated grids) produces
    # RoCoF values at the IEEE 1547-2018 Category III limit of ±3.0 Hz/s.
    # The ramp is clipped at the upper ride-through boundary (61.5 Hz) and held there
    # for the remainder of the evaluation window, simulating a post-fault frequency
    # arrest before governor/droop response takes effect.
    # Key discriminator: estimators must track a sustained linear RoCoF without lag,
    # bias, or catastrophic RLS/PLL filter windup.  EKF2 (with ω̇ state) has structural
    # advantage; window-based methods (IpDFT, TFT) have structural latency.
    # -------------------------------------------------------------------------
    f_b = np.ones(n) * 60.0
    mask_ramp = (t > 0.3) & (t < 1.0)
    # PY-02: IEEE 1547-2018 Cat-III RoCoF = +3.0 Hz/s; ramp capped at 61.5 Hz
    f_b[mask_ramp] = 60.0 + 3.0 * (t[mask_ramp] - 0.3)
    f_b = np.clip(f_b, 58.5, 61.5)          # IEEE 1547-2018 Category III ride-through limits
    f_b[t >= 1.0] = f_b[t < 1.0][-1]        # hold at 61.5 Hz after cap (ramp-and-hold)
    phi_b = np.cumsum(2.0 * np.pi * f_b * (1.0 / FS_PHYSICS))
    v_b = np.sin(phi_b) + np.random.normal(0, 0.001, n)

    # Ramp-and-hold breakpoints (derived, for meta accuracy)
    _f_b_cap = float(f_b[t >= 1.0][0])               # 61.5 Hz (clipped)
    _t_b_cap = 0.3 + (_f_b_cap - 60.0) / 3.0         # t when cap reached (0.8 s)

    meta_b = {
        "description": "Scenario B — IEEE 1547-2018 Cat-III Frequency Ramp +3 Hz/s (Ramp-and-Hold)",
        "standard": (
            "IEEE 1547-2018 Table 2 Category III (RoCoF = ±3.0 Hz/s, limits 61.5/58.5 Hz); "
            "IEC 60255-118-1:2018 §5.3.3 (frequency ramp test, df/dt up to 4 Hz/s)."
        ),
        "physical_motivation": (
            "Low-inertia IBR-dominated grids routinely exhibit 2–5 Hz/s RoCoF "
            "post-fault (NERC GADS 2019-2023). This scenario applies the maximum "
            "IEEE-standardised ramp rate to expose tracking lag in all estimator families."
        ),
        "ibr_relevance": (
            "IBR plants reduce system inertia; RoCoF > 0.5 Hz/s was rare in "
            "synchronous-machine grids but is now routinely observed in high-IBR "
            "systems (Ireland, GB, parts of ERCOT). EKF2's ω̇ augmentation provides "
            "structural advantage over static-frequency estimators."
        ),
        "parameters": (
            f"+3.0 Hz/s ramp from t=0.3 s; cap at 61.5 Hz reached at t={_t_b_cap:.2f} s; "
            f"hold at {_f_b_cap:.1f} Hz for t > {_t_b_cap:.2f} s."
        ),
        "harmonics": "None (pure fundamental + noise; no harmonic content to isolate RoCoF tracking).",
        "noise": "Gaussian white noise sigma=0.001 pu.",
        "noise_sigma": 0.001,
        "dynamics": (
            f"Piecewise: 60 Hz nominal → +3.0 Hz/s linear ramp (t=0.3–{_t_b_cap:.2f} s) "
            f"→ hold at {_f_b_cap:.1f} Hz (IEEE Cat-III upper ceiling). "
            "Phase: continuous cumulative integral of f(t). "
            "Maximum instantaneous RoCoF = 3.0 Hz/s."
        ),
        "exceeds_standard": (
            "RoCoF = 3.0 Hz/s is the IEEE 1547-2018 Cat-III boundary. IBR grids "
            "with >60% IBR penetration report 2-3× higher RoCoF in field incidents. "
            "This scenario is a compliance-boundary test; Scenarios D-F exceed it."
        ),
    }
    signals["IEEE_Freq_Ramp"] = (t, v_b, f_b, meta_b)

    # --- C: AM SIDEBAND / LOW-FREQUENCY OSCILLATION TEST (IEEE 60255-118-1 §5.3.5) -
    # Physical motivation
    # -------------------
    # Models lightly-damped inter-area electromechanical oscillations (0.1–3 Hz)
    # observed in transmission corridors with high IBR penetration.  These modes
    # are excited when grid-following inverters lack adequate power-oscillation
    # damping (POD) or virtual inertia.  A 10% amplitude modulation at 2 Hz
    # creates spectral sidebands at 58 Hz and 62 Hz.  The true frequency is
    # constant (60 Hz); estimators that confuse AM with FM will show a spurious
    # 2 Hz frequency oscillation with amplitude ≈ kx×fm = 0.2 Hz (ROCOF cross-
    # coupling artefact — well-documented for SRF-PLL and SOGI-FLL).
    #
    # Note on harmonics vs sidebands:
    #   AM sidebands (58, 62 Hz) are NOT harmonics (integer multiples of 60 Hz)
    #   and NOT interharmonics by IEEE 519-2022 definition (which requires non-
    #   integer multiples from an independent asynchronous source).  They are
    #   amplitude-modulation products of a single 60 Hz source; THD = 0%.
    # -------------------------------------------------------------------------
    fm  = 2.0      # modulation frequency [Hz]
    kx  = 0.1      # AM depth = 10%
    f_c = np.ones(n) * 60.0
    mod_signal = 1.0 + kx * np.cos(2.0 * np.pi * fm * t)
    v_c = mod_signal * np.sin(2.0 * np.pi * 60.0 * t) + np.random.normal(0, 0.001, n)

    meta_c = {
        "description": "Scenario C — AM Sideband / LFO Oscillation Immunity Test (10% depth, 2 Hz)",
        "standard": (
            "IEEE Std 60255-118-1:2018 §5.3.5 (AM modulation test, kx=0.1); "
            "IEEE C37.118.1-2011 §5.4.3; IEC 61000-4-7 (modulation products)."
        ),
        "physical_motivation": (
            "Lightly-damped inter-area electromechanical oscillations at 0.1–3 Hz "
            "are characteristic of high-IBR transmission corridors (WECC, ENTSO-E "
            "system studies). IBR plants without POD controls excite these modes."
        ),
        "ibr_relevance": (
            "GFL-IBR with inadequate virtual inertia or POD can excite or sustain "
            "inter-area oscillations. Amplitude modulation at 2 Hz represents a "
            "lightly-damped (ζ~0.05) electromechanical mode — amplitude at the "
            "stability boundary before oscillatory instability."
        ),
        "parameters": (
            f"v(t) = (1 + {kx}·cos(2π·{fm}·t))·sin(2π·60·t); "
            f"AM depth kx={kx} (10%), modulation f_m={fm} Hz. "
            "True frequency: constant 60.0 Hz."
        ),
        "harmonics": (
            "AM sidebands at 58 Hz and 62 Hz (±f_m = ±2 Hz about carrier). "
            "Per IEEE 519-2022: THD = 0% (sidebands are not integer harmonics). "
            "Not interharmonics (modulation product, not asynchronous source)."
        ),
        "noise": "Gaussian white noise sigma=0.001 pu.",
        "noise_sigma": 0.001,
        "dynamics": (
            "Constant 60 Hz frequency; sinusoidal amplitude envelope. "
            "Estimators with AM-FM coupling produce spurious ~0.2 Hz peak frequency "
            "oscillation (SOGI-FLL, SRF-PLL) — the primary discrimination mechanism."
        ),
        "exceeds_standard": (
            "kx=0.1 is the IEC 60255-118-1 §5.3.5 standard test point. "
            "Pre-cascade IBR oscillation events (e.g., WECC 2020) exhibited modulation "
            "depths of 15–25% prior to instability — 1.5–2.5× the test level."
        ),
    }
    signals["IEEE_Modulation"] = (t, v_c, f_c, meta_c)

    # --- D: IBR ISLANDING / COMPOSITE HARMONIC STRESS — BEYOND-STANDARD -----------
    # Physical motivation
    # -------------------
    # Models the worst single-point IBR islanding event: an unintended island forms
    # when the main grid breaker opens, producing an instantaneous +60° phase step
    # at the local bus (fault angle at clearing).  This exceeds IEEE 1547-2018
    # Table 14 anti-islanding requirements (≤20° phase excursion before trip) by
    # 3× and represents the maximum-stress phase-discontinuity test in this suite.
    # The power-electronics switching environment is modelled by 5th/7th harmonics
    # (inverter output filter switching artefacts, THD ≈ 4.47%, within IEEE 519-2022
    # ≤5% PCC limit) and a 32.5 Hz interharmonic (asynchronous cycloconverter or
    # wind-turbine rotor-frequency component per IEC 61000-2-2).
    # Elevated noise floor (5× Scenarios A-C) represents CT/VT uncertainty under
    # high-frequency switching in an IBR substation.
    #
    # Signal construction (vectorised — NOT a sample loop):
    #   f(t) = 60.0 Hz (constant before and after jump)
    #   phi(t) = cumsum(2π·f·dt), then +60° added at the single jump sample
    #   Phase jump applied at exactly one sample via np.searchsorted — this is
    #   the only correct implementation; a window condition (e.g., 0.6999 < t < 0.7001)
    #   at 1 MHz would fire 200 times and corrupt the signal.
    # -------------------------------------------------------------------------
    f_d = np.ones(n) * 60.0
    phi_d = np.empty(n)
    phi_d[0] = 0.0
    phi_d[1:] = np.cumsum(2.0 * np.pi * f_d[1:] * (1.0 / FS_PHYSICS))
    # Exact +60 deg phase discontinuity at t=0.7 s — single sample, not a window
    _jidx_d = int(np.searchsorted(t, 0.7))
    phi_d[_jidx_d:] += 60.0 * np.pi / 180.0   # +π/3 rad added once

    # Voltage: fundamental + harmonics + asynchronous interharmonic + noise
    v_d  = np.sin(phi_d)
    # IEEE 519-2022 compliant: THD = sqrt(0.04²+0.02²) ≈ 4.47% ≤ 5% limit at PCC
    v_d += 0.04  * np.sin(5.0 * phi_d)                    # 5th harmonic (4%)
    v_d += 0.02  * np.sin(7.0 * phi_d)                    # 7th harmonic (2%)
    v_d += 0.005 * np.sin(2.0 * np.pi * 32.5 * t)         # interharmonic 32.5 Hz (0.5%)
    # Elevated noise floor: 5× Scenarios A-C (IBR substation CT/VT uncertainty)
    v_d += np.random.normal(0, 0.005, n)

    meta_d = {
        "description": "Scenario D — IBR Islanding: Instantaneous +60° Phase Jump + Composite Harmonics",
        "standard": (
            "Intentionally exceeds IEEE 1547-2018 Table 14 anti-islanding phase limit "
            "(≤20°) by 3×. Harmonics: IEEE 519-2022 compliant (THD ≤5% at PCC). "
            "Interharmonic: IEC 61000-2-2 (asynchronous source). "
            "Phase-step test level 6× IEC 60255-118-1 §5.3.6 nominal (±10°)."
        ),
        "physical_motivation": (
            "IBR islanding event: main-grid breaker opens, local island retains 60 Hz "
            "generation. The fault-clearing angle at breaker opening appears as an "
            "instantaneous phase step at the relay terminal. 60° is drawn from worst-case "
            "field incidents (NERC GADS 2019–2023 IBR fault-ride-through reports)."
        ),
        "ibr_relevance": (
            "Frequency estimation is critically impaired at phase-jump instants: "
            "EKF/UKF state vectors diverge proportional to jump magnitude; PLL "
            "cycle-slips for jumps > ~30°; IpDFT windows straddle the discontinuity "
            "producing ±1 Hz spectral-leakage artefact (visible as MAX_PEAK in results). "
            "This scenario drives the headline RMSE and T_trip discrimination claims."
        ),
        "parameters": (
            "Instantaneous +60° (π/3 rad) phase jump at t=0.7 s. "
            "Frequency: constant 60.0 Hz before and after. "
            "Phase applied at a single sample (searchsorted, not a time window)."
        ),
        "harmonics": (
            "5th harmonic (4%), 7th harmonic (2%): "
            "THD = sqrt(0.04²+0.02²) = 4.47% ≤ IEEE 519-2022 5% PCC limit. "
            "Interharmonic at 32.5 Hz (0.5%): asynchronous source (cycloconverter / "
            "wind rotor frequency); excluded from THD per IEEE 519-2022 definition."
        ),
        "noise": "Gaussian noise sigma=0.005 pu (5× Scenarios A–C; IBR substation elevated noise floor).",
        "noise_sigma": 0.005,
        "dynamics": (
            "Piecewise constant frequency (60 Hz); instantaneous C0-discontinuous phase "
            "step at t=0.7 s. Frequency estimators observe an apparent Dirac-like "
            "frequency spike of magnitude Δφ/(2π·dt_DSP) ≈ 0.5/DT_DSP_cycles Hz "
            "at the jump sample — the primary estimator-discrimination event."
        ),
        "exceeds_standard": (
            "Phase jump 60° > 3× IEEE 1547-2018 anti-islanding detection threshold (20°). "
            "Noise floor 5× IEC 60255-118-1 standard test level. "
            "Harmonic environment at IEEE 519-2022 PCC limit (not the tighter equipment "
            "limit). Designed as a maximum-discrimination beyond-standard stress test."
        ),
    }
    signals["IBR_Nightmare"] = (t, v_d, f_d, meta_d)

    # --- E: IBR MULTI-EVENT CLASSIC  (Composite Stress Benchmark, 5 s) ----------
    # -------------------------------------------------------------------------
    # Physical motivation
    # -------------------
    # Models the worst-case composite IBR-grid stress event as described in
    # Sections III-E of the paper: a harmonic-rich steady state followed by a
    # +40 deg phase jump, a sustained -3.0 Hz/s RoCoF segment that drives
    # frequency to the IEEE 1547-2018 UFLS limit of 57.5 Hz, a second +80 deg
    # phase jump at the nadir, and a second-order underdamped ring-down recovery.
    # This sequence is physically motivated by cascaded switching events in a
    # low-inertia IBR-dominated grid during a generation-loss incident.
    #
    # Frequency trajectory (piecewise):
    #   Seg 1 [0, T_J1=1.0 s]:   f = 60.0 Hz (nominal)
    #   Seg 2 [T_J1, T_NAD]:     f = 60 + ROCOF*(t - T_J1)   ROCOF=-3.0 Hz/s
    #                              T_NAD = T_J1 + (F_NADIR-60)/ROCOF = 1.833 s
    #   Seg 3 [T_NAD, 5.0 s]:    f = 60 - 2.5*exp(-0.8*tau)*cos(2*pi*0.6*tau)
    #                              tau = t - T_NAD  (ring-down recovery)
    #
    # Phase discontinuities:
    #   +40 deg at t = T_J1 = 1.0 s   (breaker opening)
    #   +80 deg at t = T_NAD = 1.833 s (re-energisation at nadir)
    #
    # Distortion/noise (same as IBR_Nightmare for fair comparison):
    #   5th harmonic 4%, 7th harmonic 2%, interharmonic 32.5 Hz 0.5%
    #   Gaussian noise sigma=0.003 pu + impulsive sigma=0.05 pu (p=5e-4)
    # -------------------------------------------------------------------------

    T_E_DUR_CL = 5.0
    t_ecl  = np.arange(0, T_E_DUR_CL, 1.0 / FS_PHYSICS)
    n_ecl  = len(t_ecl)

    # Key time instants
    _T_J1_CL  = 1.000                        # +40 deg phase jump [s]
    _ROCOF_CL = -3.0                          # Hz/s
    _F_NAD_CL = 57.5                          # nadir frequency [Hz]
    _T_NAD_CL = _T_J1_CL + (_F_NAD_CL - F_NOM) / _ROCOF_CL   # = 1.8333 s
    # Ring-down parameters (underdamped, zeta=0.08, f_n~0.63 Hz)
    _ALPHA_RD = 0.8                           # decay rate [s^-1]
    _F_OSC_RD = 0.6                           # oscillation frequency [Hz]
    _AMP_RD   = 2.5                           # ring-down amplitude [Hz]

    # --- Build piecewise frequency trajectory ---
    f_ecl = np.empty(n_ecl)
    # Segment 1: nominal
    seg1 = t_ecl < _T_J1_CL
    f_ecl[seg1] = F_NOM
    # Segment 2: RoCoF ramp
    seg2 = (t_ecl >= _T_J1_CL) & (t_ecl < _T_NAD_CL)
    f_ecl[seg2] = F_NOM + _ROCOF_CL * (t_ecl[seg2] - _T_J1_CL)
    # Segment 3: ring-down recovery
    # NOTE: The junction at T_NAD is C0-continuous (f matches: both = F_NAD_CL = 57.5 Hz)
    # but C1-discontinuous: RoCoF jumps from -3.0 Hz/s (seg2 slope) to +2.0 Hz/s
    # (d/dtau[60-2.5*exp(-0.8*0)*cos(0)] = 2.5*0.8 = 2.0 Hz/s).
    # This 5 Hz/s apparent RoCoF spike is physically justified: the +80 deg phase jump
    # at T_NAD models re-energization (breaker reclosing), which instantaneously reverses
    # the frequency trajectory. Estimators that cannot handle phase jumps will exhibit
    # large transient errors at this junction — this is the intended discrimination event.
    seg3 = t_ecl >= _T_NAD_CL
    tau_rd = t_ecl[seg3] - _T_NAD_CL
    f_ecl[seg3] = F_NOM - _AMP_RD * np.exp(-_ALPHA_RD * tau_rd) * np.cos(
        2.0 * np.pi * _F_OSC_RD * tau_rd)

    # --- Phase accumulation ---
    phi_ecl = np.empty(n_ecl)
    phi_ecl[0] = 0.0
    phi_ecl[1:] = np.cumsum(2.0 * np.pi * f_ecl[1:] * (1.0 / FS_PHYSICS))
    # Phase jump +40 deg at T_J1
    _jidx1_cl = int(np.searchsorted(t_ecl, _T_J1_CL))
    phi_ecl[_jidx1_cl:] += 40.0 * np.pi / 180.0
    # Phase jump +80 deg at T_NAD
    _jidx2_cl = int(np.searchsorted(t_ecl, _T_NAD_CL))
    phi_ecl[_jidx2_cl:] += 80.0 * np.pi / 180.0

    # --- Voltage: fundamental + harmonics + interharmonic + noise ---
    v_ecl  = np.sin(phi_ecl)
    v_ecl += 0.04  * np.sin(5.0 * phi_ecl)
    v_ecl += 0.02  * np.sin(7.0 * phi_ecl)
    v_ecl += 0.005 * np.sin(2.0 * np.pi * 32.5 * t_ecl)
    v_ecl += np.random.normal(0, 0.003, n_ecl)
    _imp_cl = np.random.rand(n_ecl) < 5.0e-4
    v_ecl += _imp_cl * np.random.normal(0, 0.05, n_ecl)

    meta_ecl = {
        "description": (
            "Scenario E — IBR Multi-Event Classic: Cascaded Generation-Loss Stress "
            "(Phase Jumps + Sustained RoCoF + Underdamped Recovery, 5 s)"
        ),
        "standard": (
            "Intentionally exceeds multiple IEEE/IEC limits simultaneously: "
            "RoCoF = 3.0 Hz/s at IEEE 1547-2018 Cat-III boundary; "
            "nadir 57.5 Hz at IEEE 1547-2018 UFLS imminent-trip threshold; "
            "phase jumps 40° and 80° exceed anti-islanding threshold (20°) by 2× and 4×; "
            "harmonic environment: IEEE 519-2022 PCC limit (THD ≈ 4.47%); "
            "impulsive noise: beyond IEC 60255-118-1 test conditions."
        ),
        "physical_motivation": (
            "Physically motivated composite IBR-grid event: a large generation-loss "
            "incident (e.g., offshore wind farm disconnection) causes a +40° phase "
            "transient at the bus, followed by sustained RoCoF under low inertia until "
            "the UFLS relay nadir (57.5 Hz). A second switching event (+80° at nadir) "
            "models re-energization or breaker reclosing at worst-case angle, producing "
            "underdamped inter-area ring-down.  This sequence is derived from composite "
            "cascaded events recorded in NERC GADS 2020–2023 IBR fault reports."
        ),
        "ibr_relevance": (
            "In IBR-heavy grids (H_sys < 2 s), the 3.0 Hz/s RoCoF is achievable within "
            "1–2 s of a major generation loss. The double phase jump stresses estimators "
            "that reset on phase events: a first reset at T_J1 is followed immediately "
            "by a second at T_NAD (0.833 s later), before most filter transients settle. "
            "The underdamped ring-down (zeta≈0.08, f_n≈0.63 Hz) excites the same "
            "frequency band as low-frequency oscillation modes, compounding the stress."
        ),
        "character": (
            "Maximum-discrimination composite stress scenario. All four disturbance "
            "types present simultaneously: phase jump, RoCoF, frequency nadir, and "
            "electromechanical ring-down. No IEEE/IEC compliance test combines all four."
        ),
        "parameters": (
            f"[0–{_T_J1_CL} s] nominal {F_NOM} Hz + 5th/7th harmonics + 32.5 Hz; "
            f"+40° jump at t={_T_J1_CL} s (breaker opening); "
            f"[{_T_J1_CL}–{_T_NAD_CL:.3f} s] RoCoF={_ROCOF_CL} Hz/s ramp to nadir {_F_NAD_CL} Hz; "
            f"+80° jump at t={_T_NAD_CL:.3f} s (re-energisation); "
            f"[{_T_NAD_CL:.3f}–5.0 s] ring-down: "
            f"f(τ) = 60 − {_AMP_RD}·exp(−{_ALPHA_RD}τ)·cos(2π·{_F_OSC_RD}τ), "
            f"τ = t − {_T_NAD_CL:.3f} s."
        ),
        "harmonics": (
            "5th harmonic (4%), 7th harmonic (2%): "
            "THD = sqrt(0.04²+0.02²) = 4.47% ≤ IEEE 519-2022 5% PCC limit. "
            "Interharmonic 32.5 Hz (0.5%): asynchronous source (excluded from THD). "
            "Same harmonic environment as Scenario D to isolate dynamic-vs-phase-jump effects."
        ),
        "noise": (
            "Gaussian background sigma=0.003 pu + "
            "impulsive sigma=0.05 pu (Bernoulli p=5e-4/sample). "
            "Impulsive component models IGBT gate-drive noise and CT saturation spikes."
        ),
        "noise_sigma": 0.003,  # base Gaussian sigma for SNR calculation (statistical_analysis.py)
        "dynamics": (
            f"Piecewise: nominal → RoCoF={_ROCOF_CL} Hz/s ramp → nadir {_F_NAD_CL} Hz "
            f"→ underdamped ring-down (α={_ALPHA_RD} s⁻¹, f_osc={_F_OSC_RD} Hz, A={_AMP_RD} Hz); "
            f"+40° at t={_T_J1_CL} s, +80° at t={_T_NAD_CL:.3f} s. "
            "C0-continuous at T_NAD (f matches); C1-discontinuous (RoCoF jumps "
            f"−3.0→+{_AMP_RD*_ALPHA_RD:.1f} Hz/s — physically justified by re-energisation event)."
        ),
        "exceeds_standard": (
            f"Phase jumps +40°/+80° exceed IEEE 1547-2018 anti-islanding threshold (20°) "
            "by 2× and 4×. RoCoF at Cat-III boundary. Nadir at UFLS threshold. "
            "Impulsive noise beyond IEC 60255-118-1. "
            "No IEEE/IEC compliance test combines all four stressors — this is by design."
        ),
    }
    signals["IBR_MultiEvent_Classic"] = (t_ecl, v_ecl, f_ecl, meta_ecl)

    # Backward-compatibility alias: IBR_MultiEvent -> IBR_MultiEvent_Classic.
    # Scripts that still reference the old key continue to work after fresh run.
    signals["IBR_MultiEvent"] = signals["IBR_MultiEvent_Classic"]

    # --- F: IBR PRIMARY FREQUENCY RESPONSE  (Control-Response Benchmark, 5 s) --
    # -------------------------------------------------------------------------
    # Physical motivation
    # -------------------
    # Models the frequency dynamics of a low-inertia IBR-dominated grid after a
    # sudden generation-loss event, plus a phase-angle disturbance representative
    # of a breaker-reclosing event during the recovery phase:
    #
    #   (i)   Pre-disturbance: nominal 60 Hz + harmonic-rich waveform.
    #   (ii)  Inertial dip: fast frequency drop as grid inertia absorbs imbalance
    #         (80 ms, nadir -0.43 Hz below nominal -> 59.57 Hz).
    #   (iii) Governor overshoot: primary controllers over-inject power,
    #         frequency transiently exceeds nominal (80 ms, +0.37 Hz -> 60.37 Hz).
    #   (iv)  Secondary control dip: droop and inter-area response produce a
    #         slower broader secondary dip (500 ms, -0.28 Hz -> 59.72 Hz).
    #   (v)   AGC recovery: drives frequency back toward nominal (300 ms).
    #   (vi)  Underdamped settling: lightly-damped (zeta=0.08) electromechanical
    #         oscillation at 0.8 Hz, visible for ~2.4 cycles over the 3 s tail.
    #   (vii) Phase jump: +40 deg at t=2.5 s (breaker reclosing), stressing
    #         phase-tracking estimators independently of the frequency transient.
    #
    # Mathematical construction
    # -------------------------
    # Segments (ii)-(v): cosine smoothstep, C1-smooth at every boundary.
    # Segment (vi): two-term Prony settling, analytically C1-matched at junction:
    #   f(tau) = f_nom + C_DC*exp(-ALPHA_DC*tau) + C_OSC*exp(-ALPHA_OSC*tau)*cos(omega_d*tau)
    #   C0 constraint: C_DC + C_OSC = F_REC - f_nom
    #   C1 constraint: -ALPHA_DC*C_DC - ALPHA_OSC*C_OSC = 0  (zero slope)
    #   => C_OSC = (F_REC-f_nom)*ALPHA_DC / (ALPHA_DC - ALPHA_OSC)
    #      C_DC  = (F_REC-f_nom) - C_OSC
    # Phase jump (vii): offset added to accumulated phase array at jump sample.
    # Harmonics: 5th (4%), 7th (2%), interharmonic 32.5 Hz (0.5%) [IEEE 519].
    # Noise: Gaussian background sigma=0.003 pu + sparse impulsive sigma=0.05 pu.
    # -------------------------------------------------------------------------

    t_e = np.arange(0, 5.0, 1.0 / FS_PHYSICS)
    n_e = len(t_e)

    f_nom_e = F_NOM   # 60.0 Hz

    # Segment boundary times [s]
    T_FLAT_END = 1.00   # end of pre-disturbance flat nominal
    T_DIP_END  = 1.08   # end of fast inertial dip          (80 ms)
    T_OVR_END  = 1.16   # end of governor overshoot         (80 ms)
    T_SD_END   = 1.66   # end of secondary control dip      (500 ms)
    T_REC_END  = 1.96   # end of AGC recovery               (300 ms)
    #              t > T_REC_END : underdamped settling (~3.04 s)

    # Frequency setpoints at each breakpoint
    F_DIP = f_nom_e - 0.43   # 59.57 Hz  inertial dip nadir
    F_OVR = f_nom_e + 0.37   # 60.37 Hz  governor overshoot peak
    F_SD  = f_nom_e - 0.28   # 59.72 Hz  secondary dip nadir
    F_REC = f_nom_e + 0.01   # 60.01 Hz  near-nominal post-AGC

    # Underdamped Prony settling (C1-matched at T_REC_END)
    # f(tau) = f_nom + C_DC*exp(-ALPHA_DC*tau) + C_OSC*exp(-ALPHA_OSC*tau)*cos(OMEGA_D*tau)
    ZETA_S    = 0.08
    F_N_S     = 0.80                                   # natural frequency [Hz]
    OMEGA_N_S = 2.0 * np.pi * F_N_S                   # 5.027 rad/s
    ALPHA_OSC = ZETA_S * OMEGA_N_S                    # 0.402 s^-1
    OMEGA_D   = OMEGA_N_S * np.sqrt(1.0 - ZETA_S**2) # 5.010 rad/s (damped)
    ALPHA_DC  = 0.50                                   # DC decay rate [s^-1]
    _DELTA_S  = F_REC - f_nom_e                       # +0.01 Hz
    # Solve the 2x2 C0/C1 system analytically:
    C_OSC = _DELTA_S * ALPHA_DC / (ALPHA_DC - ALPHA_OSC)  # +0.0511 Hz amplitude
    C_DC  = _DELTA_S - C_OSC                              # -0.0411 Hz

    # C1 cosine-smoothstep helper (vectorised, no loop)
    def _cstp(ta, t0, t1, fa, fb):
        tau = np.clip((ta - t0) / (t1 - t0), 0.0, 1.0)
        return fa + (fb - fa) * 0.5 * (1.0 - np.cos(np.pi * tau))

    # Assemble ground-truth frequency trajectory
    f_e = np.empty(n_e)
    s_flat   = t_e <= T_FLAT_END
    s_dip    = (t_e > T_FLAT_END) & (t_e <= T_DIP_END)
    s_ovr    = (t_e > T_DIP_END)  & (t_e <= T_OVR_END)
    s_sd     = (t_e > T_OVR_END)  & (t_e <= T_SD_END)
    s_rec    = (t_e > T_SD_END)   & (t_e <= T_REC_END)
    s_settle = t_e > T_REC_END

    f_e[s_flat]  = f_nom_e
    f_e[s_dip]   = _cstp(t_e[s_dip],  T_FLAT_END, T_DIP_END, f_nom_e, F_DIP)
    f_e[s_ovr]   = _cstp(t_e[s_ovr],  T_DIP_END,  T_OVR_END, F_DIP,   F_OVR)
    f_e[s_sd]    = _cstp(t_e[s_sd],   T_OVR_END,  T_SD_END,  F_OVR,   F_SD)
    f_e[s_rec]   = _cstp(t_e[s_rec],  T_SD_END,   T_REC_END, F_SD,    F_REC)

    _tau_s = t_e[s_settle] - T_REC_END
    f_e[s_settle] = (f_nom_e
                     + C_DC  * np.exp(-ALPHA_DC  * _tau_s)
                     + C_OSC * np.exp(-ALPHA_OSC * _tau_s)
                              * np.cos(OMEGA_D * _tau_s))

    # Phase accumulation: vectorised cumulative integral
    phi_e = np.empty(n_e)
    phi_e[0] = 0.0
    phi_e[1:] = np.cumsum(2.0 * np.pi * f_e[1:] * (1.0 / FS_PHYSICS))

    # Phase jump: +40 deg at t=2.5 s (breaker reclosing during recovery)
    PHASE_JUMP_TIME = 2.5
    PHASE_JUMP_RAD  = 40.0 * np.pi / 180.0
    _jump_idx = int(np.searchsorted(t_e, PHASE_JUMP_TIME))
    phi_e[_jump_idx:] += PHASE_JUMP_RAD

    # Amplitude envelope: cosine-arch dip to 0.94 pu during frequency excursion
    _AMP_NADIR = 0.94
    amp_e = np.ones(n_e)
    _s_exc   = (t_e > T_FLAT_END) & (t_e <= T_REC_END)
    _tau_exc = (t_e[_s_exc] - T_FLAT_END) / (T_REC_END - T_FLAT_END)
    amp_e[_s_exc] = 1.0 - (1.0 - _AMP_NADIR) * np.sin(np.pi * _tau_exc)

    # Voltage: fundamental + harmonics + interharmonic (IEEE 519, THD ~4.5%)
    v_base  = amp_e * np.sin(phi_e)
    v_base += 0.04  * np.sin(5.0 * phi_e)             # 5th harmonic (4%)
    v_base += 0.02  * np.sin(7.0 * phi_e)             # 7th harmonic (2%)
    v_base += 0.005 * np.sin(2.0 * np.pi * 32.5 * t_e)  # interharmonic 32.5 Hz (0.5%)

    # Noise: Gaussian background + sparse impulsive spikes
    SIGMA_GAUSS_E = 0.003
    SIGMA_IMP_E   = 0.05
    P_IMP         = 5.0e-4
    noise_gauss  = np.random.normal(0, SIGMA_GAUSS_E, n_e)
    impulse_mask = np.random.rand(n_e) < P_IMP
    noise_imp    = impulse_mask * np.random.normal(0, SIGMA_IMP_E, n_e)
    v_e = v_base + noise_gauss + noise_imp

    _rocof_dip = 0.5 * np.pi * abs(F_DIP - f_nom_e) / (T_DIP_END - T_FLAT_END)
    _rocof_ovr = 0.5 * np.pi * abs(F_OVR - F_DIP)   / (T_OVR_END - T_DIP_END)
    meta_e = {
        "description": (
            "Scenario F — IBR Primary Frequency Response: C1-Smooth Underdamped "
            "Control-Response Benchmark (5 s)"
        ),
        "standard": (
            "Models physically representative primary-frequency-response dynamics "
            "per IEEE 1547-2018 §6.4 (frequency response); NERC BAL-003-2 inertia "
            "requirements. Peak RoCoF ~{:.1f} Hz/s (inertial dip) and ~{:.1f} Hz/s "
            "(overshoot) are within IEEE 1547-2018 Cat-III ride-through limits. "
            "Phase jump +40° exceeds anti-islanding threshold (20°) by 2×. "
            "Harmonics IEEE 519-2022 compliant (THD ≈ 4.47%).".format(
                _rocof_dip, _rocof_ovr)
        ),
        "physical_motivation": (
            "Represents the full primary-frequency-response cycle of a low-inertia "
            "IBR-dominated grid after a sudden generation loss: (i) fast inertial dip "
            f"(nadir {F_DIP} Hz, {int((T_DIP_END-T_FLAT_END)*1000)} ms), (ii) governor "
            f"overshoot ({F_OVR} Hz, {int((T_OVR_END-T_DIP_END)*1000)} ms), "
            f"(iii) secondary droop/inter-area dip ({F_SD} Hz, "
            f"{int((T_SD_END-T_OVR_END)*1000)} ms), (iv) AGC recovery, (v) underdamped "
            f"inter-area oscillation (ζ={ZETA_S}, f_n={F_N_S} Hz). "
            f"A +40° phase jump at t={PHASE_JUMP_TIME} s models breaker reclosing during recovery."
        ),
        "ibr_relevance": (
            f"With system inertia H < 2 s (IBR-dominated), RoCoF peaks of "
            f"{_rocof_dip:.1f}–{_rocof_ovr:.1f} Hz/s are achievable within the inertial window "
            f"({int((T_DIP_END-T_FLAT_END)*1000)} ms), challenging slow window-based estimators. "
            "The lightly-damped settling (ζ=0.08, f_n=0.8 Hz) represents inter-area "
            "oscillations routinely observed in high-IBR grids (WECC, GB, Iberian Peninsula). "
            "C1-smooth trajectory (no artificial phase jumps in frequency) tests RoCoF "
            "tracking accuracy independently from jump-handling capability (cf. Scenario D/E)."
        ),
        "character": (
            "Physically realistic trajectory without artificial discontinuities in frequency "
            "(C1-smooth cosine smoothstep for transient segments; analytically C1-matched "
            "Prony settling). The +40° phase jump tests phase-tracking independently. "
            "Designed as the physically-grounded counterpart to the worst-case Scenario E."
        ),
        "parameters": (
            f"[0–{T_FLAT_END} s] nominal {f_nom_e} Hz + harmonics; "
            f"[{T_FLAT_END}–{T_DIP_END} s] C1 dip to {F_DIP} Hz (peak RoCoF ≈{_rocof_dip:.1f} Hz/s); "
            f"[{T_DIP_END}–{T_OVR_END} s] C1 overshoot to {F_OVR} Hz (peak RoCoF ≈{_rocof_ovr:.1f} Hz/s); "
            f"[{T_OVR_END}–{T_SD_END} s] C1 secondary dip to {F_SD} Hz; "
            f"[{T_SD_END}–{T_REC_END} s] C1 AGC recovery to {F_REC} Hz; "
            f"[{T_REC_END}–5.0 s] Prony settling: ζ={ZETA_S}, f_n={F_N_S} Hz, "
            f"amplitude ≈{abs(C_OSC):.4f} Hz; "
            f"+40° phase jump at t={PHASE_JUMP_TIME} s (breaker reclosing); "
            "amplitude envelope: cosine-arch dip to 0.94 pu during frequency excursion."
        ),
        "harmonics": (
            "5th harmonic (4%), 7th harmonic (2%): "
            "THD = sqrt(0.04²+0.02²) = 4.47% ≤ IEEE 519-2022 5% PCC limit. "
            "Interharmonic 32.5 Hz (0.5%): asynchronous source (excluded from THD)."
        ),
        "noise": (
            f"Gaussian background sigma={SIGMA_GAUSS_E} pu + "
            f"impulsive sigma={SIGMA_IMP_E} pu (Bernoulli p={P_IMP}/sample). "
            "Same noise model as Scenario E for direct comparison."
        ),
        "noise_sigma": SIGMA_GAUSS_E,  # base Gaussian sigma for SNR calculation (statistical_analysis.py)
        "dynamics": (
            f"C1-smooth cosine-smoothstep segments (dip → overshoot → secondary dip → recovery) "
            f"+ two-term Prony settling analytically C1-matched at t={T_REC_END} s "
            f"(ζ={ZETA_S}, f_n={F_N_S} Hz). "
            f"+40° phase jump at t={PHASE_JUMP_TIME} s. "
            f"Amplitude envelope: cosine arch dip to {_AMP_NADIR} pu during transient."
        ),
        "exceeds_standard": (
            f"Phase jump +40° > 2× IEEE 1547-2018 anti-islanding threshold (20°). "
            "RoCoF values within IEEE Cat-III limits (not a beyond-standard RoCoF test). "
            "Underdamped settling with ζ=0.08 is more aggressive than typical grid "
            "stability requirements (ζ≥0.05 per ENTSO-E NC RfG) but within physical range. "
            "This scenario stresses the estimator's smooth-trajectory tracking capability "
            "rather than its jump-handling — complementary to Scenarios D and E."
        ),
    }

    signals["IBR_PrimaryFrequencyResponse"] = (t_e, v_e, f_e, meta_e)

    return signals


# =============================================================
# 2. SIGNAL PROCESSING HELPERS (Filters & Normalizers)
# =============================================================
class IIR_Bandpass:
    """2nd Order IIR Bandpass Filter (Butterworth).
    Center: 60Hz, Bandwidth: 40Hz (Wide enough for ramps).
    """
    def __init__(self):
        w0 = 2 * np.pi * 60.0 / FS_DSP
        bw = 2 * np.pi * 40.0 / FS_DSP
        R = 1.0 - (bw / 2.0)
        self.a1 = 2.0 * R * np.cos(w0)
        self.a2 = -(R * R)
        self.b0 = (1.0 - self.a2) / 2.0 * 0.5
        self.b2 = -self.b0
        self.x = deque([0.0, 0.0], maxlen=2)
        self.y = deque([0.0, 0.0], maxlen=2)
        
    def step(self, v_in):
        v_out = (
            self.b0 * v_in +
            self.b2 * self.x[1] +
            self.a1 * self.y[0] +
            self.a2 * self.y[1]
        )
        self.x.appendleft(v_in)
        self.y.appendleft(v_out)
        return v_out


class FastRMS_Normalizer:
    """Fast Automatic Gain Control (AGC) using sliding RMS.
    Decouples amplitude dynamics from frequency estimation.
    """
    def __init__(self):
        # Window: 1/2 cycle for fast response vs ripple trade-off
        self.win_len = int(FS_DSP / 60.0 / 2.0)
        self.buf = deque(maxlen=self.win_len)
    
    def step(self, val):
        sq_val = val * val
        self.buf.append(sq_val)
        rms = np.sqrt(np.mean(self.buf))
        if rms < 0.1:
            rms = 0.1  # Safety floor
        return val / (rms * 1.41421356)  # Normalize peak to ~1.0


# =============================================================
# 3. ESTIMATION ALGORITHMS (BENCHMARK SUITE)
# =============================================================

# --- 3.1 BASELINE: IpDFT (Interpolated DFT) ---
class TunableIpDFT:
    def __init__(self, cycles):
        self.sz = int((FS_DSP / 60.0) * cycles)
        self.buf = deque(maxlen=self.sz)
        self.win = np.hanning(self.sz)
        self.res = FS_DSP / self.sz
        self.name = f"IpDFT_{cycles}cyc"

    def structural_latency_samples(self) -> int:
        return self.sz

    def step(self, z):
        self.buf.append(z)
        if len(self.buf) < self.sz:
            return 60.0
        sp_c = np.fft.rfft(np.array(self.buf) * self.win)
        sp = np.abs(sp_c)
        k = int(np.argmax(sp))
        if k == 0 or k == len(sp) - 1:
            return k * self.res
        # Hann-window-compatible magnitude interpolation.
        # The complex Jacobsen form (Re{(X[k+1]-X[k-1])/(2X[k]-X[k-1]-X[k+1])})
        # is NOT appropriate for Hann-windowed DFT: it yields wrong sign/direction
        # on off-bin tones because the phase of the complex ratio does not track
        # the Hann window's even symmetry around the bin centre.
        # Magnitude-interpolation via the three-point parabola ( Quinn, 1994;
        # see also MacLeod, 1998 ) is correct for Hann-windowed DFT and gives
        # sub-bin accuracy on pure tones.
        sp_km1 = sp[k - 1]
        sp_k   = sp[k]
        sp_kp1 = sp[k + 1]
        denom_mag = sp_km1 + 2.0 * sp_k + sp_kp1
        if denom_mag < 1e-10:
            return k * self.res
        delta = 2.0 * (sp_kp1 - sp_km1) / denom_mag
        delta = float(np.clip(delta, -0.5, 0.5))
        return (k + delta) * self.res


# --- 3.2 INDUSTRY STANDARD: SRF-PLL (with MAF) ---
class StandardPLL:
    def __init__(self, kp, ki):
        self.kp = kp
        self.ki = ki
        self.integrator = 0.0
        self.theta = 0.0
        self.name = "SRF-PLL"
        self.maf_win = int(FS_DSP / 60.0)  # 1-cycle Moving Average Filter
        self.buf = deque(maxlen=self.maf_win)

    def structural_latency_samples(self) -> int:
        return self.maf_win

    def step(self, z):
        # Phase Detector (Park-based simplification)
        pd_out = z * np.cos(self.theta)
        # PI Controller
        self.integrator += self.ki * pd_out * DT_DSP
        w_dev = self.kp * pd_out + self.integrator
        w_raw = 2 * np.pi * 60.0 + w_dev
        # Oscillator
        self.theta += w_raw * DT_DSP
        self.theta %= 2 * np.pi
        # Output Filter (MAF on instantaneous frequency)
        self.buf.append(w_raw)
        if len(self.buf) < self.maf_win:
            return w_raw / (2 * np.pi)
        else:
            return np.mean(self.buf) / (2 * np.pi)


# --- 3.3 PROPOSED: EKF (Extended Kalman Filter) ---
class ClassicEKF:
    def __init__(self, q_param, r_param):
        # State: [Phase, Freq(rad/s), Amplitude]
        self.x = np.array([0.0, 2 * np.pi * 60.0, 1.0])
        self.P = np.eye(3) * 1.0
        self.Q = np.diag([1e-6, q_param, 1e-4])
        self.R = np.array([[r_param]])
        self.I = np.eye(3)
        self.init = False
        self.name = "EKF"

    def structural_latency_samples(self) -> int:
        return 0

    def step(self, z):
        if not self.init:
            if abs(z) < 0.99:
                self.x[0] = np.arcsin(z)
                self.init = True
            return 60.0
        # Predict
        self.x[0] += self.x[1] * DT_DSP
        F = np.eye(3)
        F[0, 1] = DT_DSP
        self.P = F @ self.P @ F.T + self.Q
        # Update
        theta, amp = self.x[0], self.x[2]
        y_pred = amp * np.sin(theta)
        inn = z - y_pred
        H = np.array([[amp * np.cos(theta), 0.0, np.sin(theta)]])
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T * (1.0 / (S[0, 0] + 1e-12))
        self.x += (K * inn).flatten()
        self.P = (self.I - K @ H) @ self.P
        # Constraints
        self.x[1] = np.clip(self.x[1], 2 * np.pi * 40.0, 2 * np.pi * 80.0)
        self.x[0] %= 2 * np.pi
        self.x[2] = max(0.1, self.x[2])
        return self.x[1] / (2 * np.pi)


# --- 3.4 SOTA 1: SOGI-FLL (Robust Discretization) ---
class SOGI_FLL:
    """
    Second-Order Generalized Integrator Frequency-Locked Loop (SOGI-FLL).

    Implementation follows Rodriguez et al. (IEEE Trans. Power Electron., 2009)
    with half-cycle moving-average normalization of ||v||^2 to eliminate the
    2ω DC bias in the FLL error signal.

    FLL adaptation law:
        w_dot = -gamma * (z - v_alpha) * v_beta / <||v||^2>_half
    where <·>_half denotes a half-cycle (≈83 samples @ 10 kHz) moving average.

    Front-end: IIR_Bandpass + FastRMS_Normalizer, identical to all other
    estimators in this benchmark. Without this front-end the SOGI FLL
    interprets amplitude transients (e.g. the IEEE Mag-Step scenario) as
    frequency deviations, producing spuriously large RMSE values.

    Discretization: Improved Euler (Heun's method) for stability at fs=10 kHz.
    Output: moving-average smoothed frequency estimate [Hz], window
    configurable via smooth_win (default = 1 full cycle = 167 samples @ 10 kHz).
    """

    def __init__(self, k_gain, gamma, smooth_win=None):
        """
        k_gain    : SOGI damping gain (dimensionless, typically sqrt(2) ≈ 1.414)
        gamma     : FLL adaptation gain (dimensionless, continuous-time equivalent).
                    Optimal range for fs=10 kHz after the w-normalization fix: 5–150.
                    Lower values → tighter steady-state accuracy, slower tracking.
                    Higher values → faster transient response, higher ripple.
        smooth_win: output MA window [samples].
                    Default: 1 full 60-Hz cycle (167 samples @ 10 kHz).
        """
        self.k = float(k_gain)
        self.gamma = float(gamma)
        self.w = 2.0 * np.pi * F_NOM   # angular frequency estimate [rad/s]
        self.v_alpha = 0.0
        self.v_beta  = 0.0
        self.name = "SOGI-FLL"

        # ── Front-end (same as EKF, RLS, Teager, Koopman, TFT …) ──────────
        # Without IIR_Bandpass + AGC the SOGI sees raw amplitude steps and
        # misinterprets them as frequency deviations. Adding the identical
        # front-end used by all other methods ensures a fair comparison.
        self.bp   = IIR_Bandpass()
        self.norm = FastRMS_Normalizer()

        # ── Output smoothing ──────────────────────────────────────────────
        # A short MA filter removes the 2ω ripple from the FLL output.
        # Default: 1 full 60-Hz cycle = 167 samples @ 10 kHz.
        if smooth_win is None:
            smooth_win = max(1, int(FS_DSP / 60.0))
        self.smooth_win = int(smooth_win)
        self.f_buf = deque(maxlen=self.smooth_win)

        # ── Half-cycle MA buffer for ||v||^2 normalization ────────────────
        # Averaging mag_sq over N/2 samples (≈83 @ 10 kHz) cancels the 2ω
        # ripple in the FLL denominator, removing the DC frequency bias.
        _half = max(1, int(FS_DSP / 60.0 / 2))
        self._mag_buf = deque([0.5], maxlen=_half)   # pre-load with A²=1 assumption

    def structural_latency_samples(self) -> int:
        return self.smooth_win

    def step(self, z_raw):
        # ── Pre-processing ────────────────────────────────────────────────
        z_bp = self.bp.step(z_raw)    # 2nd-order IIR bandpass (40 Hz BW @ 60 Hz)
        z    = self.norm.step(z_bp)   # Fast AGC: normalises peak amplitude to ~1.0

        # ── SOGI update (Improved Euler / Heun's method) ──────────────────
        e = z - self.v_alpha

        # Predictor step
        v_alpha_pred = self.v_alpha + DT_DSP * (self.w * e * self.k - self.w * self.v_beta)
        v_beta_pred  = self.v_beta  + DT_DSP * (self.w * self.v_alpha)

        # Corrector step (average of predictor and corrector slopes)
        e_pred = z - v_alpha_pred
        dot_alpha = 0.5 * (
            (self.w * e       * self.k - self.w * self.v_beta) +
            (self.w * e_pred  * self.k - self.w * v_beta_pred)
        )
        dot_beta = 0.5 * (
            self.w * self.v_alpha +
            self.w * v_alpha_pred
        )

        self.v_alpha += DT_DSP * dot_alpha
        self.v_beta  += DT_DSP * dot_beta

        # ── FLL frequency adaptation ──────────────────────────────────────
        # Rodriguez-normalized SOGI-FLL (Rodriguez et al., IEEE TPE 2009):
        #   w_dot = -gamma * epsilon * v_beta / ||v||^2
        # where epsilon = z - v_alpha is the tracking error and ||v||^2 is
        # normalized by a half-cycle moving average to suppress the 2ω ripple
        # that would otherwise introduce a DC bias in the frequency estimate.
        #
        # NOTE: The formulation WITHOUT the extra self.w factor is the correct
        # discrete-time equivalent.  Including self.w inflates the effective
        # gain by 2π·f₀ ≈ 377, pushing the FLL into an over-damped regime
        # that converges to a biased frequency (≈ -0.79 Hz error at 60 Hz).
        mag_sq = self.v_alpha ** 2 + self.v_beta ** 2
        self._mag_buf.append(mag_sq)
        mag_avg = float(np.mean(self._mag_buf))
        if mag_avg < 1e-4:
            mag_avg = 1e-4

        w_dot = -self.gamma * (z - self.v_alpha) * self.v_beta / mag_avg
        self.w += w_dot * DT_DSP
        self.w = float(np.clip(self.w, 2.0 * np.pi * 40.0, 2.0 * np.pi * 80.0))

        # ── Output: MA-smoothed instantaneous frequency [Hz] ─────────────
        f_inst = self.w / (2.0 * np.pi)
        self.f_buf.append(f_inst)
        if len(self.f_buf) < self.smooth_win:
            return F_NOM            # return nominal until buffer is full
        return float(np.mean(self.f_buf))


# =============================================================
# Helper: AR(2) -> frecuencia (con dt configurable)
# =============================================================
def _ar2_to_freq(theta0, dt=DT_DSP):
    """
    Convierte el primer coeficiente AR(2) (a1) en frecuencia [Hz]
    usando el mapeo estándar:
        a1 = 2 cos(w * dt)
    donde dt es el paso de muestreo efectivo.

    Incluye clamps suaves para evitar NaN y valores fuera de rango físico.
    """
    # Clamp de estabilidad
    a1 = float(np.clip(theta0, -1.9999, 1.9999))

    # Relación a1 = 2 cos(w dt)
    val = a1 / 2.0
    val = float(np.clip(val, -0.9999, 0.9999))

    try:
        # w [rad/s]
        w = math.acos(val) / dt
    except ValueError:
        # Por seguridad numérica
        return 60.0

    f_inst = w / (2.0 * math.pi)

    if math.isnan(f_inst) or not np.isfinite(f_inst):
        return 60.0

    # rango físico razonable para aplicaciones de red
    return float(np.clip(f_inst, 40.0, 80.0))


# --- 3.5 SOTA 2: RLS (decimado, normalizado, sin VFF) ---
# --- 3.5 SOTA 2: RLS (decimado, normalizado, sin VFF) ---
class RLS_Estimator:
    """
    RLS AR(2) para estimación de frecuencia.

    - Front-end (IIR + AGC) corre a FS_DSP (=10 kHz).
    - Núcleo RLS trabaja sobre una señal decimada:
         FS_eff = FS_DSP / decim
      Esto mejora la condición numérica del mapeo a1 -> f
      y, con la inicialización correcta, arranca ya en ~60 Hz.
    """
    def __init__(self, lam, win_smooth, decim=50):
        """
        lam        : factor de olvido fijo (0.98–0.999 recomendado)
        win_smooth : ventana de suavizado sobre frecuencia (en muestras RLS)
        decim      : factor de decimación interna (entero ≥1).
                      FS_eff = FS_DSP / decim.  Default 50 → FS_eff = 200 Hz.
                      Must satisfy FS_eff > 2·F_MAX (≈ 160 Hz).
        """
        self.lam = float(lam)

        # Decimación e instante de muestreo efectivo
        self.decim = int(decim)
        if self.decim < 1:
            self.decim = 1
        self.DT_eff = self.decim * DT_DSP

        # Coeficientes AR(2) iniciales consistentes con 60 Hz
        # y(n) = 2 cos(w0 dt) y(n-1) - y(n-2)
        w0 = 2.0 * math.pi * 60.0
        a1_60 = 2.0 * math.cos(w0 * self.DT_eff)
        self.theta = np.array([a1_60, -1.0], dtype=float)

        # Covarianza inicial moderada (no tan agresiva como 100·I)
        self.P = np.eye(2) * 10.0

        self.y_buf = deque([0.0, 0.0], maxlen=2)
        self.name = "RLS"

        self._cnt = 0

        self.smooth_win = int(win_smooth)
        self.f_buf = deque(maxlen=self.smooth_win)

        # Front-end
        self.bp = IIR_Bandpass()
        self.norm = FastRMS_Normalizer()

        self._last_f = 60.0

    def structural_latency_samples(self) -> int:
        return self.decim + self.smooth_win * self.decim

    def step(self, z_raw):
        # Front-end continuo a FS_DSP
        z_bp = self.bp.step(z_raw)
        z = self.norm.step(z_bp)

        # Decimación interna: solo actualizamos RLS cada 'decim' muestras
        self._cnt += 1
        if self._cnt < self.decim:
            return self._last_f
        self._cnt = 0

        # A partir de aquí estamos en tiempo efectivo FS_eff
        if len(self.y_buf) < 2:
            self.y_buf.append(z)
            self._last_f = 60.0
            return self._last_f

        # Vector de entrada AR(2): [y(n-1), y(n-2)]
        phi_raw = np.array([self.y_buf[1], self.y_buf[0]], dtype=float)
        # Normalización para estabilidad numérica (no cambia la solución)
        norm_phi = np.linalg.norm(phi_raw) + 1e-9
        phi = phi_raw / norm_phi
        d = z / norm_phi

        # Predicción y error
        y_pred = float(self.theta @ phi)
        e = d - y_pred

        # RLS estándar
        Pphi = self.P @ phi
        denom = self.lam + float(phi @ Pphi)
        if denom <= 0.0:
            denom = 1e-9
        K = Pphi / denom
        self.theta = self.theta + K * e
        self.P = (self.P - np.outer(K, Pphi)) / self.lam

        # Simetrizar P para estabilidad numérica
        self.P = 0.5 * (self.P + self.P.T)

        self.y_buf.append(z)

        # AR(2) -> frecuencia con dt efectivo
        f_inst = _ar2_to_freq(self.theta[0], dt=self.DT_eff)

        self.f_buf.append(f_inst)
        self._last_f = f_inst

        if len(self.f_buf) < self.smooth_win:
            return 60.0
        return float(np.mean(self.f_buf))


# --- 3.6 SOTA 3: TEAGER (DESA-2 + Normalization + robust clamp) ---
class Teager_Estimator:
    def __init__(self, smooth_win):
        self.buf = deque(maxlen=5)
        self.win = int(smooth_win)
        self.f_buf = deque(maxlen=self.win)
        self.name = "Teager"
        self.bp = IIR_Bandpass()
        self.norm = FastRMS_Normalizer()

    def structural_latency_samples(self) -> int:
        return self.win

    def step(self, z_raw):
        z_bp = self.bp.step(z_raw)
        z = self.norm.step(z_bp)
        self.buf.append(z)
        if len(self.buf) < 5:
            return 60.0
        
        # DESA-2 Algorithm
        x_n = self.buf[2]
        x_nm1 = self.buf[1]
        x_np1 = self.buf[3]
        x_nm2 = self.buf[0]
        x_np2 = self.buf[4]
        
        # Teager Operators
        psi_x = x_n ** 2 - x_nm1 * x_np1
        y_n = x_np1 - x_nm1
        y_nm1 = x_n - x_nm2
        y_np1 = x_np2 - x_n
        psi_y = y_n ** 2 - y_nm1 * y_np1
        
        # Umbral algo más conservador para SNR baja
        if psi_x <= 1e-4:
            f = self.f_buf[-1] if len(self.f_buf) > 0 else 60.0
        else:
            val = 1.0 - psi_y / (2.0 * psi_x)
            if abs(val) > 1.0:
                val = math.copysign(1.0, val)
            w = 0.5 * math.acos(val)  # DESA-2 factor
            f = (w / DT_DSP) / (2 * math.pi)
            
        if f > 80 or f < 40 or math.isnan(f):
            f = 60.0

        # Clamp ligero para evitar saltos aislados absurdos
        if len(self.f_buf) > 0:
            prev = self.f_buf[-1]
            if abs(f - prev) > 5.0:
                f = prev
        
        self.f_buf.append(f)
        if len(self.f_buf) < self.win:
            return 60.0
        return float(np.mean(self.f_buf))


# --- 3.7 SOTA 4: TFT (K=2 Quadratic Model) ---
class TFT_Estimator:
    def __init__(self, win_cycles):
        self.N = int((FS_DSP / 60.0) * win_cycles)
        self.buf = deque(maxlen=self.N)
        self.t_vec = np.arange(self.N) * DT_DSP
        self.t_vec = self.t_vec - np.mean(self.t_vec)
        self.name = "TFT"
        w = 2 * np.pi * 60.0
        
        # Basis Matrix (K=2)
        self.H = np.zeros((self.N, 6))
        self.H[:, 0] = np.cos(w * self.t_vec)
        self.H[:, 1] = np.sin(w * self.t_vec)
        self.H[:, 2] = self.t_vec * np.cos(w * self.t_vec)
        self.H[:, 3] = self.t_vec * np.sin(w * self.t_vec)
        self.H[:, 4] = (self.t_vec ** 2) * np.cos(w * self.t_vec)
        self.H[:, 5] = (self.t_vec ** 2) * np.sin(w * self.t_vec)
        self.H_pinv = np.linalg.pinv(self.H)

    def structural_latency_samples(self) -> int:
        return self.N

    def step(self, z):
        self.buf.append(z)
        if len(self.buf) < self.N:
            return 60.0
        y = np.array(self.buf)
        coeffs = self.H_pinv @ y
        a0, b0, a1, b1 = coeffs[0:4]
        
        num = b0 * a1 - a0 * b1
        den = a0 ** 2 + b0 ** 2
        if den < 1e-6:
            return 60.0
        df = (1.0 / (2 * np.pi)) * (num / den)
        return 60.0 + df


# --- 3.8 SOTA+: RLS con Variable Forgetting Factor (VFF-RLS) ---
class RLS_VFF_Estimator:
    """
    RLS AR(2) con factor de olvido variable tipo VFF.

    Opera sobre una señal:
      - Filtrada (IIR_Bandpass)
      - Normalizada (FastRMS)
      - Decimada por 'decim'
    """

    def __init__(self,
                 lam_min=0.98,
                 lam_max=0.9995,
                 Ka=3.0,
                 Kb=None,
                 win_smooth=20,
                 decim=50,
                 alpha=None):
        """
        lam_min   : cota inferior de lambda_k
        lam_max   : cota superior (<= 1.0)
        Ka, Kb    : parámetros de las ventanas exponenciales
                    (si Kb es None, se toma Kb = Ka)
        win_smooth: tamaño de ventana de suavizado sobre frecuencia (en muestras RLS)
        decim     : factor de decimación interna
        alpha     : alias opcional para Ka (compatibilidad hacia atrás)
        """
        # Alias alpha -> Ka (para compatibilidad con código previo)
        if alpha is not None:
            Ka = alpha

        self.lam_min = float(lam_min)
        self.lam_max = float(min(lam_max, 1.0))
        self.Ka = float(Ka)
        self.Kb = float(Ka if Kb is None else Kb)

        # Decimación y dt efectivo
        self.decim = int(decim) if int(decim) > 0 else 1
        self._cnt = 0
        self.DT_eff = self.decim * DT_DSP

        # Coeficientes AR(2) iniciales consistentes con 60 Hz
        w0 = 2.0 * math.pi * 60.0
        a1_60 = 2.0 * math.cos(w0 * self.DT_eff)
        self.theta = np.array([a1_60, -1.0], dtype=float)

        # Matriz de correlación inversa inicial: P(0) = delta^-1 I
        # delta algo más grande para que no sea tan agresivo
        delta = 0.1       # antes 0.01 -> 100·I (muy brusco)
        self.P = np.eye(2) / delta

        self.y_buf = deque([0.0, 0.0], maxlen=2)
        self.name = "RLS-VFF"

        # Suavizado sobre frecuencia estimada
        self.smooth_win = int(win_smooth)
        self.f_buf = deque(maxlen=self.smooth_win)

        # Front-end
        self.bp = IIR_Bandpass()
        self.norm = FastRMS_Normalizer()

        # Variables para VFF
        self.lambda_k = 1.0
        self.sigma_e = 1.0
        self.sigma_q = 1.0
        self.sigma_v = 1.0

        # Coeficientes de ventana exponencial (filter_len = 2 en AR(2))
        filter_len = 2.0
        self.alpha = 1.0 - 1.0 / (self.Ka * filter_len)
        self.beta  = 1.0 - 1.0 / (self.Kb * filter_len)

        # Ultimo valor de frecuencia, para rellenar en muestras decimadas
        self._last_f = 60.0

    def structural_latency_samples(self) -> int:
        return self.decim + self.smooth_win * self.decim

    def _update_lambda(self):
        """
        Actualiza lambda_k según las potencias estimadas.
        """
        se = max(self.sigma_e, 1e-12)
        sv = max(self.sigma_v, 1e-12)

        gamma = math.sqrt(se) / math.sqrt(sv)

        if 1.0 < gamma <= 2.0:
            lam = 1.0
        else:
            sq = max(self.sigma_q, 1e-12)
            num = math.sqrt(sq) * math.sqrt(sv)
            den = abs(math.sqrt(se) - math.sqrt(sv)) + 1e-8
            lam = min(num / den, 1.0)

        lam = float(np.clip(lam, self.lam_min, self.lam_max))
        # Extra seguridad: no bajar de 0.97 para evitar explosiones
        lam = max(lam, 0.97)

        self.lambda_k = lam
        return lam

    def step(self, z_raw):
        # Front-end a FS_DSP
        z_bp = self.bp.step(z_raw)
        z = self.norm.step(z_bp)

        # Decimación: sólo actualizamos RLS una de cada 'decim' muestras
        self._cnt += 1
        if self._cnt < self.decim:
            return self._last_f
        self._cnt = 0

        # Llenado inicial del buffer AR(2)
        if len(self.y_buf) < 2:
            self.y_buf.append(z)
            self._last_f = 60.0
            return self._last_f

        # Vector de entrada u = [y(k-1), y(k-2)] (normalizado)
        u = np.array([self.y_buf[1], self.y_buf[0]], dtype=float)
        norm_u = np.linalg.norm(u) + 1e-9
        u_n = u / norm_u
        d = z / norm_u

        # Kalman gain con lambda_k actual
        lam = self.lambda_k
        Pu = self.P @ u_n
        q = float(u_n @ Pu)  # u^T P u (para sigma_q)

        den = 1.0 + lam**-1 * q
        if den <= 0.0:
            den = 1e-9
        kalman = (lam**-1 * Pu) / den

        # Error a priori
        y_hat = float(self.theta @ u_n)
        e = d - y_hat

        # Actualización de coeficientes AR(2)
        self.theta = self.theta + kalman * e

        # Actualización de P (RLS clásico con lambda)
        self.P = lam**-1 * self.P - lam**-1 * (np.outer(kalman, u_n) @ self.P)

        # Mantenimiento numérico: simetrizar P suavemente
        self.P = 0.5 * (self.P + self.P.T)

        # Actualización de potencias
        self.sigma_e = self.alpha * self.sigma_e + (1.0 - self.alpha) * (e * e)
        self.sigma_q = self.alpha * self.sigma_q + (1.0 - self.alpha) * (q * q)
        self.sigma_v = self.beta  * self.sigma_v + (1.0 - self.beta)  * (e * e)

        # Nuevo lambda_k para la próxima iteración
        self._update_lambda()

        # Actualizar buffer de salida (para el siguiente paso)
        self.y_buf.append(z)

        # AR(2) -> frecuencia con dt efectivo (por decimación)
        f_inst = _ar2_to_freq(self.theta[0], dt=self.DT_eff)

        # Guardar para muestras decimadas que no actualizan RLS
        self.f_buf.append(f_inst)
        self._last_f = f_inst

        if len(self.f_buf) < self.smooth_win:
            return 60.0

        return float(np.mean(self.f_buf))


# --- 3.9 SOTA+: Unscented Kalman Filter (UKF) ---
class UKF_Estimator:
    """
    UKF para el mismo modelo de estado que el EKF:
        x = [theta, omega, A]
        z = A * sin(theta)
    Evita Jacobianos explícitos y es robusto a no linealidades fuertes.
    """
    def __init__(self, q_param, r_param, smooth_win=10,
                 alpha=0.3, beta=2.0, kappa=0.0):
        self.name = "UKF"
        self.n = 3
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lmbda = self.alpha**2 * (self.n + self.kappa) - self.n

        self.Wm = np.zeros(2 * self.n + 1)
        self.Wc = np.zeros(2 * self.n + 1)
        self.Wm[0] = self.lmbda / (self.n + self.lmbda)
        self.Wc[0] = self.Wm[0] + (1 - self.alpha**2 + self.beta)
        self.Wm[1:] = 1.0 / (2 * (self.n + self.lmbda))
        self.Wc[1:] = self.Wm[1:]

        self.x = np.array([0.0, 2 * np.pi * 60.0, 1.0])
        self.P = np.diag([1e-2, (2*np.pi)**2, 1e-2])

        self.Q = np.diag([1e-6, q_param, 1e-4])
        self.R = np.array([[r_param]])

        self.init = False
        self.smooth_win = int(smooth_win)
        self.f_buf = deque(maxlen=self.smooth_win)

    def structural_latency_samples(self) -> int:
        return self.smooth_win

    def _sigma_points(self, x, P):
        n = self.n
        sigma = np.zeros((2*n + 1, n))
        sigma[0] = x
        c = n + self.lmbda
        try:
            A = np.linalg.cholesky(c * P)
        except np.linalg.LinAlgError:
            # Regulariza si P no es SPD
            A = np.linalg.cholesky(c * (P + 1e-9 * np.eye(n)))

        for i in range(n):
            sigma[i+1]   = x + A[:, i]
            sigma[i+1+n] = x - A[:, i]
        return sigma

    def _f(self, x):
        # Dinámica del estado
        theta, omega, A = x
        theta_new = theta + omega * DT_DSP
        return np.array([theta_new, omega, A])

    def _h(self, x):
        theta, omega, A = x
        return np.array([A * math.sin(theta)])

    def step(self, z):
        if not self.init:
            if abs(z) < 0.99:
                self.x[0] = math.asin(z)
                self.init = True
            return 60.0

        # 1) Sigma points
        sigma = self._sigma_points(self.x, self.P)

        # 2) Predicción
        sigma_f = np.array([self._f(s) for s in sigma])
        x_pred = np.sum(self.Wm[:, None] * sigma_f, axis=0)

        P_pred = self.Q.copy()
        for i in range(2*self.n + 1):
            dx = (sigma_f[i] - x_pred).reshape(-1, 1)
            P_pred += self.Wc[i] * (dx @ dx.T)

        # 3) Predicción de medición
        sigma_h = np.array([self._h(s) for s in sigma_f])
        z_pred = np.sum(self.Wm[:, None] * sigma_h, axis=0)

        S = self.R.copy()
        Pxz = np.zeros((self.n, 1))
        for i in range(2*self.n + 1):
            dz = (sigma_h[i] - z_pred).reshape(-1, 1)
            dx = (sigma_f[i] - x_pred).reshape(-1, 1)
            S   += self.Wc[i] * (dz @ dz.T)
            Pxz += self.Wc[i] * (dx @ dz.T)

        # 4) Actualización
        K = Pxz @ np.linalg.inv(S + 1e-12*np.eye(1))
        inn = np.array([[z]]) - z_pred.reshape(-1, 1)
        self.x = x_pred + (K @ inn).flatten()
        self.P = P_pred - K @ S @ K.T

        # Constraints
        self.x[1] = float(np.clip(self.x[1], 2*np.pi*40.0, 2*np.pi*80.0))
        self.x[0] = float(self.x[0] % (2*np.pi))
        self.x[2] = max(0.1, float(self.x[2]))

        f_inst = self.x[1] / (2*np.pi)
        if f_inst > 80 or f_inst < 40 or np.isnan(f_inst):
            f_inst = 60.0

        self.f_buf.append(f_inst)
        if len(self.f_buf) < self.smooth_win:
            return 60.0
        return float(np.mean(self.f_buf))


# --- 3.10 SOTA+: Koopman / RK-DPMU simplificado ---
class Koopman_RKDPmu:
    """
    Estimador tipo Koopman / RK-DPMU:
    - Ventana deslizante de estados embebidos en 2D
      s_k = [x_k, x_{k-1}]^T
    - Ajuste de operador lineal K mediante mínimos cuadrados:
      S1 ≈ K S0
    - Frecuencia = arg(eig_max(K)) / (2π * DT_DSP)
    """
    def __init__(self, window_samples=200, smooth_win=20):
        self.N = int(window_samples)
        self.buf = deque(maxlen=self.N + 2)  # necesitamos al menos N+2 puntos
        self.name = "Koopman-RKDPmu"
        self.smooth_win = int(smooth_win)
        self.f_buf = deque(maxlen=self.smooth_win)
        self.bp = IIR_Bandpass()
        self.norm = FastRMS_Normalizer()

    def structural_latency_samples(self) -> int:
        return self.N + self.smooth_win

    def step(self, z_raw):
        z_bp = self.bp.step(z_raw)
        z = self.norm.step(z_bp)

        self.buf.append(z)
        if len(self.buf) < (self.N + 2):
            return 60.0

        x = np.array(self.buf)
        # Construimos estados embebidos 2D
        x0 = x[:-2]
        x1 = x[1:-1]
        x2 = x[2:]

        S0 = np.vstack((x1, x0))   # shape (2, L-2)
        S1 = np.vstack((x2, x1))   # shape (2, L-2)

        # Koopman K ~ S1 S0^+
        # np.linalg.pinv uses SVD-based truncation — numerically stable without
        # the scalar offset (S0 + 1e-9 adds a constant to every element, which
        # shifts all eigenvalues and introduces systematic frequency bias).
        S0_pinv = np.linalg.pinv(S0)
        K = S1 @ S0_pinv  # 2x2

        eigvals, _ = np.linalg.eig(K)
        # Tomamos el modo con mayor módulo (dominante)
        idx = np.argmax(np.abs(eigvals))
        lam = eigvals[idx]
        angle = np.angle(lam)

        # Frecuencia instantánea (rad/s -> Hz)
        w = angle / DT_DSP
        f_inst = w / (2*np.pi)

        if f_inst > 80 or f_inst < 40 or np.isnan(f_inst):
            f_inst = 60.0

        # Suavizado
        self.f_buf.append(f_inst)
        if len(self.f_buf) < self.smooth_win:
            return 60.0
        return float(np.mean(self.f_buf))


# --- 3.11 ML: Physics-Informed GRU Frequency Estimator (PI-GRU) ---
class PIGRU_FreqEstimator:
    """
    Physics-Informed GRU Frequency Estimator (PI-GRU).

    - Entrada: ventana deslizante de muestras de tensión ya pre-filtradas
      (bandpass 60 Hz) y normalizadas (AGC RMS).
    - Modelo: red GRU ligera entrenada offline (PyTorch).
      Debe recibir un tensor de forma (batch=1, T, 1) y devolver
      un escalar o un vector 1D con frecuencia instantánea en Hz.
    - Latencia estructural ≈ window_len / FS_DSP.
    """

    def __init__(self,
                 model=None,
                 window_len_samples=40,
                 smooth_win=10,
                 device="cpu",
                 name="PI-GRU"):
        """
        model: instancia de torch.nn.Module ya entrenada (o ScriptModule),
               con forward(x: [1,T,1]) -> [1] o [1,T_out].
        window_len_samples: tamaño de ventana en muestras a 10 kHz.
        smooth_win: tamaño de ventana de suavizado sobre la salida.
        device: 'cpu' o 'cuda' según dónde esté el modelo.
        """
        self.name = name
        self.window_len = int(window_len_samples)
        self.buf = deque(maxlen=self.window_len)
        self.smooth_win = int(smooth_win)
        self.f_buf = deque(maxlen=self.smooth_win)
        self.bp = IIR_Bandpass()
        self.norm = FastRMS_Normalizer()

        self._torch_available = (torch is not None)
        self._warned = False

        if self._torch_available and (model is not None):
            self.model = model.to(device)
            self.model.eval()
            self.device = device
            for p in self.model.parameters():
                p.requires_grad_(False)
        else:
            self.model = None
            self.device = "cpu"

    def structural_latency_samples(self) -> int:
        return self.window_len + self.smooth_win

    def _fallback(self):
        """Salida de emergencia cuando no hay modelo/torch."""
        if not self._warned:
            print("[PI-GRU WARNING] PyTorch o el modelo no están disponibles. "
                  "El estimador devuelve 60 Hz como dummy baseline.")
            self._warned = True
        return 60.0

    def step(self, z_raw):
        # Si no hay soporte ML, devolvemos baseline seguro
        if (not self._torch_available) or (self.model is None):
            return self._fallback()

        # Front-end físico: bandpass + AGC
        z_bp = self.bp.step(z_raw)
        z = self.norm.step(z_bp)

        self.buf.append(z)
        if len(self.buf) < self.window_len:
            return 60.0

        # Preparar tensor para el modelo: shape (1, T, 1)
        x_np = np.array(self.buf, dtype=np.float32).reshape(1, -1, 1)
        x_t = torch.from_numpy(x_np).to(self.device)

        with torch.no_grad():
            y = self.model(x_t)

        if y.ndim == 0:
            f_inst = float(y.cpu().item())
        else:
            y_np = y.detach().cpu().numpy().flatten()
            f_inst = float(y_np[-1])

        if (np.isnan(f_inst) or f_inst < 40.0 or f_inst > 80.0):
            f_inst = 60.0

        if len(self.f_buf) > 0:
            prev = self.f_buf[-1]
            if abs(f_inst - prev) > 5.0:
                f_inst = prev

        self.f_buf.append(f_inst)
        if len(self.f_buf) < self.smooth_win:
            return 60.0
        return float(np.mean(self.f_buf))


# =============================================================
# 4. METRICS (SIN PLOTTING)
# =============================================================
def _max_contiguous_time(mask_bool, dt):
    """Longest continuous time where mask_bool is True."""
    max_len = 0
    current = 0
    for val in mask_bool:
        if val:
            current += 1
            if current > max_len:
                max_len = current
        else:
            current = 0
    return max_len * dt


def calculate_metrics(est_trace, true_trace, exec_time, structural_samples=None):
    """
    Compute accuracy, protection-risk, and computational complexity metrics.

    Error definition
    ----------------
    e(k) = f_hat(k) - f(k)   [Hz]
        where f_hat(k) is the estimated frequency and f(k) is the ground truth
        at discrete time index k.  The first 150 ms (1500 samples at 10 kHz,
        fixed duration) are discarded to exclude the estimator cold-start
        transient before computing any metric.

    IEC/IEEE 60255-118-1 inspired metrics
    ---------------------------------------
    The standard defines three primary error quantities for PMU / protection
    frequency measurement devices:
        FE  (Frequency Error)         = |f_hat(k) - f(k)|         [Hz]
        RFE (Rate of Frequency Error) = |Δf_hat/Δt - Δf/Δt|      [Hz/s]
        TVE (Total Vector Error)      requires full phasor — N/A here.

    We report FE_max (≡ MAX_PEAK), FE_mean (≡ MAE), RFE_max and RFE_rms,
    computed on the post-transient window for consistency with the standard's
    intent.  Full IEC compliance (specific test waveform parameters) is beyond
    scope; these metrics capture the spirit of the FE / RFE pass/fail criteria
    for P-class protection-class devices (IEC/IEEE 60255-118-1, Table 2).

    Trip-risk definition
    --------------------
    |e(k)| > TRIP_THRESHOLD = 0.5 Hz approximates the under/over-frequency
    relay dead-band for IEC/IEEE 60255-118-1 P-class devices.
    """
    # Discard cold-start transient: use the larger of 150 ms (general warm-up)
    # and the algorithm's own structural latency (e.g. IpDFT with 10 cycles needs
    # 167 ms before the first valid output, so a fixed 150 ms would include biased
    # samples).  Cap at 90 % of the signal to prevent empty evaluation windows.
    n_signal = len(est_trace)
    baseline_samples = int(0.15 * FS_DSP)
    struct_samples   = int(structural_samples) if structural_samples else 0
    start_idx = min(max(baseline_samples, struct_samples), int(0.9 * n_signal))

    # ── Core error signal ───────────────────────────────────────────────────
    e     = est_trace[start_idx:] - true_trace[start_idx:]   # [Hz], signed
    abs_e = np.abs(e)
    n_eff = len(abs_e) if len(abs_e) > 0 else 1

    # ── Standard accuracy metrics ────────────────────────────────────────────
    rmse      = float(np.sqrt(np.mean(abs_e ** 2)))
    mae       = float(np.mean(abs_e))             # = FE_mean per IEC spirit
    max_peak  = float(np.max(abs_e))              # = FE_max  per IEC spirit

    settling_time = float(np.sum(abs_e > SETTLING_THRESHOLD) / FS_DSP)
    energy        = float(np.sum(abs_e ** 2) / FS_DSP)

    # ── Trip-risk metrics ─────────────────────────────────────────────────────
    trip_mask          = abs_e > TRIP_THRESHOLD
    total_trip_time    = float(np.sum(trip_mask) / FS_DSP)
    max_cont_trip_time = float(_max_contiguous_time(trip_mask, 1.0 / FS_DSP))

    # ── IEC/IEEE 60255-118-1 inspired: RFE (Rate-of-Frequency Error) ─────────
    # RFE(k) = |d/dt f_hat(k) - d/dt f(k)| = |d/dt e(k)|  [Hz/s]
    # Estimated via forward finite differences at FS_DSP.
    if len(e) > 2:
        de_dt   = np.diff(e) * FS_DSP          # [Hz/s]  (causal, 1-sample lag)
        rfe_max = float(np.max(np.abs(de_dt)))
        rfe_rms = float(np.sqrt(np.mean(de_dt ** 2)))
    else:
        rfe_max = 0.0
        rfe_rms = 0.0

    # ── Computational cost ────────────────────────────────────────────────────
    n_total         = max(len(est_trace), 1)
    t_per_sample_us = float(exec_time / n_total * 1e6)

    # ── Structural latency ────────────────────────────────────────────────────
    if structural_samples is not None and structural_samples > 0:
        latency_ms = float(structural_samples / FS_DSP * 1e3)
    else:
        latency_ms = float(1.0 / FS_DSP * 1e3)

    # ── PY-05: Probabilistic Compliance Bound (PCB) ─────────────────────────
    # PCB = mean(|FE|) + 3 * std(|FE|) — 3-sigma bound on frequency error
    # IEEE limit: 0.05 Hz (50 mHz) for P-class protection devices
    mean_abs_e = float(np.mean(abs_e))
    std_abs_e = float(np.std(abs_e))
    pcb = mean_abs_e + 3.0 * std_abs_e  # Probabilistic Compliance Bound
    IEEE_LIMIT = 0.05  # Hz (50 mHz)
    pcb_compliant = pcb <= IEEE_LIMIT

    return {
        # ── Standard benchmark ───────────────────────────────────────────────
        "RMSE":                  round(rmse,      6),
        "MAE":                   round(mae,       6),
        "MAX_PEAK":              round(max_peak,  6),
        "SETTLING":              round(settling_time, 4),
        "ENERGY":                round(energy,    6),
        "TRIP_TIME_0p5":         round(total_trip_time,    4),
        "MAX_CONTIGUOUS_0p5":    round(max_cont_trip_time, 4),
        # ── IEC/IEEE 60255-118-1 inspired ────────────────────────────────────
        # FE_max  ≡ MAX_PEAK (same quantity, IEC naming for paper tables)
        # FE_mean ≡ MAE
        "FE_max_Hz":             round(max_peak,  6),
        "FE_mean_Hz":            round(mae,       6),
        "RFE_max_Hz_s":          round(rfe_max,   4),
        "RFE_rms_Hz_s":          round(rfe_rms,   4),
        # ── Computational / latency ───────────────────────────────────────────
        "COMPLEXITY":            float(exec_time),
        "TIME_PER_SAMPLE_US":    round(t_per_sample_us, 4),
        "STRUCTURAL_LATENCY_MS": round(latency_ms, 3),
        # ── PY-05: Probabilistic Compliance Bound (PCB) ──────────────────────
        "PCB":                   round(pcb, 6),
        "PCB_COMPLIANT":         pcb_compliant,
    }


# =============================================================
# 5. MASSIVE HYPERPARAMETER TUNING HELPERS
# =============================================================

def _bayesian_tune_helper(eval_fn, bounds, n_lhs=50, maxiter=100, seed=42):
    """
    Latin Hypercube Sampling initialisation + differential_evolution refinement.

    Parameters
    ----------
    eval_fn : callable(x: 1-D array) -> float  (RMSE, lower = better)
    bounds  : list of (lo, hi) tuples
    n_lhs   : number of LHS initial candidates
    maxiter : max DE iterations
    seed    : RNG seed

    Returns
    -------
    best_x  : 1-D np.ndarray of optimised parameters, or None on failure
    best_rmse : float
    """
    try:
        from scipy.optimize import differential_evolution
        from scipy.stats.qmc import LatinHypercube
    except ImportError:
        return None, 1e9  # fallback to caller's grid search

    try:
        rng     = np.random.default_rng(seed)
        sampler = LatinHypercube(d=len(bounds), seed=rng)
        lhs_unit = sampler.random(n=n_lhs)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        init_pop = lo + lhs_unit * (hi - lo)

        result = differential_evolution(
            eval_fn,
            bounds=bounds,
            init=init_pop,
            maxiter=maxiter,
            tol=1e-7,
            seed=seed,
            workers=1,
            polish=True,
            mutation=(0.5, 1.5),
            recombination=0.9,
            popsize=1,        # init_pop already sets population
        )
        return result.x, float(result.fun)
    except Exception:
        return None, 1e9


def tune_ipdft(v, f, c_vals):
    if USE_BAYESIAN_TUNING and len(c_vals) > 1:
        c_lo, c_hi = float(min(c_vals)), float(max(c_vals))
        def _eval(x):
            c = max(1, int(round(float(x[0]))))
            algo = TunableIpDFT(c)
            tr = np.array([algo.step(s) for s in v])
            return calculate_metrics(tr, f, 0.0, structural_samples=algo.structural_latency_samples())["RMSE"]
        best_x, _ = _bayesian_tune_helper(_eval, [(c_lo, c_hi)])
        if best_x is not None:
            c_opt = max(1, int(round(float(best_x[0]))))
            return f"{c_opt} cycles"

    best = {"RMSE": 1e9}
    for c in c_vals:
        algo = TunableIpDFT(c)
        tr = np.array([algo.step(x) for x in v])
        m = calculate_metrics(tr, f, 0.0, structural_samples=algo.structural_latency_samples())
        if m["RMSE"] < best["RMSE"]:
            best = {"RMSE": m["RMSE"], "p": f"{c} cycles"}
    return best["p"]


def tune_pll(v, f, kp_vals, ki_vals, sc_name=None):
    if USE_BAYESIAN_TUNING and len(kp_vals) > 1 and len(ki_vals) > 1:
        kp_lo, kp_hi = float(min(kp_vals)), float(max(kp_vals))
        ki_lo, ki_hi = float(min(ki_vals)), float(max(ki_vals))
        def _eval(x):
            algo = StandardPLL(float(x[0]), float(x[1]))
            tr = np.array([algo.step(s) for s in v])
            return calculate_metrics(tr, f, 0.0,
                                     structural_samples=algo.structural_latency_samples())["RMSE"]
        best_x, _ = _bayesian_tune_helper(_eval, [(kp_lo, kp_hi), (ki_lo, ki_hi)])
        if best_x is not None:
            kp_opt, ki_opt = float(best_x[0]), float(best_x[1])
            return f"Kp{kp_opt:.4f},Ki{ki_opt:.4f}", (kp_opt, ki_opt)

    best = {"RMSE": 1e9}
    for kp in kp_vals:
        for ki in ki_vals:
            algo = StandardPLL(kp, ki)
            tr = np.array([algo.step(s) for s in v])
            m = calculate_metrics(tr, f, 0.0,
                                  structural_samples=algo.structural_latency_samples())
            if m["RMSE"] < best["RMSE"]:
                best = {
                    "RMSE": m["RMSE"],
                    "p": f"Kp{kp},Ki{ki}",
                    "v": (kp, ki)
                }
    return best["p"], best.get("v", (10, 50))


def tune_ekf(v, f, q_vals, r_vals, sc_name=None):
    if USE_BAYESIAN_TUNING and len(q_vals) > 1 and len(r_vals) > 1:
        q_lo, q_hi = float(min(q_vals)), float(max(q_vals))
        r_lo, r_hi = float(min(r_vals)), float(max(r_vals))
        def _eval(x):
            algo = ClassicEKF(float(x[0]), float(x[1]))
            tr = np.array([algo.step(s) for s in v])
            return calculate_metrics(tr, f, 0.0, structural_samples=algo.structural_latency_samples())["RMSE"]
        best_x, _ = _bayesian_tune_helper(_eval, [(q_lo, q_hi), (r_lo, r_hi)])
        if best_x is not None:
            q_opt, r_opt = float(best_x[0]), float(best_x[1])
            return f"Q{q_opt:.6f},R{r_opt:.6f}", (q_opt, r_opt)

    best = {"RMSE": 1e9}
    for q in q_vals:
        for r in r_vals:
            algo = ClassicEKF(q, r)
            tr = np.array([algo.step(s) for s in v])
            m = calculate_metrics(tr, f, 0.0, structural_samples=algo.structural_latency_samples())
            if m["RMSE"] < best["RMSE"]:
                best = {
                    "RMSE": m["RMSE"],
                    "p": f"Q{q},R{r}",
                    "v": (q, r)
                }
    return best["p"], best.get("v", (0.1, 1.0))


def tune_ekf2(v, f, sc_name=None):
    if USE_BAYESIAN_TUNING:
        # Continuous bounds covering EKF2.tuning_grid() range
        bounds = [(1e-5, 20.0), (1e-4, 20.0), (0.05, 5.0)]   # q, r, inn_ref
        def _eval(x):
            params = {
                "q_param":  float(x[0]),
                "r_param":  float(x[1]),
                "inn_ref":  float(x[2]),
            }
            algo = EKF2(**params)
            tr = np.array([algo.step(s) for s in v])
            return calculate_metrics(tr, f, 0.0, structural_samples=algo.structural_latency_samples())["RMSE"]
        best_x, best_rmse = _bayesian_tune_helper(_eval, bounds)
        if best_x is not None and best_rmse < 1e8:
            best_params = {
                "q_param": float(best_x[0]),
                "r_param": float(best_x[1]),
                "inn_ref": float(best_x[2]),
            }
            return EKF2.describe_params(best_params), best_params

    best = {"RMSE": 1e9, "params": None}

    for params in EKF2.tuning_grid():
        algo = EKF2(**params)
        tr = np.array([algo.step(x) for x in v])
        m = calculate_metrics(tr, f, 0.0, structural_samples=algo.structural_latency_samples())

        if m["RMSE"] < best["RMSE"]:
            best["RMSE"] = m["RMSE"]
            best["params"] = params

    if best["params"] is None:
        default_params = {"q_param": 0.1, "r_param": 1.0, "inn_ref": 0.5}
        return EKF2.describe_params(default_params), default_params

    p_str = EKF2.describe_params(best["params"])
    return p_str, best["params"]


def tune_sogi(v, f, k_vals, g_vals, smooth_win_vals=None):
    """
    Grid search (or Bayesian when USE_BAYESIAN_TUNING=True) for SOGI_FLL.

    k_vals        : SOGI damping gain (dimensionless). Typical range: 0.5 – 2.0.
    g_vals        : FLL adaptation gain gamma (dimensionless after w-fix).
                    Effective range for fs=10 kHz: 5–150.
    smooth_win_vals: output MA window [samples]. None defaults to {None} (auto).
    """
    if smooth_win_vals is None:
        smooth_win_vals = [None]   # auto = int(fs/60/6) ≈ 28 samples

    if USE_BAYESIAN_TUNING and len(k_vals) > 1 and len(g_vals) > 1:
        k_lo, k_hi = float(min(k_vals)), float(max(k_vals))
        g_lo, g_hi = float(min(g_vals)), float(max(g_vals))
        sw_fixed = smooth_win_vals[0]   # keep sw fixed to first value (usually None)
        def _eval(x):
            algo = SOGI_FLL(float(x[0]), float(x[1]), smooth_win=sw_fixed)
            tr = np.array([algo.step(s) for s in v])
            return calculate_metrics(tr, f, 0.0,
                                     structural_samples=algo.structural_latency_samples())["RMSE"]
        best_x, _ = _bayesian_tune_helper(_eval, [(k_lo, k_hi), (g_lo, g_hi)])
        if best_x is not None:
            k_opt, g_opt = float(best_x[0]), float(best_x[1])
            return f"k{k_opt:.4f},g{g_opt:.4f},sw{sw_fixed}", (k_opt, g_opt, sw_fixed)

    best = {"RMSE": 1e9}
    for k in k_vals:
        for g in g_vals:
            for sw in smooth_win_vals:
                algo = SOGI_FLL(k, g, smooth_win=sw)
                tr = np.array([algo.step(s) for s in v])
                m = calculate_metrics(tr, f, 0.0,
                                      structural_samples=algo.structural_latency_samples())
                if m["RMSE"] < best["RMSE"]:
                    best = {
                        "RMSE": m["RMSE"],
                        "p": f"k{k},g{g},sw{sw}",
                        "v": (k, g, sw)
                    }
    return best["p"], best.get("v", (1.414, 10.0, None))


def tune_rls(v, f, lam_vals, win_vals, sc_name=None):
    best = {"RMSE": 1e9}
    for l in lam_vals:
        for w in win_vals:
            # decim=50 matches deployment in main.py (FS_eff = 10 kHz / 50 = 200 Hz)
            algo = RLS_Estimator(lam=l, win_smooth=w, decim=50)
            tr = np.array([algo.step(x) for x in v])
            m = calculate_metrics(
                tr, f, 0.0,
                structural_samples=algo.structural_latency_samples()
            )
            if m["RMSE"] < best["RMSE"]:
                best = {
                    "RMSE": m["RMSE"],
                    "p": f"Lam{l},Win{w}",
                    "v": (l, w)
                }
    return best["p"], best.get("v", (0.995, 80))


def tune_teager(v, f, win_vals):
    best = {"RMSE": 1e9}
    for w in win_vals:
        algo = Teager_Estimator(w)
        tr = np.array([algo.step(x) for x in v])
        m = calculate_metrics(tr, f, 0.0,
                              structural_samples=algo.structural_latency_samples())
        if m["RMSE"] < best["RMSE"]:
            best = {
                "RMSE": m["RMSE"],
                "p": f"Win{w}",
                "v": w
            }
    return best["p"], best.get("v", 10)


def tune_tft(v, f, win_vals):
    best = {"RMSE": 1e9}
    for w in win_vals:
        algo = TFT_Estimator(w)
        tr = np.array([algo.step(x) for x in v])
        m = calculate_metrics(tr, f, 0.0,
                              structural_samples=algo.N)
        if m["RMSE"] < best["RMSE"]:
            best = {
                "RMSE": m["RMSE"],
                "p": f"Cycles{w}",
                "v": w
            }
    return best["p"], best.get("v", 2)


def tune_vff_rls(v, f, lam_min_vals, alpha_vals, sc_name=None):
    """
    Tuning para RLS_VFF_Estimator:
      - lam_min: mínimo factor de olvido
      - alpha : aquí se usa como Ka (y Kb=Ka) de las ventanas exponenciales
    lam_max se fija alto (0.9995) para buen tracking en steady state.
    """
    best = {"RMSE": 1e9}
    for lam_min in lam_min_vals:
        for Ka in alpha_vals:
            # decim=50, win_smooth=20 match deployment in main.py
            algo = RLS_VFF_Estimator(
                lam_min=lam_min,
                lam_max=0.9995,
                Ka=Ka,
                Kb=None,
                win_smooth=20,
                decim=50
            )
            tr = np.array([algo.step(x) for x in v])
            m = calculate_metrics(
                tr, f, 0.0,
                structural_samples=algo.structural_latency_samples()
            )
            if m["RMSE"] < best["RMSE"]:
                best = {
                    "RMSE": m["RMSE"],
                    "p": f"lamMin{lam_min},Ka{Ka}",
                    "v": (lam_min, Ka)
                }
    return best["p"], best.get("v", (0.985, 3.0))


def tune_ukf(v, f, q_vals, r_vals, sc_name=None):
    # DISCLOSURE: UKF_Estimator includes smooth_win=10 output smoothing (deque mean).
    # ClassicEKF does NOT include the same post-filter; it returns raw state freq.
    # This asymmetry is intentional for this benchmark configuration and must be
    # disclosed in the paper (methodology section) so reviewers understand the
    # comparison is not between equivalent unfiltered UKF vs EKF.
    if USE_BAYESIAN_TUNING and len(q_vals) > 1 and len(r_vals) > 1:
        q_lo, q_hi = float(min(q_vals)), float(max(q_vals))
        r_lo, r_hi = float(min(r_vals)), float(max(r_vals))
        def _eval(x):
            algo = UKF_Estimator(q_param=float(x[0]), r_param=float(x[1]),
                                 smooth_win=10)
            tr = np.array([algo.step(s) for s in v])
            return calculate_metrics(tr, f, 0.0, structural_samples=algo.structural_latency_samples())["RMSE"]
        best_x, _ = _bayesian_tune_helper(_eval, [(q_lo, q_hi), (r_lo, r_hi)])
        if best_x is not None:
            q_opt, r_opt = float(best_x[0]), float(best_x[1])
            return f"Q{q_opt:.6f},R{r_opt:.6f}", (q_opt, r_opt)

    best = {"RMSE": 1e9}
    for q in q_vals:
        for r in r_vals:
            algo = UKF_Estimator(q_param=q, r_param=r, smooth_win=10)
            tr = np.array([algo.step(x) for x in v])
            m = calculate_metrics(tr, f, 0.0, structural_samples=algo.structural_latency_samples())
            if m["RMSE"] < best["RMSE"]:
                best = {
                    "RMSE": m["RMSE"],
                    "p": f"Q{q},R{r}",
                    "v": (q, r)
                }
    return best["p"], best.get("v", (1.0, 0.01))


def tune_koopman(v, f, win_vals, sc_name=None):
    if USE_BAYESIAN_TUNING and len(win_vals) > 1:
        w_lo, w_hi = float(min(win_vals)), float(max(win_vals))
        def _eval(x):
            w = max(10, int(round(float(x[0]))))
            algo = Koopman_RKDPmu(window_samples=w, smooth_win=w)
            tr = np.array([algo.step(s) for s in v])
            # BUG-T1 fix: actual cold-start = N + smooth_win (both use w)
            return calculate_metrics(tr, f, 0.0, structural_samples=algo.structural_latency_samples())["RMSE"]
        best_x, _ = _bayesian_tune_helper(_eval, [(w_lo, w_hi)])
        if best_x is not None:
            w_opt = max(10, int(round(float(best_x[0]))))
            return f"Win{w_opt}", w_opt

    best = {"RMSE": 1e9}
    for w in win_vals:
        algo = Koopman_RKDPmu(window_samples=w, smooth_win=w)
        tr = np.array([algo.step(x) for x in v])
        # BUG-T1 fix: actual cold-start = N (window fill) + smooth_win (output buffer)
        m = calculate_metrics(tr, f, 0.0,
                              structural_samples=algo.structural_latency_samples())
        if m["RMSE"] < best["RMSE"]:
            best = {
                "RMSE": m["RMSE"],
                "p": f"Win{w}",
                "v": w
            }
    return best["p"], best.get("v", 200)



# --- 3.X: Linear Kalman Filter (LKF-AR2 Hybrid) ---
class LKF_Estimator:
    """
    Linear Kalman Filter + AR(2) hybrid frequency estimator.

    - LKF filtra la señal (estado x = [y(k), y(k-1)]).
    - Cada cierto número de muestras (decim) hace un pequeño ajuste AR(2)
      para obtener a1 y a2, y luego usa _ar2_to_freq(a1, dt) para f_inst.
    """

    def __init__(self, q_val=1e-4, r_val=1e-2, smooth_win=20, decim=10):
        """
        q_val      : varianza de ruido de proceso (Q)
        r_val      : varianza de ruido de medición (R)
        smooth_win : ventana de suavizado sobre frecuencia (en muestras LKF)
        decim      : factor de decimación para el ajuste AR(2)
        """
        self.name = "LKF"

        # Estado LKF: x = [y(k), y(k-1)]
        self.x = np.zeros(2)

        # Matrices LKF (modelo simple de retardo)
        # x(k|k-1) = [y(k-1); y(k-2)]
        self.A = np.array([[1.0, 0.0],
                           [1.0, 0.0]])  # se usa sólo para la forma del filtro
        self.C = np.array([[1.0, 0.0]])

        self.P = np.eye(2) * 1.0
        self.Q = np.eye(2) * float(q_val)
        self.R = np.array([[float(r_val)]])

        # Para AR(2)
        self.decim = int(decim) if decim >= 1 else 1
        self._cnt = 0
        self.buf = deque(maxlen=3)  # y[n], y[n-1], y[n-2]

        # Inicialización AR(2) consistente con 60 Hz
        w0 = 2.0 * math.pi * 60.0
        a1_60 = 2.0 * math.cos(w0 * DT_DSP * self.decim)
        self.theta = np.array([a1_60, -1.0], dtype=float)

        # Suavizado de frecuencia
        self.smooth_win = int(smooth_win)
        self.f_buf = deque(maxlen=self.smooth_win)

        # Front-end
        self.bp = IIR_Bandpass()
        self.norm = FastRMS_Normalizer()

        self._last_f = 60.0

    def structural_latency_samples(self) -> int:
        return self.decim + self.smooth_win * self.decim

    def step(self, z_raw):
        # --- Pre-filtrado 60Hz y AGC ---
        z_bp = self.bp.step(z_raw)
        z = self.norm.step(z_bp)

        # --- PREDICCIÓN LKF ---
        # Modelo: x_pred = [y_prev; y_prev_prev]
        x_pred = np.array([self.x[0], self.x[1]])
        P_pred = self.P + self.Q

        # --- UPDATE LKF ---
        y_pred = x_pred[0]
        inn = z - y_pred
        S = P_pred[0, 0] + self.R[0, 0]
        if S <= 0.0:
            S = 1e-9
        K = P_pred[:, 0] / S
        self.x = x_pred + K * inn
        self.P = (np.eye(2) - np.outer(K, self.C)) @ P_pred

        # Señal filtrada
        y_filt = float(self.x[0])

        # --- AR(2) cada 'decim' muestras ---
        self.buf.append(y_filt)
        self._cnt += 1
        if len(self.buf) < 3:
            return 60.0
        if self._cnt < self.decim:
            return self._last_f
        self._cnt = 0

        # Datos AR(2): y[n] = a1 y[n-1] + a2 y[n-2]
        y0, y1, y2 = self.buf[-1], self.buf[-2], self.buf[-3]
        phi = np.array([y1, y2])
        denom = (phi @ phi) + 1e-6
        a1_est = (y0 * y1) / denom
        # Modelo linealizado: a2 ~ -1 para oscilador subamortiguado
        a2_est = -1.0

        self.theta = np.array([a1_est, a2_est], dtype=float)

        # AR(2) -> frecuencia usando el dt efectivo (decimado)
        f_inst = _ar2_to_freq(self.theta[0], dt=DT_DSP * self.decim)

        if math.isnan(f_inst) or f_inst < 40.0 or f_inst > 80.0:
            f_inst = self._last_f

        self.f_buf.append(f_inst)
        self._last_f = f_inst

        if len(self.f_buf) < self.smooth_win:
            return 60.0

        return float(np.mean(self.f_buf))


def tune_lkf(v, f, q_vals, r_vals, sc_name=None):
    """
    Grid search para el LKF_Estimator, análogo a tune_ukf():

      - q_vals: lista de posibles Q (ruido de proceso)
      - r_vals: lista de posibles R (ruido de medición)

    Devuelve:
      p_str: string con la combinación ganadora
      (q_opt, r_opt): tupla de valores óptimos
    """
    best = {"RMSE": 1e9}

    for q in q_vals:
        for r in r_vals:
            algo = LKF_Estimator(q_val=q, r_val=r, smooth_win=20, decim=10)
            tr = np.array([algo.step(x) for x in v])

            m = calculate_metrics(
                tr, f, 0.0,
                structural_samples=algo.structural_latency_samples()
            )

            if m["RMSE"] < best["RMSE"]:
                best = {
                    "RMSE": m["RMSE"],
                    "p": f"Q{q},R{r}",
                    "v": (q, r)
                }

    # Fallback razonable por si algo raro pasa
    return best["p"], best.get("v", (1e-4, 1e-2))