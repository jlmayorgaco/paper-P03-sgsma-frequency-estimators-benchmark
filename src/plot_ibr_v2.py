import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import importlib
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
np.random.seed(42)

import estimators as est
importlib.reload(est)


# =============================================================================
# Helpers
# =============================================================================
def numerical_rocof(t: np.ndarray, f: np.ndarray) -> np.ndarray:
    dt = np.mean(np.diff(t))
    return np.gradient(f, dt)


def extrema_in_window(
    tvec: np.ndarray,
    fvec: np.ndarray,
    t0: float,
    t1: float,
    mode: str = "min",
) -> tuple[float, float]:
    m = (tvec >= t0) & (tvec <= t1)
    tt = tvec[m]
    ff = fvec[m]
    if len(tt) == 0:
        return np.nan, np.nan
    idx = np.argmin(ff) if mode == "min" else np.argmax(ff)
    return float(tt[idx]), float(ff[idx])


# =============================================================================
# OLD STYLE: more aggressive, less physical / more synthetic-looking
# =============================================================================
def build_oldstyle_ibr_multievent_frequency(
    t: np.ndarray,
    f_nom: float = 60.0,
) -> tuple[np.ndarray, dict]:
    t0 = 1.00
    tau = np.maximum(t - t0, 0.0)
    gate = (t >= t0).astype(float)

    # More aggressive shape
    A1 = 0.78
    alpha1 = 5.8
    f1 = 2.55
    phi1 = np.pi

    A2 = 0.34
    alpha2 = 8.5
    f2 = 4.20
    phi2 = -0.35

    A3 = 0.20
    alpha3 = 2.7
    f3 = 1.45
    phi3 = 0.55

    mode1 = A1 * np.exp(-alpha1 * tau) * np.sin(2.0 * np.pi * f1 * tau + phi1)
    mode2 = A2 * np.exp(-alpha2 * tau) * np.sin(2.0 * np.pi * f2 * tau + phi2)
    mode3 = A3 * np.exp(-alpha3 * tau) * np.sin(2.0 * np.pi * f3 * tau + phi3)

    tau_s = np.maximum(t - (t0 + 0.42), 0.0)
    A_osc1 = 0.092
    alpha_osc1 = 1.50
    f_osc1 = 1.65
    phi_osc1 = 0.10

    A_osc2 = 0.023
    alpha_osc2 = 3.40
    f_osc2 = 3.00
    phi_osc2 = 0.70

    osc1 = A_osc1 * np.exp(-alpha_osc1 * tau_s) * np.sin(2.0 * np.pi * f_osc1 * tau_s + phi_osc1)
    osc2 = A_osc2 * np.exp(-alpha_osc2 * tau_s) * np.sin(2.0 * np.pi * f_osc2 * tau_s + phi_osc2)

    tau_sub = np.maximum(t - (t0 + 0.60), 0.0)
    A_sub = 0.010
    alpha_sub = 0.72
    f_sub = 0.42
    phi_sub = -0.20
    sub = A_sub * np.exp(-alpha_sub * tau_sub) * np.sin(2.0 * np.pi * f_sub * tau_sub + phi_sub)

    f = f_nom + gate * (mode1 + mode2 + mode3 + osc1 + osc2 + sub)

    meta = {
        "label": "oldstyle",
        "notes": "More aggressive, more synthetic-looking, stronger dip/overshoot.",
    }
    return f, meta


# =============================================================================
# NEW STYLE: smoother, more physical, better overall shape
# =============================================================================
def build_newstyle_ibr_multievent_frequency(
    t: np.ndarray,
    f_nom: float = 60.0,
) -> tuple[np.ndarray, dict]:
    t0 = 1.00
    tau = np.maximum(t - t0, 0.0)
    gate = (t >= t0).astype(float)

    # Smoother, more sine-like, faster rebound
    A1 = 0.46
    alpha1 = 5.2
    f1 = 3.6
    phi1 = np.pi

    A2 = 0.17
    alpha2 = 3.0
    f2 = 1.8
    phi2 = 0.10

    A3 = 0.05
    alpha3 = 7.5
    f3 = 5.2
    phi3 = -0.25

    mode1 = A1 * np.exp(-alpha1 * tau) * np.sin(2.0 * np.pi * f1 * tau + phi1)
    mode2 = A2 * np.exp(-alpha2 * tau) * np.sin(2.0 * np.pi * f2 * tau + phi2)
    mode3 = A3 * np.exp(-alpha3 * tau) * np.sin(2.0 * np.pi * f3 * tau + phi3)

    tau_s = np.maximum(t - (t0 + 0.22), 0.0)
    A_set = 0.030
    alpha_set = 1.45
    f_set = 1.55
    phi_set = 0.20
    settle = A_set * np.exp(-alpha_set * tau_s) * np.sin(2.0 * np.pi * f_set * tau_s + phi_set)

    tau_sub = np.maximum(t - (t0 + 0.55), 0.0)
    A_sub = 0.008
    alpha_sub = 0.70
    f_sub = 0.42
    phi_sub = -0.20
    sub = A_sub * np.exp(-alpha_sub * tau_sub) * np.sin(2.0 * np.pi * f_sub * tau_sub + phi_sub)

    f = f_nom + gate * (mode1 + mode2 + mode3 + settle + sub)

    meta = {
        "label": "newstyle",
        "notes": "Cleaner, more physical, smoother PID-ish response.",
    }
    return f, meta


# =============================================================================
# Load current active scenario from repo
# =============================================================================
sigs = est.get_test_signals(seed=42)
t_base, v_base, f_base, meta_base = sigs["IBR_MultiEvent"]

RATIO = 100
t_p = t_base[::RATIO]
f_active = f_base[::RATIO]
v_p = v_base[::RATIO]

f_nom = float(np.round(np.median(f_active[: min(1000, len(f_active))]), 0))

f_oldstyle, meta_old = build_oldstyle_ibr_multievent_frequency(t_p, f_nom=f_nom)
f_newstyle, meta_new = build_newstyle_ibr_multievent_frequency(t_p, f_nom=f_nom)

rocof_active = numerical_rocof(t_p, f_active)
rocof_oldstyle = numerical_rocof(t_p, f_oldstyle)
rocof_newstyle = numerical_rocof(t_p, f_newstyle)


# =============================================================================
# FIG 1: Compare oldstyle vs newstyle
# =============================================================================
fig1, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
fig1.suptitle(
    "IBR Multi-Event: oldstyle vs newstyle proposal",
    fontsize=14,
    fontweight="bold",
)

# Full trajectory
ax = axes[0]
ax.plot(t_p, f_oldstyle, color="tab:blue", lw=1.6, label="Oldstyle (more aggressive)")
ax.plot(t_p, f_newstyle, color="tab:red", lw=1.6, label="Newstyle (cleaner / more physical)")
ax.axhline(f_nom, color="k", lw=0.7, ls="--", alpha=0.4, label=f"Nominal {f_nom:.0f} Hz")
ax.set_ylabel("Frequency [Hz]")
ax.set_ylim(f_nom - 0.70, f_nom + 0.50)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)
ax.set_title("Full trajectory")

# Transient zoom
ax = axes[1]
m = (t_p >= 0.95) & (t_p <= 2.10)
ax.plot(t_p[m], f_oldstyle[m], color="tab:blue", lw=1.9, label="Oldstyle")
ax.plot(t_p[m], f_newstyle[m], color="tab:red", lw=1.9, label="Newstyle")
ax.axhline(f_nom, color="k", lw=0.7, ls="--", alpha=0.4)
ax.set_ylabel("Frequency [Hz]")
ax.set_ylim(f_nom - 0.70, f_nom + 0.50)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)
ax.set_title("Transient zoom")

# Settling
ax = axes[2]
m = t_p >= 1.60
ax.plot(t_p[m], f_oldstyle[m], color="tab:blue", lw=1.6, label="Oldstyle")
ax.plot(t_p[m], f_newstyle[m], color="tab:red", lw=1.6, label="Newstyle")
ax.axhline(f_nom, color="k", lw=0.7, ls="--", alpha=0.4)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Frequency [Hz]")
ax.set_ylim(f_nom - 0.10, f_nom + 0.10)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)
ax.set_title("Settling comparison")

plt.tight_layout()
out1 = "figures_estimatores_benchmark_test5/IBR_MultiEvent_oldstyle_vs_newstyle.png"
fig1.savefig(out1, dpi=160, bbox_inches="tight")
print("Saved:", out1)


# =============================================================================
# FIG 2: Compare against active scenario
# =============================================================================
fig2, axes2 = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
fig2.suptitle(
    "IBR Multi-Event: active vs oldstyle vs newstyle",
    fontsize=14,
    fontweight="bold",
)

# Full
ax = axes2[0]
ax.plot(t_p, f_active, color="gray", lw=1.3, ls="--", label="Active scenario")
ax.plot(t_p, f_oldstyle, color="tab:blue", lw=1.5, label="Oldstyle")
ax.plot(t_p, f_newstyle, color="tab:red", lw=1.5, label="Newstyle")
ax.axhline(f_nom, color="k", lw=0.7, ls=":", alpha=0.4)
ax.set_ylabel("Frequency [Hz]")
ax.set_ylim(f_nom - 0.70, f_nom + 0.50)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)
ax.set_title("Full trajectory")

# Zoom
ax = axes2[1]
m = (t_p >= 0.95) & (t_p <= 2.10)
ax.plot(t_p[m], f_active[m], color="gray", lw=1.3, ls="--", label="Active")
ax.plot(t_p[m], f_oldstyle[m], color="tab:blue", lw=1.8, label="Oldstyle")
ax.plot(t_p[m], f_newstyle[m], color="tab:red", lw=1.8, label="Newstyle")
ax.axhline(f_nom, color="k", lw=0.7, ls=":", alpha=0.4)
ax.set_ylabel("Frequency [Hz]")
ax.set_ylim(f_nom - 0.70, f_nom + 0.50)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)
ax.set_title("Transient zoom")

# RoCoF
ax = axes2[2]
ax.plot(t_p, rocof_active, color="gray", lw=1.0, ls="--", label="Active")
ax.plot(t_p, rocof_oldstyle, color="tab:blue", lw=1.0, label="Oldstyle")
ax.plot(t_p, rocof_newstyle, color="tab:red", lw=1.0, label="Newstyle")
ax.axhline(0.0, color="k", lw=0.7, ls="--", alpha=0.4)
ax.set_xlabel("Time [s]")
ax.set_ylabel("RoCoF [Hz/s]")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)
ax.set_title("RoCoF comparison")

plt.tight_layout()
out2 = "figures_estimatores_benchmark_test5/IBR_MultiEvent_active_old_new_compare.png"
fig2.savefig(out2, dpi=160, bbox_inches="tight")
print("Saved:", out2)
plt.close("all")


print("\nVerdict:")
print("  Newstyle = better overall shape / more physical / cleaner control-like response")
print("  Oldstyle = more aggressive but more synthetic-looking")
print("Done.")