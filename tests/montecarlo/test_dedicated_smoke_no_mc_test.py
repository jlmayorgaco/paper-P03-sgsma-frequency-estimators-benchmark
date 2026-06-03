import os
# --- PROTECCIÓN MULTIPROCESSING ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import time
import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optuna

# Configuración de Rutas
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Importar todos los escenarios
from scenarios.ieee_single_sinwave import IEEESingleSinWaveScenario
from scenarios.ieee_freq_step import IEEEFreqStepScenario
from scenarios.ieee_mag_step import IEEEMagStepScenario
from scenarios.ieee_freq_ramp import IEEEFreqRampScenario
from scenarios.nerc_phase_jump_60 import NERCPhaseJump60Scenario
from scenarios.ieee_phase_jump_20 import IEEEPhaseJump20Scenario
from scenarios.ieee_phase_jump_60 import IEEEPhaseJump60Scenario
from scenarios.ieee_modulation import IEEEModulationScenario
from scenarios.ieee_modulation_am import IEEEModulationAMScenario
from scenarios.ieee_modulation_fm import IEEEModulationFMScenario
from scenarios.ieee_modulation_fm import IEEEModulationFMScenario
from scenarios.ibr_power_imbalance_ringdown import IBRPowerImbalanceRingdownScenario
from scenarios.ibr_multi_event import IBRMultiEventScenario

from estimators.common import F_NOM

# Mostrar info de Optuna en consola para seguir el progreso del tuning
optuna.logging.set_verbosity(optuna.logging.INFO)

# ==========================================
# 1. CARGA DINÁMICA DE ESTIMADORES
# ==========================================
AVAILABLE_ESTIMATORS = {}
def try_import(module_name, class_name):
    try:
        mod = __import__(f"estimators.{module_name}", fromlist=[class_name])
        cls = getattr(mod, class_name)
        # Priorizar el atributo 'name' de la clase si existe
        est_name = cls.name if hasattr(cls, 'name') else class_name
        AVAILABLE_ESTIMATORS[est_name] = cls
    except Exception: pass

print("--- Cargando estimadores ---")
try_import("ipdft", "IPDFT_Estimator")
try_import("sogi_pll", "SOGIPLLEstimator")
try_import("ra_ekf", "RAEKF_Estimator")
try_import("pll", "PLL_Estimator")
try_import("ekf", "EKF_Estimator")
try_import("rls", "RLS_Estimator")
try_import("zcd", "ZCDEstimator")
try_import("tft", "TFT_Estimator")
try_import("prony", "Prony_Estimator")
try_import("music", "MUSIC_Estimator")
try_import("esprit", "ESPRIT_Estimator")
try_import("koopman", "Koopman_Estimator")

# ==========================================
# 2. ESPACIOS DE BÚSQUEDA (Refinados)
# ==========================================
SEARCH_SPACES = {
    "PLL": lambda trial: {"kp_scale": trial.suggest_float("kp_scale", 0.001, 1.0, log=True), "ki_scale": trial.suggest_float("ki_scale", 0.001, 1.0, log=True)},
    "SOGI-PLL": lambda trial: {"settle_time": trial.suggest_float("settle_time", 0.01, 0.2), "k_sogi": trial.suggest_float("k_sogi", 0.1, 2.0)},
    "EKF": lambda trial: {"q_omega": trial.suggest_float("q_omega", 1e-4, 1.0, log=True), "r_meas": trial.suggest_float("r_meas", 1e-4, 1.0, log=True)},
    "RA-EKF": lambda trial: {"q_rocof": trial.suggest_float("q_rocof", 1e-5, 1e-2, log=True), "r_meas": trial.suggest_float("r_meas", 1e-4, 1.0, log=True)},
    "RLS": lambda trial: {"lambda_fixed": trial.suggest_float("lambda_fixed", 0.98, 0.9999)},
}

# ==========================================
# 3. LISTA DE ESCENARIOS A EVALUAR
# ==========================================
SCENARIOS_TO_TEST = [
    IEEESingleSinWaveScenario,
    IEEEFreqStepScenario,
    IEEEMagStepScenario,
    IEEEFreqRampScenario,
    NERCPhaseJump60Scenario,
    IEEEPhaseJump20Scenario,
    IEEEPhaseJump60Scenario,
    IEEEModulationScenario,
    IEEEModulationAMScenario,
    IEEEModulationFMScenario,
    IBRPowerImbalanceRingdownScenario,
    IBRMultiEventScenario
]

# ==========================================
# 4. EJECUCIÓN DEL SMOKE TEST MULTI-ESCENARIO
# ==========================================
def run_dedicated_smoke_test(n_trials=15):
    BASE_OUT_DIR = ROOT / "tests" / "montecarlo" / "artifacts" / "dedicated_smoke_fixed"
    
    SIM_DURATION = 5.0
    # Reducido a 0.2s para que el RMSE capture los eventos (escalones/rampas) que ocurren en t=0.3s o 0.5s
    SETTLE_TIME = 0.2 
    
    print(f"\n[{time.strftime('%H:%M:%S')}] --- INICIANDO SMOKE TEST MULTI-ESCENARIO ({SIM_DURATION}s) ---")

    for scenario_cls in SCENARIOS_TO_TEST:
        scenario_name = scenario_cls.SCENARIO_NAME
        OUT_DIR = BASE_OUT_DIR / scenario_name
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'#'*70}")
        print(f"[*] PREPARANDO ESCENARIO: {scenario_name}")
        print(f"{'#'*70}")
        
        # Generar señal del escenario actual
        sc = scenario_cls.run(duration_s=SIM_DURATION, noise_sigma=0.0, seed=42)
        t = sc.t
        v = sc.v
        f_true = sc.f_true
        dt_real = float(t[1] - t[0])
        
        # =========================================================================
        # FIX COMPUTACIONAL: Decimar la señal si viene a 1 MHz o más
        # =========================================================================
        if dt_real < 1e-5: # Si el dt es menor a 10 microsegundos
            target_dt = 1e-4 # Objetivo: 10 kHz
            dec_factor = max(1, int(target_dt / dt_real))
            
            t = t[::dec_factor]
            v = v[::dec_factor]
            f_true = f_true[::dec_factor]
            dt_real = float(t[1] - t[0])
            print(f"\n[!] SEÑAL DECIMADA: fs original era muy alto. Nuevo fs = {1.0/dt_real:.0f} Hz")
        
        mask_ss = t >= SETTLE_TIME
        summary_results = []

        for est_name, est_cls in AVAILABLE_ESTIMATORS.items():
            print(f"\n{'-'*50}")
            print(f"[*] Evaluando: {est_name} en {scenario_name}...")
            print(f"{'-'*50}")
            
            # --- FIX #1 y #2: Inyección de nominal_f y dt real ---
            init_sig = inspect.signature(est_cls.__init__).parameters
            params = est_cls.default_params() if hasattr(est_cls, 'default_params') else {}
            
            # Inyectar F_NOM (60Hz) buscando el nombre correcto de variable
            for key in ("nominal_f", "f_nominal", "f0", "fn", "f_nom"):
                if key in init_sig:
                    params[key] = F_NOM
                    print(f"    -> Inyectado {key}={F_NOM}")
                    break
            
            # Inyectar dt real del escenario
            if "dt" in init_sig:
                params["dt"] = dt_real
                print(f"    -> Inyectado dt={dt_real:.2e}")

            # --- TUNING ÓPTIMO ---
            if est_name in SEARCH_SPACES:
                print(f"    -> Optimizando hiperparámetros (Iniciando Optuna)...")
                def objective(trial):
                    test_p = params.copy()
                    test_p.update(SEARCH_SPACES[est_name](trial))
                    try:
                        est_t = est_cls(**test_p)
                        f_hat_t = est_t.step_vectorized(v) if hasattr(est_t, "step_vectorized") else np.array([est_t.step(float(vi)) for vi in v])
                        rmse = np.sqrt(np.mean((f_hat_t[mask_ss] - f_true[mask_ss])**2))
                        return rmse if np.isfinite(rmse) else 1e6
                    except: return 1e6

                def print_callback(study, trial):
                    print(f"       [Trial {trial.number}/{n_trials}] RMSE: {trial.value:.4f} | Params: {trial.params}")

                study = optuna.create_study(direction="minimize")
                study.optimize(objective, n_trials=n_trials, callbacks=[print_callback])
                params.update(study.best_params)
                print(f"    -> Mejor RMSE en Tuning: {study.best_value:.6f}")
            else:
                print(f"    -> Saltando Tuning (No hay espacio de búsqueda definido para {est_name})")

            # --- SIMULACIÓN FINAL Y DIAGNÓSTICO DE NaNs ---
            est = est_cls(**params)
            
            print(f"    -> Ejecutando simulación final ({len(v)} muestras). Por favor espera...")
            start_eval_time = time.time()
            
            f_hat_raw = est.step_vectorized(v) if hasattr(est, "step_vectorized") else np.array([est.step(float(vi)) for vi in v])
            
            eval_time = time.time() - start_eval_time
            print(f"    -> Simulación completada en {eval_time:.2f} segundos.")
            
            # FIX #5: Reportar NaNs antes de limpiar
            nans = np.isnan(f_hat_raw).sum()
            infs = np.isinf(f_hat_raw).sum()
            if nans > 0 or infs > 0:
                print(f"    [!] ADVERTENCIA: Detectados {nans} NaNs y {infs} Infs")
            
            f_hat = np.nan_to_num(f_hat_raw, nan=0.0) # Limpiar para el plot
            
            # Métricas
            rmse_ss = float(np.sqrt(np.mean((f_hat[mask_ss] - f_true[mask_ss])**2)))
            print(f"    -> RMSE Final (Post-Transient): {rmse_ss:.6f} Hz")

            summary_results.append({
                "Estimator": est_name,
                "RMSE_SS_Hz": rmse_ss,
                "NaN_Count": nans,
                "Params": str(params)
            })

            # --- PLOTEO ---
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            ax1.plot(t, v, color='navy', lw=0.5)
            ax1.set_ylabel("Voltaje [pu]")
            ax1.set_title(f"[{scenario_name}] {est_name} | RMSE: {rmse_ss:.6f} Hz")
            
            ax2.plot(t, f_true, 'k--', alpha=0.5, label="Referencia")
            ax2.plot(t, f_hat, 'r-', lw=1.2, label="Estimación")
            ax2.axvline(x=SETTLE_TIME, color='green', ls=':', label="Inicio Evaluación")
            
            # Zoom inteligente adaptativo según el escenario
            f_min, f_max = np.min(f_true), np.max(f_true)
            if f_max - f_min > 0.5:  # Escenarios con rampas o escalones severos
                ax2.set_ylim(f_min - 1.0, f_max + 1.0)
            elif rmse_ss > 1.0: 
                ax2.set_ylim(F_NOM - 5, F_NOM + 5)
            else: 
                ax2.set_ylim(F_NOM - 0.2, F_NOM + 0.2)
            
            ax2.set_ylabel("Frecuencia [Hz]")
            ax2.set_xlabel("Tiempo [s]")
            ax2.legend(loc="upper right")
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            fig.savefig(OUT_DIR / f"smoke_{est_name}.png", dpi=300)
            plt.close(fig)

        # Guardar resultados del escenario
        df = pd.DataFrame(summary_results).sort_values("RMSE_SS_Hz")
        df.to_csv(OUT_DIR / f"summary_{scenario_name}.csv", index=False)
        print(f"\n[OK] Test completado para {scenario_name}. Resultados en: {OUT_DIR}")

    print(f"\n[{time.strftime('%H:%M:%S')}] --- TODOS LOS ESCENARIOS HAN SIDO EJECUTADOS EXITOSAMENTE ---")

if __name__ == "__main__":
    run_dedicated_smoke_test(n_trials=20)