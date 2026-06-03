import os
# --- PROTECCIÓN MULTIPROCESSING ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import json
import time
import gc
import multiprocessing
from pathlib import Path
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import optuna
from tqdm import tqdm

# Silenciar Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
ORIGINAL_CPU_COUNT = multiprocessing.cpu_count

# ==========================================
# Configuración de Rutas e Importaciones
# ==========================================
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis.monte_carlo_engine import MonteCarloEngine
from estimators.common import F_NOM 

# ==========================================
# IMPORTACIÓN DE ESCENARIOS
# ==========================================
from scenarios.ieee_single_sinwave import IEEESingleSinWaveScenario
from scenarios.ieee_mag_step import IEEEMagStepScenario
from scenarios.ieee_freq_step import IEEEFreqStepScenario
from scenarios.ieee_freq_ramp import IEEEFreqRampScenario
from scenarios.ieee_modulation import IEEEModulationScenario
from scenarios.ieee_modulation_am import IEEEModulationAMScenario
from scenarios.ieee_modulation_fm import IEEEModulationFMScenario
from scenarios.ieee_oob_interference import IEEEOOBInterferenceScenario
from scenarios.ieee_phase_jump_20 import IEEEPhaseJump20Scenario
from scenarios.ieee_phase_jump_60 import IEEEPhaseJump60Scenario
from scenarios.nerc_phase_jump_60 import NERCPhaseJump60Scenario
from scenarios.ibr_multi_event import IBRMultiEventScenario
from scenarios.ibr_power_imbalance_ringdown import IBRPowerImbalanceRingdownScenario

# Escenarios activos para el Pipeline
SCENARIOS = [
    IEEESingleSinWaveScenario,
    IEEEMagStepScenario,
]

OUT_DIR = ROOT / "tests" / "montecarlo" / "artifacts" / "master_benchmark_all"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# DICCIONARIO DE ESPACIOS DE BÚSQUEDA (RANGOS SEGUROS)
# =====================================================================
SEARCH_SPACES = {
    "PLL": lambda trial: {"kp_scale": trial.suggest_float("kp_scale", 0.001, 1.0, log=True), "ki_scale": trial.suggest_float("ki_scale", 0.001, 1.0, log=True)},
    "SOGI-PLL": lambda trial: {"settle_time": trial.suggest_float("settle_time", 0.01, 0.2), "k_sogi": trial.suggest_float("k_sogi", 0.1, 2.0)},
    "Type3-SOGI-PLL": lambda trial: {"settle_time": trial.suggest_float("settle_time", 0.01, 0.2), "k_sogi": trial.suggest_float("k_sogi", 0.1, 2.0)},
    "EPLL": lambda trial: {"kp": trial.suggest_float("kp", 1.0, 60.0), "ki": trial.suggest_float("ki", 1.0, 200.0)},
    "SOGI-FLL": lambda trial: {"k_sogi": trial.suggest_float("k_sogi", 0.5, 2.0), "gamma_fll": trial.suggest_float("gamma_fll", 5.0, 150.0)},
    "EKF": lambda trial: {"q_omega": trial.suggest_float("q_omega", 1e-4, 1.0, log=True), "r_meas": trial.suggest_float("r_meas", 1e-4, 1.0, log=True)},
    "RA-EKF": lambda trial: {"q_rocof": trial.suggest_float("q_rocof", 1e-4, 10.0, log=True), "r_meas": trial.suggest_float("r_meas", 1e-4, 1.0, log=True), "gamma": trial.suggest_float("gamma", 1.0, 5.0)},
    "UKF": lambda trial: {"q_omega": trial.suggest_float("q_omega", 1e-4, 1.0, log=True), "r_meas": trial.suggest_float("r_meas", 1e-4, 1.0, log=True)},
    "LKF": lambda trial: {"q": trial.suggest_float("q", 1e-6, 1e-2, log=True), "r": trial.suggest_float("r", 1e-4, 1.0, log=True)},
    "LKF2": lambda trial: {"q_dc": trial.suggest_float("q_dc", 1e-5, 1e-2, log=True), "q_vc": trial.suggest_float("q_vc", 1e-5, 1e-2, log=True), "q_vs": trial.suggest_float("q_vs", 1e-5, 1e-2, log=True), "r": trial.suggest_float("r", 1e-4, 1.0, log=True)},
    "RLS": lambda trial: {"lambda_fixed": trial.suggest_float("lambda_fixed", 0.90, 0.999)},
    "IpDFT": lambda trial: {"window_cycles": trial.suggest_categorical("window_cycles", [2.0, 3.0, 4.0, 6.0])},
    "TFT": lambda trial: {"window_cycles": trial.suggest_categorical("window_cycles", [2.0, 3.0, 4.0, 6.0])},
    "TKEO": lambda trial: {"output_smoothing": trial.suggest_float("output_smoothing", 0.01, 0.2)}
}

# ==========================================
# Importación de TODOS los Estimadores
# ==========================================
AVAILABLE_ESTIMATORS = {}
def try_import(module_name, class_name):
    try:
        mod = __import__(f"estimators.{module_name}", fromlist=[class_name])
        cls = getattr(mod, class_name)
        est_name = cls.name if hasattr(cls, 'name') else class_name
        AVAILABLE_ESTIMATORS[est_name] = cls
    except (ImportError, AttributeError):
        pass 

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
try_import("pi_gru", "PI_GRU_Estimator") 

def get_timestamp(): return datetime.now().strftime("%H:%M:%S")

# =========================================================================================
# STAGE 0: SMOKE TEST (Duración Extendida 5s + Optimal Tuning en Estado Estacionario)
# =========================================================================================
# =========================================================================================
# STAGE 0: SMOKE TEST (VERSIÓN CORREGIDA - INITIAL FREQUENCY FIX)
# =========================================================================================
def stage0_smoke_test(n_tune_trials=20):
    SIM_DURATION = 5.0 
    SETTLE_TIME_THRESHOLD = 1.0 
    
    print(f"\n[{get_timestamp()}] === STAGE 0: SMOKE TEST (Onda {F_NOM}Hz | Fix Initial State) ===")
    smoke_dir = OUT_DIR / "00_Smoke_Test"
    smoke_dir.mkdir(exist_ok=True)
    
    # 1. Generación de señal ideal (Lógica basada en tus tests de validación)
    sc = IEEESingleSinWaveScenario.run(
        duration_s=SIM_DURATION,
        amplitude=1.0,
        freq_hz=F_NOM,
        phase_rad=0.0,
        noise_sigma=0.0,
        seed=42
    )
    
    results = []
    timeseries_data = {"t_s": sc.t, "v_pu": sc.v, "f_true": sc.f_true}
    steady_state_mask = sc.t >= SETTLE_TIME_THRESHOLD
    
    for est_name, est_cls in AVAILABLE_ESTIMATORS.items():
        try:
            print(f"    [*] Procesando {est_name}...")
            
            # --- CORRECCIÓN 1: Asegurar f_nominal en 60Hz ---
            best_params = est_cls.default_params() if hasattr(est_cls, 'default_params') else {}
            # Intentamos inyectar la frecuencia nominal en cualquier variante de nombre que use la clase
            for key in ["f_nominal", "f0", "fn", "freq_nominal"]:
                if key in inspect.signature(est_cls.__init__).parameters:
                    best_params[key] = F_NOM

            # --- CORRECCIÓN 2: Tuning centrado en convergencia ---
            if est_name in SEARCH_SPACES:
                print(f"        -> Optimizando para 60Hz ({n_tune_trials} trials)...")
                def objective(trial):
                    test_params = SEARCH_SPACES[est_name](trial)
                    merged = best_params.copy()
                    merged.update(test_params)
                    # Forzamos de nuevo f_nominal en el trial
                    for key in ["f_nominal", "f0", "fn"]:
                        if key in merged: merged[key] = F_NOM
                    
                    try:
                        est_t = est_cls(**merged)
                        # Procesar solo 2 segundos para el tuning (más rápido)
                        v_short = sc.v[:20000] # asumiendo 10kHz
                        f_trial = est_t.step_vectorized(v_short) if hasattr(est_t, "step_vectorized") else np.array([est_t.step(float(v)) for v in v_short])
                        
                        # Penalizamos si se queda en 0 o 50Hz
                        mask_short = (np.linspace(0, 2, len(f_trial)) >= 0.5)
                        rmse = np.sqrt(np.mean((f_trial[mask_short] - F_NOM)**2))
                        return rmse if np.isfinite(rmse) else 1e6
                    except Exception: return 1e6

                study = optuna.create_study(direction="minimize")
                study.optimize(objective, n_trials=n_tune_trials, show_progress_bar=False)
                if study.best_value < 100: # Si encontró algo decente
                    best_params.update(study.best_params)

            # --- EVALUACIÓN FINAL ---
            est = est_cls(**best_params)
            f_hat = est.step_vectorized(sc.v) if hasattr(est, "step_vectorized") else np.array([est.step(float(v)) for v in sc.v])
            
            # Limpieza de NaNs (para el RLS si explota)
            f_hat = np.nan_to_num(f_hat, nan=0.0, posinf=F_NOM+100, neginf=0.0)
            
            rmse_ss = float(np.sqrt(np.mean((f_hat[steady_state_mask] - sc.f_true[steady_state_mask])**2)))
            
            results.append({"Estimator": est_name, "RMSE_SS_Hz": rmse_ss, "Params": str(best_params)})
            timeseries_data[f"{est_name}_f_hat"] = f_hat
            
            print(f"        -> OK. RMSE Final: {rmse_ss:.6f} Hz")

            # --- PLOTEO INDIVIDUAL CON ESCALA FIJA ---
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
            ax1.plot(sc.t, sc.v, color="navy", lw=0.5)
            ax1.set_ylabel("Voltaje [pu]")
            ax1.set_title(f"Smoke Test: {est_name}\nRMSE Post-Asentamiento: {rmse_ss:.4f} Hz")
            
            ax2.plot(sc.t, sc.f_true, 'k--', alpha=0.5, label="Target (60Hz)")
            ax2.plot(sc.t, f_hat, 'r-', lw=1.2, label="Estimación")
            ax2.axvline(x=SETTLE_TIME_THRESHOLD, color='green', ls=':', label="Zona de Evaluación")
            
            # Ajuste de escala Y para ver qué pasó
            if rmse_ss > 10:
                ax2.set_ylim(-5, 75) # Ver si está en 0 o si explotó
            else:
                ax2.set_ylim(F_NOM - 2, F_NOM + 2) # Zoom a los 60Hz
                
            ax2.set_ylabel("Frecuencia [Hz]")
            ax2.set_xlabel("Tiempo [s]")
            ax2.legend(loc="upper right")
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            fig.savefig(smoke_dir / f"smoke_test_{est_name}.png", dpi=300)
            plt.close(fig)
            
        except Exception as e:
            print(f"        -> FALLÓ CRÍTICAMENTE: {e}")
            
    if results:
        pd.DataFrame(results).sort_values("RMSE_SS_Hz").to_csv(smoke_dir / "smoke_test_metrics.csv", index=False)

# =========================================================================================
# STAGES 1, 2 Y 3 (Simulación Monte Carlo, ETL y Ploteo Final)
# =========================================================================================
def stage1_simulate(n_mc_runs=10, n_tune_trials=10, force_recompute=False):
    print(f"\n[{get_timestamp()}] === STAGE 1: SIMULACIÓN MONTE CARLO ===")
    for scenario_cls in SCENARIOS:
        scen_name = scenario_cls.get_name() if hasattr(scenario_cls, 'get_name') else scenario_cls.__name__
        scen_dir = OUT_DIR / scen_name
        scen_dir.mkdir(exist_ok=True)
        for est_name, est_cls in AVAILABLE_ESTIMATORS.items():
            est_dir = scen_dir / est_name.replace("-", "_")
            est_dir.mkdir(exist_ok=True)
            meta_path = est_dir / "run_meta.json"
            if meta_path.exists() and not force_recompute: continue
            
            print(f"[*] Simulando {est_name} en {scen_name}...")
            best_params = est_cls.default_params() if hasattr(est_cls, 'default_params') else {}
            # Tuning simplificado para MC
            if est_name in SEARCH_SPACES:
                def objective(trial):
                    p = SEARCH_SPACES[est_name](trial)
                    try:
                        eng = MonteCarloEngine(scenario_cls, est_cls, n_runs=2, base_seed=42, estimator_params=p)
                        res = eng.run()
                        return res.summary_df["m1_rmse_hz"].mean()
                    except Exception: return float('inf')
                study = optuna.create_study(direction="minimize")
                study.optimize(objective, n_trials=n_tune_trials, show_progress_bar=False)
                if study.best_value < float('inf'): best_params.update(study.best_params)

            try:
                engine = MonteCarloEngine(scenario_cls, est_cls, n_runs=n_mc_runs, base_seed=100, estimator_params=best_params)
                start_time = time.time()
                result = engine.run()
                engine.save_csv(result, est_dir)
                meta = {"Metadata": {"Scenario": scen_name, "Estimator": est_name, "Execution_Time_s": time.time()-start_time}, "Tuning": {"Optimal_Params": best_params}}
                with open(meta_path, "w") as f: json.dump(meta, f, indent=4)
            except Exception as e: print(f"Error: {e}")
            gc.collect()

def stage2_process_metrics():
    print(f"\n[{get_timestamp()}] === STAGE 2: ETL ===")
    all_metrics = []
    for scen_dir in OUT_DIR.iterdir():
        if not scen_dir.is_dir() or scen_dir.name == "00_Smoke_Test": continue
        for est_dir in scen_dir.iterdir():
            if not est_dir.is_dir(): continue
            summary_csv = list(est_dir.glob("*_summary.csv"))
            if summary_csv:
                df = pd.read_csv(summary_csv[0])
                all_metrics.append({"Scenario": scen_dir.name, "Estimator": est_dir.name, "RMSE": df["m1_rmse_hz"].mean(), "CPU_Time": df["m13_cpu_time_us"].mean(), "Trip_Risk": df["m5_trip_risk_s"].mean()})
            signals_csv = list(est_dir.glob("*_signals.csv"))
            if signals_csv and not (est_dir / "aggregated_signals.csv").exists():
                df_sig = pd.read_csv(signals_csv[0])
                df_agg = df_sig.groupby("t_s").agg(v_mean=("v_pu", "mean"), f_true=("f_true_hz", "mean"), f_hat_mean=("f_hat_hz", "mean"), f_hat_std=("f_hat_hz", "std")).reset_index()
                df_agg.to_csv(est_dir / "aggregated_signals.csv", index=False)
    if all_metrics: pd.DataFrame(all_metrics).to_csv(OUT_DIR / "GLOBAL_Metrics_Master.csv", index=False)

def stage3_plot_results():
    print(f"\n[{get_timestamp()}] === STAGE 3: PLOTS ===")
    master_csv = OUT_DIR / "GLOBAL_Metrics_Master.csv"
    if not master_csv.exists(): return
    df_all = pd.read_csv(master_csv)
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    
    # Heatmap
    table_rmse = df_all.pivot(index="Estimator", columns="Scenario", values="RMSE")
    plt.figure(figsize=(10, 6))
    sns.heatmap(table_rmse, annot=True, fmt=".3f", cmap="Reds", norm=matplotlib.colors.LogNorm())
    plt.title("Benchmark Global: RMSE [Hz]")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "Plot_A_Matrix_RMSE.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    # 0. Smoke Test: Verificamos precisión en 60Hz con tuning óptimo y 5 segundos
    stage0_smoke_test(n_tune_trials=10)
    
    # 1. Simulación completa (Opcional, consume tiempo)
    # stage1_simulate(n_mc_runs=5, n_tune_trials=10)
    # stage2_process_metrics()
    # stage3_plot_results()