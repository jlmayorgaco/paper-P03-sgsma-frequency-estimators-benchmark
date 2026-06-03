import sys
import json
import time
import gc
import multiprocessing
from pathlib import Path
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

import numpy as np
import pandas as pd
from tqdm import tqdm

# ==========================================
# Configuración de Rutas 
# ==========================================
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenarios.ieee_mag_step import IEEEMagStepScenario
from analysis.monte_carlo_engine import MonteCarloEngine 

OUT_DIR = ROOT / "tests" / "montecarlo" / "artifacts" / "ieee_mag_step_benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# Importación Dinámica de Estimadores
# ==========================================
AVAILABLE_ESTIMATORS = []

def try_import(module_name, class_name):
    try:
        mod = __import__(f"estimators.{module_name}", fromlist=[class_name])
        cls = getattr(mod, class_name)
        AVAILABLE_ESTIMATORS.append(cls)
    except (ImportError, AttributeError):
        pass 

# Batería completa
try_import("ipdft", "IPDFT_Estimator")
try_import("rls", "RLS_Estimator")
try_import("lkf", "LKF_Estimator")
try_import("lkf2", "LKF2_Estimator")
try_import("ekf", "EKF_Estimator")
try_import("pll", "PLL_Estimator")
try_import("sogi_pll", "SOGIPLLEstimator")
try_import("ra_ekf", "RAEKF_Estimator")
try_import("zcd", "ZCDEstimator")
try_import("ukf", "UKF_Estimator")
try_import("type3_sogi_pll", "Type3SOGIPLLEstimator")
try_import("epll", "EPLL_Estimator")
try_import("music", "MUSIC_Estimator")
try_import("tft", "TFT_Estimator")
try_import("sogi_fll", "SOGIFLLEstimator")
try_import("prony", "Prony_Estimator")
try_import("esprit", "ESPRIT_Estimator")
try_import("koopman", "Koopman_Estimator")
try_import("tkeo", "TKEO_Estimator")
#try_import("pi_gru", "PI_GRU_Estimator")

def get_timestamp():
    return datetime.now().strftime("%H:%M:%S")

# ==========================================
# STAGE 1: SIMULACIÓN AISLADA (RAM-SAFE & VERBOSE)
# ==========================================
def stage1_simulate(n_runs=10, force_recompute=False):
    print(f"\n[{get_timestamp()}] --- [STAGE 1] Iniciando Simulaciones ({n_runs} runs/método) ---")
    
    if not AVAILABLE_ESTIMATORS:
        print(f"[{get_timestamp()}] [ERROR] No hay estimadores cargados.")
        return

    # Usamos tqdm pero gestionamos los logs con tqdm.write para no romper la barra
    pbar = tqdm(AVAILABLE_ESTIMATORS, desc="Progreso Global", position=0, leave=True)
    
    for est_cls in pbar:
        est_name = est_cls.name if hasattr(est_cls, 'name') else est_cls.__name__
        est_dir = OUT_DIR / est_name.lower().replace("-", "_")
        est_dir.mkdir(exist_ok=True)
        
        metrics_path = est_dir / "metrics.json"
        
        # 0. Verificación de caché
        if metrics_path.exists() and not force_recompute:
            pbar.set_postfix_str(f"Saltando {est_name} (Ya calculado)")
            continue
            
        pbar.set_postfix_str(f"Procesando {est_name}...")
        tqdm.write(f"\n[{get_timestamp()}] >>> INICIANDO: {est_name} <<<")
        
        try:
            # 1. Instanciación
            tqdm.write(f"[{get_timestamp()}] Paso 1/4: Preparando Motor Monte Carlo para {est_name}...")
            engine = MonteCarloEngine(
                scenario_cls=IEEEMagStepScenario,
                estimator_cls=est_cls,
                n_runs=n_runs,
                base_seed=42 
            )
            
            # 2. Simulación
            tqdm.write(f"[{get_timestamp()}] Paso 2/4: Simulando {n_runs} iteraciones (Esto puede tomar tiempo)...")
            start_time = time.time()
            result = engine.run()
            total_time = time.time() - start_time
            tqdm.write(f"[{get_timestamp()}]           -> ¡Simulación completada en {total_time:.2f} segundos!")
            
            summary_df = result.summary_df
            
            # 3. Guardado en Disco (CSVs)
            tqdm.write(f"[{get_timestamp()}] Paso 3/4: Escribiendo series de tiempo en disco (CSV)...")
            engine.save_csv(result, est_dir)
            
            # 4. Extracción de Métricas y JSON
            tqdm.write(f"[{get_timestamp()}] Paso 4/4: Extrayendo métricas de precisión y hardware...")
            est_params = est_cls.default_params() if hasattr(est_cls, 'default_params') else {}
            
            metrics = {
                "Metadata": {
                    "Estimator": est_name,
                    "Scenario": IEEEMagStepScenario.get_name(),
                    "N_Runs": n_runs,
                    "Timestamp": datetime.now().isoformat(),
                    "Total_MC_Execution_Time_s": round(total_time, 4)
                },
                "Estimator_Config": est_params,
                "Performance_Hardware": {
                    "CPU_Time_us_mean": float(summary_df["m13_cpu_time_us"].mean()),
                    "CPU_Time_us_max": float(summary_df["m13_cpu_time_us"].max()),
                },
                "Precision_Metrics": {
                    "RMSE_Hz_mean": float(summary_df["m1_rmse_hz"].mean()),
                    "RMSE_Hz_std": float(summary_df["m1_rmse_hz"].std()),
                    "Max_Peak_Hz_mean": float(summary_df["m3_max_peak_hz"].mean()),
                    "Settling_Time_s_mean": float(summary_df["m8_settling_time_s"].mean()),
                    "MAE_Hz_mean": float(summary_df["m2_mae_hz"].mean())
                }
            }
            
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=4)
            
            # 5. Limpieza de RAM
            tqdm.write(f"[{get_timestamp()}] Limpiando memoria RAM para el siguiente estimador...")
            del summary_df
            del result.signals_df
            del result.summary_df
            del result
            del engine
            gc.collect()
            tqdm.write(f"[{get_timestamp()}] <<< {est_name} COMPLETADO CON ÉXITO >>>\n")
                
        except Exception as e:
            tqdm.write(f"[{get_timestamp()}] [ERROR FATAL] Fallo en {est_name}: {e}\n")

    print(f"\n[{get_timestamp()}] --- [STAGE 1] Finalizado ---")


# ==========================================
# STAGE 2: AGREGACIÓN Y RANKING
# ==========================================
def stage2_aggregate():
    print(f"\n[{get_timestamp()}] --- [STAGE 2] Agregando Resultados ---")
    
    benchmark_results = []
    
    for metrics_file in OUT_DIR.rglob("metrics.json"):
        with open(metrics_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            flat_data = {
                "Estimator": data["Metadata"]["Estimator"],
                "RMSE_Hz": data["Precision_Metrics"]["RMSE_Hz_mean"],
                "Max_Peak_Hz": data["Precision_Metrics"]["Max_Peak_Hz_mean"],
                "Settling_Time_s": data["Precision_Metrics"]["Settling_Time_s_mean"],
                "CPU_Time_us": data["Performance_Hardware"]["CPU_Time_us_mean"],
                "MC_Total_Time_s": data["Metadata"]["Total_MC_Execution_Time_s"]
            }
            benchmark_results.append(flat_data)
            
    if not benchmark_results:
        print(f"[{get_timestamp()}] [AVISO] No se encontraron archivos metrics.json.")
        return None
        
    df_bench = pd.DataFrame(benchmark_results)
    df_bench = df_bench.sort_values(by="RMSE_Hz")
    
    csv_path = OUT_DIR / "00_benchmark_summary.csv"
    df_bench.to_csv(csv_path, index=False)
    
    print(f"[{get_timestamp()}] Ranking generado con {len(df_bench)} estimadores.")
    return df_bench


# ==========================================
# STAGE 3: GRAFICACIÓN
# ==========================================
def stage3_plot():
    print(f"\n[{get_timestamp()}] --- [STAGE 3] Generando Gráficas ---")
    csv_path = OUT_DIR / "00_benchmark_summary.csv"
    
    if not csv_path.exists():
        print(f"[{get_timestamp()}] [ERROR] No existe el summary CSV.")
        return
        
    df_bench = pd.read_csv(csv_path)
    
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--"
    })

    fig, ax1 = plt.subplots(figsize=(12, 6))

    x = np.arange(len(df_bench))
    width = 0.35

    ax1.bar(x - width/2, df_bench["RMSE_Hz"], width, label='RMSE (Hz)', color='crimson', alpha=0.8)
    ax1.set_ylabel('Mean RMSE [Hz]', color='crimson', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='crimson')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_bench["Estimator"], rotation=45, ha="right")

    ax2 = ax1.twinx()
    ax2.bar(x + width/2, df_bench["CPU_Time_us"], width, label='CPU Time (us)', color='teal', alpha=0.8)
    ax2.set_ylabel('Mean CPU Time [μs]', color='teal', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='teal')
    
    plt.title("Performance Comparison: Magnitude Step Scenario")
    fig.tight_layout()

    plot_path = OUT_DIR / "00_benchmark_comparison_plot.png"
    plt.savefig(plot_path)
    plt.close()
    
    print(f"[{get_timestamp()}] Plot guardado en: {plot_path}")


if __name__ == "__main__":
    stage1_simulate(n_runs=10, force_recompute=False) 
    stage2_aggregate()
    stage3_plot()