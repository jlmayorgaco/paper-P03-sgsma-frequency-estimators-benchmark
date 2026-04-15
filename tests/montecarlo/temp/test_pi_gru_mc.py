import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import time
import gc
import multiprocessing
from pathlib import Path
import pandas as pd

# =======================================================
# PARCHE ANTI-DEADLOCK PARA WINDOWS + PYTORCH
# Obligamos al motor a usar 1 solo proceso secuencial
# =======================================================
multiprocessing.cpu_count = lambda: 1  

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis.monte_carlo_engine import MonteCarloEngine
from estimators.pi_gru import PI_GRU_Estimator

try:
    from scenarios.ieee_freq_step import IEEEFreqStepScenario as TestScenario
except ImportError:
    from scenarios.ieee_single_sinwave import IEEESingleSinWaveScenario as TestScenario


def run_pi_gru_montecarlo():
    print("\n" + "="*50)
    print(" VALIDACIÓN AISLADA DE MONTE CARLO: PI-GRU ")
    print("="*50)
    
    n_runs = 10  
    
    out_dir = ROOT / "tests" / "montecarlo" / "artifacts" / "pi_gru_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    scenario_name = TestScenario.get_name() if hasattr(TestScenario, 'get_name') else TestScenario.__name__
    
    print(f"[*] Escenario Objetivo: {scenario_name}")
    print(f"[*] Iteraciones configuradas: {n_runs}\n")
    
    try:
        print("[*] Instanciando MonteCarloEngine (Forzado a 1 proceso)...")
        engine = MonteCarloEngine(
            scenario_cls=TestScenario,
            estimator_cls=PI_GRU_Estimator,
            n_runs=n_runs,
            base_seed=123
        )
        
        start_time = time.time()
        result = engine.run()
        total_time = time.time() - start_time
        
        print(f"\n[+] ¡ÉXITO! Las {n_runs} corridas terminaron en {total_time:.2f} segundos.")
        
        summary_df = result.summary_df
        print(f"[+] Métricas Promedio Extraídas:")
        print(f"    - RMSE: {summary_df['m1_rmse_hz'].mean():.4f} Hz")
        print(f"    - CPU Time: {summary_df['m13_cpu_time_us'].mean():.2f} us")
        
        del result, engine, summary_df
        gc.collect()
        
    except Exception as e:
        print(f"\n[ERROR FATAL] Fallo durante la ejecución: {e}")

if __name__ == "__main__":
    run_pi_gru_montecarlo()