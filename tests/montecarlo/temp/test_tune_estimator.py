import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
from pathlib import Path
import optuna
import logging

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenarios.ieee_mag_step import IEEEMagStepScenario
from estimators.pll import PLL_Estimator
from analysis.monte_carlo_engine import MonteCarloEngine

# Silenciamos los logs de Optuna para que no ensucien la consola, solo veremos el progreso
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    # 1. DEFINIR EL ESPACIO DE BÚSQUEDA
    # Le decimos a Optuna qué parámetros buscar y en qué rangos
    test_params = {
        "kp_scale": trial.suggest_float("kp_scale", 0.01, 1.0, log=True),
        "ki_scale": trial.suggest_float("ki_scale", 0.001, 0.5, log=True),
        "pd_lpf_alpha": trial.suggest_float("pd_lpf_alpha", 0.001, 0.2),
    }
    
    # 2. EJECUTAR MONTE CARLO
    # Usamos pocas corridas (ej. 5) para que el tuning sea rápido. 
    # El objetivo es encontrar el orden de magnitud correcto.
    try:
        engine = MonteCarloEngine(
            scenario_cls=IEEEMagStepScenario,
            estimator_cls=PLL_Estimator,
            n_runs=5, 
            base_seed=42,
            estimator_params=test_params
        )
        
        result = engine.run()
        
        # 3. EXTRAER LA MÉTRICA A MINIMIZAR (RMSE)
        rmse = result.summary_df["m1_rmse_hz"].mean()
        
        # Opcional: Penalizar si el tiempo de asentamiento es muy alto
        # settling_time = result.summary_df["m8_settling_time_s"].mean()
        # return rmse + (settling_time * 10.0) 
        
        return rmse
        
    except Exception as e:
        # Si una combinación de parámetros hace que el filtro explote (NaNs),
        # le devolvemos un error infinito para que Optuna no vuelva a intentarlo.
        return float('inf')

if __name__ == "__main__":
    print("\n[*] Iniciando Auto-Tuning con Optuna para PLL_Estimator...")
    
    # Creamos un estudio buscando "minimizar" el error
    study = optuna.create_study(direction="minimize")
    
    # Lanzamos 50 intentos inteligentes
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    print("\n[+] ¡Tuning Completado!")
    print(f"Mejor RMSE alcanzado: {study.best_value:.4f} Hz")
    print("Mejores parámetros encontrados:")
    for key, value in study.best_params.items():
        print(f"    '{key}': {value:.4f}")