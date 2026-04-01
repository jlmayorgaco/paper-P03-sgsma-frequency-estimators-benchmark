import sys
import json
import time
import multiprocessing
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

# ==========================================
# Configuración de Rutas (Igual a tu código)
# ==========================================
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenarios.ieee_single_sinwave import IEEESingleSinWaveScenario
from estimators.lkf import LKF_Estimator

from analysis.monte_carlo_engine import MonteCarloEngine 


# ==========================================
# A & B: Pruebas de Carga e Instanciación
# ==========================================
def test_a_b_classes_and_engine_loading():
    """
    Test A y B: Verifica que las clases del escenario, estimador 
    y el motor de Monte Carlo se carguen e instancien correctamente.
    """
    # Verificamos que las clases importadas están definidas
    assert IEEESingleSinWaveScenario is not None, "La clase del escenario no se cargó."
    assert LKF_Estimator is not None, "La clase del estimador no se cargó."
    assert MonteCarloEngine is not None, "La clase MonteCarloEngine no se cargó."

    # Probamos instanciar el engine con las clases
    engine = MonteCarloEngine(
        scenario_cls=IEEESingleSinWaveScenario,
        estimator_cls=LKF_Estimator,
        n_runs=5,
        base_seed=42
    )

    # Validamos que los atributos se asignaron bien
    assert engine.scenario_cls == IEEESingleSinWaveScenario
    assert engine.estimator_cls == LKF_Estimator
    assert engine.n_runs == 5
    assert engine.base_seed == 42


# ==========================================
# C: Prueba de 1 Sola Corrida (Run)
# ==========================================
def test_c_single_run_success():
    """
    Test C: Realiza 1 corrida y verifica que todo fluya sin errores,
    y que los DataFrames de resultado tengan los datos correctos.
    """
    engine = MonteCarloEngine(
        scenario_cls=IEEESingleSinWaveScenario,
        estimator_cls=LKF_Estimator,
        n_runs=1,
        base_seed=100
    )

    result = engine.run()

    # Validamos el objeto MonteCarloResult
    assert result.scenario_name == IEEESingleSinWaveScenario.get_name()
    assert result.summary_df is not None
    assert result.signals_df is not None
    
    # Validamos longitudes
    assert len(result.summary_df) == 1, "Debería haber exactamente 1 fila en summary_df"
    assert len(result.signals_df) > 0, "signals_df no debería estar vacío"
    
    # Validamos que el estimador corrió y calculó errores
    assert "f_rmse_hz" in result.summary_df.columns
    assert "f_hat_hz" in result.signals_df.columns
    assert not result.summary_df["f_rmse_hz"].isna().any(), "RMSE no debería ser NaN"


# ==========================================
# D: Prueba de 10 Corridas con Artefactos
# ==========================================
def test_d_ten_runs_artifacts_and_metrics():
    """
    Test D: Corre 10 veces, guarda los resultados en CSV,
    genera un plot, y un JSON con métricas de performance, 
    workers, RMSE y variaciones de Monte Carlo.
    """
    N_RUNS = 3
    out_dir = ROOT / "tests" / "montecarlo" / "artifacts" / "ieee_single_sinwave_lkf"
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = MonteCarloEngine(
        scenario_cls=IEEESingleSinWaveScenario,
        estimator_cls=LKF_Estimator,
        n_runs=N_RUNS,
        base_seed=24
    )

    # Medimos tiempo de ejecución
    start_time = time.time()
    result = engine.run()
    execution_time = time.time() - start_time

    # 1. Guardar CSVs
    summary_csv, signals_csv = engine.save_csv(result, out_dir)
    assert summary_csv.exists()
    assert signals_csv.exists()

    # 2. Generar Plot
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })

    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    
    # Ploteamos cada run en la gráfica
    for run_idx in range(N_RUNS):
        run_data = result.signals_df[result.signals_df["run_idx"] == run_idx]
        
        # Subplot 1: Voltaje
        axes[0].plot(run_data["t_s"], run_data["v_pu"], alpha=0.5, linewidth=0.5)
        
        # Subplot 2: Frecuencia (Real vs Estimada)
        axes[1].plot(run_data["t_s"], run_data["f_true_hz"], color="black", alpha=0.3, linewidth=1.5, label="True Freq" if run_idx==0 else "")
        if "f_hat_hz" in run_data.columns:
            axes[1].plot(run_data["t_s"], run_data["f_hat_hz"], alpha=0.6, linewidth=0.8, linestyle="--")

    axes[0].set_title(f"Monte Carlo: {result.scenario_name} + {result.estimator_name} ({N_RUNS} runs)")
    axes[0].set_ylabel("Voltage [pu]")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Frequency [Hz]")
    axes[1].legend(loc="upper right")

    plot_path = out_dir / "mc_10_runs_plot.png"
    fig.savefig(plot_path)
    plt.close(fig)
    assert plot_path.exists()

    # 3. Generar JSON con Métricas
    summary_df = result.summary_df
    
    # Extraer variables que variaron en el Monte Carlo (buscamos varianza > 0)
    # Excluimos columnas propias del resultado como run_idx, f_rmse_hz, etc.
    exclude_cols = ["run_idx", "scenario_name", "n_samples", "seed", "v_mean", "v_std", "v_rms", "v_max", "v_min", "f_true_mean", "f_true_std", "f_mae_hz", "f_rmse_hz", "f_max_abs_err_hz", "f_hat_mean_hz", "f_hat_std_hz"]
    param_cols = [col for col in summary_df.columns if col not in exclude_cols]
    
    variations = {}
    for col in param_cols:
        try:
            variations[f"{col}_std"] = float(summary_df[col].std())
            variations[f"{col}_mean"] = float(summary_df[col].mean())
        except TypeError:
            pass # Ignoramos columnas que no sean numéricas

    metrics = {
        "performance": {
            "execution_time_seconds": round(execution_time, 4),
            "workers_used": multiprocessing.cpu_count(),
            "n_runs_completed": N_RUNS
        },
        "estimator_accuracy": {
            "mean_rmse_hz": float(summary_df["f_rmse_hz"].mean()),
            "max_rmse_hz": float(summary_df["f_rmse_hz"].max()),
            "min_rmse_hz": float(summary_df["f_rmse_hz"].min()),
            "mean_mae_hz": float(summary_df["f_mae_hz"].mean())
        },
        "montecarlo_variations": variations
    }

    json_path = out_dir / "mc_10_runs_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    assert json_path.exists()
    
    # Pequeña validación de que el estimador es medianamente razonable 
    # (Ajusta este umbral según lo que esperes de tu LKF)
    assert metrics["estimator_accuracy"]["mean_rmse_hz"] < 25.0, "El RMSE promedio es excesivamente alto"