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

import platform
from datetime import datetime

# ==========================================
# Configuración de Rutas 
# ==========================================
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenarios.ieee_single_sinwave import IEEESingleSinWaveScenario
from estimators.ipdft import IPDFT_Estimator

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
    assert IPDFT_Estimator is not None, "La clase del estimador no se cargó."
    assert MonteCarloEngine is not None, "La clase MonteCarloEngine no se cargó."

    # Probamos instanciar el engine con las clases
    engine = MonteCarloEngine(
        scenario_cls=IEEESingleSinWaveScenario,
        estimator_cls=IPDFT_Estimator,
        n_runs=5,
        base_seed=42
    )

    # Validamos que los atributos se asignaron bien
    assert engine.scenario_cls == IEEESingleSinWaveScenario
    assert engine.estimator_cls == IPDFT_Estimator
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
        estimator_cls=IPDFT_Estimator,
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
    
    # Validamos que el estimador corrió y calculó la nueva arquitectura de errores
    assert "m1_rmse_hz" in result.summary_df.columns, "La nueva arquitectura de métricas no se ejecutó"
    assert "f_hat_hz" in result.signals_df.columns
    assert not result.summary_df["m1_rmse_hz"].isna().any(), "RMSE no debería ser NaN"


# ==========================================
# D: Prueba de Corridas con Artefactos y JSON de Métricas
# ==========================================
def test_d_ten_runs_artifacts_and_metrics():
    """
    Test D: Corre N veces, guarda los resultados en CSV,
    genera un plot, y un JSON con las métricas avanzadas (M1 a M17)
    promediadas del Monte Carlo.
    """
    N_RUNS = 30  # Usamos 30 para evaluar Monte Carlo estocástico
    out_dir = ROOT / "tests" / "montecarlo" / "artifacts" / "ieee_single_sinwave_ipdft"
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = MonteCarloEngine(
        scenario_cls=IEEESingleSinWaveScenario,
        estimator_cls=IPDFT_Estimator,
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
    
    for run_idx in range(N_RUNS):
        run_data = result.signals_df[result.signals_df["run_idx"] == run_idx]
        axes[0].plot(run_data["t_s"], run_data["v_pu"], alpha=0.5, linewidth=0.5)
        axes[1].plot(run_data["t_s"], run_data["f_true_hz"], color="black", alpha=0.3, linewidth=1.5, label="True Freq" if run_idx==0 else "")
        if "f_hat_hz" in run_data.columns:
            axes[1].plot(run_data["t_s"], run_data["f_hat_hz"], alpha=0.6, linewidth=0.8, linestyle="--")

    axes[0].set_title(f"Monte Carlo: {result.scenario_name} + {result.estimator_name} ({N_RUNS} runs)")
    axes[0].set_ylabel("Voltage [pu]")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Frequency [Hz]")
    axes[1].legend(loc="upper right")

    plot_path = out_dir / f"mc_{N_RUNS}_runs_plot.png"
    fig.savefig(plot_path)
    plt.close(fig)
    assert plot_path.exists()

    # 3. Generar JSON con Métricas IBR-Centric y Estadísticas de Monte Carlo
    summary_df = result.summary_df
    
    # Excluimos las columnas fijas o de métricas al calcular variaciones de entrada
    exclude_cols = ["run_idx", "scenario_name", "n_samples", "seed", "v_mean", "v_std", "v_rms", "v_max", "v_min", "f_true_mean", "f_true_std"]
    metric_cols = [col for col in summary_df.columns if col.startswith("m")] # m1, m2, etc.
    param_cols = [col for col in summary_df.columns if col not in exclude_cols and col not in metric_cols]
    
    # Estadísticas de perturbación de entrada
    variations = {}
    for col in param_cols:
        try:
            variations[f"{col}_std"] = float(summary_df[col].std())
            variations[f"{col}_mean"] = float(summary_df[col].mean())
        except TypeError:
            pass 

    # Diccionario final consolidado (Directo al Paper)
    metrics = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "python_version": platform.python_version(),
            "os_system": f"{platform.system()} {platform.release()}",
            "cpu_architecture": platform.processor(),
            "total_system_cores": multiprocessing.cpu_count(),
            "estimator_name": result.estimator_name,
            "scenario_name": result.scenario_name
        },
        "performance": {
            "execution_time_seconds": round(execution_time, 4),
            "workers_used": min(multiprocessing.cpu_count(), N_RUNS),
            "n_runs_completed": N_RUNS,
            "avg_time_per_run_seconds": round(execution_time / max(N_RUNS, 1), 4)
        },
        "precision_baseline": {
            "m1_rmse_hz_mean": float(summary_df["m1_rmse_hz"].mean()),
            "m1_rmse_hz_std": float(summary_df["m1_rmse_hz"].std()),
            "m1_rmse_hz_p95": float(summary_df["m1_rmse_hz"].quantile(0.95)),
            "m2_mae_hz_mean": float(summary_df["m2_mae_hz"].mean()),
            "m3_max_peak_hz_mean": float(summary_df["m3_max_peak_hz"].mean()),
        },
        "protection_risk": {
            "m5_trip_risk_s_mean": float(summary_df["m5_trip_risk_s"].mean()),
            "m5_trip_risk_s_max": float(summary_df["m5_trip_risk_s"].max()),
            "m7_pcb_hz_mean": float(summary_df["m7_pcb_hz"].mean()),
            "m8_settling_time_s_mean": float(summary_df["m8_settling_time_s"].mean()),
        },
        "rocof_and_ibr": {
            "m9_rfe_max_hz_s_mean": float(summary_df["m9_rfe_max_hz_s"].mean()),
            "m11_rnaf_db_mean": float(summary_df["m11_rnaf_db"].mean()),
            "m12_isi_pu_mean": float(summary_df["m12_isi_pu"].mean()),
        },
        "hardware_and_paper": {
            "m13_cpu_time_us_mean": float(summary_df["m13_cpu_time_us"].mean()),
            "m16_heatmap_pass_rate": float(summary_df["m16_heatmap_pass"].mean()), # % de corridas que pasan
            "hw_class_assigned": summary_df["m17_hw_class"].iloc[0] 
        },
        "montecarlo_variations": variations
    }

    json_path = out_dir / f"mc_{N_RUNS}_runs_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    assert json_path.exists()
    
    # Validación del límite de error usando la nueva métrica
    assert metrics["precision_baseline"]["m1_rmse_hz_mean"] < 1.0, "El RMSE promedio es excesivamente alto"