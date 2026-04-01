
from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm  # <-- Importamos tqdm para la barra de progreso


@dataclass
class MonteCarloResult:
    scenario_name: str
    estimator_name: str | None
    summary_df: pd.DataFrame
    signals_df: pd.DataFrame
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class MonteCarloEngine:
    scenario_cls: type
    estimator_cls: type | None = None
    estimator_params: dict[str, Any] | None = None
    n_runs: int = 30
    base_seed: int = 12345

    def sample_from_space(self, rng: np.random.Generator, spec: dict[str, Any]) -> Any:
        kind = spec["kind"]

        if kind == "uniform":
            return float(rng.uniform(spec["low"], spec["high"]))

        if kind == "choice":
            return rng.choice(spec["values"])

        if kind == "fixed":
            return spec["value"]

        raise ValueError(f"Unsupported Monte Carlo sampling kind: {kind}")

    def sample_params(self, run_idx: int) -> dict[str, Any]:
        rng = np.random.default_rng(self.base_seed + run_idx)

        params = self.scenario_cls.get_default_params()
        mc_space = self.scenario_cls.get_monte_carlo_space()

        for key, spec in mc_space.items():
            params[key] = self.sample_from_space(rng, spec)

        params["seed"] = self.base_seed + run_idx
        return params

    def _run_estimator(self, v: np.ndarray, run_idx: int = 0) -> dict[str, np.ndarray]:
        if self.estimator_cls is None:
            return {}

        params = dict(self.estimator_params or {})
        est = self.estimator_cls(**params)

        if hasattr(est, "reset"):
            est.reset()

        # =========================================================
        # 1. RUTA RÁPIDA (Fast Path): Soporte para Numba/Vectorizado
        # =========================================================
        if hasattr(est, "step_vectorized"):
            # Pasamos todo el bloque. El estimador iterará internamente en C/C++
            f_hat = est.step_vectorized(v)

        # =========================================================
        # 2. RUTA GENERAL (Fallback): Python Puro muestra por muestra
        # =========================================================
        else:
            f_hat = np.empty(len(v), dtype=float)
            
            # Cache the method lookup (prevents Python from searching for .step 1.5M times)
            step_func = est.step 
            
            # Iterate directly over the numpy array
            for k in range(len(v)):
                f_hat[k] = float(step_func(float(v[k])))

        return {"f_hat": np.asarray(f_hat, dtype=float)}
    
    def run_once(self, run_idx: int) -> tuple[dict[str, Any], pd.DataFrame]:
        """
        Este método es el que se ejecutará en paralelo.
        """
        params = self.sample_params(run_idx)
        sc = self.scenario_cls.run(**params)

        est_out = self._run_estimator(sc.v)

        row = {
            "run_idx": run_idx,
            "scenario_name": sc.name,
            **params,
            "n_samples": len(sc.t),
            "v_mean": float(np.mean(sc.v)),
            "v_std": float(np.std(sc.v)),
            "v_rms": float(np.sqrt(np.mean(sc.v ** 2))),
            "v_max": float(np.max(sc.v)),
            "v_min": float(np.min(sc.v)),
            "f_true_mean": float(np.mean(sc.f_true)),
            "f_true_std": float(np.std(sc.f_true)),
        }

        signal_df = pd.DataFrame(
            {
                "run_idx": run_idx,
                "t_s": sc.t,
                "v_pu": sc.v,
                "f_true_hz": sc.f_true,
            }
        )

        if "f_hat" in est_out:
            f_hat = np.asarray(est_out["f_hat"], dtype=float)
            signal_df["f_hat_hz"] = f_hat

            err = f_hat - np.asarray(sc.f_true, dtype=float)
            abs_err = np.abs(err)

            row["f_mae_hz"] = float(np.mean(abs_err))
            row["f_rmse_hz"] = float(np.sqrt(np.mean(err ** 2)))
            row["f_max_abs_err_hz"] = float(np.max(abs_err))
            row["f_hat_mean_hz"] = float(np.mean(f_hat))
            row["f_hat_std_hz"] = float(np.std(f_hat))

        for key, value in params.items():
            signal_df[key] = value

        return row, signal_df

    def run(self) -> MonteCarloResult:
        """
        Ejecuta la simulación usando múltiples cores del CPU con barra de progreso.
        """
        num_workers = multiprocessing.cpu_count()
        
        print(f"Lanzando {self.n_runs} iteraciones en {num_workers} procesos...")

        summary_rows = []
        signal_dfs = []

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Enviamos todos los trabajos al pool y guardamos las "promesas" (Futures)
            futures = [executor.submit(self.run_once, i) for i in range(self.n_runs)]
            
            # Usamos tqdm para iterar sobre as_completed
            # Esto actualiza la barra cada vez que una tarea finaliza, sin importar el orden
            for future in tqdm(as_completed(futures), total=self.n_runs, desc="Monte Carlo Progress"):
                row, signal_df = future.result()  # future.result() bloquea hasta que ese worker termine
                summary_rows.append(row)
                signal_dfs.append(signal_df)

        # Ordenamos los resultados porque as_completed los devuelve a medida que terminan (desordenados)
        summary_df = pd.DataFrame(summary_rows).sort_values(by="run_idx").reset_index(drop=True)
        signals_df = pd.concat(signal_dfs, ignore_index=True).sort_values(by=["run_idx", "t_s"]).reset_index(drop=True)

        estimator_name = None
        if self.estimator_cls is not None:
            estimator_name = getattr(self.estimator_cls, "name", self.estimator_cls.__name__)

        return MonteCarloResult(
            scenario_name=self.scenario_cls.get_name(),
            estimator_name=estimator_name,
            summary_df=summary_df,
            signals_df=signals_df,
            meta={
                "n_runs": self.n_runs,
                "base_seed": self.base_seed,
                "estimator_params": dict(self.estimator_params or {}),
            },
        )

    def save_csv(self, result: MonteCarloResult, out_dir: Path) -> tuple[Path, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)

        base = result.scenario_name
        if result.estimator_name:
            base += f"__{result.estimator_name}"

        summary_path = out_dir / f"{base}_summary.csv"
        signals_path = out_dir / f"{base}_signals.csv"

        result.summary_df.to_csv(summary_path, index=False)
        result.signals_df.to_csv(signals_path, index=False)

        return summary_path, signals_path

if __name__ == "__main__":
    pass
