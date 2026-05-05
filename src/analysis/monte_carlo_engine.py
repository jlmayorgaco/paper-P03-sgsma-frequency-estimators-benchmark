from __future__ import annotations

# =====================================================================
# T-000 SAMPLE-RATE AUDIT (2026-04-12)
# =====================================================================
# FINDING: sc.v reaches step_vectorized() at FS_PHYSICS = 1,000,000 Hz.
#
# Evidence (diagnostic run):
#   generate(): N=100001, dt=1.00e-06, fs=1,000,000
#   run()     : N=100001, dt=1.00e-06, fs=1,000,000
#
# Signal flow:
#   Scenario.generate() -> 1 MHz  (t step = 1e-6 s)
#   Scenario.run()      -> 1 MHz  (no decimation - just calls generate())
#   _run_estimator(sc.v)-> 1 MHz  (no decimation here either)
#   est.step_vectorized(v) receives 1 MHz samples
#
# Estimator internal time base:
#   All estimators import DT_DSP = 1/FS_DSP = 1e-4 s (10 kHz)
#   and use it as self.dt in state-transition matrices.
#
# MISMATCH CONFIRMED:
#   Estimators receive 1 MHz samples but compute with a 10 kHz dt.
#   This causes a factor-100 error in the time base used for physics
#   (frequency extraction from phase, Kalman prediction steps, etc.).
#
# Additional finding:
#   calculate_all_metrics() is called with hardcoded fs_dsp=10000.0
#   (line ~159) even though the actual signal length corresponds to 1 MHz.
#   Metric windows (e.g. 150 ms = 1500 samples at 10 kHz) are therefore
#   computed on 150,000 samples instead - the evaluation window is correct
#   in time but the sample count is 100x larger than intended.
#
# Exception - smoke tests DO correctly decimate:
#   test_dedicated_smoke_no_mc_test.py:127-138 checks dt_real < 1e-5
#   and decimates by factor 100 before calling step_vectorized().
#   Smoke-test results are therefore valid; MonteCarloEngine is NOT.
#
# Fix responsibility: T-100 will implement Option C (decimate in
# Scenario.run()) so that run() always returns 10 kHz output.
# =====================================================================

import os
# =====================================================================
# CRITICAL OPTIMIZATION: Prevent NumPy deadlocks on multi-core CPUs.
# Force math libraries in C to use 1 thread per process
# para que el ProcessPoolExecutor de Python escale perfectamente.
# Debe ir ANTES de importar numpy.
# =====================================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

# Import new IBR-centric metrics architecture
from analysis.metrics import calculate_all_metrics
try:
    from estimators.base import MemoryStore
except ModuleNotFoundError:
    # Fallback for environments where another global `estimators` package shadows this repo.
    from src.estimators.base import MemoryStore  # type: ignore


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
    n_cost_reps: int = 20
    enforce_standardized_step: bool = True
    capture_signals: bool = True

    @staticmethod
    def _resolve_worker_count(n_runs: int) -> int:
        """
        Resolve process count for MC execution.
        Default policy avoids spawning more workers than runs.
        Optional override: BENCHMARK_MC_MAX_WORKERS.
        """
        max_default = max(1, min(int(n_runs), multiprocessing.cpu_count()))
        raw = os.getenv("BENCHMARK_MC_MAX_WORKERS")
        if raw is None:
            return max_default
        try:
            requested = int(raw)
        except ValueError:
            return max_default
        if requested <= 0:
            return max_default
        return max(1, min(max_default, requested))

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

    def _run_estimator(self, v: np.ndarray, t: np.ndarray | None = None, run_idx: int = 0) -> dict[str, Any]:
        if self.estimator_cls is None:
            return {}

        params = dict(self.estimator_params or {})
        est = self.estimator_cls(**params)
        mem = MemoryStore()

        if hasattr(est, "reset"):
            est.reset()

        # Extraemos la latencia estructural aquí para evitar doble instanciación
        struct_samples = 0
        if hasattr(est, "structural_latency_samples"):
            struct_samples = est.structural_latency_samples()

        has_step = hasattr(est, "step")
        has_step_vectorized = hasattr(est, "step_vectorized")

        if self.enforce_standardized_step and has_step:
            f_hat = np.empty(len(v), dtype=float)
            step_func = est.step
            for k in range(len(v)):
                t_k = float(t[k]) if t is not None else None
                f_hat[k] = float(step_func(float(v[k]), t_k, mem))
        else:
            if has_step_vectorized:
                f_hat = est.step_vectorized(v)
            elif has_step:
                f_hat = np.empty(len(v), dtype=float)
                step_func = est.step
                for k in range(len(v)):
                    f_hat[k] = float(step_func(float(v[k])))
            else:
                raise AttributeError(
                    f"Estimator {self.estimator_cls.__name__} must define step(...) or step_vectorized(...)."
                )

        runtime_metrics: dict[str, Any] = {}
        if hasattr(est, "runtime_summary"):
            try:
                runtime_metrics = dict(est.runtime_summary(mem))
            except Exception:
                runtime_metrics = {}

        return {
            "f_hat": np.asarray(f_hat, dtype=float),
            "struct_samples": struct_samples,
            "runtime_metrics": runtime_metrics,
        }

    def _measure_exec_time(self, v: np.ndarray, t: np.ndarray | None = None) -> float:
        """
        Mean CPU time for one full estimator pass.
        """
        if self.estimator_cls is None:
            return 0.0

        n_reps = max(1, int(self.n_cost_reps))
        timings_s: list[float] = []

        for _ in range(n_reps):
            start_cpu = time.process_time()
            self._run_estimator(v, t=t)
            timings_s.append(time.process_time() - start_cpu)

        return float(np.mean(timings_s))
    
    def run_once(self, run_idx: int) -> tuple[dict[str, Any], pd.DataFrame]:
        params = self.sample_params(run_idx)
        sc = self.scenario_cls.run(**params)

        # T-100 assertion: Scenario.run() must return 10 kHz data (Option C fix).
        if len(sc.t) > 1:
            dt_actual = float(sc.t[1] - sc.t[0])
            assert abs(dt_actual - 1e-4) < 1e-9, (
                f"[T-100] Scenario {sc.name} returned dt={dt_actual:.2e} "
                f"(expected 1e-4 s = 10 kHz). Scenario.run() decimation may be broken."
            )
            fs_dsp = 1.0 / dt_actual
        else:
            fs_dsp = 10000.0

        # T-202 WARMUP: Trigger Numba JIT compilation BEFORE the timed window.
        # Without this, the first MC run includes compilation time (up to several
        # seconds for Numba kernels), inflating TIME_PER_SAMPLE_US for whichever
        # estimator happens to run first. The warmup creates a throwaway instance,
        # runs it on the first 100 samples to trigger @njit compilation, then
        # discards it. The real timed run below uses a fresh instance on sc.v.
        if self.estimator_cls is not None:
            _params_warmup = dict(self.estimator_params or {})
            _est_warmup = self.estimator_cls(**_params_warmup)
            _warmup_v = sc.v[:min(100, len(sc.v))]
            try:
                if hasattr(_est_warmup, "step_vectorized"):
                    _est_warmup.step_vectorized(_warmup_v)
                elif hasattr(_est_warmup, "step"):
                    for _z in _warmup_v:
                        _est_warmup.step(float(_z))
            except Exception:
                pass  # warmup failure is non-fatal; timing may be slightly inflated
            del _est_warmup, _warmup_v, _params_warmup

        # CPU time measurement
        est_out = self._run_estimator(sc.v, t=sc.t)
        exec_time_s = self._measure_exec_time(sc.v, t=sc.t)

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

        n_len = len(sc.t)
        signal_dict: dict[str, Any] | None = None
        if self.capture_signals:
            # Build the full dictionary in memory before DataFrame creation.
            signal_dict = {
                "run_idx": np.full(n_len, run_idx, dtype=int),
                "t_s": sc.t,
                "v_pu": sc.v,
                "f_true_hz": sc.f_true,
            }

            # Add Monte Carlo parameters to each signal sample.
            for key, value in params.items():
                signal_dict[key] = np.full(n_len, value)

        if "f_hat" in est_out:
            f_hat = est_out["f_hat"]
            if signal_dict is not None:
                signal_dict["f_hat_hz"] = f_hat

            struct_samples = est_out.get("struct_samples", 0)
            noise_sigma = params.get("noise_sigma", 0.0)

            # -------------------------------------------------------------
            # New metrics architecture (integration with metrics.py)
            # -------------------------------------------------------------
            advanced_metrics = calculate_all_metrics(
                f_hat=f_hat,
                f_true=np.asarray(sc.f_true, dtype=float),
                fs_dsp=fs_dsp,
                exec_time_s=exec_time_s,
                structural_samples=struct_samples,
                noise_sigma=noise_sigma,
                interharmonic_hz=32.5
            )
            
            # Agregamos M1 a M17 a la fila de resultados
            row.update(advanced_metrics)
            rt = est_out.get("runtime_metrics", {})
            row["m18_mem_peak_kb"] = round(float(rt.get("memory_peak_bytes", 0.0)) / 1024.0, 4)
            row["m19_mem_mean_kb"] = round(float(rt.get("memory_mean_bytes", 0.0)) / 1024.0, 4)
            row["m20_runtime_jitter_us"] = round(float(rt.get("runtime_jitter_us", 0.0)), 4)
            row["m21_startup_valid_samples"] = int(rt.get("startup_valid_samples", 0))
            row["m22_invalid_output_rate"] = round(float(rt.get("invalid_output_rate", 0.0)), 6)
            row["m23_memory_key_count"] = int(rt.get("memory_key_count", 0))

        if signal_dict is not None:
            signal_df = pd.DataFrame(signal_dict)
        else:
            signal_df = pd.DataFrame()

        return row, signal_df

    def run(self) -> MonteCarloResult:
        num_workers = self._resolve_worker_count(self.n_runs)
        print(f"Lanzando {self.n_runs} iteraciones en {num_workers} procesos...")

        summary_rows = []
        signal_dfs = []

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.run_once, i) for i in range(self.n_runs)]
            for future in tqdm(as_completed(futures), total=self.n_runs, desc="Monte Carlo Progress"):
                row, signal_df = future.result() 
                summary_rows.append(row)
                signal_dfs.append(signal_df)

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



