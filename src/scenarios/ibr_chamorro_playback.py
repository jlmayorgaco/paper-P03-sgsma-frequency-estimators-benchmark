from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from .base import Scenario, ScenarioData
from .offline_processing import extract_true_frequency_non_causal
from estimators.common import FS_PHYSICS


class IBRChamorroPlaybackScenario(Scenario):
    """
    Playback of the Chamorro et al. Simulink Event.
    
    Target: 50 Hz system, 4.0 seconds duration.
    Features an Auto-Time-Warp algorithm to correct Simulink export 
    time-scaling errors by detecting the FFT peak and aligning it to F_NOM.
    """

    SCENARIO_NAME: ClassVar[str] = "IBR_Chamorro_Playback"
    F_NOM_CHAMORRO: ClassVar[float] = 50.0

    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {
        "csv_filename": "chamorro_data.csv", 
        "time_col": "Tiempo",
        "voltage_col": "Va",                 
        "normalize_pu": True,                
        "target_duration_s": 4.0,            
        "noise_sigma": 0.0,
        "seed": None,
    }

    MONTE_CARLO_SPACE: ClassVar[dict[str, Any]] = {
        "noise_sigma": {
            "kind": "uniform",
            "low": 0.0,
            "high": 0.005,
        }
    }

    @classmethod
    def validate_params(cls, params: dict[str, Any]) -> None:
        if not params.get("csv_filename"):
            raise ValueError("A CSV filename must be provided.")

    @classmethod
    def generate(
        cls,
        csv_filename: str = "chamorro_data.csv",
        time_col: str = "Tiempo",
        voltage_col: str = "Va",
        normalize_pu: bool = True,
        target_duration_s: float = 4.0,
        noise_sigma: float = 0.0,
        seed: int | None = None,
    ) -> ScenarioData:
        rng = np.random.default_rng(seed)
        
        current_dir = Path(__file__).resolve().parent
        csv_path = current_dir / "data" / csv_filename
        
        if not csv_path.exists():
            raise FileNotFoundError(f"No se encuentra el archivo en: {csv_path}")

        # 1. Cargar datos crudos y empezar el tiempo en 0
        df = pd.read_csv(csv_path)
        t_raw = df[time_col].values
        v_raw = df[voltage_col].values
        t_raw_zeroed = t_raw - t_raw[0]

        # 2. AUTO-CALIBRACIÓN DE TIEMPO (Warping)
        dt_raw = np.mean(np.diff(t_raw_zeroed))
        n_samples = len(v_raw)
        
        # Detectar frecuencia real del CSV vía FFT
        freqs = np.fft.fftfreq(n_samples, d=dt_raw)
        fft_mag = np.abs(np.fft.fft(v_raw))
        valid_idx = (freqs > 10) & (freqs < 100) # Buscar fundamental entre 10 y 100 Hz
        f_peak = freqs[valid_idx][np.argmax(fft_mag[valid_idx])]
        
        # Estirar o comprimir el tiempo para forzar que el pico sea exactamente 50.0 Hz
        time_stretch_factor = f_peak / cls.F_NOM_CHAMORRO
        t_warped = t_raw_zeroed * time_stretch_factor

        # 3. Normalización a p.u.
        v_norm = np.copy(v_raw)
        if normalize_pu:
            dt_warped = np.mean(np.diff(t_warped))
            samples_init = int(0.2 / dt_warped) # 10 ciclos iniciales a 50Hz
            v_peak = np.max(np.abs(v_raw[:samples_init])) if samples_init < len(v_raw) else np.max(np.abs(v_raw))
            if v_peak > 0:
                v_norm = v_raw / v_peak

        # 4. Extraer f_true sobre la escala de tiempo calibrada
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f_true_raw = extract_true_frequency_non_causal(
                t_warped, 
                v_norm, 
                nominal_f=cls.F_NOM_CHAMORRO
            )

        # 5. Interpolar para el Benchmark (Estricto a 4.0s)
        t_benchmark = np.arange(0.0, target_duration_s, 1.0 / FS_PHYSICS, dtype=float)
        
        interp_v = interp1d(t_warped, v_norm, kind='cubic', bounds_error=False, fill_value="extrapolate")
        interp_f = interp1d(t_warped, f_true_raw, kind='linear', bounds_error=False, fill_value="extrapolate")
        
        v_benchmark = interp_v(t_benchmark)
        f_benchmark = interp_f(t_benchmark)

        if noise_sigma > 0.0:
            v_benchmark += rng.normal(0.0, noise_sigma, size=t_benchmark.shape)

        meta = {
            "description": "Chamorro Playback (Auto-Warped to 50Hz)",
            "parameters": {
                "csv_filename": csv_filename,
                "f_original_detected": round(float(f_peak), 2),
                "time_stretch_factor": round(float(time_stretch_factor), 3),
            },
            "fs_physics_hz": FS_PHYSICS,
        }

        return ScenarioData(
            name=cls.SCENARIO_NAME, t=t_benchmark, v=v_benchmark, f_true=f_benchmark, meta=meta
        )