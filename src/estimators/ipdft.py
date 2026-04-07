from __future__ import annotations

import math
import numpy as np
from numba import njit

from .base import BaseFrequencyEstimator
from .common import DT_DSP


# =====================================================================
# Numba JIT-compiled core logic (Ruta Rápida)
# =====================================================================
@njit(cache=True)
def _ipdft_direct_vectorized(
    v_array: np.ndarray,
    z_buf: np.ndarray,
    basis_real: np.ndarray,
    basis_imag: np.ndarray,
    sz: int,
    k_nom: int,
    res: float,
    decim: int,
    f_nom: float  # <--- Añadido parámetro de frecuencia nominal
) -> np.ndarray:
    
    n_samples = len(v_array)
    f_hat_array = np.empty(n_samples, dtype=np.float64)
    
    buf_idx = 0
    buf_count = 0
    last_f = f_nom  # <--- Inicializamos con f_nom, no 60.0
    
    cnt = 0

    for i in range(n_samples):
        # Decimación Interna: Simulamos el ADC operando a FS_DSP (10 kHz)
        if cnt == 0:
            v_in = v_array[i]
            z_buf[buf_idx] = v_in
            buf_idx = (buf_idx + 1) % sz
            buf_count += 1

            # Solo computamos la DFT si el buffer ya tiene al menos 1 ciclo completo
            if buf_count >= sz:
                m1_re = 0.0; m1_im = 0.0
                m2_re = 0.0; m2_im = 0.0
                m3_re = 0.0; m3_im = 0.0

                # DFT puntual en los 3 bins de interés (Equivalente a np.fft.rfft)
                for m in range(sz):
                    idx = (buf_idx + m) % sz
                    val = z_buf[idx]
                    
                    m1_re += val * basis_real[0, m]
                    m1_im += val * basis_imag[0, m]
                    
                    m2_re += val * basis_real[1, m]
                    m2_im += val * basis_imag[1, m]
                    
                    m3_re += val * basis_real[2, m]
                    m3_im += val * basis_imag[2, m]

                sp_km1 = math.hypot(m1_re, m1_im)
                sp_k   = math.hypot(m2_re, m2_im)
                sp_kp1 = math.hypot(m3_re, m3_im)

                # Interpolación Parabólica Magnitud (Quinn/MacLeod) idéntica a tu código
                denom_mag = sp_km1 + 2.0 * sp_k + sp_kp1
                if denom_mag >= 1e-10:
                    delta = 2.0 * (sp_kp1 - sp_km1) / denom_mag
                    # Clip de seguridad
                    if delta > 0.5: delta = 0.5
                    if delta < -0.5: delta = -0.5
                    
                    last_f = (k_nom + delta) * res
                    
            f_hat_array[i] = last_f
            cnt = decim - 1  # Reiniciamos el contador de muestras ignoradas
        else:
            cnt -= 1
            f_hat_array[i] = last_f

    return f_hat_array


# =====================================================================
# Clase Python
# =====================================================================
class IPDFT_Estimator(BaseFrequencyEstimator):
    """
    Interpolated Discrete Fourier Transform (IPDFT) Frequency Estimator.
    Matemáticamente idéntico a TunableIpDFT, optimizado para simulación a 1 MHz.
    """
    name = "IPDFT"

    def __init__(self, nominal_f: float = 50.0, cycles: float = 2.0, decim: int = 100) -> None:
        self.nominal_f = float(nominal_f)  # <--- Ahora soporta 50 Hz, 60 Hz, etc.
        self.cycles = float(cycles)
        
        # Factor de decimación (1 MHz -> 10 kHz = 100)
        self.decim = int(decim)
        
        # Frecuencia de muestreo base del DSP (10000.0 Hz)
        fs_dsp = 1.0 / DT_DSP
        
        # Tamaño de la ventana ajustado a la frecuencia nominal
        self.sz = int(round((fs_dsp / self.nominal_f) * self.cycles))
        if self.sz < 3:
            self.sz = 3
            
        self.res = fs_dsp / self.sz
        self.k_nom = int(round(self.cycles))

        # Ventana de Hann pre-calculada
        self.win = np.hanning(self.sz)
        
        # Vectores base de Fourier (reales e imaginarios, pre-multiplicados por Hann)
        self.basis_real = np.empty((3, self.sz), dtype=np.float64)
        self.basis_imag = np.empty((3, self.sz), dtype=np.float64)

        k_vals = [self.k_nom - 1, self.k_nom, self.k_nom + 1]
        for i, k in enumerate(k_vals):
            for m in range(self.sz):
                angle = 2.0 * math.pi * k * m / self.sz
                self.basis_real[i, m] = math.cos(angle) * self.win[m]
                self.basis_imag[i, m] = -math.sin(angle) * self.win[m]

        self.reset()

    def reset(self) -> None:
        self._z_buf = np.zeros(self.sz, dtype=np.float64)

    def structural_latency_samples(self) -> int:
        # La latencia en la matriz del Monte Carlo debe mapearse a la física (1 MHz)
        return self.sz * self.decim

    @classmethod
    def default_params(cls) -> dict[str, float | int]:
        return {"nominal_f": 50.0, "cycles": 2.0, "decim": 100}

    @staticmethod
    def describe_params(params: dict[str, float | int]) -> str:
        return f"f_nom={params.get('nominal_f', 50.0)}Hz, Cycles={params.get('cycles', 2.0)}"

    def step(self, z: float) -> float:
        v_array = np.array([z], dtype=np.float64)
        return float(self.step_vectorized(v_array)[0])

    def step_vectorized(self, v_array: np.ndarray) -> np.ndarray:
        return _ipdft_direct_vectorized(
            v_array=v_array.astype(np.float64),
            z_buf=self._z_buf,
            basis_real=self.basis_real,
            basis_imag=self.basis_imag,
            sz=self.sz,
            k_nom=self.k_nom,
            res=self.res,
            decim=self.decim,
            f_nom=self.nominal_f
        )
    
    def estimate(self, t: np.ndarray, v: np.ndarray) -> np.ndarray:
        self.reset()
        return self.step_vectorized(v)