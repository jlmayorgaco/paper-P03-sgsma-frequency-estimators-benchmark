from __future__ import annotations

import math
import numpy as np
from numba import njit

from .base import BaseFrequencyEstimator
from .common import DT_DSP

REFERENCE_KEYS = ("grandke1983_ipdft",)

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
    f_nom: float,
    f_min_hz: float,
    f_max_hz: float,
    delta_limit_bins: float,
    smooth_alpha: float,
    buf_idx: int,
    buf_count: int,
    last_f: float,
) -> tuple[np.ndarray, int, int, int, float]:
    """
    Núcleo del IPDFT optimizado.
    Procesa muestra a muestra sin diezmado (downsampling), actualizando la ventana 
    circular y recalculando la frecuencia mediante interpolación parabólica.
    """
    n_samples = len(v_array)
    f_hat_array = np.empty(n_samples, dtype=np.float64)
    clamp_count = 0

    for i in range(n_samples):
        # 1. Actualizar ventana circular
        v_in = v_array[i]
        z_buf[buf_idx] = v_in
        buf_idx = (buf_idx + 1) % sz
        buf_count += 1

        # 2. Solo computamos si el buffer ya tiene al menos 1 ciclo completo
        if buf_count >= sz:
            m1_re = 0.0
            m1_im = 0.0
            m2_re = 0.0
            m2_im = 0.0
            m3_re = 0.0
            m3_im = 0.0

            # DFT puntual en los 3 bins de interés
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

            # 3. Interpolación Parabólica Magnitud (Quinn/MacLeod modificado)
            denom_mag = sp_km1 + 2.0 * sp_k + sp_kp1
            if denom_mag >= 1e-10:
                # Calculamos el desplazamiento
                delta = 2.0 * (sp_kp1 - sp_km1) / denom_mag
                
                # Mantenemos un clamp holgado (±3.0 bines) para que la fórmula 
                # matemática no colapse antes de llegar al Sanity Check.
                if delta > delta_limit_bins:
                    delta = delta_limit_bins
                    clamp_count += 1
                elif delta < -delta_limit_bins:
                    delta = -delta_limit_bins
                    clamp_count += 1
                
                f_candidate = (k_nom + delta) * res
                
                # Filtro de Cordura Eléctrica (Sanity Check)
                # Restringido explícitamente de 40 Hz a 90 Hz.
                if f_min_hz <= f_candidate <= f_max_hz:
                    if smooth_alpha > 0.0:
                        last_f = (1.0 - smooth_alpha) * last_f + smooth_alpha * f_candidate
                    else:
                        last_f = f_candidate
                else:
                    clamp_count += 1
                    
        f_hat_array[i] = last_f

    return f_hat_array, clamp_count, buf_idx, buf_count, last_f


# =====================================================================
# Clase Python
# =====================================================================
class IPDFT_Estimator(BaseFrequencyEstimator):
    """
    Interpolated Discrete Fourier Transform (IPDFT) Frequency Estimator.
    Matemáticamente idéntico a TunableIpDFT, corregido para evitar aliasing y 
    seguir fallas severas en el rango seguro de 40-90 Hz.
    """
    name = "IPDFT"

    def __init__(
        self,
        nominal_f: float = 60.0,
        cycles: float = 2.0,
        decim: int = 1,
        dt: float = DT_DSP,
        window: str = "hann",
        f_min_hz: float = 40.0,
        f_max_hz: float = 90.0,
        delta_limit_bins: float = 3.0,
        output_smoothing: float = 0.0,
        verbose: bool = False,
    ) -> None:
        self.nominal_f = float(nominal_f)
        self.cycles = float(cycles)
        self.decim = max(1, int(decim))
        self.dt = float(dt)
        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")
        self.window = str(window).lower()
        self.f_min_hz = float(f_min_hz)
        self.f_max_hz = float(f_max_hz)
        if self.f_min_hz >= self.f_max_hz:
            raise ValueError("f_min_hz must be lower than f_max_hz.")
        self.delta_limit_bins = float(delta_limit_bins)
        self.output_smoothing = float(output_smoothing)
        self.verbose = bool(verbose)
        
        self._update_internals()
        self.reset()

    def _update_internals(self) -> None:
        """Recalcula las bases de Fourier dinámicamente si el dt cambia"""
        fs_dsp = 1.0 / (self.dt * self.decim)
        
        # Tamaño de la ventana ajustado a la frecuencia nominal y tasa de muestreo real
        self.sz = int(round((fs_dsp / self.nominal_f) * self.cycles))
        if self.sz < 3:
            self.sz = 3
            
        self.res = fs_dsp / self.sz
        self.k_nom = int(round(self.cycles))

        if self.window == "boxcar":
            self.win = np.ones(self.sz, dtype=np.float64)
        elif self.window == "blackman":
            self.win = np.blackman(self.sz)
        else:
            self.window = "hann"
            self.win = np.hanning(self.sz)
        
        # Vectores base pre-multiplicados
        self.basis_real = np.empty((3, self.sz), dtype=np.float64)
        self.basis_imag = np.empty((3, self.sz), dtype=np.float64)

        k_vals = [self.k_nom - 1, self.k_nom, self.k_nom + 1]
        for i, k in enumerate(k_vals):
            for m in range(self.sz):
                angle = 2.0 * math.pi * k * m / self.sz
                self.basis_real[i, m] = math.cos(angle) * self.win[m]
                self.basis_imag[i, m] = -math.sin(angle) * self.win[m]

    def reset(self) -> None:
        self._z_buf = np.zeros(self.sz, dtype=np.float64)
        self._total_clamps = 0
        self._buf_idx   = 0
        self._buf_count = 0
        self._last_f    = self.nominal_f

    def structural_latency_samples(self) -> int:
        return self.sz * self.decim

    @classmethod
    def default_params(cls) -> dict[str, float | int | str | bool]:
        return {
            "nominal_f": 60.0,
            "cycles": 2.0,
            "decim": 1,
            "dt": DT_DSP,
            "window": "hann",
            "f_min_hz": 40.0,
            "f_max_hz": 90.0,
            "delta_limit_bins": 3.0,
            "output_smoothing": 0.0,
            "verbose": False,
        }

    @staticmethod
    def describe_params(params: dict[str, float | int | str | bool]) -> str:
        return (
            f"f_nom={params.get('nominal_f', 60.0)}Hz, "
            f"Cycles={params.get('cycles', 2.0)}, "
            f"decim={params.get('decim', 1)}, "
            f"dt={params.get('dt', DT_DSP)}, "
            f"window={params.get('window', 'hann')}"
        )

    def step(self, z: float) -> float:
        v_array = np.array([z], dtype=np.float64)
        return float(self.step_vectorized(v_array)[0])

    def step_vectorized(self, v_array: np.ndarray) -> np.ndarray:
        arr = v_array.astype(np.float64)
        arr_ds = arr[::self.decim] if self.decim > 1 else arr

        f_hat_ds, clamps, self._buf_idx, self._buf_count, self._last_f = _ipdft_direct_vectorized(
            v_array=arr_ds,
            z_buf=self._z_buf,
            basis_real=self.basis_real,
            basis_imag=self.basis_imag,
            sz=self.sz,
            k_nom=self.k_nom,
            res=self.res,
            f_nom=self.nominal_f,
            f_min_hz=self.f_min_hz,
            f_max_hz=self.f_max_hz,
            delta_limit_bins=self.delta_limit_bins,
            smooth_alpha=self.output_smoothing,
            buf_idx=self._buf_idx,
            buf_count=self._buf_count,
            last_f=self._last_f,
        )
        self._total_clamps += clamps
        if self.decim == 1:
            return f_hat_ds

        out = np.empty(len(arr), dtype=np.float64)
        j = 0
        hold = self._last_f
        for i in range(len(arr)):
            if i % self.decim == 0 and j < len(f_hat_ds):
                hold = f_hat_ds[j]
                j += 1
            out[i] = hold
        return out
    
    def estimate(self, t: np.ndarray, v: np.ndarray) -> np.ndarray:
        dt_new = float(t[1] - t[0])
        # Sincronizar bases matemáticas si la simulación cambia de velocidad
        if abs(self.dt - dt_new) > 1e-10:
            self.dt = dt_new
            self._update_internals()
        self.reset()
            
        res = self.step_vectorized(v)
        
        if self.verbose and self._total_clamps > 0:
            print(f"      [!] IPDFT Info: Sanity Check limit hit {self._total_clamps} times. Transient or out-of-bounds frequency detected.")
            
        return res
