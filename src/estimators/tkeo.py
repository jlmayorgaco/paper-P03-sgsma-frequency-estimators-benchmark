from __future__ import annotations

import math
import numpy as np
from numba import njit

from .base import BaseFrequencyEstimator
from .common import DT_DSP

@njit(cache=True)
def _psi(x0: float, xm1: float, xp1: float) -> float:
    """Operador de Energía de Teager-Kaiser: Psi(x) = x(n)^2 - x(n-1)x(n+1)"""
    return x0**2 - xm1 * xp1

@njit(cache=True)
def _tkeo_vectorized_core(
    v_array: np.ndarray,
    dt: float,
    f_out: float,
    smooth_alpha: float,
    input_smooth_alpha: float,
    x_smooth: float,
    have_x_smooth: bool,
    buffer_x: np.ndarray,
    buffer_y: np.ndarray,
    samples_seen: int,
    noise_power: float,
    derivative_noise_factor: float,
) -> tuple[np.ndarray, float, float, bool, np.ndarray, np.ndarray, int]:
    """
    Núcleo del estimador TKEO utilizando un algoritmo DES más robusto.
    Requiere evaluar la energía tanto de la señal (x) como de su derivada aproximada (y).
    """
    n = len(v_array)
    f_est = np.empty(n, dtype=np.float64)
    two_pi = 2.0 * math.pi

    for i in range(n):
        z = v_array[i]
        if input_smooth_alpha < 1.0:
            if not have_x_smooth:
                x_smooth = z
                have_x_smooth = True
            else:
                x_smooth = (1.0 - input_smooth_alpha) * x_smooth + input_smooth_alpha * z
            z = x_smooth

        # 1. Update signal buffer
        buffer_x[0] = buffer_x[1]
        buffer_x[1] = buffer_x[2]
        buffer_x[2] = buffer_x[3]
        buffer_x[3] = z

        # Calculate derivative approximation: y[n] = x[n] - x[n-1]
        y_n = buffer_x[3] - buffer_x[2]
        
        # Update derivative buffer
        buffer_y[0] = buffer_y[1]
        buffer_y[1] = buffer_y[2]
        buffer_y[2] = y_n

        samples_seen += 1

        # We need sufficient history to calculate energy operators safely
        # Ensure we don't calculate on initial zeros to avoid division by zero
        if samples_seen <= 3:
             f_est[i] = f_out
             continue

        # Energy of the signal at n-1 (center of our available window for symmetry)
        psi_x = _psi(buffer_x[2], buffer_x[1], buffer_x[3]) - noise_power
        
        # Energy of the derivative at n-1
        psi_y = _psi(buffer_y[1], buffer_y[0], buffer_y[2]) - derivative_noise_factor * noise_power

        # Security check: avoid division by zero or extremely small numbers
        if psi_x > 1e-10 and psi_y >= 0:
            # The classic DES ratio:
            # sin^2(w * dt / 2) = Psi(y_n) / (4 * Psi(x_n))
            # However, a more direct and stable formulation often used is:
            # cos(w * dt) = 1 - [Psi(x_n - x_{n-1}) + Psi(x_{n+1} - x_n)] / (4 * Psi(x_n))
            # Or using the ratio directly: cos(w * dt) = 1 - (Psi(y_n) / (2 * Psi(x_n)))
            
            ratio = psi_y / (2.0 * psi_x)
            arg = 1.0 - ratio
            
            # Clamp to domain of arccos
            if arg > 1.0: 
                arg = 1.0
            elif arg < -1.0: 
                arg = -1.0
            
            f_raw = math.acos(arg) / (two_pi * dt)
            
            # Sanity check: prevent absurd jumps if noise causes strange energy ratios
            if math.isnan(f_raw) or f_raw > 120.0 or f_raw < 10.0:
                 f_raw = f_out
        else:
            f_raw = f_out

        # Output filtering (crucial because TKEO acts like a high-pass filter for noise)
        f_out = (1.0 - smooth_alpha) * f_out + smooth_alpha * f_raw
        f_est[i] = f_out

    return f_est, f_out, x_smooth, have_x_smooth, buffer_x, buffer_y, samples_seen

class TKEO_Estimator(BaseFrequencyEstimator):
    """
    Teager-Kaiser Energy Operator (TKEO).
    Estimador de latencia sub-ciclo y costo O(1).
    Actualizado para usar un algoritmo de separación de energía (DES) más estable.
    """
    name = "TKEO"

    def __init__(
        self,
        nominal_f: float = 60.0,
        output_smoothing: float = 0.01,
        input_smoothing: float = 1.0,
        noise_power: float = 0.0,
        derivative_noise_factor: float = 2.0,
        dt: float = DT_DSP,
    ) -> None:
        self.nominal_f = float(nominal_f)
        self.output_smoothing = float(output_smoothing)
        self.input_smoothing = float(input_smoothing)
        if not np.isfinite(self.input_smoothing) or not (0.0 < self.input_smoothing <= 1.0):
            raise ValueError("input_smoothing must be in (0, 1].")
        self.noise_power = float(noise_power)
        self.derivative_noise_factor = float(derivative_noise_factor)
        if not np.isfinite(self.noise_power) or self.noise_power < 0.0:
            raise ValueError("noise_power must be >= 0.")
        if not np.isfinite(self.derivative_noise_factor) or self.derivative_noise_factor < 0.0:
            raise ValueError("derivative_noise_factor must be >= 0.")
        self.dt = float(dt)
        self.reset()

    def reset(self) -> None:
        # Require a buffer of 4 for the signal to compute symmetrical energies
        self.buffer_x = np.zeros(4, dtype=np.float64)
        # Require a buffer of 3 for the derivative
        self.buffer_y = np.zeros(3, dtype=np.float64)
        self.f_out = self.nominal_f
        self.samples_seen = 0
        self.x_smooth = 0.0
        self.have_x_smooth = False

    @classmethod
    def default_params(cls) -> dict[str, float]:
        return {
            "nominal_f": 60.0,
            "output_smoothing": 0.01,
            "input_smoothing": 1.0,
            "noise_power": 0.0,
            "derivative_noise_factor": 2.0,
        }

    def structural_latency_samples(self) -> int:
        return 3 # Latency increased slightly due to deeper buffering required for stability

    def _step(self, z: float) -> float:
        v_array = np.array([z], dtype=np.float64)
        return float(self.step_vectorized(v_array)[0])

    def step_vectorized(self, v_array: np.ndarray) -> np.ndarray:
        v_array = np.asarray(v_array, dtype=np.float64)
        if len(v_array) == 0:
            return np.empty(0, dtype=np.float64)

        (
            f_est,
            self.f_out,
            self.x_smooth,
            self.have_x_smooth,
            self.buffer_x,
            self.buffer_y,
            self.samples_seen,
        ) = _tkeo_vectorized_core(
            v_array=v_array,
            dt=self.dt,
            f_out=self.f_out,
            smooth_alpha=self.output_smoothing,
            input_smooth_alpha=self.input_smoothing,
            x_smooth=self.x_smooth,
            have_x_smooth=self.have_x_smooth,
            buffer_x=self.buffer_x,
            buffer_y=self.buffer_y,
            samples_seen=self.samples_seen,
            noise_power=self.noise_power,
            derivative_noise_factor=self.derivative_noise_factor,
        )
        return f_est

    def estimate(self, t: np.ndarray, v: np.ndarray) -> np.ndarray:
        self.dt = float(t[1] - t[0])
        self.reset()
        return self.step_vectorized(v)
