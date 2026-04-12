import os
# --- CORRECCIÓN MULTIPROCESSING EXTREMA ---
# Estas variables deben establecerse ANTES de importar torch
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from tqdm import tqdm

import math
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
from pathlib import Path

torch.set_num_threads(1)

from .base import BaseFrequencyEstimator
from .common import DT_DSP

class IIR_Bandpass:
    def __init__(self, fs=10000.0, lowcut=40.0, highcut=80.0):
        self.b, self.a = scipy.signal.butter(2, [lowcut, highcut], btype='bandpass', fs=fs)
        self.z = scipy.signal.lfilter_zi(self.b, self.a)
        
    def step(self, x: float) -> float:
        y, self.z = scipy.signal.lfilter(self.b, self.a, [x], zi=self.z)
        return float(y[0])

class FastRMS_Normalizer:
    def __init__(self, window_size=167):
        self.window_size = window_size
        self.buffer = np.zeros(window_size, dtype=np.float64)
        self.idx = 0
        self.sum_sq = 0.0
        
    def step(self, x: float) -> float:
        old_val = self.buffer[self.idx]
        self.sum_sq += (x * x) - (old_val * old_val)
        self.buffer[self.idx] = x
        self.idx = (self.idx + 1) % self.window_size
        rms = math.sqrt(max(self.sum_sq / self.window_size, 1e-12))
        return float(x / rms) if rms > 1e-6 else 0.0

class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        weights = self.attention(x) 
        context = torch.sum(weights * x, dim=1) 
        return context, weights

class PIDRE_Model(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(32)
        )
        self.gru = nn.GRU(
            input_size=32, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )
        self.attn = AttentionBlock(hidden_dim)
        self.fc_freq = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, 1)
        )
        self.fc_amp = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.GELU(), nn.Linear(32, 1), nn.Softplus()
        )

    def forward(self, x):
        x_conv = x.permute(0, 2, 1)
        feat = self.conv(x_conv)
        feat = feat.permute(0, 2, 1) 
        out, _ = self.gru(feat) 
        context, _ = self.attn(out) 
        delta_freq = self.fc_freq(context).squeeze(-1) 
        return delta_freq

class PI_GRU_Estimator(BaseFrequencyEstimator):
    name = "PI-GRU"

    def __init__(self, nominal_f: float = 60.0, dt: float = DT_DSP) -> None:
        self.nominal_f = float(nominal_f)
        self.dt = float(dt)
        self.window_len = 100 
        
        self.device = torch.device("cpu")
        self.model = None # Lazy initialization

    def _init_model_if_needed(self):
        # Solo instancia la red si no existe. Esto salva el multiprocesamiento.
        if self.model is None:
            self.model = PIDRE_Model(input_dim=1, hidden_dim=128, num_layers=2).to(self.device)
            self.model.eval() 
            
            base_dir = Path(__file__).parent
            weights_path = base_dir / "pi_gru_weights.pt"
            
            if weights_path.exists():
                try:
                    self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True), strict=False)
                    # Print silenciado para no ensuciar la terminal
                except Exception:
                    pass

    def reset(self) -> None:
        # Aseguramos que la red cargue de forma segura
        self._init_model_if_needed()
        
        self.buffer = np.zeros(self.window_len, dtype=np.float32)
        self.f_out = self.nominal_f
        self.step_count = 0
        
        self.bp = IIR_Bandpass(fs=1.0/self.dt, lowcut=self.nominal_f-20, highcut=self.nominal_f+20)
        self.agc = FastRMS_Normalizer(window_size=int(round(1.0 / self.nominal_f / self.dt)))
        
        self.ma_len = int(round(1.0 / self.nominal_f / self.dt))
        self.out_buffer = np.ones(self.ma_len, dtype=np.float64) * self.nominal_f

    def structural_latency_samples(self) -> int:
        return (self.window_len // 2) + (self.ma_len // 2)

    def step(self, z: float) -> float:
        z_bp = self.bp.step(z)
        z_norm = self.agc.step(z_bp)
        
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = z_norm
        self.step_count += 1
        
        wait_samples = int(3.0 / self.nominal_f / self.dt) 
        if self.step_count > wait_samples:
            with torch.no_grad():
                x_t = torch.tensor(self.buffer, dtype=torch.float32, device=self.device).view(1, -1, 1)
                f_pred = self.model(x_t).item() + self.nominal_f
                
            self.out_buffer[:-1] = self.out_buffer[1:]
            self.out_buffer[-1] = f_pred
            self.f_out = np.mean(self.out_buffer)
            
        return self.f_out

    def step_vectorized(self, v_array: np.ndarray) -> np.ndarray:
        n = len(v_array)
        f_est = np.empty(n, dtype=np.float64)
        
        # ¡Aquí está la magia! Una barra de progreso para cada simulación
        for i in tqdm(range(n), desc=f"PI-GRU (Inferencia Muestra a Muestra)", leave=False):
            f_est[i] = self.step(v_array[i])
            
        return f_est

    def estimate(self, t: np.ndarray, v: np.ndarray) -> np.ndarray:
        self.reset()
        return self.step_vectorized(v)