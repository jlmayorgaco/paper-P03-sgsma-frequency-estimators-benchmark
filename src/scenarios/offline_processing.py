import numpy as np
from scipy.signal import hilbert, butter, sosfiltfilt

def extract_true_frequency_non_causal(
    t: np.ndarray, 
    v: np.ndarray, 
    nominal_f: float = 50.0
) -> np.ndarray:
    
    dt = t[1] - t[0]
    fs = 1.0 / dt
    nyq = 0.5 * fs
    
    # 1. ELIMINACIÓN DE DC Y RUIDO (Bandpass SOS)
    # Centramos la señal y filtramos agresivamente fuera de la banda de interés
    v_centered = v - np.mean(v)
    
    low = (nominal_f - 20.0) / nyq
    high = (nominal_f + 20.0) / nyq
    sos_bp = butter(4, [low, high], btype='band', output='sos')
    v_clean = sosfiltfilt(sos_bp, v_centered)
    
    # 2. TRANSFORMADA DE HILBERT
    z = hilbert(v_clean)
    
    # 3. EXTRACCIÓN DE FASE
    phase = np.unwrap(np.angle(z))
    
    # 4. DERIVADA (FRECUENCIA)
    f_raw = np.gradient(phase, dt) / (2.0 * np.pi)
    
    # 5. SUAVIZADO DINÁMICO (Low-pass 15Hz)
    # La frecuencia en sistemas de potencia no cambia a más de 15-20 Hz de ancho de banda
    lp_cutoff = 15.0 / nyq 
    sos_lp = butter(4, lp_cutoff, btype='low', output='sos')
    f_smooth = sosfiltfilt(sos_lp, f_raw)
    
    return f_smooth