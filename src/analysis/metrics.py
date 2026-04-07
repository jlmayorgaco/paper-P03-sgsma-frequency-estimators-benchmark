import numpy as np

# =====================================================================
# Bloque 0: Funciones Utilitarias Privadas
# =====================================================================

def _max_contiguous_time(mask_bool: np.ndarray, dt: float) -> float:
    """Calcula el tiempo continuo máximo donde la máscara booleana es True."""
    padded = np.pad(mask_bool, (1, 1), mode='constant')
    edges = np.diff(padded.astype(int))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    
    if len(starts) == 0:
        return 0.0
    return float(np.max(ends - starts) * dt)


def _calc_isi(error_array: np.ndarray, fs_dsp: float, target_freq_hz: float) -> float:
    """
    Interharmonic Susceptibility Index (ISI).
    Extrae la amplitud de la fuga espectral en el error de frecuencia 
    asociada a un interarmónico específico.
    """
    if len(error_array) < 2:
        return 0.0
        
    error_ac = error_array - np.mean(error_array)
    fft_err = np.abs(np.fft.rfft(error_ac))
    freqs = np.fft.rfftfreq(len(error_ac), d=1.0/fs_dsp)
    
    idx_target = np.argmin(np.abs(freqs - target_freq_hz))
    ripple_amplitude = (fft_err[idx_target] * 2.0) / len(error_array)
    
    return float(ripple_amplitude)


def _prepare_steady_state_vectors(f_hat: np.ndarray, f_true: np.ndarray, fs_dsp: float, structural_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Recorta el transitorio de 'cold-start' inicial.
    Retorna (f_hat_steady, f_true_steady).
    """
    n_signal = len(f_hat)
    baseline_samples = int(0.15 * fs_dsp)
    start_idx = min(max(baseline_samples, structural_samples), int(0.9 * n_signal))
    
    f_hat_steady = f_hat[start_idx:]
    f_true_steady = f_true[start_idx:]
    
    if len(f_hat_steady) < 2:
        return np.zeros(2), np.zeros(2)
        
    return f_hat_steady, f_true_steady


# =====================================================================
# Bloque 1: Precisión Clásica (IEC/IEEE Baseline)
# =====================================================================

def m1_rmse_hz(error: np.ndarray) -> float:
    """Root Mean Square Error."""
    return float(np.sqrt(np.mean(error**2)))

def m2_mae_hz(error: np.ndarray) -> float:
    """Mean Absolute Error (Equivalente al FE promedio IEC)."""
    return float(np.mean(np.abs(error)))

def m3_max_peak_hz(error: np.ndarray) -> float:
    """Error máximo absoluto (Peak Error)."""
    return float(np.max(np.abs(error)))

def m4_std_error_hz(error: np.ndarray) -> float:
    """Desviación estándar del error absoluto (Variabilidad)."""
    return float(np.std(np.abs(error)))


# =====================================================================
# Bloque 2: Riesgo de Protecciones (Protective Relay Risk)
# =====================================================================

def m5_trip_risk_s(error: np.ndarray, dt: float, threshold: float = 0.5) -> float:
    """Tiempo acumulado violando la banda muerta del relé (> 0.5 Hz)."""
    trip_mask = np.abs(error) > threshold
    return float(np.sum(trip_mask) * dt)

def m6_max_contig_trip_s(error: np.ndarray, dt: float, threshold: float = 0.5) -> float:
    """Máxima duración continua de una violación de la banda muerta."""
    trip_mask = np.abs(error) > threshold
    return _max_contiguous_time(trip_mask, dt)

def m7_pcb_hz(error: np.ndarray) -> float:
    """Probabilistic Compliance Bound (mu + 3*sigma)."""
    return float(np.mean(np.abs(error)) + 3.0 * np.std(np.abs(error)))

def m8_settling_time_s(error: np.ndarray, dt: float, threshold: float = 0.2) -> float:
    """Tiempo desde el inicio de la ventana hasta que el error cae y no vuelve a superar 0.2 Hz."""
    settle_mask = np.abs(error) > threshold
    if np.any(settle_mask):
        last_breach_idx = np.where(settle_mask)[0][-1]
        return float((last_breach_idx + 1) * dt)
    return 0.0


# =====================================================================
# Bloque 3: RoCoF y Propuestas IBR-Centric
# =====================================================================

def m9_m10_rfe_metrics(f_hat_steady: np.ndarray, f_true_steady: np.ndarray, dt: float) -> tuple[float, float]:
    """Retorna (RFE_max_hz_s, RFE_rms_hz_s)."""
    rocof_hat = np.gradient(f_hat_steady, dt)
    rocof_true = np.gradient(f_true_steady, dt)
    rfe_array = rocof_hat - rocof_true
    
    return float(np.max(np.abs(rfe_array))), float(np.sqrt(np.mean(rfe_array**2)))

def m11_rnaf_db(f_hat_steady: np.ndarray, f_true_steady: np.ndarray, dt: float, noise_sigma: float) -> float:
    """RoCoF Noise Amplification Factor [dB]."""
    rocof_hat = np.gradient(f_hat_steady, dt)
    rocof_true = np.gradient(f_true_steady, dt)
    rfe_array = rocof_hat - rocof_true
    
    var_rfe = float(np.var(rfe_array))
    var_noise = noise_sigma**2
    
    if var_noise > 0 and var_rfe > 0:
        return float(10.0 * np.log10(var_rfe / var_noise))
    return 0.0

def m12_isi_pu(error: np.ndarray, fs_dsp: float, target_freq_hz: float = 32.5) -> float:
    """Interharmonic Susceptibility Index (ISI)."""
    return _calc_isi(error, fs_dsp, target_freq_hz)


# =====================================================================
# Bloque 4: Viabilidad de Hardware y Latencia
# =====================================================================

def m13_cpu_time_us(exec_time_s: float, n_samples: int) -> float:
    """Tiempo de CPU promedio por muestra en microsegundos."""
    return float((exec_time_s / max(n_samples, 1)) * 1e6)

def m14_struct_latency_ms(structural_samples: int, fs_dsp: float) -> float:
    """Latencia estructural algorítmica en milisegundos."""
    return float((structural_samples / fs_dsp) * 1000.0)

def m15_pcb_compliant(pcb: float, limit: float = 0.05) -> bool:
    """Retorna True si el PCB está dentro del límite IEEE (ej. 50 mHz)."""
    return bool(pcb <= limit)


# =====================================================================
# Bloque 5: Específicos del Paper (Figuras y Tablas)
# =====================================================================

def m16_heatmap_pass(rmse: float, max_peak: float, trip_risk_s: float) -> bool:
    """Evaluación estricta para la Fig. 2g (Compliance Heatmap)."""
    return bool((rmse < 0.05) and (max_peak < 0.5) and (trip_risk_s < 0.1))

def m17_hw_class(cpu_time_us: float) -> str:
    """Clasificación de despliegue de hardware para la Tabla V."""
    if cpu_time_us < 20.0: return "P1"
    if cpu_time_us <= 40.0: return "P2"
    return "M1"


# =====================================================================
# Orquestador Principal (Punto de Entrada)
# =====================================================================

def calculate_all_metrics(
    f_hat: np.ndarray, 
    f_true: np.ndarray, 
    fs_dsp: float, 
    exec_time_s: float, 
    structural_samples: int, 
    noise_sigma: float = 0.0,
    interharmonic_hz: float = 32.5
) -> dict:
    """
    Calcula el set completo de métricas llamando a las funciones individuales (m1 -> m17).
    """
    
    # 0. Preparar vectores
    n_signal = len(f_hat)
    dt = 1.0 / fs_dsp
    f_hat_st, f_true_st = _prepare_steady_state_vectors(f_hat, f_true, fs_dsp, structural_samples)
    error_st = f_hat_st - f_true_st

    # 1. Precisión Clásica
    rmse       = m1_rmse_hz(error_st)
    mae        = m2_mae_hz(error_st)
    max_peak   = m3_max_peak_hz(error_st)
    std_error  = m4_std_error_hz(error_st)

    # 2. Riesgo de Protecciones
    trip_risk  = m5_trip_risk_s(error_st, dt)
    max_contig = m6_max_contig_trip_s(error_st, dt)
    pcb        = m7_pcb_hz(error_st)
    settling   = m8_settling_time_s(error_st, dt)

    # 3. RoCoF e IBR-Centric
    rfe_max, rfe_rms = m9_m10_rfe_metrics(f_hat_st, f_true_st, dt)
    rnaf_db    = m11_rnaf_db(f_hat_st, f_true_st, dt, noise_sigma)
    isi_pu     = m12_isi_pu(error_st, fs_dsp, interharmonic_hz)

    # 4. Viabilidad de Hardware
    cpu_us     = m13_cpu_time_us(exec_time_s, n_signal)
    struct_ms  = m14_struct_latency_ms(structural_samples, fs_dsp)
    compliant  = m15_pcb_compliant(pcb)

    # 5. Específicos del Paper
    hmap_pass  = m16_heatmap_pass(rmse, max_peak, trip_risk)
    hw_class   = m17_hw_class(cpu_us)

    # Retorno consolidado
    return {
        "m1_rmse_hz":           round(rmse, 6),
        "m2_mae_hz":            round(mae, 6),
        "m3_max_peak_hz":       round(max_peak, 6),
        "m4_std_error_hz":      round(std_error, 6),
        
        "m5_trip_risk_s":       round(trip_risk, 6),
        "m6_max_contig_trip_s": round(max_contig, 6),
        "m7_pcb_hz":            round(pcb, 6),
        "m8_settling_time_s":   round(settling, 6),
        
        "m9_rfe_max_hz_s":      round(rfe_max, 4),
        "m10_rfe_rms_hz_s":     round(rfe_rms, 4),
        "m11_rnaf_db":          round(rnaf_db, 4),
        "m12_isi_pu":           round(isi_pu, 6),
        
        "m13_cpu_time_us":      round(cpu_us, 4),
        "m14_struct_latency_ms":round(struct_ms, 3),
        "m15_pcb_compliant":    compliant,
        
        "m16_heatmap_pass":     hmap_pass,
        "m17_hw_class":         hw_class
    }