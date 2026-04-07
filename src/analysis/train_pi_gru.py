import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import importlib
from pathlib import Path
from tqdm import tqdm

# 1. Rutas
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Importamos la arquitectura y los filtros EXACTOS de producción
from estimators.pi_gru import PIDRE_Model, IIR_Bandpass, FastRMS_Normalizer
from estimators.common import DT_DSP

def get_scenario_instance(module_name):
    try:
        module = importlib.import_module(f"scenarios.{module_name}")
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and "Scenario" in name and name != "BaseScenario":
                return obj()
    except Exception:
        pass
    return None

def generate_synthetic_augmentation(duration=0.5):
    t = np.arange(0, duration, DT_DSP)
    X_syn, Y_syn = [], []
    
    # 1. Tonos Puros y Ruido
    for f_target in np.arange(55.0, 65.5, 0.5):
        v = np.sin(2 * np.pi * f_target * t) + np.random.normal(0, 0.005, len(t))
        X_syn.append((v, f_target * np.ones_like(t)))

    # 2. Rampas Dinámicas (RoCoF)
    for rocof in [-10.0, -5.0, 5.0, 10.0]:
        f_ramp = 60.0 + rocof * t
        phase = (np.cumsum(f_ramp) - f_ramp[0]) * DT_DSP * 2 * np.pi
        v_ramp = np.sin(phase) + np.random.normal(0, 0.005, len(t))
        X_syn.append((v_ramp, f_ramp))

    # 3. Armónicos (Típico en IBRs)
    v_harm = np.sin(2 * np.pi * 60 * t) + 0.05 * np.sin(2 * np.pi * 300 * t)
    X_syn.append((v_harm, 60.0 * np.ones_like(t)))

    return X_syn

def apply_dsp_pipeline(v, nominal_f=60.0, dt=DT_DSP):
    """
    Replica EXACTAMENTE el preprocesamiento de pi_gru.py para que la red
    aprenda a lidiar con el desfase y la normalización RMS reales.
    """
    bp = IIR_Bandpass(fs=1.0/dt, lowcut=nominal_f-20, highcut=nominal_f+20)
    agc_window = int(round(1.0 / nominal_f / dt)) # 167 muestras para 60Hz a 10kHz
    agc = FastRMS_Normalizer(window_size=agc_window)
    
    v_processed = np.zeros_like(v)
    for i in range(len(v)):
        z_bp = bp.step(v[i])
        z_norm = agc.step(z_bp)
        v_processed[i] = z_norm
        
    return v_processed

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[PI-GRU] ENTRENAMIENTO ARQUITECTURA PIDRE | Device: {device}")

    X_data, Y_data = [], []
    window_len = 100 
    
    # Tiempo de estabilización de los filtros (1 ciclo)
    settling_samples = int(round(1.0 / 60.0 / DT_DSP)) 

    print(f"\n[1/3] Procesando Escenarios y Aumentación con Filtros DSP...")
    
    # Procesar Aumentación Sintética
    syn_signals = generate_synthetic_augmentation()
    for v_syn, f_syn in syn_signals:
        # Pasamos la señal por el pipeline DSP
        v_processed = apply_dsp_pipeline(v_syn)
        
        # Empezamos a extraer ventanas DESPUÉS de que el filtro se estabilice
        start_idx = max(settling_samples, window_len)
        for i in range(start_idx, len(v_processed), 10):
            v_win = v_processed[i-window_len:i]
            X_data.append(v_win.reshape(-1, 1))
            Y_data.append(f_syn[i] - 60.0) # Entrenamos Delta f

    # Procesar Escenarios Físicos en Disco
    scenario_dir = SRC / "scenarios"
    all_files = [f.stem for f in scenario_dir.glob("*.py") if f.stem not in ["base", "__init__", "offline_processing"]]

    for filename in all_files:
        sc = get_scenario_instance(filename)
        if sc is None: continue
        try:
            data = sc.generate()
            v, f_true = np.array(data.v), getattr(data, 'f_true', getattr(data, 'f', getattr(data, 'freq', None)))
            if f_true is None: continue

            # Pasamos la señal del escenario por el pipeline DSP
            v_processed = apply_dsp_pipeline(v)

            stride = 10 if "chamorro" in filename else 50
            start_idx = max(settling_samples, window_len)
            for i in range(start_idx, len(v_processed), stride):
                v_win = v_processed[i-window_len:i]
                X_data.append(v_win.reshape(-1, 1))
                Y_data.append(f_true[i] - 60.0)
        except Exception:
            pass

    X_train = torch.tensor(np.array(X_data), dtype=torch.float32).to(device)
    Y_train = torch.tensor(np.array(Y_data), dtype=torch.float32).to(device)
    print(f"Dataset Total: {len(X_train)} ventanas de {window_len} muestras.")

    # Instanciar el modelo robusto
    model = PIDRE_Model(input_dim=1, hidden_dim=128, num_layers=2).to(device)
    criterion = nn.MSELoss() 
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    epochs = 40 
    batch_size = 256 

    print(f"\n[2/3] Entrenando PIDRE_Model...")
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        indices = torch.randperm(X_train.size(0))
        epoch_loss = 0
        for i in range(0, X_train.size(0), batch_size):
            idx = indices[i:i+batch_size]
            optimizer.zero_grad()
            
            pred = model(X_train[idx])
            loss = criterion(pred, Y_train[idx]) 
            loss.backward()
            
            # Clip de gradientes para evitar explosiones
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            
        pbar.set_description(f"Loss: {epoch_loss/(len(X_train)/batch_size):.6f}")

    print("\n[3/3] Exportando modelo...")
    out_dir = SRC / "estimators"
    # Guardamos en formato nativo de PyTorch (.pt)
    torch.save(model.state_dict(), out_dir / "pi_gru_weights.pt")
    print(f">>> FINALIZADO. Guardado en {out_dir / 'pi_gru_weights.pt'}")

if __name__ == "__main__":
    main()