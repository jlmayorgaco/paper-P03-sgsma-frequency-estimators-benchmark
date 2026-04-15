import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Rutas
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from scenarios.ieee_single_sinwave import IEEESingleSinWaveScenario
from estimators.pll import PLL_Estimator

print("--- MINIMAL DEBUG TEST ---")

# 1. Señal perfecta (Como en tu test que sí funcionó)
sc = IEEESingleSinWaveScenario.run(
    duration_s=1.5, 
    amplitude=1.0, 
    freq_hz=60.0, 
    noise_sigma=0.0
)

# 2. Instanciamos UN solo estimador con parámetros fijos (los que Optuna halló antes)
pll = PLL_Estimator(kp_scale=0.0641, ki_scale=0.4665, pd_lpf_alpha=0.01)

# 3. Procesamos la señal
f_hat = np.array([pll.step(float(v)) for v in sc.v])

# 4. Graficamos directo
plt.figure(figsize=(8, 4))
plt.plot(sc.t, sc.f_true, 'k--', lw=2, label="Real (60 Hz)")
plt.plot(sc.t, f_hat, 'r-', lw=2, label="Estimación PLL")
plt.ylim(55, 65) # Ventana estricta para ver si llega o no
plt.xlabel("Tiempo [s]")
plt.ylabel("Frecuencia [Hz]")
plt.legend()
plt.grid(True)

# Guardar
out_path = ROOT / "tests" / "montecarlo" / "minimal_debug.png"
plt.savefig(out_path)
print(f"Gráfica guardada en: {out_path}")