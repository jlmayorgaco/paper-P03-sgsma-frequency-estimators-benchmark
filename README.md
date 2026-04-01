# 📊 Benchmarking Dynamic Frequency Estimators for Low-Inertia IBR Grids  
### A Latency–Robustness Trade-off Analysis

This repository contains the **complete codebase, simulation framework, and LaTeX source** for the paper:

> *Benchmarking Dynamic Frequency Estimators for Low-Inertia IBR Grids: A Latency–Robustness Trade-off Analysis*

---

## 🚀 Overview

As power systems transition toward **low-inertia grids dominated by inverter-based resources (IBRs)**, frequency dynamics become faster and more volatile.

This repository provides a **reproducible benchmarking platform** to evaluate frequency estimation algorithms under realistic and extreme conditions, focusing on the trade-off between:

- ⚡ **Latency** → fast response for protection  
- 🛡️ **Robustness** → resilience to noise, harmonics, and transients  

---

## 🧠 Key Features

- 🔬 **Multi-family estimator benchmark**
  - PLL-based: SRF-PLL, SOGI-FLL  
  - Window-based: IpDFT, TFT  
  - Recursive: RLS, VFF-RLS  
  - Model-based: EKF, UKF, RA-EKF  
  - Data-driven: Koopman, PI-GRU  

- ⚡ **Proposed RA-EKF**
  - Explicit RoCoF state  
  - Innovation-driven covariance scaling  
  - Event-gating for phase discontinuities  

- 🧪 **Stress-test scenarios**
  - IEC/IEEE-inspired tests (step, ramp, modulation)  
  - Composite islanding (phase jumps + harmonics)  
  - Multi-event IBR disturbance sequence  

- 📊 **Advanced performance metrics**
  - RMSE  
  - Peak error  
  - Settling time  
  - **Trip-risk duration (protection-critical)**  

---

## 🏗️ Repository Structure

```
.
├── src/
│   ├── estimators/        # Estimator implementations
│   ├── scenarios/         # Test scenario generators
│   ├── benchmark/         # Evaluation pipeline
│   └── utils/             # Helpers and signal processing
│
├── results/
│   ├── figures/           # IEEE-ready plots
│   ├── tables/            # Benchmark tables
│   └── reports/           # JSON outputs
│
├── latex/
│   ├── main.tex
│   ├── sections/
│   ├── figures/
│   └── bibliography.bib
│
├── configs/
├── scripts/
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

```bash
# Run full benchmark
python main.py

# Generate plots and tables
python scripts/generate_results.py
```

---

## 📈 Key Insights

- Window-based methods → best steady-state accuracy  
- Model-based methods (RA-EKF) → best dynamic tracking  
- PLL-based → low computational cost but higher trip-risk  

👉 The proposed **RA-EKF significantly reduces trip-risk under phase discontinuities** while maintaining low latency.

---

## 🧪 Reproducibility

- Deterministic simulations  
- Structured parameter search  
- JSON-based result storage  
- Fully automated figure generation  

---

## 🎯 Research Scope

This repository is intended as a **benchmarking platform for**:

- Frequency estimation in low-inertia grids  
- Protection and relay applications  
- DSP / FPGA implementation studies  
- Future estimator development  

---

## 📌 Future Work

- Hardware-in-the-loop validation  
- FPGA / real-time deployment  
- Distributed estimation (multi-agent / DKF)  
- Extended statistical analysis (Monte Carlo, hypothesis testing)  

---

## 📚 Citation

```bibtex
@inproceedings{mayorga2026benchmark,
  title={Benchmarking Dynamic Frequency Estimators for Low-Inertia IBR Grids: A Latency-Robustness Trade-off Analysis},
  author={Mayorga Taborda, Jorge Luis and Africano Rodriguez, Yessica and Jimenez, Fernando},
  booktitle={IEEE SGSMA 2026},
  year={2026}
}
```

---

## 🧑‍💻 Author

**Jorge Luis Mayorga Taborda**  
MSc Robotics & Control  
Universidad de los Andes  

---

## ⭐ Philosophy

> “You cannot improve what you don’t benchmark — and you cannot benchmark what you don’t stress.”
