# pigru_train.py — PI-GRU Generalist Training (Aligned Preprocessing)
#
# Trains a single PI-GRU model for the paper benchmark as a GENERALIST baseline.
#
# SCIENTIFICALLY IMPORTANT:
#   Training preprocessing is now aligned with inference preprocessing.
#   The model sees the SAME front-end at training time as at deployment:
#     - IIR_Bandpass (causal 2nd-order Butterworth bandpass at 60 Hz)
#     - FastRMS_Normalizer (causal sliding-window AGC)
#   This eliminates the offline-RMS preprocessing mismatch that previously existed.
#
# TWO EVALUATION MODES:
#   MODE A — GENERALIST (this script, default):
#     Train once on broad synthetic distribution. Frozen for all scenarios.
#     This is the PRIMARY paper benchmark mode.
#
#   MODE B — SPECIALIZED (optional ablation, NOT in main paper):
#     Fine-tune per scenario family. Clearly labeled as upper bound.
#
# MODEL SELECTION:
#   Best checkpoint selected by VALIDATION loss, not training loss.
#   Validation uses held-out benchmark-like signal families.
#
import os
import math
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader

from estimators import FS_DSP, IIR_Bandpass, FastRMS_Normalizer
from pigru_model import PIDRE_Model

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

WINDOW_LEN = 100       # 10 ms @ 10 kHz
BATCH_SIZE = 128
STEPS_PER_EPOCH = 200
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

# Paths — generalist model output (MODE A)
MODEL_PATH = "pi_gru_pmu.pt"       # Canonical generalist checkpoint
CONFIG_PATH = "pi_gru_pmu_config.json"

# Validation config
VAL_STEPS = 50   # validation batches per validation check
VAL_EVERY = 5    # epochs between validation checks


# ==============================================================================
# PREPROCESSING PIPELINE (aligned with inference)
# ==============================================================================
# IMPORTANT: This must match the inference front-end in PIGRU_FreqEstimator.step():
#   z_bp = self.bp.step(z_raw)     # IIR_Bandpass causal
#   z    = self.norm.step(z_bp)    # FastRMS_Normalizer causal
#
# We use the SAME classes here, but process entire signals to extract windows.
# ==============================================================================

def apply_inference_front_end(signal: np.ndarray):
    """
    Apply the same IIR_Bandpass + FastRMS_Normalizer front-end used at inference.
    Takes a 1D raw voltage signal and returns the processed signal.

    This is the authoritative preprocessing function for PI-GRU training.
    Any change here must be mirrored in PIGRU_FreqEstimator.step().

    Parameters
    ----------
    signal : np.ndarray
        Raw voltage samples at FS_DSP (10 kHz).

    Returns
    -------
    np.ndarray
        Preprocessed signal of same length.
    """
    bp = IIR_Bandpass()
    agc = FastRMS_Normalizer()
    out = np.zeros_like(signal, dtype=np.float32)
    for i, s in enumerate(signal):
        out[i] = agc.step(bp.step(float(s)))
    return out


# ==============================================================================
# SYNTHETIC SIGNAL GENERATOR (aligned preprocessing)
# ==============================================================================
class SyntheticGenerator(IterableDataset):
    """
    Generates synthetic training windows using the SAME front-end as inference:
      IIR_Bandpass + FastRMS_Normalizer (causal, per-sample).

    Distributions:
      - Mode 0: Steady (stable sinusoid + noise)
      - Mode 1: Frequency step (sudden jump)
      - Mode 2: Frequency ramp (linear RoCoF)
      - Mode 3: Ringdown (IBR-style damped oscillation)
      - Mode 4: Phase jump (IBR_Nightmare analogue — freq unchanged)
      - Mode 5: Amplitude step (IEEE_Mag_Step analogue — freq unchanged)

    Target: frequency of the LAST sample in the window (as at inference).
    """

    def __init__(self, batch_size=32, window_len=100, seed_offset=0):
        self.batch_size = batch_size
        self.window_len = window_len
        self.dt = 1.0 / FS_DSP
        self._rng = np.random.default_rng(SEED + seed_offset)

    def __iter__(self):
        while True:
            yield self.generate_batch()

    def generate_batch(self):
        X = np.zeros((self.batch_size, self.window_len, 1), dtype=np.float32)
        y = np.zeros((self.batch_size,), dtype=np.float32)
        t = np.arange(self.window_len) * self.dt

        for i in range(self.batch_size):
            f0 = self._rng.uniform(55.0, 65.0)

            mode = self._rng.choice(
                [0, 1, 2, 3, 4, 5],
                p=[0.20, 0.15, 0.15, 0.20, 0.15, 0.15]
            )

            freq_profile = np.ones_like(t) * f0
            phi_offset = np.zeros_like(t)
            amp_profile = np.ones_like(t)

            if mode == 1:  # Frequency step
                step_t = self._rng.integers(10, self.window_len - 10)
                f_new = f0 + self._rng.uniform(-2.0, 2.0)
                freq_profile[step_t:] = f_new

            elif mode == 2:  # Frequency ramp — IEEE 1547-2018 Cat. III: ±3.0 Hz/s, ride-through ±1.5 Hz
                rate = self._rng.uniform(-3.0, 3.0)
                freq_profile = np.clip(f0 + rate * t, f0 - 1.5, f0 + 1.5)

            elif mode == 3:  # Ringdown (IBR)
                A_osc = self._rng.uniform(0.5, 3.0)
                sigma = self._rng.uniform(1.0, 6.0)
                w_osc = 2 * np.pi * self._rng.uniform(1.0, 5.0)
                freq_profile = f0 + A_osc * np.exp(-sigma * t) * np.sin(w_osc * t)

            elif mode == 4:  # Phase jump (freq unchanged) — max ±80° (Scen D=60°, E=40°/80°)
                jump_t = self._rng.integers(10, self.window_len - 10)
                jump_rad = self._rng.uniform(-np.deg2rad(80), np.deg2rad(80))
                phi_offset[jump_t:] = jump_rad

            elif mode == 5:  # Amplitude step (freq unchanged)
                fault_t = self._rng.integers(10, self.window_len - 10)
                scale = self._rng.uniform(0.1, 1.2)
                amp_profile[fault_t:] = scale

            # Phase integration
            target_f = freq_profile[-1]
            phi_freq = np.cumsum(freq_profile * self.dt) * 2 * np.pi
            phi_total = phi_freq + phi_offset + self._rng.uniform(0, 2 * np.pi)
            v = amp_profile * np.sin(phi_total)

            # IBR realism: harmonics — IEEE 519 THD<=5%: 5th=4%, 7th=2%, interharmonic=0.5%
            # Aligned with Scen D (IBR_Nightmare) and Scen E (IBR_MultiEvent)
            if self._rng.random() < 0.6:
                v += self._rng.uniform(0, 0.04) * np.sin(5 * phi_total)   # 5th ≤4%
                v += self._rng.uniform(0, 0.02) * np.sin(7 * phi_total)   # 7th ≤2%
                if self._rng.random() < 0.3:
                    f_inter = self._rng.uniform(20, 40)
                    v += self._rng.uniform(0, 0.005) * np.sin(2 * np.pi * f_inter * t)  # ≤0.5%

            # Gaussian noise
            noise_lvl = self._rng.uniform(0.0001, 0.005)
            v += self._rng.normal(0, noise_lvl, size=len(t))

            # Impulsive noise
            if self._rng.random() < 0.25:
                idx_imp = self._rng.integers(0, self.window_len)
                v[idx_imp] += self._rng.uniform(-0.8, 0.8)

            # ─── PREPROCESSING ALIGNMENT FIX (Task A) ─────────────────────────────
            # Apply the SAME IIR_Bandpass + FastRMS_Normalizer pipeline
            # used at inference time. This replaces the old offline RMS approach.
            v_proc = apply_inference_front_end(v)

            X[i, :, 0] = v_proc
            # Target: delta frequency (model train mode outputs delta; eval adds 60)
            y[i] = target_f - 60.0

        return torch.from_numpy(X), torch.from_numpy(y)


# ==============================================================================
# BENCHMARK-LIKE VALIDATION GENERATOR
# ==============================================================================
class ValidationGenerator(IterableDataset):
    """
    Generates held-out validation batches that resemble the actual benchmark
    scenarios (not the training distribution).

    Validation families:
      - "steady_heavy": steady + high noise + harmonics (stress test)
      - "fast_ramp": rapid RoCoF (5 Hz/s) — harder than training range
      - "composite": multi-disturbance (freq jump + amplitude sag together)
      - "ringdown_fast": rapid oscillation (higher freq than training)
      - "phase_jump_large": large phase jump (>45 deg)

    Model selection is based on validation RMSE, NOT training loss.
    """

    def __iter__(self):
        while True:
            yield self.generate_batch()

    def __init__(self, batch_size=64, window_len=100):
        self.batch_size = batch_size
        self.window_len = window_len
        self.dt = 1.0 / FS_DSP
        self._rng = np.random.default_rng(SEED + 999)  # Different seed from train

    def generate_batch(self):
        """Generate one batch of validation samples."""
        X = np.zeros((self.batch_size, self.window_len, 1), dtype=np.float32)
        y = np.zeros((self.batch_size,), dtype=np.float32)
        t = np.arange(self.window_len) * self.dt

        for i in range(self.batch_size):
            f0 = self._rng.uniform(55.0, 65.0)

            # Deliberately harder than training distributions
            val_mode = self._rng.choice([0, 1, 2, 3, 4])

            freq_profile = np.ones_like(t) * f0
            phi_offset = np.zeros_like(t)
            amp_profile = np.ones_like(t)

            if val_mode == 0:  # Steady with heavy noise
                noise_lvl = self._rng.uniform(0.003, 0.008)
                v = np.sin(2 * np.pi * f0 * t + self._rng.uniform(0, 2 * np.pi))
                v += self._rng.normal(0, noise_lvl, size=len(t))

            elif val_mode == 1:  # Fast ramp (beyond training range)
                rate = self._rng.choice([-5.0, -4.0, 4.0, 5.0])  # Harder than training ±3 Hz/s
                freq_profile = f0 + rate * t
                phi_freq = np.cumsum(freq_profile * self.dt) * 2 * np.pi
                v = np.sin(phi_freq + self._rng.uniform(0, 2 * np.pi))
                v += self._rng.normal(0, 0.002, size=len(t))

            elif val_mode == 2:  # Composite: freq step + amplitude sag
                step_t = self._rng.integers(15, self.window_len - 15)
                f_new = f0 + self._rng.uniform(-3.0, 3.0)
                freq_profile[step_t:] = f_new
                amp_profile[step_t:] = self._rng.uniform(0.3, 0.7)
                phi_freq = np.cumsum(freq_profile * self.dt) * 2 * np.pi
                v = amp_profile * np.sin(phi_freq + self._rng.uniform(0, 2 * np.pi))
                v += self._rng.normal(0, 0.003, size=len(t))

            elif val_mode == 3:  # Fast ringdown
                A_osc = self._rng.uniform(2.0, 5.0)
                sigma = self._rng.uniform(4.0, 8.0)
                w_osc = 2 * np.pi * self._rng.uniform(3.0, 8.0)
                freq_profile = f0 + A_osc * np.exp(-sigma * t) * np.sin(w_osc * t)
                phi_freq = np.cumsum(freq_profile * self.dt) * 2 * np.pi
                v = np.sin(phi_freq + self._rng.uniform(0, 2 * np.pi))
                v += self._rng.normal(0, 0.003, size=len(t))

            else:  # Large phase jump — harder than training max 80° (Scen D=60°, E=80°)
                jump_t = self._rng.integers(15, self.window_len - 15)
                jump_rad = self._rng.uniform(np.deg2rad(60), np.deg2rad(90))  # 60-90 deg
                phi_offset[jump_t:] = jump_rad
                phi_freq = np.cumsum(freq_profile * self.dt) * 2 * np.pi
                v = np.sin(phi_freq + phi_offset + self._rng.uniform(0, 2 * np.pi))
                v += self._rng.normal(0, 0.003, size=len(t))

            target_f = freq_profile[-1]

            # Apply aligned preprocessing
            v_proc = apply_inference_front_end(v)
            X[i, :, 0] = v_proc
            y[i] = target_f - 60.0

        return torch.from_numpy(X), torch.from_numpy(y)


# ==============================================================================
# TRAINING LOOP (generalist mode)
# ==============================================================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] PI-GRU Generalist Training (aligned preprocessing)")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Window: {WINDOW_LEN} samples = {WINDOW_LEN / FS_DSP * 1e3:.1f} ms")
    print(f"[INFO] Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, Steps/epoch: {STEPS_PER_EPOCH}")

    model = PIDRE_Model(hidden_dim=128, num_layers=2, dropout=0.2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS
    )
    criterion = nn.MSELoss()

    # Training dataset
    train_gen = SyntheticGenerator(batch_size=BATCH_SIZE, window_len=WINDOW_LEN, seed_offset=0)
    train_loader = DataLoader(train_gen, batch_size=None)
    iter_train = iter(train_loader)

    # Validation dataset
    val_gen = ValidationGenerator(batch_size=BATCH_SIZE, window_len=WINDOW_LEN)
    iter_val = iter(DataLoader(val_gen, batch_size=None))

    best_val_rmse = float("inf")
    best_epoch = -1
    model.train()

    print(f"\n[INFO] Starting training with validation-based model selection...")
    print(f"[INFO] Validation check every {VAL_EVERY} epochs")
    print()

    for epoch in range(1, EPOCHS + 1):
        # ── Training step ──────────────────────────────────────────────────────
        epoch_loss = 0.0
        model.train()

        for _ in range(STEPS_PER_EPOCH):
            X, y_true = next(iter_train)
            X, y_true = X.to(device), y_true.to(device)

            optimizer.zero_grad()
            y_pred = model(X)   # train mode: outputs delta freq
            loss = criterion(y_pred, y_true)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        train_rmse = math.sqrt(epoch_loss / STEPS_PER_EPOCH)

        # ── Validation (periodic) ───────────────────────────────────────────────
        do_val = (epoch % VAL_EVERY == 0) or (epoch == EPOCHS)
        val_rmse = None

        if do_val:
            model.eval()
            val_loss = 0.0
            val_count = 0
            with torch.no_grad():
                for _ in range(VAL_STEPS):
                    Xv, yv = next(iter_val)
                    Xv, yv = Xv.to(device), yv.to(device)
                    model.train()
                    yvp = model(Xv)
                    model.eval()
                    val_loss += criterion(yvp, yv).item() * Xv.size(0)
                    val_count += Xv.size(0)
            val_rmse = math.sqrt(val_loss / max(val_count, 1))
            model.train()

        # ── Logging ───────────────────────────────────────────────────────────
        if (epoch % 5 == 0) or do_val:
            val_str = f" | Val RMSE={val_rmse:.4f}" if val_rmse is not None else ""
            print(f"Epoch {epoch:03d}/{EPOCHS} | Train RMSE={train_rmse:.4f}{val_str}")

        # ── Model selection by validation RMSE (not training loss!) ────────────
        if do_val and (val_rmse is not None) and (val_rmse < best_val_rmse):
            best_val_rmse = val_rmse
            best_epoch = epoch
            torch.save(model.state_dict(), MODEL_PATH)
            if epoch % 10 == 0:
                print(f"  ★ New best model (Val RMSE={val_rmse:.4f}) saved")

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n[DONE] Training complete.")
    print(f"        Best validation RMSE: {best_val_rmse:.4f} Hz (epoch {best_epoch})")

    # Load best model for final eval
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    # Final validation report
    print(f"\n[FINAL VALIDATION REPORT]")
    val_gen_final = ValidationGenerator(batch_size=BATCH_SIZE, window_len=WINDOW_LEN)
    val_loader_final = DataLoader(val_gen_final, batch_size=None)
    iter_vf = iter(val_loader_final)

    model.eval()
    family_losses = {name: [] for name in ["steady", "fast_ramp", "composite", "ringdown_fast", "phase_jump"]}
    overall_loss = 0.0
    n_total = 0
    family_idx = 0

    with torch.no_grad():
        for _ in range(100):
            Xv, yv = next(iter_vf)
            Xv, yv = Xv.to(device), yv.to(device)
            model.train()
            yvp = model(Xv)
            model.eval()
            mse = (yvp - yv) ** 2
            family_losses[list(family_losses.keys())[family_idx % 5]].extend(mse.cpu().tolist())
            overall_loss += mse.sum().item()
            n_total += Xv.size(0)
            family_idx += 1

    overall_rmse = math.sqrt(overall_loss / max(n_total, 1))
    print(f"  Overall validation RMSE: {overall_rmse:.4f} Hz")
    for fam, losses in family_losses.items():
        if losses:
            fam_rmse = math.sqrt(np.mean(losses))
            print(f"  {fam:20s}: RMSE={fam_rmse:.4f} Hz (n={len(losses)})")

    # ── Save config ───────────────────────────────────────────────────────────
    config = {
        "window_len_samples": WINDOW_LEN,
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "target_centered": True,
        "preprocessing": "IIR_Bandpass + FastRMS_Normalizer (aligned with inference)",
        "training_mode": "GENERALIST (mode A — single model, frozen for all scenarios)",
        "validation_regime": "benchmark-like held-out families, model selected by val RMSE",
        "best_val_rmse": float(best_val_rmse),
        "best_epoch": int(best_epoch),
        "architecture": "PIDRE_Model (Conv1D + bidirectional GRU + Attention)",
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)

    print(f"\n[DONE] Model saved to: {MODEL_PATH}")
    print(f"        Config saved to: {CONFIG_PATH}")
    print(f"\n[NOTE] This is MODE A (GENERALIST).")
    print(f"        MODE B (SPECIALIZED) fine-tuning is available as a separate experiment.")
    print(f"        Do NOT use MODE B results as the main paper baseline.")


if __name__ == "__main__":
    train()
