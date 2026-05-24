from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import scipy.signal
import torch
import torch.nn as nn

from .base import BaseFrequencyEstimator
from .common import DT_DSP
from .ipdft import IPDFT_Estimator
from .zcd import ZCDEstimator


torch.set_num_threads(1)

DEFAULT_MODEL_CONFIG: dict[str, int | float] = {
    "input_dim": 3,
    "conv_channels": 48,
    "hidden_dim": 160,
    "fc_hidden_dim": 96,
    "num_layers": 2,
    "dropout": 0.15,
}
LEGACY_FEATURE_CONFIG: dict[str, Any] = {
    "mode": "legacy_waveform",
    "clip_hz": 12.0,
}
HYBRID_FEATURE_CONFIG: dict[str, Any] = {
    "mode": "hybrid_priors",
    "clip_hz": 12.0,
}
DEFAULT_TARGET_MODE = "absolute_delta"
RESIDUAL_TARGET_MODE = "residual_to_prior"


def _infer_model_config_from_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, int | float]:
    model_config = dict(DEFAULT_MODEL_CONFIG)
    conv_w = state_dict.get("conv.0.weight")
    gru_w = state_dict.get("gru.weight_ih_l0")
    if conv_w is not None:
        model_config["conv_channels"] = int(conv_w.shape[0])
        model_config["input_dim"] = int(conv_w.shape[1])
    if gru_w is not None:
        model_config["hidden_dim"] = int(gru_w.shape[0] // 3)
    fc_w = state_dict.get("fc_freq.0.weight")
    if fc_w is not None:
        model_config["fc_hidden_dim"] = int(fc_w.shape[0])
    layer_ids = {
        int(key.split("gru.weight_ih_l", 1)[1])
        for key in state_dict
        if key.startswith("gru.weight_ih_l")
    }
    if layer_ids:
        model_config["num_layers"] = max(layer_ids) + 1
    model_config["dropout"] = 0.0
    return model_config


def _normalize_feature_config(feature_config: dict[str, Any] | None) -> dict[str, Any]:
    config = dict(LEGACY_FEATURE_CONFIG)
    if feature_config:
        config.update(feature_config)
    return config


def _clip_frequency_delta(x: np.ndarray | float, clip_hz: float) -> np.ndarray | float:
    return np.clip(x, -abs(float(clip_hz)), abs(float(clip_hz)))


def _quiet_prior_traces(
    v_array: np.ndarray,
    nominal_f: float,
    dt: float,
) -> dict[str, np.ndarray]:
    v_array = np.asarray(v_array, dtype=np.float64)

    zcd = ZCDEstimator(nominal_f=nominal_f, dt=dt)
    zcd.reset()
    zcd_f = zcd.step_vectorized(v_array)

    ipdft = IPDFT_Estimator(nominal_f=nominal_f)
    if abs(ipdft.dt - dt) > 1e-12:
        ipdft.dt = float(dt)
        ipdft._update_internals()
    ipdft.reset()
    ipdft_f = ipdft.step_vectorized(v_array)

    nominal_trace = np.full(len(v_array), nominal_f, dtype=np.float64)
    stacked = np.column_stack([nominal_trace, zcd_f, ipdft_f])
    prior_f = np.median(stacked, axis=1)

    return {
        "prior_f": prior_f.astype(np.float64, copy=False),
        "zcd_f": np.asarray(zcd_f, dtype=np.float64),
        "ipdft_f": np.asarray(ipdft_f, dtype=np.float64),
    }


class IIR_Bandpass:
    def __init__(self, fs: float = 10000.0, lowcut: float = 40.0, highcut: float = 80.0):
        self.b, self.a = scipy.signal.butter(2, [lowcut, highcut], btype="bandpass", fs=fs)
        self.z = scipy.signal.lfilter_zi(self.b, self.a)

    def step(self, x: float) -> float:
        y, self.z = scipy.signal.lfilter(self.b, self.a, [x], zi=self.z)
        return float(y[0])


class FastRMS_Normalizer:
    def __init__(self, window_size: int = 167):
        self.window_size = int(window_size)
        self.buffer = np.zeros(self.window_size, dtype=np.float64)
        self.idx = 0
        self.sum_sq = 0.0

    def step(self, x: float) -> float:
        old_val = self.buffer[self.idx]
        self.sum_sq += (x * x) - (old_val * old_val)
        self.buffer[self.idx] = x
        self.idx = (self.idx + 1) % self.window_size
        rms = math.sqrt(max(self.sum_sq / self.window_size, 1e-12))
        return float(x / rms) if rms > 1e-6 else 0.0


def build_pi_gru_features(
    v_array: np.ndarray,
    nominal_f: float = 60.0,
    dt: float = DT_DSP,
    input_dim: int = 3,
    feature_config: dict[str, Any] | None = None,
    return_aux: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Causal DSP front-end shared by training and inference.

    Feature channels:
    1. RMS-normalized band-pass output
    2. First difference of the normalized signal
    3. Raw band-pass output
    """
    feature_config = _normalize_feature_config(feature_config)
    feature_mode = str(feature_config.get("mode", LEGACY_FEATURE_CONFIG["mode"]))
    clip_hz = float(feature_config.get("clip_hz", LEGACY_FEATURE_CONFIG["clip_hz"]))

    fs = 1.0 / float(dt)
    bp = IIR_Bandpass(fs=fs, lowcut=nominal_f - 20.0, highcut=nominal_f + 20.0)
    rms_window = int(round(1.0 / nominal_f / dt))
    agc = FastRMS_Normalizer(window_size=rms_window)

    n = len(v_array)
    base_features = np.empty((n, 3), dtype=np.float64)
    prev_norm = 0.0

    for i, sample in enumerate(np.asarray(v_array, dtype=np.float64)):
        z_bp = bp.step(float(sample))
        z_norm = agc.step(z_bp)
        d_norm = z_norm - prev_norm
        base_features[i, 0] = z_norm
        base_features[i, 1] = d_norm
        base_features[i, 2] = z_bp
        prev_norm = z_norm

    aux = {
        "prior_f": np.full(n, nominal_f, dtype=np.float64),
        "zcd_f": np.full(n, nominal_f, dtype=np.float64),
        "ipdft_f": np.full(n, nominal_f, dtype=np.float64),
    }
    features = base_features

    if feature_mode == HYBRID_FEATURE_CONFIG["mode"]:
        aux = _quiet_prior_traces(v_array=v_array, nominal_f=nominal_f, dt=dt)
        extra_features = np.column_stack(
            [
                _clip_frequency_delta(aux["prior_f"] - nominal_f, clip_hz),
                _clip_frequency_delta(aux["zcd_f"] - nominal_f, clip_hz),
                _clip_frequency_delta(aux["ipdft_f"] - nominal_f, clip_hz),
                _clip_frequency_delta(aux["zcd_f"] - aux["ipdft_f"], clip_hz),
            ]
        )
        features = np.concatenate([base_features, extra_features], axis=1)
    elif feature_mode != LEGACY_FEATURE_CONFIG["mode"]:
        raise ValueError(f"Unsupported PI-GRU feature mode: {feature_mode}")

    if input_dim > features.shape[1]:
        raise ValueError(
            f"Requested input_dim={input_dim} but feature mode {feature_mode!r} only provides {features.shape[1]} channels."
        )
    if input_dim <= 1:
        sliced = features[:, [0]]
    elif input_dim == 2:
        sliced = features[:, :2]
    else:
        sliced = features[:, :input_dim]

    sliced = np.asarray(sliced, dtype=np.float64)
    if return_aux:
        return sliced, aux
    return sliced


def _causal_moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.asarray(x, dtype=np.float64)

    x = np.asarray(x, dtype=np.float64)
    csum = np.cumsum(x, dtype=np.float64)
    y = np.empty_like(x, dtype=np.float64)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        total = csum[i] - (csum[start - 1] if start > 0 else 0.0)
        y[i] = total / float(i - start + 1)
    return y


def _buffered_output_average(x: np.ndarray, nominal_f: float, window: int) -> np.ndarray:
    if window <= 1:
        return np.asarray(x, dtype=np.float64)

    x = np.asarray(x, dtype=np.float64)
    buf = np.full(window, nominal_f, dtype=np.float64)
    buf_sum = float(nominal_f * window)
    out = np.empty_like(x, dtype=np.float64)
    idx = 0
    for i, value in enumerate(x):
        buf_sum += float(value) - float(buf[idx])
        buf[idx] = float(value)
        idx = (idx + 1) % window
        out[i] = buf_sum / float(window)
    return out


class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        weights = self.attention(x)
        context = torch.sum(weights * x, dim=1)
        return context, weights


class PIDRE_Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        conv_channels: int = 48,
        hidden_dim: int = 160,
        fc_hidden_dim: int = 96,
        num_layers: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.conv_channels = int(conv_channels)
        self.hidden_dim = int(hidden_dim)
        self.fc_hidden_dim = int(fc_hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

        self.conv = nn.Sequential(
            nn.Conv1d(self.input_dim, self.conv_channels, kernel_size=5, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(self.conv_channels),
        )
        self.gru = nn.GRU(
            input_size=self.conv_channels,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.attn = AttentionBlock(self.hidden_dim)
        self.fc_freq = nn.Sequential(
            nn.Linear(self.hidden_dim, self.fc_hidden_dim),
            nn.GELU(),
            nn.Linear(self.fc_hidden_dim, 1),
        )
        # Retained for checkpoint compatibility and future multitask training.
        self.fc_amp = nn.Sequential(
            nn.Linear(self.hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),
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

    def __init__(
        self,
        nominal_f: float = 60.0,
        dt: float = DT_DSP,
        weights_filename: str = "pi_gru_weights_hybrid.pt",
        show_progress: bool = False,
        batch_size: int = 2048,
        output_ma_samples: int | None = None,
        window_len: int = 100,
        warmup_cycles: float = 3.0,
    ) -> None:
        self.nominal_f = float(nominal_f)
        self.dt = float(dt)
        self.weights_filename = str(weights_filename)
        self.show_progress = bool(show_progress)
        self.batch_size = int(batch_size)
        self.window_len = int(window_len)
        self.warmup_cycles = float(warmup_cycles)
        default_ma = max(8, int(round(0.5 / self.nominal_f / self.dt)))
        self.output_ma_samples = int(default_ma if output_ma_samples is None else output_ma_samples)
        self.ma_len = self.output_ma_samples

        self.device = torch.device("cpu")
        self.model: PIDRE_Model | None = None
        self.model_config = dict(DEFAULT_MODEL_CONFIG)
        self.feature_config = _normalize_feature_config(None)
        self.target_mode = DEFAULT_TARGET_MODE
        self.feature_dim = int(self.model_config["input_dim"])

    @classmethod
    def default_params(cls) -> dict[str, float | str | bool | int]:
        return {
            "nominal_f": 60.0,
            "dt": DT_DSP,
            "weights_filename": "pi_gru_weights_hybrid.pt",
            "show_progress": False,
            "batch_size": 2048,
            "output_ma_samples": None,
            "window_len": 100,
            "warmup_cycles": 3.0,
        }

    def _resolve_weights_path(self) -> Path:
        base_dir = Path(__file__).parent
        candidates = [
            self.weights_filename,
            "pi_gru_weights_hybrid.pt",
            "pi_gru_weights.pt",
            "pi_gru_weights_retrained.pt",
            "pi_gru_pmu.pt",
        ]
        tried: list[str] = []
        for candidate in candidates:
            if candidate in tried:
                continue
            tried.append(candidate)
            path = base_dir / candidate
            if path.exists():
                return path
        raise FileNotFoundError("PI-GRU weights not found. Tried: " + ", ".join(tried))

    def _load_checkpoint(
        self,
        weights_path: Path,
    ) -> tuple[dict[str, Any], dict[str, Any], str, dict[str, torch.Tensor]]:
        try:
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(weights_path, map_location=self.device)
        except Exception as exc:
            raise RuntimeError(f"Failed to load PI-GRU weights from {weights_path}") from exc

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model_config = dict(DEFAULT_MODEL_CONFIG)
            model_config.update(checkpoint.get("model_config", {}))
            feature_config = _normalize_feature_config(checkpoint.get("feature_config"))
            target_mode = str(checkpoint.get("target_mode", DEFAULT_TARGET_MODE))
            state_dict = checkpoint["state_dict"]
            return model_config, feature_config, target_mode, state_dict

        if isinstance(checkpoint, dict):
            return _infer_model_config_from_state_dict(checkpoint), _normalize_feature_config(None), DEFAULT_TARGET_MODE, checkpoint

        raise RuntimeError(f"Unsupported PI-GRU checkpoint format in {weights_path}")

    def _init_model_if_needed(self) -> None:
        if self.model is not None:
            return

        weights_path = self._resolve_weights_path()
        model_config, feature_config, target_mode, state_dict = self._load_checkpoint(weights_path)
        self.model_config = model_config
        self.feature_config = feature_config
        self.target_mode = target_mode
        self.feature_dim = int(self.model_config.get("input_dim", DEFAULT_MODEL_CONFIG["input_dim"]))
        self.model = PIDRE_Model(**self.model_config).to(self.device).double()
        self.model.eval()

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            raise RuntimeError(
                f"PI-GRU weights mismatch for {weights_path.name}: missing={missing}, unexpected={unexpected}"
            )

    def _warmup_samples(self) -> int:
        return max(self.window_len, int(round(self.warmup_cycles / self.nominal_f / self.dt)))

    def _predict_delta(self, window_features: np.ndarray) -> float:
        assert self.model is not None
        x_t = torch.from_numpy(window_features.astype(np.float64, copy=False)).to(self.device).unsqueeze(0)
        with torch.no_grad():
            delta_f = float(self.model(x_t).item())
        return delta_f

    def _predict_delta_batch(self, feature_windows: np.ndarray) -> np.ndarray:
        assert self.model is not None
        if len(feature_windows) == 0:
            return np.empty(0, dtype=np.float64)

        outputs: list[np.ndarray] = []
        for start in range(0, len(feature_windows), self.batch_size):
            batch = feature_windows[start:start + self.batch_size]
            x_t = torch.from_numpy(batch.astype(np.float64, copy=False)).to(self.device)
            with torch.no_grad():
                y = self.model(x_t).detach().cpu().numpy().astype(np.float64, copy=False)
            outputs.append(y.reshape(-1))
        return np.concatenate(outputs)

    def reset(self) -> None:
        self._init_model_if_needed()
        self.buffer = np.zeros((self.window_len, self.feature_dim), dtype=np.float64)
        self.f_out = self.nominal_f
        self.step_count = 0
        self.prev_norm = 0.0
        self.bp = IIR_Bandpass(fs=1.0 / self.dt, lowcut=self.nominal_f - 20.0, highcut=self.nominal_f + 20.0)
        self.agc = FastRMS_Normalizer(window_size=int(round(1.0 / self.nominal_f / self.dt)))
        if str(self.feature_config.get("mode")) == HYBRID_FEATURE_CONFIG["mode"]:
            self.prior_zcd = ZCDEstimator(nominal_f=self.nominal_f, dt=self.dt)
            self.prior_ipdft = IPDFT_Estimator(nominal_f=self.nominal_f)
            if abs(self.prior_ipdft.dt - self.dt) > 1e-12:
                self.prior_ipdft.dt = float(self.dt)
                self.prior_ipdft._update_internals()
            self.prior_ipdft.reset()
        else:
            self.prior_zcd = None
            self.prior_ipdft = None
        self.out_buffer = np.ones(self.output_ma_samples, dtype=np.float64) * self.nominal_f
        self.out_idx = 0

    def structural_latency_samples(self) -> int:
        return (self.window_len // 2) + (self.output_ma_samples // 2)

    def _feature_step(self, z: float) -> tuple[np.ndarray, float]:
        z_bp = self.bp.step(float(z))
        z_norm = self.agc.step(z_bp)
        d_norm = z_norm - self.prev_norm
        self.prev_norm = z_norm
        feat_values: list[float] = [z_norm, d_norm, z_bp]
        prior_f = self.nominal_f

        if self.prior_zcd is not None and self.prior_ipdft is not None:
            zcd_f = float(self.prior_zcd.step(float(z)))
            ipdft_f = float(self.prior_ipdft.step(float(z)))
            prior_f = float(np.median([self.nominal_f, zcd_f, ipdft_f]))
            clip_hz = float(self.feature_config.get("clip_hz", LEGACY_FEATURE_CONFIG["clip_hz"]))
            feat_values.extend(
                [
                    float(_clip_frequency_delta(prior_f - self.nominal_f, clip_hz)),
                    float(_clip_frequency_delta(zcd_f - self.nominal_f, clip_hz)),
                    float(_clip_frequency_delta(ipdft_f - self.nominal_f, clip_hz)),
                    float(_clip_frequency_delta(zcd_f - ipdft_f, clip_hz)),
                ]
            )

        feat = np.asarray(feat_values, dtype=np.float64)
        return feat[:self.feature_dim], prior_f

    def step(self, z: float) -> float:
        if not hasattr(self, "bp"):
            self.reset()

        feat, prior_f = self._feature_step(float(z))
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = feat
        self.step_count += 1

        if self.step_count >= self._warmup_samples():
            delta_f = self._predict_delta(self.buffer)
            if self.target_mode == RESIDUAL_TARGET_MODE:
                f_pred = prior_f + delta_f
            else:
                f_pred = delta_f + self.nominal_f
        else:
            f_pred = self.nominal_f

        self.out_buffer[self.out_idx] = f_pred
        self.out_idx = (self.out_idx + 1) % len(self.out_buffer)
        self.f_out = float(np.mean(self.out_buffer))
        return self.f_out

    def step_vectorized(self, v_array: np.ndarray) -> np.ndarray:
        self._init_model_if_needed()
        v_array = np.asarray(v_array, dtype=np.float64)
        n = len(v_array)
        if n == 0:
            return np.empty(0, dtype=np.float64)

        features, aux = build_pi_gru_features(
            v_array,
            nominal_f=self.nominal_f,
            dt=self.dt,
            input_dim=self.feature_dim,
            feature_config=self.feature_config,
            return_aux=True,
        )
        raw = np.full(n, self.nominal_f, dtype=np.float64)

        if n >= self.window_len:
            windows = np.lib.stride_tricks.sliding_window_view(features, window_shape=self.window_len, axis=0)
            windows = np.transpose(windows, (0, 2, 1)).astype(np.float64, copy=True)
            delta = self._predict_delta_batch(windows)
            if self.target_mode == RESIDUAL_TARGET_MODE:
                prior_f = np.asarray(aux["prior_f"], dtype=np.float64)
                raw[self.window_len - 1:] = prior_f[self.window_len - 1:] + delta
            else:
                raw[self.window_len - 1:] = self.nominal_f + delta

        raw[:max(0, self._warmup_samples() - 1)] = self.nominal_f
        return _buffered_output_average(raw, self.nominal_f, self.output_ma_samples)

    def estimate(self, t: np.ndarray, v: np.ndarray) -> np.ndarray:
        del t
        return self.step_vectorized(v)
