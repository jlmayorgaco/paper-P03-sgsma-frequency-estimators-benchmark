from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from estimators.common import DT_DSP
from estimators.pi_gru import (
    DEFAULT_MODEL_CONFIG,
    DEFAULT_TARGET_MODE,
    HYBRID_FEATURE_CONFIG,
    LEGACY_FEATURE_CONFIG,
    PIDRE_Model,
    RESIDUAL_TARGET_MODE,
    _normalize_feature_config,
    build_pi_gru_features,
)
from pipelines.full_mc_benchmark import SCENARIOS


def _integrate_phase(f_true: np.ndarray, dt: float, phase_offset: np.ndarray | None = None) -> np.ndarray:
    phase = np.cumsum(2.0 * np.pi * np.asarray(f_true, dtype=np.float64) * float(dt))
    if phase_offset is not None:
        phase = phase + np.asarray(phase_offset, dtype=np.float64)
    return phase


def _deterministic_synthetic_cases(
    nominal_f: float = 60.0,
    dt: float = DT_DSP,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    t = np.arange(0.0, 0.8, dt)
    rng = np.random.default_rng(42)
    cases: list[tuple[str, np.ndarray, np.ndarray]] = []

    for f_target in np.arange(nominal_f - 5.0, nominal_f + 5.5, 0.5):
        noise = rng.normal(0.0, 0.003, size=len(t))
        f_true = np.full_like(t, f_target)
        phase = _integrate_phase(f_true, dt)
        v = np.sin(phase) + noise
        cases.append((f"tone_{f_target:.1f}", v, f_true))

    for rocof in [-15.0, -10.0, -5.0, -2.5, 2.5, 5.0, 10.0, 15.0]:
        f_true = nominal_f + rocof * t
        phase = _integrate_phase(f_true, dt)
        v = np.sin(phase) + rng.normal(0.0, 0.003, size=len(t))
        cases.append((f"ramp_{rocof:+.1f}", v, f_true))

    for jump_deg in [20.0, 40.0, 60.0, 80.0]:
        t_step = 0.2
        f_true = np.full_like(t, nominal_f)
        phase_offset = np.zeros_like(t)
        phase_offset[t >= t_step] = np.deg2rad(jump_deg)
        phase = _integrate_phase(f_true, dt, phase_offset=phase_offset)
        v = np.sin(phase) + rng.normal(0.0, 0.0025, size=len(t))
        cases.append((f"phase_jump_{jump_deg:.0f}", v, f_true))

    for harm_level in [0.02, 0.05, 0.10]:
        f_true = np.full_like(t, nominal_f)
        phase = _integrate_phase(f_true, dt)
        v = (
            np.sin(phase)
            + harm_level * np.sin(5.0 * phase)
            + 0.5 * harm_level * np.sin(7.0 * phase)
            + rng.normal(0.0, 0.003, size=len(t))
        )
        cases.append((f"harm_{harm_level:.2f}", v, f_true))

    for am_depth in [0.05, 0.10, 0.20]:
        envelope = 1.0 + am_depth * np.sin(2.0 * np.pi * 2.0 * t)
        f_true = np.full_like(t, nominal_f)
        phase = _integrate_phase(f_true, dt)
        v = envelope * np.sin(phase) + rng.normal(0.0, 0.003, size=len(t))
        cases.append((f"am_{am_depth:.2f}", v, f_true))

    return cases


def _random_synthetic_cases(
    n_cases: int,
    nominal_f: float = 60.0,
    dt: float = DT_DSP,
    seed: int = 42,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    if n_cases <= 0:
        return []

    rng = np.random.default_rng(seed)
    cases: list[tuple[str, np.ndarray, np.ndarray]] = []

    for case_idx in range(n_cases):
        duration = float(rng.uniform(0.8, 1.8))
        t = np.arange(0.0, duration, dt)
        n = len(t)

        f_true = np.full(n, nominal_f + rng.uniform(-1.0, 1.0), dtype=np.float64)
        amplitude = np.full(n, rng.uniform(0.85, 1.15), dtype=np.float64)
        phase_offset = np.zeros(n, dtype=np.float64)

        if rng.random() < 0.85:
            start = float(rng.uniform(0.12, 0.45 * duration))
            stop = float(rng.uniform(start + 0.08, min(duration, start + 0.60)))
            rocof = float(rng.uniform(-20.0, 20.0))
            ramp_mask = (t >= start) & (t <= stop)
            if np.any(ramp_mask):
                f_true[ramp_mask] += rocof * (t[ramp_mask] - start)
                f_true[t > stop] += rocof * (stop - start)

        if rng.random() < 0.55:
            mod_depth = float(rng.uniform(0.05, 0.7))
            mod_freq = float(rng.uniform(0.2, 4.0))
            f_true += mod_depth * np.sin(2.0 * np.pi * mod_freq * t)

        if rng.random() < 0.50:
            event_t = float(rng.uniform(0.15, 0.75 * duration))
            tau = np.maximum(t - event_t, 0.0)
            ring_amp = float(rng.uniform(0.15, 1.5))
            ring_freq = float(rng.uniform(1.0, 6.0))
            decay = float(rng.uniform(2.0, 10.0))
            f_true += ring_amp * np.exp(-decay * tau) * np.sin(2.0 * np.pi * ring_freq * tau)

        if rng.random() < 0.75:
            n_jumps = int(rng.integers(1, 3))
            for _ in range(n_jumps):
                jump_t = float(rng.uniform(0.1, 0.85 * duration))
                jump_deg = float(rng.uniform(-90.0, 90.0))
                phase_offset[t >= jump_t] += np.deg2rad(jump_deg)

        if rng.random() < 0.60:
            amp_t = float(rng.uniform(0.1, 0.7 * duration))
            amp_scale = float(rng.uniform(0.75, 1.30))
            amplitude[t >= amp_t] *= amp_scale

        if rng.random() < 0.55:
            am_depth = float(rng.uniform(0.02, 0.20))
            am_freq = float(rng.uniform(0.5, 5.0))
            amplitude *= 1.0 + am_depth * np.sin(2.0 * np.pi * am_freq * t)

        f_true = np.clip(f_true, nominal_f - 12.0, nominal_f + 12.0)
        phase = _integrate_phase(f_true, dt, phase_offset=phase_offset)

        v = amplitude * np.sin(phase)
        if rng.random() < 0.85:
            h5 = float(rng.uniform(0.0, 0.10))
            h7 = float(rng.uniform(0.0, 0.06))
            v += h5 * np.sin(5.0 * phase) + h7 * np.sin(7.0 * phase)
        if rng.random() < 0.35:
            ih_amp = float(rng.uniform(0.0, 0.08))
            ih_freq = float(rng.uniform(25.0, 40.0))
            ih_phase = float(rng.uniform(0.0, 2.0 * np.pi))
            v += ih_amp * np.sin(2.0 * np.pi * ih_freq * t + ih_phase)

        noise_sigma = float(rng.uniform(0.0, 0.015))
        impulsive_count = int(rng.integers(0, 4))
        if impulsive_count > 0:
            spikes = np.zeros_like(v)
            spike_idx = rng.integers(0, len(v), size=impulsive_count)
            spikes[spike_idx] = rng.normal(0.0, 0.15, size=impulsive_count)
            v = v + spikes
        v = v + rng.normal(0.0, noise_sigma, size=len(v))

        cases.append((f"random_synth_{case_idx:03d}", v.astype(np.float64), f_true.astype(np.float64)))

    return cases


def _synthetic_cases(
    random_synth_cases: int,
    nominal_f: float = 60.0,
    dt: float = DT_DSP,
    seed: int = 42,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    cases = _deterministic_synthetic_cases(nominal_f=nominal_f, dt=dt)
    cases.extend(_random_synthetic_cases(random_synth_cases, nominal_f=nominal_f, dt=dt, seed=seed + 101))
    return cases


def _sample_from_space(rng: np.random.Generator, spec: dict[str, Any]) -> Any:
    kind = spec["kind"]
    if kind == "uniform":
        return float(rng.uniform(spec["low"], spec["high"]))
    if kind == "choice":
        return rng.choice(spec["values"])
    if kind == "fixed":
        return spec["value"]
    raise ValueError(f"Unsupported Monte Carlo sampling kind: {kind}")


def _monte_carlo_scenario_cases(mc_aug_per_scenario: int, seed: int) -> list[tuple[str, np.ndarray, np.ndarray]]:
    if mc_aug_per_scenario <= 0:
        return []

    rng = np.random.default_rng(seed)
    cases: list[tuple[str, np.ndarray, np.ndarray]] = []
    for sc_cls in SCENARIOS:
        mc_space = sc_cls.get_monte_carlo_space()
        if not mc_space:
            continue

        defaults = sc_cls.get_default_params()
        for mc_idx in range(mc_aug_per_scenario):
            params = dict(defaults)
            for key, spec in mc_space.items():
                params[key] = _sample_from_space(rng, spec)
            params["seed"] = int(seed + 1000 * (mc_idx + 1) + len(cases))
            try:
                data = sc_cls.run(**params)
            except Exception as exc:
                print(f"[skip-mc] {sc_cls.get_name()} mc={mc_idx}: {exc}")
                continue
            cases.append((f"{sc_cls.get_name()}__mc_{mc_idx:02d}", np.asarray(data.v), np.asarray(data.f_true)))
    return cases


def _selection_indices(
    v: np.ndarray,
    f_true: np.ndarray,
    dt: float,
    warmup_samples: int,
    base_stride: int,
    event_stride: int,
    nominal_f: float,
) -> np.ndarray:
    idx = np.arange(warmup_samples, len(f_true), base_stride, dtype=int)
    rocof = np.abs(np.gradient(f_true, dt))
    dv = np.abs(np.diff(v, prepend=float(v[0])))
    dynamic = (
        (rocof > 0.2)
        | (np.abs(f_true - nominal_f) > 0.05)
        | (dv > max(0.15, float(np.percentile(dv, 99.0))))
    )
    event_idx = np.nonzero(dynamic)[0]
    event_idx = event_idx[event_idx >= warmup_samples]
    if len(event_idx):
        event_idx = event_idx[::max(1, event_stride)]
        idx = np.unique(np.concatenate([idx, event_idx]))
    return idx


def _sample_weight_vector(
    v: np.ndarray,
    f_true: np.ndarray,
    dt: float,
    nominal_f: float,
    aux: dict[str, np.ndarray],
) -> np.ndarray:
    rocof = np.abs(np.gradient(f_true, dt))
    dv = np.abs(np.diff(v, prepend=float(v[0])))
    freq_dev = np.abs(f_true - nominal_f)
    prior_error = np.abs(np.asarray(aux["prior_f"], dtype=np.float64) - np.asarray(f_true, dtype=np.float64))
    prior_disagreement = np.abs(
        np.asarray(aux["zcd_f"], dtype=np.float64) - np.asarray(aux["ipdft_f"], dtype=np.float64)
    )
    weights = (
        1.0
        + 0.8 * np.clip(freq_dev / 0.5, 0.0, 4.0)
        + 0.5 * np.clip(rocof / 5.0, 0.0, 4.0)
        + 0.7 * (dv > max(0.12, float(np.percentile(dv, 99.0)))).astype(np.float64)
        + 1.2 * np.clip(prior_error / 0.5, 0.0, 5.0)
        + 0.6 * np.clip(prior_disagreement / 0.5, 0.0, 5.0)
    )
    return np.asarray(weights, dtype=np.float32)


def _target_vector(
    f_true: np.ndarray,
    target_mode: str,
    nominal_f: float,
    aux: dict[str, np.ndarray],
) -> np.ndarray:
    if target_mode == RESIDUAL_TARGET_MODE:
        return np.asarray(f_true, dtype=np.float64) - np.asarray(aux["prior_f"], dtype=np.float64)
    if target_mode == DEFAULT_TARGET_MODE:
        return np.asarray(f_true, dtype=np.float64) - float(nominal_f)
    raise ValueError(f"Unsupported PI-GRU target mode: {target_mode}")


def _windows_from_case(
    name: str,
    v: np.ndarray,
    f_true: np.ndarray,
    window_len: int,
    warmup_samples: int,
    base_stride: int,
    event_stride: int,
    max_windows_per_source: int,
    feature_config: dict[str, Any],
    target_mode: str,
    nominal_f: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    features, aux = build_pi_gru_features(
        v,
        nominal_f=nominal_f,
        dt=DT_DSP,
        input_dim=int(DEFAULT_MODEL_CONFIG["input_dim"]) if feature_config["mode"] == LEGACY_FEATURE_CONFIG["mode"] else 7,
        feature_config=feature_config,
        return_aux=True,
    )
    target = _target_vector(f_true=f_true, target_mode=target_mode, nominal_f=nominal_f, aux=aux)
    sample_weights = _sample_weight_vector(v=v, f_true=f_true, dt=DT_DSP, nominal_f=nominal_f, aux=aux)
    indices = _selection_indices(
        v=v,
        f_true=f_true,
        dt=DT_DSP,
        warmup_samples=warmup_samples,
        base_stride=base_stride,
        event_stride=event_stride,
        nominal_f=nominal_f,
    )

    x_list: list[np.ndarray] = []
    y_list: list[float] = []
    w_list: list[float] = []
    src_list: list[str] = []
    for idx in indices:
        start = idx - window_len + 1
        if start < 0:
            continue
        x_list.append(features[start:idx + 1])
        y_list.append(float(target[idx]))
        w_list.append(float(sample_weights[idx]))
        src_list.append(name)

    if not x_list:
        return (
            np.empty((0, window_len, features.shape[1]), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            [],
        )

    if max_windows_per_source > 0 and len(x_list) > max_windows_per_source:
        keep = np.linspace(0, len(x_list) - 1, num=max_windows_per_source, dtype=int)
        x_list = [x_list[i] for i in keep]
        y_list = [y_list[i] for i in keep]
        w_list = [w_list[i] for i in keep]
        src_list = [src_list[i] for i in keep]

    return (
        np.asarray(x_list, dtype=np.float32),
        np.asarray(y_list, dtype=np.float32),
        np.asarray(w_list, dtype=np.float32),
        src_list,
    )


def _collect_dataset(
    window_len: int,
    base_stride: int,
    event_stride: int,
    max_windows_per_source: int,
    mc_aug_per_scenario: int,
    random_synth_cases: int,
    feature_config: dict[str, Any],
    target_mode: str,
    nominal_f: float,
    input_dim: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    warmup_samples = max(window_len, int(round(3.0 / nominal_f / DT_DSP)))
    x_blocks: list[np.ndarray] = []
    y_blocks: list[np.ndarray] = []
    w_blocks: list[np.ndarray] = []
    source_names: list[str] = []

    for name, v, f_true in _synthetic_cases(
        random_synth_cases=random_synth_cases,
        nominal_f=nominal_f,
        dt=DT_DSP,
        seed=seed,
    ):
        x, y, w, src = _windows_from_case(
            name,
            v,
            f_true,
            window_len,
            warmup_samples,
            base_stride,
            event_stride,
            max_windows_per_source,
            feature_config=feature_config,
            target_mode=target_mode,
            nominal_f=nominal_f,
        )
        if len(x):
            x_blocks.append(x[:, :, :input_dim])
            y_blocks.append(y)
            w_blocks.append(w)
            source_names.extend(src)

    for name, v, f_true in _monte_carlo_scenario_cases(mc_aug_per_scenario=mc_aug_per_scenario, seed=seed):
        x, y, w, src = _windows_from_case(
            name,
            v,
            f_true,
            window_len,
            warmup_samples,
            base_stride,
            event_stride,
            max_windows_per_source,
            feature_config=feature_config,
            target_mode=target_mode,
            nominal_f=nominal_f,
        )
        if len(x):
            x_blocks.append(x[:, :, :input_dim])
            y_blocks.append(y)
            w_blocks.append(w)
            source_names.extend(src)
            print(f"[data] {name:40s} windows={len(x)}")

    for sc_cls in SCENARIOS:
        try:
            data = sc_cls.run()
        except Exception as exc:
            print(f"[skip] {sc_cls.__name__}: {exc}")
            continue
        x, y, w, src = _windows_from_case(
            sc_cls.get_name(),
            np.asarray(data.v, dtype=np.float64),
            np.asarray(data.f_true, dtype=np.float64),
            window_len,
            warmup_samples,
            base_stride,
            event_stride,
            max_windows_per_source,
            feature_config=feature_config,
            target_mode=target_mode,
            nominal_f=nominal_f,
        )
        if len(x):
            x_blocks.append(x[:, :, :input_dim])
            y_blocks.append(y)
            w_blocks.append(w)
            source_names.extend(src)
            print(f"[data] {sc_cls.get_name():40s} windows={len(x)}")

    if not x_blocks:
        raise RuntimeError("PI-GRU training dataset is empty.")

    X = np.concatenate(x_blocks, axis=0)
    y = np.concatenate(y_blocks, axis=0)
    w = np.concatenate(w_blocks, axis=0)
    return X, y, w, source_names


def _split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    split = int(round((1.0 - val_ratio) * len(indices)))
    train_idx = indices[:split]
    val_idx = indices[split:]
    return X[train_idx], y[train_idx], w[train_idx], X[val_idx], y[val_idx], w[val_idx]


def _load_initial_state(
    init_from: str | None,
    device: torch.device,
    model_config: dict[str, Any],
    feature_config: dict[str, Any],
    target_mode: str,
) -> dict[str, torch.Tensor] | None:
    if not init_from:
        return None

    init_path = Path(init_from)
    if not init_path.is_absolute():
        init_path = SRC / "estimators" / init_from
    if not init_path.exists():
        raise FileNotFoundError(f"Initial checkpoint not found: {init_path}")

    checkpoint = torch.load(init_path, map_location=device)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise RuntimeError(f"Initial checkpoint {init_path} does not contain state_dict.")

    ckpt_model_config = dict(DEFAULT_MODEL_CONFIG)
    ckpt_model_config.update(checkpoint.get("model_config", {}))
    ckpt_feature_config = _normalize_feature_config(checkpoint.get("feature_config"))
    ckpt_target_mode = str(checkpoint.get("target_mode", DEFAULT_TARGET_MODE))

    if ckpt_model_config != model_config:
        raise RuntimeError(
            f"Cannot init PI-GRU from {init_path.name}: model config mismatch {ckpt_model_config} != {model_config}"
        )
    if ckpt_feature_config != feature_config:
        raise RuntimeError(
            f"Cannot init PI-GRU from {init_path.name}: feature config mismatch {ckpt_feature_config} != {feature_config}"
        )
    if ckpt_target_mode != target_mode:
        raise RuntimeError(
            f"Cannot init PI-GRU from {init_path.name}: target mode mismatch {ckpt_target_mode} != {target_mode}"
        )

    print(f"[PI-GRU] warm start from {init_path}")
    return checkpoint["state_dict"]


def train(args: argparse.Namespace) -> dict[str, float | int | str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    feature_config = _normalize_feature_config(
        HYBRID_FEATURE_CONFIG if args.feature_mode == HYBRID_FEATURE_CONFIG["mode"] else LEGACY_FEATURE_CONFIG
    )
    feature_config["clip_hz"] = float(args.clip_hz)
    input_dim = 7 if feature_config["mode"] == HYBRID_FEATURE_CONFIG["mode"] else int(args.input_dim)
    target_mode = str(args.target_mode)
    if feature_config["mode"] == HYBRID_FEATURE_CONFIG["mode"] and target_mode == DEFAULT_TARGET_MODE:
        print("[PI-GRU] warning: hybrid prior features work best with residual target mode.")

    model_config = dict(DEFAULT_MODEL_CONFIG)
    model_config.update(
        {
            "input_dim": int(input_dim),
            "conv_channels": int(args.conv_channels),
            "hidden_dim": int(args.hidden_dim),
            "fc_hidden_dim": int(args.fc_hidden_dim),
            "num_layers": int(args.num_layers),
            "dropout": float(args.dropout),
        }
    )

    print(f"[PI-GRU] device={device} model_config={model_config}")
    print(f"[PI-GRU] feature_config={feature_config} target_mode={target_mode}")
    print("[PI-GRU] building dataset from active scenarios, Monte Carlo augmentation, and synthetic stress cases ...")
    X, y, w, source_names = _collect_dataset(
        window_len=args.window_len,
        base_stride=args.base_stride,
        event_stride=args.event_stride,
        max_windows_per_source=args.max_windows_per_source,
        mc_aug_per_scenario=args.mc_aug_per_scenario,
        random_synth_cases=args.random_synth_cases,
        feature_config=feature_config,
        target_mode=target_mode,
        nominal_f=args.nominal_f,
        input_dim=input_dim,
        seed=args.seed,
    )
    print(f"[PI-GRU] total windows={len(X)} feature_shape={X.shape[1:]} target_std={float(np.std(y)):.4f}")

    X_train, y_train, w_train, X_val, y_val, w_val = _split_dataset(
        X, y, w, val_ratio=args.val_ratio, seed=args.seed
    )
    train_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
        torch.from_numpy(w_train),
    )
    val_x = torch.from_numpy(X_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)
    val_w = torch.from_numpy(w_val).to(device)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    model = PIDRE_Model(**model_config).to(device)
    initial_state = _load_initial_state(
        init_from=args.init_from,
        device=device,
        model_config=model_config,
        feature_config=feature_config,
        target_mode=target_mode,
    )
    if initial_state is not None:
        model.load_state_dict(initial_state, strict=True)

    criterion = nn.SmoothL1Loss(beta=0.1, reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(2, args.patience // 2),
    )

    best_val = float("inf")
    best_epoch = -1
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_weight_sum = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for xb, yb, wb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            per_sample_loss = criterion(pred, yb)
            weighted_loss = torch.sum(per_sample_loss * wb) / torch.clamp(torch.sum(wb), min=1e-9)
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss_sum += float(torch.sum(per_sample_loss * wb).item())
            epoch_weight_sum += float(torch.sum(wb).item())
            pbar.set_postfix(loss=float(weighted_loss.item()))

        train_loss = epoch_loss_sum / max(epoch_weight_sum, 1e-9)

        model.eval()
        with torch.no_grad():
            pred_val = model(val_x)
            val_per_sample = criterion(pred_val, val_y)
            val_loss = float(torch.sum(val_per_sample * val_w).item() / max(float(torch.sum(val_w).item()), 1e-9))
            val_mae = float(torch.mean(torch.abs(pred_val - val_y)).item())

        scheduler.step(val_loss)
        lr_now = float(optimizer.param_groups[0]["lr"])
        print(
            f"[epoch {epoch:02d}] train_loss={train_loss:.5f} "
            f"val_loss={val_loss:.5f} val_mae_hz={val_mae:.5f} lr={lr_now:.2e}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"[early-stop] no improvement for {args.patience} epochs")
                break

    if best_state is None:
        raise RuntimeError("PI-GRU training did not produce a valid checkpoint.")

    out_dir = SRC / "estimators"
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / args.output_name
    checkpoint = {
        "model_config": model_config,
        "feature_config": feature_config,
        "target_mode": target_mode,
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "state_dict": best_state,
    }
    torch.save(checkpoint, weights_path)

    metadata = {
        "weights_path": str(weights_path),
        "model_config": model_config,
        "feature_config": feature_config,
        "target_mode": target_mode,
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "scenario_count": int(len(SCENARIOS)),
        "synthetic_source_count": int(
            len({s for s in source_names if s.startswith(("tone_", "ramp_", "phase_", "harm_", "am_", "random_synth_"))})
        ),
        "random_synth_cases": int(args.random_synth_cases),
        "mc_aug_per_scenario": int(args.mc_aug_per_scenario),
        "max_windows_per_source": int(args.max_windows_per_source),
    }
    meta_path = weights_path.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[saved] {weights_path}")
    print(f"[saved] {meta_path}")
    return {
        "best_val_loss": float(best_val),
        "best_epoch": int(best_epoch),
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain PI-GRU using active benchmark scenarios and synthetic stress cases.")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--nominal-f", type=float, default=60.0)
    parser.add_argument("--input-dim", type=int, default=3)
    parser.add_argument("--conv-channels", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--fc-hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--window-len", type=int, default=100)
    parser.add_argument("--base-stride", type=int, default=8)
    parser.add_argument("--event-stride", type=int, default=1)
    parser.add_argument("--max-windows-per-source", type=int, default=768)
    parser.add_argument("--mc-aug-per-scenario", type=int, default=2)
    parser.add_argument("--random-synth-cases", type=int, default=120)
    parser.add_argument("--feature-mode", choices=[LEGACY_FEATURE_CONFIG["mode"], HYBRID_FEATURE_CONFIG["mode"]], default=HYBRID_FEATURE_CONFIG["mode"])
    parser.add_argument("--target-mode", choices=[DEFAULT_TARGET_MODE, RESIDUAL_TARGET_MODE], default=RESIDUAL_TARGET_MODE)
    parser.add_argument("--clip-hz", type=float, default=12.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init-from", type=str, default=None)
    parser.add_argument("--output-name", type=str, default="pi_gru_weights_hybrid.pt")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
