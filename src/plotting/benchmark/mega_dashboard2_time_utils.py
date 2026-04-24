from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
LEGACY_OUTPUT_DIR = ROOT / "tests" / "montecarlo" / "outputs"


ESTIMATOR_COLORS = {
    "EKF": "#1565C0",
    "UKF": "#8E24AA",
    "RA-EKF": "#D81B60",
    "SOGI-PLL": "#2E7D32",
    "SOGI-FLL": "#66A61E",
    "PLL": "#8D6E63",
    "IPDFT": "#E65100",
    "TFT": "#F57C00",
    "Koopman (RK-DPMU)": "#B71C1C",
    "ZCD": "#00897B",
    "Prony": "#6D4C41",
    "ESPRIT": "#5E35B1",
}

SHORT_LABELS = {
    "Koopman (RK-DPMU)": "Koopman",
}


def _candidate_roots(base_results_dir: Path) -> list[Path]:
    roots = [Path(base_results_dir)]
    if LEGACY_OUTPUT_DIR not in roots:
        roots.append(LEGACY_OUTPUT_DIR)
    return roots


def resolve_signal_csv(base_results_dir: Path, scenario: str, estimator: str) -> Path:
    for root in _candidate_roots(base_results_dir):
        candidate = root / scenario / estimator / f"{scenario}__{estimator}_signals.csv"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Missing signals CSV for scenario={scenario}, estimator={estimator}. "
        f"Searched roots: {', '.join(str(r) for r in _candidate_roots(base_results_dir))}"
    )


def load_signal_frame(base_results_dir: Path, scenario: str, estimator: str) -> pd.DataFrame:
    return pd.read_csv(resolve_signal_csv(base_results_dir, scenario, estimator))


def _prepare_interp_series(t_rel: np.ndarray, values: np.ndarray, t_window: tuple[float, float]) -> tuple[np.ndarray, np.ndarray] | None:
    mask = np.isfinite(t_rel) & np.isfinite(values)
    if not np.any(mask):
        return None

    t_use = t_rel[mask]
    y_use = values[mask]

    mask_window = (t_use >= t_window[0]) & (t_use <= t_window[1])
    if mask_window.sum() < 2:
        return None

    t_use = t_use[mask_window]
    y_use = y_use[mask_window]

    order = np.argsort(t_use)
    t_use = t_use[order]
    y_use = y_use[order]

    unique_mask = np.concatenate(([True], np.diff(t_use) > 0.0))
    t_use = t_use[unique_mask]
    y_use = y_use[unique_mask]

    if t_use.size < 2:
        return None
    return t_use, y_use


def _collapse_samples(samples: list[np.ndarray]) -> np.ndarray | None:
    if not samples:
        return None
    arr = np.vstack(samples)
    out = np.full(arr.shape[1], np.nan, dtype=float)
    for idx in range(arr.shape[1]):
        col = arr[:, idx]
        col = col[np.isfinite(col)]
        if col.size:
            out[idx] = float(np.median(col))
    return out


def aggregate_aligned_curves(
    base_results_dir: Path,
    scenario: str,
    estimators: list[str],
    *,
    align_col: str | None,
    align_value: float,
    t_window: tuple[float, float],
    n_points: int = 900,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    t_common = np.linspace(t_window[0], t_window[1], n_points)
    true_curve = None
    curves: dict[str, np.ndarray] = {}

    for est_idx, estimator in enumerate(estimators):
        df = load_signal_frame(base_results_dir, scenario, estimator)
        est_samples: list[np.ndarray] = []
        true_samples: list[np.ndarray] = []

        for _, g in df.groupby("run_idx", sort=True):
            if align_col is not None and align_col in g.columns:
                event_t = float(g[align_col].iloc[0])
            else:
                event_t = align_value

            t_rel = g["t_s"].to_numpy(dtype=float) - event_t
            f_hat = g["f_hat_hz"].to_numpy(dtype=float)
            prepared_hat = _prepare_interp_series(t_rel, f_hat, t_window)
            if prepared_hat is None:
                continue

            t_use, y_use = prepared_hat
            est_samples.append(np.interp(t_common, t_use, y_use, left=np.nan, right=np.nan))

            if est_idx == 0:
                f_true = g["f_true_hz"].to_numpy(dtype=float)
                prepared_true = _prepare_interp_series(t_rel, f_true, t_window)
                if prepared_true is not None:
                    t_true, y_true = prepared_true
                    true_samples.append(np.interp(t_common, t_true, y_true, left=np.nan, right=np.nan))

        curve = _collapse_samples(est_samples)
        if curve is not None:
            curves[estimator] = curve
        if est_idx == 0:
            true_curve = _collapse_samples(true_samples)

    if true_curve is None or not curves:
        raise ValueError(f"Unable to assemble aligned curves for {scenario}.")

    return t_common, true_curve, curves


def compute_data_limits(*arrays: np.ndarray, min_span: float = 0.8, pad_frac: float = 0.08) -> tuple[float, float]:
    finite_chunks = []
    for arr in arrays:
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            finite_chunks.append(arr)

    if not finite_chunks:
        return 59.0, 61.0

    merged = np.concatenate(finite_chunks)
    lo = float(np.min(merged))
    hi = float(np.max(merged))
    span = max(hi - lo, min_span)
    pad = pad_frac * span
    center = 0.5 * (lo + hi)
    return center - 0.5 * span - pad, center + 0.5 * span + pad


def plot_clean_tracking_panel(
    ax,
    *,
    base_results_dir: Path,
    scenario: str,
    estimators: list[str],
    panel_title: str,
    align_col: str | None,
    align_value: float,
    t_window: tuple[float, float],
    event_label: str,
    legend_loc: str = "upper right",
    legend_ncol: int = 2,
    show_ylabel: bool = True,
    min_y_span: float = 0.8,
    xlabel: str = "Relative Time [s]",
    show_event_line: bool = True,
    y_limits: tuple[float, float] | None = None,
) -> None:
    t_common, f_true, curves = aggregate_aligned_curves(
        base_results_dir,
        scenario,
        estimators,
        align_col=align_col,
        align_value=align_value,
        t_window=t_window,
    )

    ax.plot(t_common, f_true, color="black", lw=1.8, ls="--", label="True", zorder=6)

    plotted_arrays = [f_true]
    for estimator in estimators:
        curve = curves.get(estimator)
        if curve is None:
            continue
        plotted_arrays.append(curve)
        ax.plot(
            t_common,
            curve,
            color=ESTIMATOR_COLORS.get(estimator, "#546E7A"),
            lw=1.35,
            alpha=0.97,
            label=SHORT_LABELS.get(estimator, estimator),
            zorder=4,
        )

    if y_limits is None:
        y_lo, y_hi = compute_data_limits(*plotted_arrays, min_span=min_y_span)
    else:
        y_lo, y_hi = y_limits

    if show_event_line:
        ax.axvline(0.0, color="#777777", ls=":", lw=0.9, alpha=0.85, zorder=3)
    if event_label:
        ax.text(
            0.02,
            0.93,
            event_label,
            transform=ax.transAxes,
            fontsize=9.2,
            color="#666666",
            va="top",
        )

    ax.set_xlim(t_window[0], t_window[1])
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(xlabel, fontsize=10.5, labelpad=2)
    if show_ylabel:
        ax.set_ylabel("Frequency [Hz]", fontsize=10.5, labelpad=2)
    ax.set_title(panel_title, fontweight="bold", fontsize=11.5, loc="left", pad=5)
    ax.tick_params(labelsize=9.5)
    ax.grid(True, alpha=0.18, lw=0.4)

    legend = ax.legend(
        loc=legend_loc,
        fontsize=8.8,
        ncol=legend_ncol,
        frameon=True,
        framealpha=0.92,
        edgecolor="#cccccc",
        handlelength=1.3,
        handletextpad=0.35,
        columnspacing=0.8,
        borderpad=0.35,
        labelspacing=0.22,
    )
    legend.get_frame().set_linewidth(0.7)
