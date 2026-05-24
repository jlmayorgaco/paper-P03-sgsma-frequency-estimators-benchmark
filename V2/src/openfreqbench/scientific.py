from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .registry import CANONICAL_METRIC_IDS

PRIMARY_RANKING_METRICS = (
    "m1_rmse_hz",
    "m2_mae_hz",
    "m3_max_peak_hz",
    "m5_trip_risk_s",
    "m8_settling_time_s",
    "m13_cpu_time_us",
    "m14_struct_latency_ms",
)

CLASSICAL_ESTIMATORS = {"ZCD", "IPDFT", "TFT", "PLL", "Prony", "ESPRIT"}


def scenario_family(scenario: str) -> str:
    name = str(scenario)
    if name == "IEEE_Single_SinWave":
        return "nominal"
    if name.startswith("IBR_Harmonics"):
        return "harmonics"
    if "Ringdown" in name:
        return "ibr_ringdown"
    if name.startswith("IBR"):
        return "ibr_event"
    if "Freq_Ramp" in name:
        return "rocof_ramp"
    if "Freq_Step" in name:
        return "frequency_step"
    if "Mag_Step" in name:
        return "magnitude_step"
    if "Phase_Jump" in name:
        return "phase_jump"
    if "Modulation" in name:
        return "modulation"
    if "OOB" in name:
        return "oob_interference"
    return "other"


def _numeric(raw: pd.DataFrame, metric: str) -> pd.Series:
    if metric not in raw.columns:
        return pd.Series([], dtype=float)
    return pd.to_numeric(raw[metric], errors="coerce")


def _bootstrap_mean_ci(values: np.ndarray, *, seed: int = 12345, iters: int = 1000) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan
    if len(values) == 1:
        val = float(values[0])
        return val, val
    rng = np.random.default_rng(seed)
    means = np.empty(iters, dtype=float)
    for idx in range(iters):
        means[idx] = float(np.mean(rng.choice(values, size=len(values), replace=True)))
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def metric_confidence_intervals(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    work = raw.copy()
    work["scenario_family"] = work["scenario"].map(scenario_family)
    metrics = [metric for metric in CANONICAL_METRIC_IDS if metric in work.columns]
    rows: list[dict[str, Any]] = []
    group_cols = ["scenario_family", "scenario", "estimator", "family"]
    for keys, block in work.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, keys))
        for metric in metrics:
            values = pd.to_numeric(block[metric], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            ci_lo, ci_hi = _bootstrap_mean_ci(values)
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "n": int(len(values)),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "p05": float(np.quantile(values, 0.05)),
                    "p95": float(np.quantile(values, 0.95)),
                    "ci95_lo": ci_lo,
                    "ci95_hi": ci_hi,
                }
            )
    return pd.DataFrame(rows)


def failure_analysis(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    work = raw.copy()
    work["scenario_family"] = work["scenario"].map(scenario_family)
    primary = [m for m in ("m1_rmse_hz", "m2_mae_hz", "m3_max_peak_hz", "m13_cpu_time_us", "m14_struct_latency_ms") if m in work.columns]
    for metric in primary:
        work[metric] = pd.to_numeric(work[metric], errors="coerce")
    invalid_rate = _numeric(work, "m22_invalid_output_rate")
    work["_invalid_output_flag"] = invalid_rate.fillna(0.0) > 0.0 if len(invalid_rate) else False
    if primary:
        finite_matrix = np.column_stack([np.isfinite(work[m].to_numpy(dtype=float)) for m in primary])
        work["_nonfinite_metric_flag"] = ~np.all(finite_matrix, axis=1)
    else:
        work["_nonfinite_metric_flag"] = False
    work["_extreme_error_flag"] = False
    if "m1_rmse_hz" in work:
        work["_extreme_error_flag"] = work["_extreme_error_flag"] | (work["m1_rmse_hz"] > 0.5)
    if "m3_max_peak_hz" in work:
        work["_extreme_error_flag"] = work["_extreme_error_flag"] | (work["m3_max_peak_hz"] > 1.0)
    latency = (
        pd.to_numeric(work["m14_struct_latency_ms"], errors="coerce").fillna(0.0)
        if "m14_struct_latency_ms" in work
        else pd.Series(np.zeros(len(work)), index=work.index)
    )
    cpu = (
        pd.to_numeric(work["m13_cpu_time_us"], errors="coerce").fillna(0.0)
        if "m13_cpu_time_us" in work
        else pd.Series(np.zeros(len(work)), index=work.index)
    )
    work["_latency_fail_flag"] = latency > 100.0
    work["_cpu_fail_flag"] = cpu > 1000.0
    if "m15_pcb_compliant" in work.columns:
        compliant = pd.to_numeric(work["m15_pcb_compliant"], errors="coerce").fillna(0.0)
        work["_pmu_noncompliant_flag"] = compliant < 1.0
    else:
        work["_pmu_noncompliant_flag"] = False
    flag_cols = [
        "_invalid_output_flag",
        "_nonfinite_metric_flag",
        "_extreme_error_flag",
        "_latency_fail_flag",
        "_cpu_fail_flag",
        "_pmu_noncompliant_flag",
    ]
    work["_collapse_flag"] = work[flag_cols].astype(bool).any(axis=1)
    grouped = work.groupby(["scenario_family", "scenario", "estimator", "family"], dropna=False)
    rows: list[dict[str, Any]] = []
    for keys, block in grouped:
        rows.append(
            {
                "scenario_family": keys[0],
                "scenario": keys[1],
                "estimator": keys[2],
                "family": keys[3],
                "n": int(len(block)),
                "invalid_output_rate": float(block["_invalid_output_flag"].mean()),
                "nonfinite_metric_rate": float(block["_nonfinite_metric_flag"].mean()),
                "extreme_error_rate": float(block["_extreme_error_flag"].mean()),
                "latency_fail_rate": float(block["_latency_fail_flag"].mean()),
                "cpu_fail_rate": float(block["_cpu_fail_flag"].mean()),
                "pmu_noncompliance_rate": float(block["_pmu_noncompliant_flag"].mean()),
                "collapse_rate": float(block["_collapse_flag"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _mean_by_estimator(raw: pd.DataFrame, metrics: tuple[str, ...]) -> pd.DataFrame:
    present = [metric for metric in metrics if metric in raw.columns]
    if not present:
        return pd.DataFrame()
    work = raw[["estimator", "family", *present]].copy()
    for metric in present:
        work[metric] = pd.to_numeric(work[metric], errors="coerce")
    return work.groupby(["estimator", "family"], as_index=False)[present].mean()


def _normalized_lower_better(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    lo = values.min()
    hi = values.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series(np.zeros(len(values)), index=series.index)
    return (values - lo) / (hi - lo)


def pareto_recommendations(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = ("m1_rmse_hz", "m13_cpu_time_us", "m14_struct_latency_ms")
    means = _mean_by_estimator(raw, metrics)
    if means.empty or not set(metrics).issubset(means.columns):
        return pd.DataFrame()
    work = means.copy()
    work["_rmse_n"] = _normalized_lower_better(work["m1_rmse_hz"])
    work["_cpu_n"] = _normalized_lower_better(work["m13_cpu_time_us"])
    work["_latency_n"] = _normalized_lower_better(work["m14_struct_latency_ms"])
    profiles = {
        "protection": {"_rmse_n": 0.4, "_latency_n": 0.4, "_cpu_n": 0.2},
        "monitoring": {"_rmse_n": 0.6, "_latency_n": 0.2, "_cpu_n": 0.2},
        "low_cost": {"_rmse_n": 0.3, "_latency_n": 0.1, "_cpu_n": 0.6},
        "offline_analysis": {"_rmse_n": 0.8, "_latency_n": 0.1, "_cpu_n": 0.1},
    }
    rows: list[dict[str, Any]] = []
    for profile, weights in profiles.items():
        score = sum(work[col] * weight for col, weight in weights.items())
        ranked = work.assign(score=score).sort_values("score").head(5)
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            rows.append(
                {
                    "profile": profile,
                    "rank": rank,
                    "estimator": row["estimator"],
                    "family": row["family"],
                    "score": float(row["score"]),
                    "rmse_hz": float(row["m1_rmse_hz"]),
                    "cpu_time_us": float(row["m13_cpu_time_us"]),
                    "struct_latency_ms": float(row["m14_struct_latency_ms"]),
                }
            )
    return pd.DataFrame(rows)


def ranking_sensitivity(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for metric in PRIMARY_RANKING_METRICS:
        if metric not in raw.columns:
            continue
        means = _mean_by_estimator(raw, (metric,))
        if means.empty:
            continue
        ranked = means.sort_values(metric, ascending=True).reset_index(drop=True)
        for rank, row in ranked.iterrows():
            rows.append(
                {
                    "metric": metric,
                    "rank": int(rank + 1),
                    "estimator": row["estimator"],
                    "family": row["family"],
                    "value": float(row[metric]),
                }
            )
    return pd.DataFrame(rows)


def ibr_robustness(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty or "m1_rmse_hz" not in raw.columns:
        return pd.DataFrame()
    work = raw.copy()
    work["scenario_family"] = work["scenario"].map(scenario_family)
    work["m1_rmse_hz"] = pd.to_numeric(work["m1_rmse_hz"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for estimator, block in work.groupby("estimator"):
        baseline = block[block["scenario_family"].isin(["nominal", "frequency_step"])]["m1_rmse_hz"].dropna()
        ibr = block[block["scenario_family"].isin(["ibr_event", "ibr_ringdown", "harmonics"])]["m1_rmse_hz"].dropna()
        if baseline.empty or ibr.empty:
            continue
        base_mean = float(baseline.mean())
        ibr_mean = float(ibr.mean())
        rows.append(
            {
                "estimator": estimator,
                "family": str(block["family"].iloc[0]) if "family" in block else "",
                "baseline_rmse_hz": base_mean,
                "ibr_rmse_hz": ibr_mean,
                "absolute_delta_hz": ibr_mean - base_mean,
                "relative_delta": (ibr_mean - base_mean) / base_mean if base_mean else math.nan,
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("ibr_rmse_hz") if not out.empty else out


def pi_gru_generalization(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty or "m1_rmse_hz" not in raw.columns or "PI-GRU" not in set(raw["estimator"].astype(str)):
        return pd.DataFrame()
    work = raw.copy()
    work["scenario_family"] = work["scenario"].map(scenario_family)
    work["m1_rmse_hz"] = pd.to_numeric(work["m1_rmse_hz"], errors="coerce")
    if "m13_cpu_time_us" in work:
        work["m13_cpu_time_us"] = pd.to_numeric(work["m13_cpu_time_us"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for scenario, block in work.groupby("scenario"):
        means = block.groupby(["estimator", "family"], as_index=False).mean(numeric_only=True)
        if "PI-GRU" not in set(means["estimator"].astype(str)):
            continue
        ranked = means.sort_values("m1_rmse_hz").reset_index(drop=True)
        pi = ranked[ranked["estimator"].astype(str) == "PI-GRU"].iloc[0]
        best = ranked.iloc[0]
        rows.append(
            {
                "scenario": scenario,
                "scenario_family": scenario_family(str(scenario)),
                "pi_gru_rank_rmse": int(ranked.index[ranked["estimator"].astype(str) == "PI-GRU"][0] + 1),
                "pi_gru_rmse_hz": float(pi["m1_rmse_hz"]),
                "best_estimator": best["estimator"],
                "best_rmse_hz": float(best["m1_rmse_hz"]),
                "pi_gru_delta_to_best_pct": float(
                    100.0 * (pi["m1_rmse_hz"] - best["m1_rmse_hz"]) / best["m1_rmse_hz"]
                )
                if best["m1_rmse_hz"]
                else math.nan,
                "pi_gru_cpu_time_us": float(pi["m13_cpu_time_us"]) if "m13_cpu_time_us" in pi else math.nan,
                "generalization_label": "competitive" if int(ranked.index[ranked["estimator"].astype(str) == "PI-GRU"][0] + 1) <= 3 else "not_top3",
            }
        )
    return pd.DataFrame(rows)


def classical_competitiveness(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty or "m1_rmse_hz" not in raw.columns:
        return pd.DataFrame()
    work = raw.copy()
    work["m1_rmse_hz"] = pd.to_numeric(work["m1_rmse_hz"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for scenario, block in work.groupby("scenario"):
        means = block.groupby(["estimator", "family"], as_index=False)["m1_rmse_hz"].mean()
        means = means.sort_values("m1_rmse_hz").reset_index(drop=True)
        if means.empty:
            continue
        best = float(means.iloc[0]["m1_rmse_hz"])
        for rank, row in means.iterrows():
            if row["estimator"] not in CLASSICAL_ESTIMATORS:
                continue
            within_10pct = bool(best == 0.0 or row["m1_rmse_hz"] <= best * 1.10)
            rows.append(
                {
                    "scenario": scenario,
                    "scenario_family": scenario_family(str(scenario)),
                    "estimator": row["estimator"],
                    "family": row["family"],
                    "rmse_rank": int(rank + 1),
                    "rmse_hz": float(row["m1_rmse_hz"]),
                    "best_rmse_hz": best,
                    "within_10pct_of_best": within_10pct,
                    "competitive": bool((rank + 1) <= 3 or within_10pct),
                }
            )
    return pd.DataFrame(rows)


def tuning_policy_summary(report: dict[str, Any]) -> pd.DataFrame:
    cfg = report.get("run_configuration", {})
    return pd.DataFrame(
        [
            {
                "run_id": cfg.get("run_id"),
                "parameter_policy": cfg.get("parameter_policy"),
                "tuned_artifacts_dir": cfg.get("tuned_artifacts_dir"),
                "comparison_ready": False,
                "note": "Use paired artifact_tuned and default/fixed-policy reports for tuning generalization deltas.",
            }
        ]
    )


def write_scientific_tables(report: dict[str, Any], raw: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "metric_confidence_intervals": metric_confidence_intervals(raw),
        "failure_analysis": failure_analysis(raw),
        "pareto_recommendations": pareto_recommendations(raw),
        "ibr_robustness": ibr_robustness(raw),
        "pi_gru_generalization": pi_gru_generalization(raw),
        "classical_competitiveness": classical_competitiveness(raw),
        "ranking_sensitivity": ranking_sensitivity(raw),
        "tuning_policy_summary": tuning_policy_summary(report),
    }
    paths: dict[str, str] = {}
    for name, table in tables.items():
        path = output_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        paths[name] = str(path)
    return paths
