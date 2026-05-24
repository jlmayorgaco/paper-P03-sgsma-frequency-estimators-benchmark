from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None


@dataclass(frozen=True)
class AdvancedStatsConfig:
    bootstrap_iters: int = 2000
    alpha: float = 0.05
    random_seed: int = 42


class AdvancedBenchmarkAnalyzer:
    """
    Advanced post-hoc statistics for Monte Carlo benchmark outputs.
    """

    def __init__(self, config: AdvancedStatsConfig | None = None) -> None:
        self.config = config or AdvancedStatsConfig()
        self._rng = np.random.default_rng(self.config.random_seed)

    @staticmethod
    def _clean_values(values: np.ndarray | list[float]) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        return arr

    def bootstrap_ci_mean(self, values: np.ndarray | list[float]) -> dict[str, float | int | bool]:
        arr = self._clean_values(values)
        if arr.size == 0:
            return {"available": False, "reason": "empty"}

        if arr.size == 1:
            val = float(arr[0])
            return {
                "available": True,
                "n": int(arr.size),
                "mean": val,
                "ci_low": val,
                "ci_high": val,
                "alpha": float(self.config.alpha),
                "bootstrap_iters": int(self.config.bootstrap_iters),
            }

        means = np.empty(self.config.bootstrap_iters, dtype=float)
        for i in range(self.config.bootstrap_iters):
            sample = self._rng.choice(arr, size=arr.size, replace=True)
            means[i] = float(np.mean(sample))
        low_q = 100.0 * (self.config.alpha / 2.0)
        high_q = 100.0 * (1.0 - self.config.alpha / 2.0)
        return {
            "available": True,
            "n": int(arr.size),
            "mean": float(np.mean(arr)),
            "ci_low": float(np.percentile(means, low_q)),
            "ci_high": float(np.percentile(means, high_q)),
            "alpha": float(self.config.alpha),
            "bootstrap_iters": int(self.config.bootstrap_iters),
        }

    @staticmethod
    def cliffs_delta(x: np.ndarray | list[float], y: np.ndarray | list[float]) -> dict[str, float | int | str]:
        ax = np.asarray(x, dtype=float)
        ay = np.asarray(y, dtype=float)
        ax = ax[np.isfinite(ax)]
        ay = ay[np.isfinite(ay)]
        if ax.size == 0 or ay.size == 0:
            return {"available": False, "reason": "empty"}

        diffs = ax[:, None] - ay[None, :]
        gt = float(np.sum(diffs > 0.0))
        lt = float(np.sum(diffs < 0.0))
        total = float(ax.size * ay.size)
        delta = (gt - lt) / total
        ad = abs(delta)
        if ad < 0.147:
            magnitude = "negligible"
        elif ad < 0.33:
            magnitude = "small"
        elif ad < 0.474:
            magnitude = "medium"
        else:
            magnitude = "large"
        return {
            "available": True,
            "delta": float(delta),
            "magnitude": magnitude,
            "n_x": int(ax.size),
            "n_y": int(ay.size),
        }

    def pairwise_win_rate(
        self,
        df_agg: pd.DataFrame,
        metric_mean_col: str,
        lower_better: bool = True,
    ) -> dict[str, Any]:
        if df_agg.empty or metric_mean_col not in df_agg.columns:
            return {"available": False, "reason": "missing metric"}

        pivot = df_agg.pivot_table(
            index="scenario",
            columns="estimator",
            values=metric_mean_col,
            aggfunc="mean",
        )
        if pivot.empty:
            return {"available": False, "reason": "empty pivot"}

        estimators = list(pivot.columns)
        matrix: list[dict[str, Any]] = []
        for est_i in estimators:
            row: dict[str, Any] = {"estimator": est_i}
            for est_j in estimators:
                if est_i == est_j:
                    row[est_j] = 0.5
                    continue
                pair = pivot[[est_i, est_j]].dropna()
                if pair.empty:
                    row[est_j] = None
                    continue
                if lower_better:
                    wins = float((pair[est_i] < pair[est_j]).mean())
                else:
                    wins = float((pair[est_i] > pair[est_j]).mean())
                row[est_j] = wins
            matrix.append(row)
        return {"available": True, "metric": metric_mean_col, "matrix": matrix}

    def estimator_bootstrap_summary(
        self,
        df_long: pd.DataFrame,
        metric_col: str,
    ) -> dict[str, Any]:
        if df_long.empty or metric_col not in df_long.columns:
            return {"available": False, "reason": "missing metric"}

        out: list[dict[str, Any]] = []
        for est_name, df_est in df_long.groupby("estimator"):
            stats = self.bootstrap_ci_mean(df_est[metric_col].to_numpy(dtype=float))
            stats["estimator"] = str(est_name)
            out.append(stats)
        out.sort(key=lambda d: d.get("mean", np.inf))
        return {"available": True, "metric": metric_col, "estimators": out}

    def friedman_test(
        self,
        df_agg: pd.DataFrame,
        metric_mean_col: str,
    ) -> dict[str, Any]:
        if scipy_stats is None:
            return {"available": False, "reason": "scipy not installed"}
        if df_agg.empty or metric_mean_col not in df_agg.columns:
            return {"available": False, "reason": "missing metric"}

        pivot = df_agg.pivot_table(
            index="scenario",
            columns="estimator",
            values=metric_mean_col,
            aggfunc="mean",
        ).dropna(axis=0, how="any")
        if pivot.shape[0] < 2 or pivot.shape[1] < 3:
            return {"available": False, "reason": "insufficient data"}

        arrays = [pivot[col].to_numpy(dtype=float) for col in pivot.columns]
        stat, p_val = scipy_stats.friedmanchisquare(*arrays)
        return {
            "available": True,
            "metric": metric_mean_col,
            "n_scenarios": int(pivot.shape[0]),
            "n_estimators": int(pivot.shape[1]),
            "statistic": float(stat),
            "p_value": float(p_val),
            "reject_H0_alpha_0p05": bool(p_val < 0.05),
            "estimators": [str(c) for c in pivot.columns],
        }

    def pairwise_wilcoxon_vs_best(
        self,
        df_agg: pd.DataFrame,
        metric_mean_col: str,
        lower_better: bool = True,
    ) -> dict[str, Any]:
        if scipy_stats is None:
            return {"available": False, "reason": "scipy not installed"}
        if df_agg.empty or metric_mean_col not in df_agg.columns:
            return {"available": False, "reason": "missing metric"}

        pivot = df_agg.pivot_table(
            index="scenario",
            columns="estimator",
            values=metric_mean_col,
            aggfunc="mean",
        ).dropna(axis=0, how="any")
        if pivot.shape[0] < 5 or pivot.shape[1] < 2:
            return {"available": False, "reason": "insufficient data"}

        mean_vals = pivot.mean(axis=0)
        best = str(mean_vals.idxmin() if lower_better else mean_vals.idxmax())
        results: list[dict[str, Any]] = []

        for est in pivot.columns:
            est = str(est)
            if est == best:
                continue
            x = pivot[best].to_numpy(dtype=float)
            y = pivot[est].to_numpy(dtype=float)
            try:
                stat, p_val = scipy_stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
                effect = self.cliffs_delta(x, y)
                results.append(
                    {
                        "best_estimator": best,
                        "other_estimator": est,
                        "statistic": float(stat),
                        "p_value": float(p_val),
                        "reject_H0_alpha_0p05": bool(p_val < 0.05),
                        "cliffs_delta": effect,
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "best_estimator": best,
                        "other_estimator": est,
                        "error": str(exc),
                    }
                )
        return {"available": True, "metric": metric_mean_col, "best_estimator": best, "comparisons": results}

    def dominance_score(
        self,
        df_agg: pd.DataFrame,
        metric_cols: list[str],
        lower_better: bool = True,
    ) -> dict[str, Any]:
        if df_agg.empty:
            return {"available": False, "reason": "empty data"}
        use_cols = [c for c in metric_cols if c in df_agg.columns]
        if not use_cols:
            return {"available": False, "reason": "missing metrics"}

        grouped = df_agg.groupby("estimator")[use_cols].mean()
        ranks = grouped.rank(axis=0, method="average", ascending=lower_better)
        score = ranks.mean(axis=1).sort_values()
        return {
            "available": True,
            "metrics": use_cols,
            "scores": [
                {"estimator": str(est), "mean_rank_score": float(val)}
                for est, val in score.items()
            ],
        }
