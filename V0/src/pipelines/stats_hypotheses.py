from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: str
    title: str
    metric: str
    group_a: str
    group_b: str
    alpha: float
    correction: str
    mode: str
    test: str


def _load_schema(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Hypothesis schema not found: {path}")
    schema = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(schema, dict):
        raise ValueError("Hypothesis schema must be a mapping.")
    return schema


def _validate_hypothesis_record(item: dict[str, Any], schema: dict[str, Any]) -> None:
    required = list(schema.get("required_fields", []))
    allowed = dict(schema.get("allowed", {}))
    for key in required:
        if key not in item:
            raise ValueError(f"Hypothesis missing required field: {key}")

    correction = str(item.get("correction", "")).lower()
    if correction not in set(allowed.get("correction", [])):
        raise ValueError(f"Invalid correction={correction!r}. Allowed: {allowed.get('correction')}")

    mode = str(item.get("mode", "")).lower()
    if mode not in set(allowed.get("mode", [])):
        raise ValueError(f"Invalid mode={mode!r}. Allowed: {allowed.get('mode')}")

    test = str(item.get("test", "")).lower()
    if test not in set(allowed.get("test", [])):
        raise ValueError(f"Invalid test={test!r}. Allowed: {allowed.get('test')}")

    alpha = float(item.get("alpha", 0.05))
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1), got {alpha}")


def _load_hypotheses(path: Path, schema_path: Path) -> list[Hypothesis]:
    schema = _load_schema(schema_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = data.get("hypotheses", [])
    out: list[Hypothesis] = []
    for item in rows:
        if not isinstance(item, dict):
            raise ValueError("Each hypothesis entry must be a mapping.")
        _validate_hypothesis_record(item, schema)
        out.append(
            Hypothesis(
                hypothesis_id=str(item["id"]),
                title=str(item.get("title", item["id"])),
                metric=str(item["metric"]),
                group_a=str(item["group_a"]),
                group_b=str(item["group_b"]),
                alpha=float(item.get("alpha", 0.05)),
                correction=str(item.get("correction", "holm")).lower(),
                mode=str(item.get("mode", "preregistered")).lower(),
                test=str(item.get("test", "mw")).lower(),
            )
        )
    return out


def _extract_long_df(report_json: dict[str, Any]) -> pd.DataFrame:
    rows = report_json.get("raw_run_records", [])
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _family_from_estimator(report_json: dict[str, Any]) -> dict[str, str]:
    mapping = report_json.get("run_configuration", {}).get("estimator_families", {})
    return {str(k): str(v) for k, v in mapping.items()}


def _resolve_selector(df: pd.DataFrame, family_map: dict[str, str], selector: str) -> pd.Series:
    if selector.startswith("family:"):
        target = selector.split(":", 1)[1].strip()
        fam_col = df["estimator"].map(lambda x: family_map.get(str(x), "Unknown"))
        return fam_col == target
    if selector.startswith("estimator:"):
        target = selector.split(":", 1)[1].strip()
        return df["estimator"].astype(str) == target
    if selector.startswith("scenario:"):
        target = selector.split(":", 1)[1].strip()
        return df["scenario"].astype(str) == target
    if selector == "all":
        return pd.Series(np.ones(len(df), dtype=bool), index=df.index)
    return pd.Series(np.zeros(len(df), dtype=bool), index=df.index)


def _mann_whitney_u(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    # Normal approximation for large enough samples.
    n1 = len(x)
    n2 = len(y)
    if n1 < 2 or n2 < 2:
        return math.nan, math.nan
    values = np.concatenate([x, y])
    ranks = pd.Series(values).rank(method="average").to_numpy(dtype=float)
    r1 = np.sum(ranks[:n1])
    u1 = r1 - (n1 * (n1 + 1) / 2.0)
    mu = n1 * n2 / 2.0
    sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    if sigma <= 0:
        return float(u1), math.nan
    z = (u1 - mu) / sigma
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return float(u1), float(p)


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return math.nan
    greater = 0
    lower = 0
    for xi in x:
        greater += int(np.sum(xi > y))
        lower += int(np.sum(xi < y))
    return (greater - lower) / float(len(x) * len(y))


def _bootstrap_ci_of_mean_diff(x: np.ndarray, y: np.ndarray, iters: int = 1000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(iters):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        diffs.append(float(np.mean(xb) - np.mean(yb)))
    lo, hi = np.quantile(np.asarray(diffs, dtype=float), [0.025, 0.975])
    return float(lo), float(hi)


def _adjust_holm(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.zeros(m, dtype=float)
    for i, idx in enumerate(order):
        adjusted[idx] = min(1.0, (m - i) * p_values[idx])
    # Monotonicity fix
    for i in range(1, m):
        adjusted[order[i]] = max(adjusted[order[i]], adjusted[order[i - 1]])
    return adjusted.tolist()


def _adjust_bh(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.zeros(m, dtype=float)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        idx = int(order[i])
        rank = i + 1
        val = min(prev, p_values[idx] * m / rank)
        adjusted[idx] = val
        prev = val
    return adjusted.tolist()


def run_hypotheses(
    hypotheses_path: Path,
    schema_path: Path,
    input_json_path: Path,
    output_dir: Path,
    allow_exploratory: bool = False,
    require_canonical_input: bool = True,
) -> dict[str, Any]:
    hypotheses = _load_hypotheses(hypotheses_path, schema_path)
    repo_root = Path(__file__).resolve().parents[2]
    canonical_json = repo_root / "artifacts" / "full_mc_benchmark" / "benchmark_full_report.json"
    if require_canonical_input and input_json_path.resolve() != canonical_json.resolve():
        raise RuntimeError(
            "Stats must run on canonical benchmark_full_report.json. "
            f"Got {input_json_path}"
        )
    report_json = json.loads(input_json_path.read_text(encoding="utf-8"))
    df = _extract_long_df(report_json)
    family_map = _family_from_estimator(report_json)

    if df.empty:
        raise RuntimeError("No raw_run_records available in benchmark JSON report.")

    results: list[dict[str, Any]] = []
    for h in hypotheses:
        if h.mode == "exploratory" and not allow_exploratory:
            results.append(
                {
                    "id": h.hypothesis_id,
                    "title": h.title,
                    "status": "blocked",
                    "reason": "Exploratory hypothesis blocked by guardrail (set allow_exploratory=true).",
                    "mode": h.mode,
                }
            )
            continue

        if h.metric not in df.columns:
            results.append(
                {
                    "id": h.hypothesis_id,
                    "title": h.title,
                    "status": "skipped",
                    "reason": f"Metric {h.metric} not found",
                    "mode": h.mode,
                }
            )
            continue

        mask_a = _resolve_selector(df, family_map, h.group_a)
        mask_b = _resolve_selector(df, family_map, h.group_b)
        xa = df.loc[mask_a, h.metric].dropna().to_numpy(dtype=float)
        xb = df.loc[mask_b, h.metric].dropna().to_numpy(dtype=float)

        if len(xa) < 3 or len(xb) < 3:
            results.append(
                {
                    "id": h.hypothesis_id,
                    "title": h.title,
                    "status": "skipped",
                    "reason": "Insufficient samples",
                    "mode": h.mode,
                    "n_a": int(len(xa)),
                    "n_b": int(len(xb)),
                }
            )
            continue

        stat, p_raw = _mann_whitney_u(xa, xb)
        eff = _cliffs_delta(xa, xb)
        ci_lo, ci_hi = _bootstrap_ci_of_mean_diff(xa, xb)

        results.append(
            {
                "id": h.hypothesis_id,
                "title": h.title,
                "mode": h.mode,
                "metric": h.metric,
                "group_a": h.group_a,
                "group_b": h.group_b,
                "test": h.test,
                "correction": h.correction,
                "alpha": h.alpha,
                "status": "ok",
                "n_a": int(len(xa)),
                "n_b": int(len(xb)),
                "statistic": stat,
                "p_value_raw": p_raw,
                "effect_cliffs_delta": eff,
                "ci95_mean_diff_lo": ci_lo,
                "ci95_mean_diff_hi": ci_hi,
                "mean_a": float(np.mean(xa)),
                "mean_b": float(np.mean(xb)),
                "mean_diff_a_minus_b": float(np.mean(xa) - np.mean(xb)),
            }
        )

    # Apply correction by family of correction types.
    for corr in ("holm", "bh"):
        idx = [i for i, r in enumerate(results) if r.get("status") == "ok" and r.get("correction") == corr]
        if not idx:
            continue
        p = [float(results[i]["p_value_raw"]) for i in idx]
        adj = _adjust_holm(p) if corr == "holm" else _adjust_bh(p)
        for k, i in enumerate(idx):
            results[i]["p_value_adj"] = float(adj[k])
            alpha = float(results[i]["alpha"])
            results[i]["reject_h0"] = bool(adj[k] < alpha)

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "statistical_tests_report.json"
    csv_path = output_dir / "statistical_tests_report.csv"
    md_path = output_dir / "statistical_tests_report.md"

    payload = {
        "source_report": str(input_json_path),
        "source_hypotheses": str(hypotheses_path),
        "source_schema": str(schema_path),
        "n_hypotheses": len(hypotheses),
        "allow_exploratory": bool(allow_exploratory),
        "n_blocked": int(sum(1 for r in results if r.get("status") == "blocked")),
        "n_skipped": int(sum(1 for r in results if r.get("status") == "skipped")),
        "n_ok": int(sum(1 for r in results if r.get("status") == "ok")),
        "results": results,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fieldnames = sorted({k for row in results for k in row.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    lines = [
        "# Statistical Tests Report",
        "",
        f"- Hypotheses file: `{hypotheses_path}`",
        f"- Benchmark report: `{input_json_path}`",
        f"- Total hypotheses: **{len(hypotheses)}**",
        "",
        "| id | metric | groups | p(raw) | p(adj) | reject | effect | mode |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in results:
        if row.get("status") != "ok":
            lines.append(
                f"| {row.get('id')} | - | - | - | - | - | - | {row.get('mode', '-') } |"
            )
            continue
        lines.append(
            "| {id} | {metric} | {ga} vs {gb} | {p0:.4g} | {p1:.4g} | {rej} | {eff:.4g} | {mode} |".format(
                id=row["id"],
                metric=row["metric"],
                ga=row["group_a"],
                gb=row["group_b"],
                p0=float(row["p_value_raw"]),
                p1=float(row.get("p_value_adj", math.nan)),
                rej=int(bool(row.get("reject_h0", False))),
                eff=float(row.get("effect_cliffs_delta", math.nan)),
                mode=row.get("mode", "preregistered"),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "json_path": str(json_path),
        "csv_path": str(csv_path),
        "md_path": str(md_path),
        "n_results": len(results),
    }
