from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _build_manifest(input_json: Path, stats_json: Path | None, output_dir: Path, report_paths: list[Path]) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    tracked = [input_json]
    if stats_json and stats_json.exists():
        tracked.append(stats_json)
    tracked.extend(report_paths)
    for p in tracked:
        if p.exists():
            files.append({"path": str(p), "sha256": _sha256(p), "size_bytes": p.stat().st_size})
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_input_json": str(input_json),
        "source_stats_json": str(stats_json) if stats_json else None,
        "output_dir": str(output_dir),
        "files": files,
    }


def build_benchmark_report(input_json: Path, output_dir: Path, stats_json: Path | None = None) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(input_json.read_text(encoding="utf-8"))
    stats_data = None
    if stats_json and stats_json.exists():
        stats_data = json.loads(stats_json.read_text(encoding="utf-8"))

    run_cfg = data.get("run_configuration", {})
    trends = data.get("advanced_analysis", {}).get("trends", {})
    robust = data.get("advanced_analysis", {}).get("robust_statistics", {})
    andes_block = data.get("andes_ieee39", {})

    md_lines = [
        "# OpenFreqBench Benchmark Report",
        "",
        "## Run Summary",
        f"- Benchmark identity: `{run_cfg.get('benchmark_identity', 'unknown')}`",
        f"- Scenarios: `{len(run_cfg.get('scenarios', []))}`",
        f"- Estimators loaded: `{len(data.get('estimators_loaded', []))}`",
        f"- MC runs: `{run_cfg.get('n_mc_runs', 'unknown')}`",
        "",
        "## Global Trends",
        f"- RMSE scenario winners available: `{len(trends.get('scenario_winners_rmse', []))}`",
        f"- Family wins by RMSE: `{trends.get('family_wins_by_rmse', {})}`",
        "",
        "## Robust Statistics",
        f"- Bootstrap config: `{robust.get('config', {})}`",
        f"- Friedman RMSE: `{robust.get('friedman_rmse', {})}`",
        f"- Friedman CPU: `{robust.get('friedman_cpu', {})}`",
        "",
        "## ANDES IEEE39",
        f"- Included: `{bool(andes_block)}`",
        f"- Events: `{andes_block.get('event_count', 0) if isinstance(andes_block, dict) else 0}`",
        "",
        "## Limitations",
        "- This report is generated automatically from canonical artifacts.",
        "- Claims must be cross-checked against raw artifacts and statistical outputs.",
    ]

    if stats_data:
        md_lines.extend(
            [
                "",
                "## Hypothesis Testing",
                f"- Hypotheses executed: `{stats_data.get('n_hypotheses', 'unknown')}`",
                f"- Result rows: `{len(stats_data.get('results', []))}`",
            ]
        )

    md_path = output_dir / "benchmark_report.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    html_path = output_dir / "benchmark_report.html"
    html_body = "<html><body><pre>" + "\n".join(md_lines).replace("&", "&amp;").replace("<", "&lt;") + "</pre></body></html>"
    html_path.write_text(html_body, encoding="utf-8")

    appendix_path = output_dir / "benchmark_appendix_tables.csv"
    rows = data.get("aggregated_metrics", [])
    if rows:
        fieldnames = sorted({k for r in rows for k in r.keys()})
        with appendix_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
    else:
        appendix_path.write_text("no_data\n", encoding="utf-8")

    manifest = _build_manifest(
        input_json=input_json,
        stats_json=stats_json,
        output_dir=output_dir,
        report_paths=[md_path, html_path, appendix_path],
    )
    manifest_path = output_dir / "benchmark_run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "markdown": str(md_path),
        "html": str(html_path),
        "appendix_csv": str(appendix_path),
        "run_manifest": str(manifest_path),
    }

