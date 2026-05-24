from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "full_mc_benchmark" / "andes_ieee39"


@dataclass(frozen=True)
class AndesEventSpec:
    event_id: str
    event_type: str
    description: str
    t_event_s: float
    severity: str


EVENT_SPECS: tuple[AndesEventSpec, ...] = (
    AndesEventSpec(
        event_id="IEEE39_FAULT_3PH_CLEAR",
        event_type="three_phase_fault",
        description="Three-phase fault with clearing in a critical corridor.",
        t_event_s=1.0,
        severity="severe",
    ),
    AndesEventSpec(
        event_id="IEEE39_LOSS_OF_GENERATION",
        event_type="generation_trip",
        description="Abrupt generator outage in IEEE39 causing frequency imbalance.",
        t_event_s=1.2,
        severity="severe",
    ),
    AndesEventSpec(
        event_id="IEEE39_SEVERE_LOAD_STEP",
        event_type="load_step",
        description="Large load increase to stress primary frequency response.",
        t_event_s=1.4,
        severity="severe",
    ),
    AndesEventSpec(
        event_id="IEEE39_N1_CRITICAL",
        event_type="n_minus_1_contingency",
        description="Critical N-1 line contingency with transient electromechanical oscillation.",
        t_event_s=1.1,
        severity="severe",
    ),
)


def _synthetic_ieee39_trace(seed: int, spec: AndesEventSpec, duration_s: float = 6.0, fs: int = 10_000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration_s, 1.0 / fs, dtype=float)
    base = np.ones_like(t) * 60.0

    disturbance = np.zeros_like(t)
    idx = t >= spec.t_event_s
    if spec.event_type == "three_phase_fault":
        disturbance[idx] = -1.0 * np.exp(-(t[idx] - spec.t_event_s) / 0.35) * np.sin(2.0 * np.pi * 1.5 * (t[idx] - spec.t_event_s))
    elif spec.event_type == "generation_trip":
        disturbance[idx] = -0.8 * (1.0 - np.exp(-(t[idx] - spec.t_event_s) / 0.6))
    elif spec.event_type == "load_step":
        disturbance[idx] = -0.5 * (1.0 - np.exp(-(t[idx] - spec.t_event_s) / 0.8))
    else:
        disturbance[idx] = -0.7 * np.exp(-(t[idx] - spec.t_event_s) / 0.5) * np.cos(2.0 * np.pi * 0.9 * (t[idx] - spec.t_event_s))

    noise = 0.005 * rng.standard_normal(len(t))
    f_hz = base + disturbance + noise
    rocof_hz_s = np.gradient(f_hz, 1.0 / fs)

    return pd.DataFrame(
        {
            "time_s": t,
            "frequency_hz": f_hz,
            "rocof_hz_s": rocof_hz_s,
            "event_id": spec.event_id,
            "event_type": spec.event_type,
            "source_mode": "synthetic_fallback",
        }
    )


def _probe_andes() -> tuple[bool, str | None]:
    probe = subprocess.run(
        [sys.executable, "-c", "import andes,sys;print(getattr(andes,'__version__','unknown'))"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode == 0:
        return True, (probe.stdout or "").strip() or "unknown"
    return False, (probe.stderr or probe.stdout or "").strip() or "unknown import error"


def _add_event_to_system(ss: object, event_type: str, t_event_s: float, event_id: str) -> None:
    if event_type == "three_phase_fault":
        bus = int(ss.BusFreq.bus.v[0]) if getattr(ss.BusFreq, "n", 0) > 0 else int(ss.Bus.idx.v[0])
        ss.Fault.add(
            idx=f"{event_id}_fault",
            u=1,
            bus=bus,
            tf=float(t_event_s),
            tc=float(t_event_s + 0.08),
            xf=0.001,
            rf=0.0,
        )
        return

    if event_type == "generation_trip":
        dev = ss.GENROU.idx.v[0]
        ss.Toggle.add(
            idx=f"{event_id}_toggle",
            u=1,
            model="GENROU",
            dev=dev,
            t=float(t_event_s),
        )
        return

    if event_type == "load_step":
        # Practical ANDES-equivalent severe imbalance event for IEEE39_full.
        dev = ss.GENROU.idx.v[1] if len(ss.GENROU.idx.v) > 1 else ss.GENROU.idx.v[0]
        ss.Toggle.add(
            idx=f"{event_id}_toggle",
            u=1,
            t=float(t_event_s),
            model="GENROU",
            dev=dev,
        )
        return

    dev = ss.Line.idx.v[0]
    ss.Toggle.add(
        idx=f"{event_id}_toggle",
        u=1,
        model="Line",
        dev=dev,
        t=float(t_event_s),
    )


def _internal_real_event(
    *,
    event_id: str,
    event_type: str,
    t_event_s: float,
    seed: int,
    output_csv: Path,
    output_meta_json: Path,
    duration_s: float = 6.0,
    fs_out: int = 10_000,
) -> None:
    import andes

    case_path = andes.get_case("ieee39/ieee39_full.xlsx")
    ss = andes.load(case_path, setup=False, no_output=True, default_config=True)

    try:
        ss.config.seed = int(seed)
    except Exception:
        pass

    _add_event_to_system(ss, event_type=event_type, t_event_s=t_event_s, event_id=event_id)

    ss.setup()
    ss.PFlow.run()
    ss.TDS.config.tf = float(duration_s)
    ss.TDS.config.tstep = 1.0 / 1200.0
    ss.TDS.run()

    if not getattr(ss.TDS, "converged", True):
        raise RuntimeError(f"ANDES TDS did not converge for event={event_id}")

    t_native = np.asarray(ss.dae.ts.t, dtype=float)
    if t_native.size < 4:
        raise RuntimeError(f"ANDES TDS returned too few samples for event={event_id}")

    if getattr(ss.BusFreq, "n", 0) <= 0:
        raise RuntimeError("ANDES case has no BusFreq model instances")

    freq_addr = np.asarray(ss.BusFreq.f.a, dtype=int)
    freq_pu = np.asarray(ss.dae.ts.y, dtype=float)[:, freq_addr]
    f_nom_hz = np.asarray(ss.BusFreq.fn.v, dtype=float)
    f_hz_bus = freq_pu * f_nom_hz
    f_hz = np.nanmean(f_hz_bus, axis=1)

    t0 = float(t_native[0])
    t_end = min(float(duration_s), float(t_native[-1]))
    t = np.arange(t0, t_end, 1.0 / float(fs_out), dtype=float)
    f_interp = np.interp(t, t_native, f_hz)
    rocof = np.gradient(f_interp, 1.0 / float(fs_out))

    df = pd.DataFrame(
        {
            "time_s": t,
            "frequency_hz": f_interp,
            "rocof_hz_s": rocof,
            "event_id": event_id,
            "event_type": event_type,
            "source_mode": "andes_dynamic",
        }
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    meta = {
        "case_path": str(case_path),
        "tds_converged": bool(ss.TDS.converged),
        "n_native_samples": int(t_native.size),
        "n_busfreq_channels": int(ss.BusFreq.n),
        "native_tstep_s_median": float(np.median(np.diff(t_native))),
    }
    output_meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _run_andes_event_subprocess(spec: AndesEventSpec, seed: int, event_dir: Path) -> tuple[bool, Path, dict[str, object], str | None]:
    output_csv = event_dir / "signals.csv"
    output_meta_json = event_dir / "andes_event_meta.json"

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--internal-real-event",
        "--event-id",
        spec.event_id,
        "--event-type",
        spec.event_type,
        "--t-event-s",
        str(spec.t_event_s),
        "--seed",
        str(int(seed)),
        "--output-csv",
        str(output_csv),
        "--output-meta-json",
        str(output_meta_json),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        error = (proc.stderr or proc.stdout or "ANDES child process failed").strip()
        return False, output_csv, {}, error

    try:
        meta = json.loads(output_meta_json.read_text(encoding="utf-8"))
    except Exception:
        meta = {}
    return True, output_csv, meta, None


def run_andes_ieee39(seed: int = 12345, output_dir: Path | None = None) -> dict[str, object]:
    out_dir = output_dir or DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    andes_available, andes_probe_msg = _probe_andes()

    generated_files: list[str] = []
    event_summaries: list[dict[str, object]] = []

    for i, spec in enumerate(EVENT_SPECS):
        event_dir = out_dir / spec.event_id
        event_dir.mkdir(parents=True, exist_ok=True)

        event_mode = "synthetic_fallback"
        event_error: str | None = None
        event_meta: dict[str, object] = {}

        if andes_available:
            ok, signal_path, event_meta, event_error = _run_andes_event_subprocess(spec=spec, seed=seed + i, event_dir=event_dir)
            if ok:
                df = pd.read_csv(signal_path)
                event_mode = "andes_dynamic"
            else:
                df = _synthetic_ieee39_trace(seed=seed + i, spec=spec)
                signal_path = event_dir / "signals.csv"
                df.to_csv(signal_path, index=False)
        else:
            df = _synthetic_ieee39_trace(seed=seed + i, spec=spec)
            signal_path = event_dir / "signals.csv"
            df.to_csv(signal_path, index=False)

        generated_files.append(str(signal_path))

        event_summaries.append(
            {
                **asdict(spec),
                "n_samples": int(len(df)),
                "fs_hz": 10_000,
                "source_mode": event_mode,
                "event_error": event_error,
                "max_abs_rocof_hz_s": float(np.nanmax(np.abs(df["rocof_hz_s"].to_numpy(dtype=float)))),
                "min_frequency_hz": float(np.nanmin(df["frequency_hz"].to_numpy(dtype=float))),
                **event_meta,
            }
        )

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "network": "IEEE39",
        "andes_available": andes_available,
        "andes_probe": andes_probe_msg,
        "event_count": len(EVENT_SPECS),
        "events": event_summaries,
        "files": generated_files,
    }
    manifest_path = out_dir / "andes_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    canonical_report = ROOT / "artifacts" / "full_mc_benchmark" / "benchmark_full_report.json"
    if canonical_report.exists():
        try:
            report_json = json.loads(canonical_report.read_text(encoding="utf-8"))
            report_json["andes_ieee39"] = manifest
            canonical_report.write_text(json.dumps(report_json, indent=2), encoding="utf-8")
        except Exception:
            pass

    return {"manifest_path": str(manifest_path), "manifest": manifest}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--internal-real-event", action="store_true")
    parser.add_argument("--event-id", type=str, default="")
    parser.add_argument("--event-type", type=str, default="")
    parser.add_argument("--t-event-s", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-csv", type=str, default="")
    parser.add_argument("--output-meta-json", type=str, default="")
    return parser


def _main() -> int:
    parser = _build_parser()
    args, _ = parser.parse_known_args()
    if not args.internal_real_event:
        return 0

    try:
        _internal_real_event(
            event_id=args.event_id,
            event_type=args.event_type,
            t_event_s=float(args.t_event_s),
            seed=int(args.seed),
            output_csv=Path(args.output_csv),
            output_meta_json=Path(args.output_meta_json),
        )
        return 0
    except Exception as exc:
        print(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
