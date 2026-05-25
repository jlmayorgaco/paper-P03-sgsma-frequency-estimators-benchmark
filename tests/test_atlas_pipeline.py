from __future__ import annotations

from pipelines import atlas_sweep


def test_atlas_builds_three_sweep_families(monkeypatch) -> None:
    monkeypatch.setenv("ATLAS_MAG_LEVELS_PCT", "5")
    monkeypatch.setenv("ATLAS_ROCOF_LEVELS_HZ_S", "0.5")
    monkeypatch.setenv("ATLAS_FREQSTEP_LEVELS_HZ", "0.1")
    scenarios = atlas_sweep.build_atlas_scenarios(["magnitude_step", "rocof", "frequency_step"])

    assert len(scenarios) == 6
    assert {item.sweep_key for item in scenarios} == {"magnitude_step", "rocof", "frequency_step"}
    assert {item.direction for item in scenarios} == {"pos", "neg"}
    assert all(item.scenario_name.startswith("Atlas_") for item in scenarios)


def test_atlas_parser_accepts_oracle_policy() -> None:
    parser = atlas_sweep.build_parser()
    args = parser.parse_args(["--sweeps", "rocof", "--policy", "oracle", "--n-runs", "2"])

    assert args.sweeps == "rocof"
    assert args.policy == "oracle"
    assert args.n_runs == 2
