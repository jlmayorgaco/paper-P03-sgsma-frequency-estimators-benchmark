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


def test_atlas_builds_p0_nondirectional_sweeps(monkeypatch) -> None:
    monkeypatch.setenv("ATLAS_HARMONICS_THD_LEVELS_PCT", "5")
    monkeypatch.setenv("ATLAS_INTERHARMONIC_LEVELS_PCT", "2")
    monkeypatch.setenv("ATLAS_NOISE_SIGMA_LEVELS_PU", "0.001")

    scenarios = atlas_sweep.build_atlas_scenarios(["harmonics", "interharmonics", "noise_snr"])

    assert len(scenarios) == 3
    assert {item.sweep_key for item in scenarios} == {"harmonics", "interharmonics", "noise_snr"}
    assert {item.direction for item in scenarios} == {"level"}
    assert all(item.scenario_name.startswith("Atlas_") for item in scenarios)


def test_atlas_p0_scenarios_isolate_primary_disturbance(monkeypatch) -> None:
    monkeypatch.setenv("ATLAS_HARMONICS_THD_LEVELS_PCT", "5")
    monkeypatch.setenv("ATLAS_INTERHARMONIC_LEVELS_PCT", "2")
    monkeypatch.setenv("ATLAS_NOISE_SIGMA_LEVELS_PU", "0.001")

    scenarios = {item.sweep_key: item for item in atlas_sweep.build_atlas_scenarios(["p0"])}
    harmonics = scenarios["harmonics"].scenario_cls.get_default_params()
    interharmonics = scenarios["interharmonics"].scenario_cls.get_default_params()
    noise = scenarios["noise_snr"].scenario_cls.get_default_params()

    assert harmonics["freq_step_hz"] == 0.0
    assert harmonics["ih325_pct"] == 0.0
    assert harmonics["ih85_pct"] == 0.0
    assert harmonics["impulse_prob"] == 0.0

    assert interharmonics["rocof_hz_s"] == 0.0
    assert interharmonics["h5_pct"] == 0.0
    assert interharmonics["ih75_pct"] == 0.02

    assert noise["freq_hz"] == 60.0
    assert noise["noise_sigma"] == 0.001
