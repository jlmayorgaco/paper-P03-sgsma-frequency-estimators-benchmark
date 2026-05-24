# Architecture

OpenFreqBench V2 is a thin public platform layer around the validated benchmark
engine.

## Layers

1. `openfreqbench.cli`: user-facing commands.
2. `openfreqbench.config`: YAML parsing and guardrails.
3. `openfreqbench.registry`: canonical scenarios, estimators, and metric profile.
4. `openfreqbench.runner`: selected matrix execution and public artifacts.
5. `openfreqbench.reproducibility`: git, dependency, source, and checkpoint hashes.
6. `analysis`, `estimators`, `scenarios`, `pipelines`, `plotting`: current benchmark core.

## Locked metric profile

The active metric profile is `canonical-single-phase-v1`. YAML files can select
metrics by id, but cannot redefine formulas. The implementation remains in
`analysis.metrics.calculate_all_metrics`.

## Extension path

Single-phase support is the first public release. Three-phase support should add
a new profile only after the signal container, scenario contract, and metric
adapter are explicit. WAMS support should live behind network-simulation extras
(`andes`, `opendss`) and must preserve reproducible event manifests.
