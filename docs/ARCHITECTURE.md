# Architecture

OpenFreqBench 2.0.0 is a public CLI and artifact layer around the validated
single-phase benchmark engine. The package source lives under `src/`; the
historical layouts remain in older branches.

## Layers

| Layer | Role |
| --- | --- |
| `openfreqbench.cli` | user-facing commands |
| `openfreqbench.config` | YAML parsing and guardrails |
| `openfreqbench.registry` | canonical scenarios, estimators, and metric profile |
| `openfreqbench.runner` | matrix execution and public artifacts |
| `openfreqbench.reports` | summaries, tables, plots, and traceability files |
| `openfreqbench.hypotheses` | preregistered and exploratory hypothesis checks |
| `openfreqbench.reproducibility` | git, dependency, source, and checkpoint hashes |
| `analysis`, `estimators`, `scenarios`, `pipelines`, `plotting` | audited benchmark core |

## Locked metric profile

The active metric profile is `canonical-single-phase-v1`. YAML files can select
registered metric ids, but cannot redefine formulas. The implementation remains
in `analysis.metrics.calculate_all_metrics`.

This is a boundary decision, not a convenience feature. Estimators and scenarios
are research inputs. Metric formulas define the benchmark and stay
platform-owned.

## Design rules

- Keep metrics platform-owned.
- Keep estimator code replaceable.
- Keep scenario generation separate from metric calculation.
- Keep raw results. Every plot or claim must trace back to CSV/JSON artifacts.
- Prefer small adapters over broad rewrites until paper artifacts are frozen.

## Current debt

`openfreqbench.runner` still has several roles: estimator loading, tuned
parameter resolution, Monte Carlo execution, artifact writing, aggregation, and
report payload assembly. That is acceptable for MVP 2.0.0 because the journal
artifact path is still being frozen.

`openfreqbench.reports` also mixes statistics, plotting, Markdown, and JSON
output. Keep the public output contract stable first. Split internals later.

The scripts under `pipelines/` are audited research workflows. Treat them as
reproduction code with clearer entry points, not as plugin APIs.

## Extension path

Single-phase support is the first public release. Three-phase support should add
a new profile only after the signal container, scenario contract, and metric
adapter are explicit.

WAMS support should live behind network-simulation extras (`andes`, `opendss`)
and must preserve reproducible event manifests.

## Refactor order

1. Freeze a journal artifact set from the current code.
2. Split `runner` into estimator loading, execution, artifact writing, and
   aggregation modules.
3. Split `reports` into statistical tables, plots, and text summaries.
4. Add explicit contracts for estimator and scenario plugins.
5. Add new metric profiles only when three-phase or WAMS data needs them.
