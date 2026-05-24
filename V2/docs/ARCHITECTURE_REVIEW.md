# Architecture review

This review covers the MVP 2.0.0 package in `V2/`. The older `src/` tree remains
the paper benchmark source. The point of V2 is to expose that work through a
stable public interface without changing the metric definitions.

## Current state

The package has the right outer shape for public use:

- CLI commands cover quick tests, estimator comparison, YAML runs, reports,
  plots, hypotheses, manifests, and quality gates.
- YAML can choose estimators, scenarios, seeds, run counts, and parameter policy.
- Metric formulas are locked by `canonical-single-phase-v1`.
- Reports keep raw records, aggregates, plots, confidence intervals, and
  reproducibility metadata.
- `V0/` keeps the copied baseline. `V2/` is the public package line.

## Code issues to watch

`openfreqbench.runner` does too much. It loads estimators, resolves tuned
parameters, runs Monte Carlo, writes artifacts, aggregates metrics, and builds
the report payload. This is acceptable for the MVP, but it is the first module
to split when V2 grows.

`openfreqbench.reports` also has several roles: statistics, plotting, Markdown,
and JSON output. Keep it stable until the Journal run is reproduced. After that,
split plotting backends from statistical summaries.

`openfreqbench.cli` is still simple enough to keep in one file. Avoid adding
business logic there. CLI handlers should keep building config objects and call
library functions.

The old paper and sweep pipelines under `pipelines/` are large scripts. Do not
rewrite them before the paper results are reproduced. Treat them as audited
research workflows and wrap them with clearer entry points.

## Design rules

- Keep metrics platform-owned. A YAML file may select a metric, not redefine it.
- Keep estimator code replaceable. A researcher should be able to add an
  estimator without touching metrics, reports, or scenario code.
- Keep scenario generation separate from metric calculation.
- Keep raw results. Every plot or claim must trace back to CSV/JSON artifacts.
- Prefer small adapters over broad rewrites. The current benchmark has working
  scientific value; the package should preserve that.

## SOLID/KISS/DRY decisions

Single responsibility is enforced at the boundary level, not by tiny classes.
The public boundary is clear: config, registry, runner, reports, hypotheses,
quality, and reproducibility.

KISS matters more than abstraction here. A contributor should understand the
main path in one pass: YAML to config, config to runner, runner to report.

DRY is applied where repetition risks inconsistent science: metric ids,
scenario names, estimator labels, and artifact names. Plot styling and report
layout can repeat a little if that keeps each figure easy to audit.

Dependency inversion is limited to estimators and scenarios. Metrics are not a
plugin point in MVP 2.0.0 because changing metrics changes the benchmark.

## Refactor order

1. Freeze a Journal artifact set from the current code.
2. Split `runner` into estimator loading, execution, artifact writing, and
   aggregation modules.
3. Split `reports` into statistical tables, plots, and text summaries.
4. Add explicit contracts for estimator and scenario plugins.
5. Add new metric profiles only when three-phase or WAMS data needs them.

Do not start with step 2 if the paper numbers are still moving.
